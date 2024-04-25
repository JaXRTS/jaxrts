from jax import numpy as jnp
import jax.interpreters
import jaxrts
import jax
from functools import partial
import jpu

from pint import Quantity

from jaxrts.units import ureg

from typing import List, Callable


@jax.jit
def phi(t, k=2 * jnp.pi):
    return t / (1 - jnp.exp(-k * jnp.sinh(t)))


@jax.jit
def f1(x):
    return jnp.exp(-(x**2))

@jax.jit
def _sinfft(f):
    h = 0.001
    dk = (2 * jnp.pi)**2 / (r[1] - r[0])
    k = dk * jnp.arange(len(r))

    M = jnp.pi / h
    N_trunc = int(20 / h)
    n_eval = jnp.arange(-N_trunc, N_trunc, 1)

    eval_points = M * phi(n_eval * h)

    phi_prime = jax.vmap(jax.grad(phi))

    def for_each_k(k):
        f_eval = jnp.interp(eval_points / k, r, f)
        return jnp.pi * jnp.nansum(
            f_eval * jnp.sin(eval_points) * phi_prime(n_eval * h) / k
        )

    out = jax.vmap(for_each_k)(k)
    return out

def boringtap(f, r, k):
    return jax.scipy.integrate.trapezoid(f[:, jnp.newaxis] * jnp.sin(k[jnp.newaxis, :] * r[:, jnp.newaxis]), r[:, jnp.newaxis], axis = 0)

r = jnp.linspace(0.01, 1000000, 1000)
dk = (2 * jnp.pi)**3/ (r[1] - r[0])
k =  dk * jnp.arange(len(r))

import matplotlib.pyplot as plt
plt.plot(k, _sinfft(r/(1+r**2)), label = "test")
plt.plot(k, jnp.pi * jnp.exp(-1 * k) / 2, label="!!!")
plt.legend()
plt.show()


@jax.jit
def V_l(
    r: Quantity | jnp.ndarray, q: Quantity, alpha: Quantity
) -> Quantity | jnp.ndarray:
    """
    q**2 / (4 * jnp.pi * ureg.epsilon_0 * r) * (1 - jnp.exp(-alpha * r))
    """

    _q = q[:, :, jnp.newaxis]
    _alpha = alpha[:, :, jnp.newaxis]
    _r = r[jnp.newaxis, jnp.newaxis, :]

    return (
        _q
        / (4 * jnp.pi * ureg.epsilon_0 * _r)
        * (1 - jpu.numpy.exp(-_alpha * _r))
    )


@jax.jit
def V_s(
    r: Quantity | jnp.ndarray, q: Quantity, alpha: Quantity
) -> Quantity | jnp.ndarray:
    """
    q**2 / (4 * jnp.pi * ureg.epsilon_0 * r) * (jnp.exp(-alpha * r))
    """

    _q = q[:, :, jnp.newaxis]
    _alpha = alpha[:, :, jnp.newaxis]
    _r = r[jnp.newaxis, jnp.newaxis, :]

    return (
        _q / (4 * jnp.pi * ureg.epsilon_0 * _r) * (jpu.numpy.exp(-_alpha * _r))
    ).to(ureg.electron_volt)


@jax.jit
def construct_alpha_matrix(ne: jnp.ndarray | Quantity):

    d = jpu.numpy.cbrt(
        3 / (4 * jnp.pi * (ne[:, jnp.newaxis] + ne[jnp.newaxis, :]) / 2)
    )

    return 2 / d


@jax.jit
def construct_q_matrix(q: jnp.ndarray) -> jnp.ndarray:

    return jpu.numpy.outer(q, q)


# @jax.jit
def pair_distribution_function_HNC(V_s, V_l, r, Ti, ni):
    r = r.m_as(ureg.angstrom)

    @jax.jit
    def _sinfft(f):
        h = 0.001
        dk = jnp.pi / (r[1] - r[0])
        k = 2 * jnp.pi / r[-1] + dk * jnp.arange(len(r))

        M = jnp.pi / h
        N_trunc = int(2 / h)
        n_eval = jnp.arange(-N_trunc, N_trunc, 1)

        eval_points = M * phi(n_eval * h)

        phi_prime = jax.vmap(jax.grad(phi))

        def for_each_k(k):
            f_eval = jnp.interp(eval_points / k, r, f)
            return jnp.pi * jnp.nansum(
                f_eval * jnp.sin(eval_points) * phi_prime(n_eval * h) / k
            )

        out = jax.vmap(for_each_k)(k)
        return out

    sinfft = jax.vmap(jax.vmap(_sinfft, in_axes = 0, out_axes=0), in_axes = 1, out_axes=1)

    dr = r[1] - r[0]
    dk = jnp.pi / (len(r) * dr)

    k = 2 * jnp.pi / r[-1] + jnp.arange(len(r)) * dk

    beta = 1 / (ureg.boltzmann_constant * Ti)
    # g_r = (-beta * V_s).m_as(ureg.dimensionless) + 1cls
    g_r = jnp.exp(-(beta * (V_s + V_l)).m_as(ureg.dimensionless)) + 0j

    V_l_k = (
        (
            sinfft(
                r[jnp.newaxis, jnp.newaxis, :] * V_l.m_as(ureg.electron_volt)
            )
            * ureg.electron_volt
        )
        * ((4 * jnp.pi) / k[jnp.newaxis, jnp.newaxis, :])
    )

    delta = 1e-8

    Ns_r0 = jnp.zeros_like(g_r)

    d = jnp.eye(ni.shape[0]) * ni

    def ozr(input_vec):
        return (
            jnp.linalg.inv(
                jnp.eye(ni.shape[0])
                - input_vec @ d.m_as(1 / ureg.centimeter**3)
            )
            @ input_vec
        )

    def condition(val):
        g_r, g_r_old, _, n_iter = val
        return (n_iter < 1000) & jnp.all(
            jnp.max(jnp.abs(g_r - g_r_old)) > delta
        )

    def step(val):
        g_r, _, Ns_r, i = val

        h_r = g_r - 1

        cs_r = h_r - Ns_r

        cs_k = (
            sinfft(r[jnp.newaxis, jnp.newaxis, :] * cs_r)
            * (4 * jnp.pi)
            / k[jnp.newaxis, jnp.newaxis, :]
        )

        c_k = cs_k - (beta * V_l_k).m_as(ureg.dimensionless)

        # Ornstein-Zernike relation
        h_k = jax.vmap(ozr, in_axes=2, out_axes=2)(c_k)

        Ns_k = h_k - cs_k

        Ns_r_new = (
            sinfft(k[jnp.newaxis, jnp.newaxis, :] * Ns_k)
            * (4 * jnp.pi)
            / r[jnp.newaxis, jnp.newaxis, :] * 2 * jnp.pi / len(r)
        )

        print(Ns_r_new)
        g_r_new = jnp.exp(Ns_r_new - (beta * V_s).m_as(ureg.dimensionless))
        print(g_r_new)

        return g_r_new, g_r, Ns_r_new, i + 1

    init = (g_r, g_r + 100, Ns_r0, 0)
    val = init
    while condition(val):
        val = step(val)
        g_r, _, _, niter = val
    # g_r, _, _, niter = jax.lax.while_loop(condition, step, (g_r, g_r + 100, Ns_r0, 0))

    return g_r, niter


def S_ii_HNC(k: Quantity, pdf, ni, r):

    integral = (
        jax.scipy.integrate.trapezoid(
            (r * jpu.numpy.sin(k * r) * (pdf - 1)).m_as(ureg.angstrom),
            r.m_as(ureg.angstrom),
        )
        * (1 * ureg.angstrom) ** 2
    )
    return jnp.eye(ni.shape[0]) + (
        (4 * jnp.pi / k) * jpu.numpy.sqrt(jpu.numpy.outer(ni, ni)) * integral
    ).m_as(ureg.dimensionless)


"""
Returns len(q) x len(q) x len(r)
"""
