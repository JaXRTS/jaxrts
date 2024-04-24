from jax import numpy as jnp
import jax.interpreters
import jaxrts
import jax
import jpu

from pint import Quantity

from jaxrts.units import ureg

from typing import List, Callable

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
        _q
        / (4 * jnp.pi * ureg.epsilon_0 * _r)
        * (jpu.numpy.exp(-_alpha * _r))
    ).to(ureg.electron_volt)

@jax.jit
def construct_alpha_matrix(ne: jnp.ndarray | Quantity):

    d = jpu.numpy.cbrt(
        3 / (4 * jnp.pi * (ne[:, jnp.newaxis] + ne[jnp.newaxis, :]) / 2)
    )

    return 2/d
@jax.jit
def construct_q_matrix(q: jnp.ndarray) -> jnp.ndarray:

    return jpu.numpy.outer(q, q)

@jax.jit
def pair_distribution_function_HNC(V_s, V_l, Ti, ni):

    beta = 1 / (ureg.boltzmann_constant * Ti)
    # g_r = (-beta * V_s).m_as(ureg.dimensionless) + 1cls
    g_r = jnp.exp(-(beta * (V_s + V_l)).m_as(ureg.dimensionless)) + 0j

    V_l_k = (
        jnp.fft.fft(V_l.m_as(ureg.electron_volt), norm="forward", axis=2)
        * ureg.electron_volt
    )

    delta = 1e-5

    g_r_old = g_r 
    # g_r_old = jnp.exp(-(beta * (V_s + V_l)).m_as(ureg.dimensionless))

    Ns_r0 = jnp.zeros_like(g_r)

    d = jnp.eye(ni.shape[0]) * ni

    def ozr(input_vec):
        return jnp.linalg.inv(jnp.eye(ni.shape[0]) - input_vec @ d.m_as(1/ureg.centimeter**3)) @ input_vec

    def condition(val):
        g_r, g_r_old, _, n_iter = val
        # return (jnp.max(jnp.abs(g_r - g_r_old)) > delta) | (n_iter < 10)
        return (n_iter < 1000) | jnp.all(jnp.greater(jnp.abs(g_r - g_r_old), delta))

    def step(val):
        g_r, _, Ns_r, i = val

        h_r = g_r - 1

        cs_r = h_r - Ns_r

        cs_k = jnp.fft.fft(cs_r, norm="forward", axis=2) # ureg.angstrom ** 3

        c_k = cs_k - (beta * V_l_k).m_as(ureg.dimensionless)

        # Ornstein-Zernike relation
        h_k = jax.vmap(ozr, in_axes=2, out_axes = 2)(c_k)

        Ns_k = h_k - cs_k

        Ns_r_new = jnp.fft.fft(Ns_k, norm="backward", axis=2)

        g_r_new = jnp.exp(Ns_r_new - (beta * V_s).m_as(ureg.dimensionless))

        return g_r_new, g_r, Ns_r_new, i + 1

    g_r, _, _, niter = jax.lax.while_loop(condition, step, (g_r, g_r_old, Ns_r0, 0))

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
