from jax import numpy as jnp
import jax.interpreters
import jaxrts
import jax
from functools import partial
import jpu

from pint import Quantity

from jaxrts.units import ureg

from typing import List, Callable


@partial(jax.jit, static_argnames=["isign"])
def four1(y, isign):
    """
    See :cite:`Press.1994`.
    """

    # y_indices = [1, ...., 2 * nn]

    # Replaces data[1..2*nn] by its discrete Fourier transform, if isign is
    # input as 1; or replaces data[1..2*nn] by nn times its inverse discrete
    # Fourier transform, if isign is input as −1. data is a complex array of
    # length nn or, equivalently, a real array of length 2*nn. nn MUST be an
    # integer power of 2 (this is not checked for!).
    nn = len(y) // 2
    n = nn << 1
    j = 1
    for i in range(1, n, 2):
        if j > i:
            temp1 = y[i - 1]
            temp2 = y[j - 1]
            y = y.at[j - 1].set(temp1)
            y = y.at[i - 1].set(temp2)
            temp1 = y[i]
            temp2 = y[j]
            y = y.at[j].set(temp1)
            y = y.at[i].set(temp2)
        m = nn
        while (m >= 2) & (j > m):
            j -= m
            m >>= 1
        j += m

    mmax = 2
    while n > mmax:
        istep = mmax << 1
        theta = isign * (2 * jnp.pi / mmax)
        wtemp = jnp.sin(0.5 * theta)
        wpr = -2.0 * wtemp**2

        wpi = jnp.sin(theta)
        wr = 1.0
        wi = 0.0
        for m in range(1, mmax, 2):
            for i in range(m, n + 1, istep):
                j = i + mmax
                tempr = wr * y[j - 1] - wi * y[j]
                tempi = wr * y[j] + wi * y[j - 1]
                y = y.at[j - 1].set(y[i - 1] - tempr)
                y = y.at[j].set(y[i] - tempi)
                y = y.at[i - 1].set(y[i - 1] + tempr)
                y = y.at[i].set(y[i] + tempi)
            wtemp = wr
            wr = wtemp * wpr - wi * wpi + wr
            wi = wi * wpr + wtemp * wpi + wi

        mmax = istep

    return y


@partial(jax.jit, static_argnames=["isign"])
def realfft(y, isign=1):
    """
    See :cite:`Press.1994`.
    """
    n = len(y)

    theta = jnp.pi / (n >> 1)

    c2 = (-isign) * 0.5
    if isign == 1:
        y = four1(y, 1)

    theta = isign * theta

    wtemp = jnp.sin(0.5 * theta)
    wpr = -2.0 * wtemp**2
    wpi = jnp.sin(theta)

    wr = 1.0 + wpr
    wi = wpi
    np3 = n + 3

    c1 = 0.5

    for i in range(2, (n >> 2) + 1, 1):
        i1 = i + i - 1
        i2 = 1 + i1
        i3 = np3 - i2
        i4 = 1 + i3

        h1r = c1 * (y[i1 - 1] + y[i3 - 1])
        h1i = c1 * (y[i2 - 1] - y[i4 - 1])

        h2r = -c2 * (y[i2 - 1] + y[i4 - 1])
        h2i = c2 * (y[i1 - 1] - y[i3 - 1])

        y = y.at[i1 - 1].set(+h1r + wr * h2r - wi * h2i)
        y = y.at[i2 - 1].set(+h1i + wr * h2i + wi * h2r)
        y = y.at[i3 - 1].set(+h1r - wr * h2r + wi * h2i)
        y = y.at[i4 - 1].set(-h1i + wr * h2i + wi * h2r)

        wtemp = wr
        wr = wtemp * wpr - wi * wpi + wr
        wi = wi * wpr + wtemp * wpi + wi

    # exit()

    if isign == 1:

        h1r = y[0]
        y = y.at[0].set(h1r + y[1])
        y = y.at[1].set(h1r - y[1])

    else:
        h1r = y[0]
        y = y.at[0].set(c1 * (h1r + y[1]))
        y = y.at[1].set(c1 * (h1r - y[1]))
        y = four1(y, -1)

    return y


@jax.jit
def realfftnp(y):
    rfft = jnp.fft.rfft(y)[:-1]
    y = y.at[::2].set(jnp.real(rfft))
    y = y.at[1::2].set(-jnp.imag(rfft))
    return y


@jax.jit
def OLDsinft(y):
    # In the original version, we modified y in place. This can be ok, but do
    # we want it, here?
    # y = y.copy()
    n = len(y)

    wr = 1.0
    wi = 0.0

    n2 = n + 2

    theta = jnp.pi / n
    wtemp = jnp.sin(0.5 * theta)
    wpr = -2.0 * wtemp**2
    wpi = jnp.sin(theta)

    y = y.at[0].set(0.0)
    for j in range(2, (n >> 1) + 2, 1):
        wtemp = wr
        wr = (wtemp) * wpr - wi * wpi + wr
        wi = wi * wpr + wtemp * wpi + wi
        y1 = wi * (y[j - 1] + y[n2 - j - 1])  # Construct the auxiliary array
        y2 = 0.5 * (y[j - 1] - y[n2 - j - 1])
        y = y.at[j - 1].set(y1 + y2)  # Terms j and N - j are related
        y = y.at[n2 - j - 1].set(y1 - y2)

    y = realfftnp(y)  # Transform the auxiliary array

    sum_val = 0.0
    y = y.at[0].set(y[0] * 0.5)
    y = y.at[1].set(0.0)
    for j in range(1, n, 2):
        # print(n)
        sum_val += y[j - 1]
        y = y.at[j - 1].set(y[j])
        y = y.at[j].set(sum_val)
    return y


@jax.jit
def sinft(y):
    """
    See :cite:`Press.1994`.
    """
    # In the original version, we modified y in place. This can be ok, but do
    # we want it, here?
    # y = y.copy()
    n = len(y)

    halfn = n >> 1

    wi = jnp.imag(jnp.exp(1j * jnp.arange(halfn + 1) / (2 * n) * jnp.pi * 2))

    y = y.at[0].set(0.0)

    f1 = wi[1 : halfn + 1] * (y[1 : halfn + 1] + y[n : halfn - 1 : -1])
    f2 = 0.5 * (y[1 : halfn + 1] - y[n : halfn - 1 : -1])

    y = y.at[1 : halfn + 1].set(f1 + f2)
    y = y.at[n : halfn - 1 : -1].set(f1 - f2)

    y = realfftnp(y)

    y = y.at[0].set(y[0] * 0.5)
    y = y.at[1].set(0.0)

    sum_val = jnp.cumsum(y[: n - 1 : 2])
    y = y.at[0 : n - 1 : 2].set(y[1:n:2])
    y = y.at[1:n:2].set(sum_val)
    return y


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
def V_l_k(
    k: Quantity | jnp.ndarray, q: Quantity, alpha: Quantity
) -> Quantity | jnp.ndarray:
    """
    q**2 / (k**2 * ε0) * (alpha**2 / (k**2 + alpha**2))
    """

    _q = q[:, :, jnp.newaxis]
    _alpha = alpha[:, :, jnp.newaxis]
    _k = k[jnp.newaxis, jnp.newaxis, :]

    return _q / (_k**2 * ureg.epsilon_0) * _alpha**2 / (_k**2 + _alpha**2)


@jax.jit
def V_s(
    r: Quantity | jnp.ndarray, q: Quantity, alpha: Quantity
) -> Quantity | jnp.ndarray:
    """
    q**2 / (4 * jnp.pi * ureg.epsilon_0 * r) * (jnp.exp(-alpha * r))
    """

    _q = q[:, :, jnp.newaxis]
    _alpha = alpha[:, :, jnp.newaxis]
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


_sinfft = jax.vmap(
    jax.vmap(sinft, in_axes=0, out_axes=0), in_axes=1, out_axes=1
)


@jax.jit
def pair_distribution_function_HNC(V_s, V_l_k, r, Ti, ni):
    """
    Calculate the Pair distribution function in the Hypernetted Chain approach,
    as it was published by :cite:`Wunsch.2011`.
    """
    delta = 1e-6

    dr = r[1] - r[0]
    dk = jnp.pi / (len(r) * dr)

    k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk

    beta = 1 / (ureg.boltzmann_constant * Ti)

    v_s = beta * V_s
    v_l_k = beta * V_l_k

    g_r = jpu.numpy.exp(-(v_s))
    Ns_r0 = jnp.zeros_like(g_r) * ureg.dimensionless

    d = jnp.eye(ni.shape[0]) * ni

    def ozr(input_vec):
        """
        Ornstein-Zernicke Relation
        """
        return jpu.numpy.matmul(
            jnp.linalg.inv(
                (jnp.eye(ni.shape[0]) - jpu.numpy.matmul(input_vec, d)).m_as(
                    ureg.dimensionless
                )
            ),
            input_vec,
        )

    def condition(val):
        """
        If this is False, the loop will stop. Abort if too many steps were
        reached, or if convergence was reached.
        """
        g_r, g_r_old, _, n_iter = val
        return (n_iter < 2000) & jnp.all(
            jnp.max(jnp.abs((g_r - g_r_old).m_as(ureg.dimensionless))) > delta
        )

    def step(val):
        g_r, _, Ns_r, i = val

        h_r = g_r - 1

        cs_r = h_r - Ns_r

        cs_k = (
            _sinfft(
                (r[jnp.newaxis, jnp.newaxis, :] * cs_r).m_as(ureg.angstrom)
            )
            * ureg.angstrom
            * (4 * jnp.pi)
            / k[jnp.newaxis, jnp.newaxis, :]
            * dr
        )

        c_k = cs_k - v_l_k

        # Ornstein-Zernike relation
        h_k = jax.vmap(ozr, in_axes=2, out_axes=2)(c_k)

        Ns_k = h_k - cs_k

        Ns_r_new = (
            _sinfft(
                (k[jnp.newaxis, jnp.newaxis, :] * Ns_k).m_as(ureg.angstrom**2)
            )
            * (1 * ureg.angstrom) ** 2
            * dk
            / (2 * jnp.pi**2 * r[jnp.newaxis, jnp.newaxis, :])
        )

        g_r_new = jpu.numpy.exp(Ns_r_new) / jpu.numpy.exp(v_s)

        return g_r_new, g_r, Ns_r_new, i + 1

    init = (g_r, g_r - 1, Ns_r0, 0)
    g_r, _, _, niter = jax.lax.while_loop(condition, step, init)

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
