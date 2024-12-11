"""
This submodule is dedicated to the using the hypernetted chain approach
to calculate static structure factors.
"""

from functools import partial

import jax
import jax.interpreters
import jpu
from jax import numpy as jnp

from jaxrts.units import Quantity, ureg


@partial(jax.jit, static_argnames=["func"])
def compute_fixpoint(func, x0):

    max_iters = 1000
    delta = 1e-8

    def body(state):
        x, i = state

        x_next = func(x)
        return x_next, i + 1

    def cond(state):
        x, i = state

        x_next = func(x)
        error = jpu.numpy.absolute((x_next - x))

        return (error >= delta * 1 * ureg.bohr**3).any() & (i < max_iters)

    x, num_iters = jax.lax.while_loop(cond, body, (x0, 0))

    return x, num_iters


# Helper functions.
@jax.jit
def psi(t):
    return t * jnp.tanh(jnp.pi * jnp.sinh(t) / 2)


@jax.jit
def dpsi(t):
    res = (jnp.pi * t * jnp.cosh(t) + jnp.sinh(jnp.pi * jnp.sinh(t))) / (
        1 + jnp.cosh(jnp.pi * jnp.sinh(t))
    )
    return jnp.where(jnp.isnan(res), 1.0, res)


# Jitted-versions of Bessel functions of various orders and kinds.
@jax.jit
def bessel_3_2(x):
    return jnp.sqrt(2 / (jnp.pi * x)) * (jnp.sin(x) / x - jnp.cos(x))


@jax.jit
def bessel_0_5(x):
    return jnp.sqrt(2 / (jnp.pi * x)) * jnp.sin(x)


@jax.jit
def bessel_neg0_5(x):
    return jnp.sqrt(2 / (jnp.pi * x)) * jnp.cos(x)


@jax.jit
def bessel_2ndkind_0_5(x):
    return (
        bessel_0_5(x) * jnp.cos(0.5 * jnp.pi) - bessel_neg0_5(x)
    ) / jnp.sin(0.5 * jnp.pi)


@partial(jax.jit, static_argnames=["N"])
def fourier_transform_ogata(k, r, f, N, h):
    """
    A numerical algorithm to calculate the discrete fourier transform based on
    work by Ogata.
    """

    runits = r.units
    units = f.units
    r = r.m_as(runits)
    k = k.m_as(1 / runits)
    f = f.m_as(units)
    r_k = jnp.arange(1, N + 1)

    y_k = jnp.pi * psi(h * r_k) / h

    def internal(onek):
        f_int = jpu.numpy.interp(
            x=y_k / onek,
            xp=r,
            fp=f * r ** (3 / 2),
            left=f[0] * r[0] ** (3 / 2),
            right=f[-1] * r[-1] ** (3 / 2),
            period=None,
        )

        dpsi_k = dpsi(h * r_k)
        w_k = bessel_2ndkind_0_5(jnp.pi * r_k) / bessel_3_2(jnp.pi * r_k)
        series_sum = jnp.pi * w_k * f_int * bessel_0_5(y_k) * dpsi_k

        res = (jpu.numpy.nansum(series_sum) / onek**3) * (onek ** (3 / 2))

        return res * (2 * jnp.pi) ** (3 / 2)

    return jax.vmap(internal, in_axes=0, out_axes=0)(k) * units * runits**3


##########


@jax.jit
def fourier_transform_sine(k, rvals, fvals):
    arg = rvals * fvals
    units = arg.units
    dr = rvals[1] - rvals[0]
    res = zaf_dst(arg.m_as(units), 4) * units * (4 * jnp.pi) / k * dr
    return res


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
def sinft(y):
    """
    See :cite:`Press.1994`.

    This function implements the Sine FFT algorithm, which is can be used
    efficiently compute the Fourier transform of a three-dimensional
    function with rotational symmetry.
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

    sum_val = jnp.cumsum(y[::2])
    y = y.at[::2].set(y[1::2])
    y = y.at[1::2].set(sum_val)
    return y


@partial(jax.jit, static_argnames=["dst_type"])
def zaf_dst(f, dst_type):
    """
    Compute the discrete sine transform (DST) using the fast Fourier transform
    (FFT).

    Taken from `Zaf
    Python<https://github.com/zafarrafii/Zaf-Python/tree/master>`_
    """
    window_length = len(f)

    if dst_type == 1:
        # Compute the DST-I using the FFT
        out = jnp.zeros(2 * window_length + 2)
        out = out.at[1 : window_length + 1].set(f)
        out = out.at[window_length + 2 :].set(-f[::-1])
        out = jnp.fft.fft(out)
        out = -jnp.imag(out[1 : window_length + 1]) / 2
        return out

    elif dst_type == 2:
        # Compute the DST-II using the FFT
        out = jnp.zeros(4 * window_length)
        out = out.at[1 : 2 * window_length : 2].set(f)
        out = out.at[2 * window_length + 1 : 4 * window_length : 2].set(
            -f[-1::-1]
        )
        out = jnp.fft.fft(out)
        out = -jnp.imag(out[1 : window_length + 1]) / 2
        return out

    elif dst_type == 3:
        # Pre-process the signal to make the DST-III matrix orthogonal
        # (copy the signal to avoid modifying it outside of the function)
        f_copy = f.copy()
        f_copy = f_copy.at[-1].set(f_copy[-1] * jnp.sqrt(2))

        # Compute the DST-III using the FFT
        out = jnp.zeros(4 * window_length)
        out = out.at[1 : window_length + 1].set(f_copy)
        out = out.at[window_length + 1 : 2 * window_length].set(f_copy[-2::-1])
        out = out.at[2 * window_length + 1 : 3 * window_length + 1].set(
            -f_copy
        )
        out = out.at[3 * window_length + 1 : 4 * window_length].set(
            -f_copy[-2::-1]
        )
        out = jnp.fft.fft(out)
        out = -jnp.imag(out[1 : 2 * window_length : 2]) / 4
        return out

    elif dst_type == 4:
        out = jnp.zeros(8 * window_length)

        # Compute the DST-IV using the FFT
        out = out.at[1 : 2 * window_length : 2].set(f)
        out = out.at[2 * window_length + 1 : 4 * window_length : 2].set(
            f[window_length - 1 :: -1]
        )
        out = out.at[4 * window_length + 1 : 6 * window_length : 2].set(-f)
        out = out.at[6 * window_length + 1 : 8 * window_length : 2].set(
            -f[window_length - 1 :: -1]
        )
        out = jnp.fft.fft(out)
        out = -jnp.imag(out[1 : 2 * window_length : 2]) / 4
        return out


_3Dfour_sine = jax.vmap(
    jax.vmap(fourier_transform_sine, in_axes=(None, None, 0), out_axes=0),
    in_axes=(None, None, 1),
    out_axes=1,
)

_3Dfour_ogata = jax.vmap(
    jax.vmap(
        lambda k, r, V: fourier_transform_ogata(
            k,
            r,
            V,
            N=100,
            h=10 * ((r[1] - r[0]) / r[-1]).m_as(ureg.dimensionless),
        ),
        in_axes=(None, None, 0),
        out_axes=0,
    ),
    in_axes=(None, None, 1),
    out_axes=1,
)

_3Dfour = _3Dfour_sine


@jax.jit
def pair_distribution_function_two_component_SVT_HNC(
    V_s, V_l_k, r, T_ab, m_ab, ni, mix=0.0
):
    """
    Please explain.
    """
    delta = 1e-6

    dr = r[1] - r[0]
    dk = jnp.pi / (len(r) * dr)

    k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk

    beta = 1 / (ureg.boltzmann_constant * T_ab)

    _T_ab = T_ab[:, :, 0]

    v_s = beta * V_s
    v_l_k = beta * V_l_k

    log_g_r0 = -(v_s).to(ureg.dimensionless)
    Ns_r0 = jnp.zeros_like(log_g_r0) * ureg.dimensionless

    d = jnp.eye(ni.shape[0]) * ni
    B = m_ab / _T_ab
    _m = 2 * jpu.numpy.diagonal(m_ab)

    m = 1.0 / (jnp.diag(_m.m_as(ureg.gram)) * 1 * ureg.gram)

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

    def svt_ozr(input_vec):
        """
        The modified Ornstein-Zernicke Relation
        """
        def func(h):

            return input_vec + m * ni * B * (
                jpu.numpy.matmul(_T_ab * input_vec, h)
                + jpu.numpy.matmul(h, _T_ab * input_vec)
            )

        h, niter = compute_fixpoint(func, ozr(input_vec))
        return h

    def condition(val):
        """
        If this is False, the loop will stop. Abort if too many steps were
        reached, or if convergence was reached.
        """
        _, Ns_r, Ns_r_old, n_iter = val
        err = jpu.numpy.sum((Ns_r - Ns_r_old) ** 2)
        return (n_iter < 2000) & jnp.all(err > delta)

    def step(val):
        log_g_r, Ns_r, _, i = val

        h_r = jpu.numpy.expm1(log_g_r)

        cs_r = h_r - Ns_r

        cs_k = _3Dfour(k, r, cs_r)
        hkold = _3Dfour(k, r, h_r)

        c_k = cs_k - v_l_k

        # Ornstein-Zernike relation
        h_k = jax.vmap(svt_ozr, in_axes=2, out_axes=2)(c_k)

        Ns_k = h_k - cs_k

        Ns_r_new_full = (
            _3Dfour(
                r,
                k,
                Ns_k,
            )
            / (2 * jnp.pi) ** 3
        )

        Ns_r_new = (1 - mix) * Ns_r_new_full + mix * Ns_r

        log_g_r_new = Ns_r_new - v_s

        return log_g_r_new, Ns_r_new, Ns_r, i + 1

    init = (log_g_r0, Ns_r0, Ns_r0 - 1, 0)
    log_g_r, _, _, niter = jax.lax.while_loop(condition, step, init)

    return jpu.numpy.exp(log_g_r), niter


@jax.jit
def pair_distribution_function_HNC(V_s, V_l_k, r, Ti, ni, mix=0.0):
    """
    Calculate the Pair distribution function in the Hypernetted Chain approach,
    as it was published by :cite:`Wunsch.2011`.

    The `mix` argument should lie within the interval [0, 1) and controls
    the amount by which the short-range nodal diagram term `N_ab` is updated
    with each iteration. `mix=0` corresponds to fully using the newly obtained
    result, while increasing `mix` mixes more of the previous iteration's value
    to `N_ab`. This addition to the HNC scheme presented by :cite:`Wunsch.2011`
    was introduced in the MCSS User Guide :cite:`Chapman.2016` and is
    especially relevant e.g., at low temperatures, where the HNC scheme becomes
    numerically unstable.
    """
    delta = 1e-6

    dr = r[1] - r[0]
    dk = jnp.pi / (len(r) * dr)

    k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk

    beta = 1 / (ureg.boltzmann_constant * Ti)

    v_s = beta * V_s
    v_l_k = beta * V_l_k

    log_g_r0 = -(v_s).to(ureg.dimensionless)
    Ns_r0 = jnp.zeros_like(log_g_r0) * ureg.dimensionless

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
        _, Ns_r, Ns_r_old, n_iter = val
        err = jpu.numpy.sum((Ns_r - Ns_r_old) ** 2)
        return (n_iter < 2000) & jnp.all(err > delta)

    def step(val):
        log_g_r, Ns_r, _, i = val

        h_r = jpu.numpy.expm1(log_g_r)

        cs_r = h_r - Ns_r

        cs_k = _3Dfour(k, r, cs_r)

        c_k = cs_k - v_l_k

        # Ornstein-Zernike relation
        h_k = jax.vmap(ozr, in_axes=2, out_axes=2)(c_k)

        Ns_k = h_k - cs_k

        Ns_r_new_full = (
            _3Dfour(
                r,
                k,
                Ns_k,
            )
            / (2 * jnp.pi) ** 3
        )

        Ns_r_new = (1 - mix) * Ns_r_new_full + mix * Ns_r

        log_g_r_new = Ns_r_new - v_s

        return log_g_r_new, Ns_r_new, Ns_r, i + 1

    init = (log_g_r0, Ns_r0, Ns_r0 - 1, 0)
    log_g_r, _, _, niter = jax.lax.while_loop(condition, step, init)

    return jpu.numpy.exp(log_g_r), niter


def geometric_mean_T(T):

    return jpu.numpy.sqrt(T[jnp.newaxis, :] * T[:, jnp.newaxis])


def mass_weighted_T(m, T):
    """
    The mass weighted temperature average of a pair, according to
    :cite:`Schwarz.2007`.

    .. math::

       \\bar{T}_{ab} = \\frac{T_a m_b + T_b m_a}{m_a + m_b}

    """
    return (
        m[:, jnp.newaxis] * T[jnp.newaxis, :]
        + m[jnp.newaxis, :] * T[:, jnp.newaxis]
    ) / (m[:, jnp.newaxis] + m[jnp.newaxis, :])


@jax.jit
def S_ii_HNC(k: Quantity, pdf, ni, r):
    """
    Calculates the static structure factor for an isotropic system from the
    pair distribution function obtained used the HNC approach.

    .. note::
        Due to the numerical implementation of :py:func:`~.sinft`, we force the
        entry ``S_ii[0]`` to the value ``S_ii[1]``, as it would be always
        unity, otherwise.
    """
    S_k = (
        jnp.eye(ni.shape[0])[:, :, jnp.newaxis]
        + (
            _3Dfour(
                k,
                r,
                pdf - 1.0,
            )
            * jpu.numpy.sqrt(jpu.numpy.outer(ni, ni))[:, :, jnp.newaxis]
        )
    ).m_as(ureg.dimensionless)

    # The first index is forced to 1. Set it to the entry [1], as this is not
    # desired
    return S_k.at[:, :, 0].set(S_k[:, :, 1]) * (1 * ureg.dimensionless)


#: The equivalent of jnp.interp for HNC-shaped arrays
hnc_interp = jax.vmap(
    jax.vmap(jpu.numpy.interp, in_axes=(None, None, 0), out_axes=0),
    in_axes=(None, None, 1),
    out_axes=1,
)
