"""Bridge closures for the hypernetted-chain equation.

The VMHNC implementation is based on Otter :cite:`Otter.2026`.
"""

from functools import partial
from math import comb, factorial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import hypernetted_chain as hnc
from .units import Quantity, ureg


class VMHNCResult(NamedTuple):
    """Correlations (shape ``(1, 1, N)``) and scalar convergence diagnostics.

    ``g_r``, ``s_k``, ``bridge_r`` are dimensionless JAX arrays; ``sigma`` is a
    length. Failed solves return NaN correlations. ``status`` is 0 on success,
    1 for invalid physical inputs/grid, 2 for inner solve/box failure, 3 for an
    unbracketed packing fraction, and 4 for exhausted outer iterations.
    The variational residual remains available for diagnosis.
    """

    g_r: jax.Array
    s_k: jax.Array
    bridge_r: jax.Array
    eta: jax.Array
    sigma: Quantity
    variational_residual: jax.Array
    iterations: jax.Array
    converged: jax.Array
    status: jax.Array


@jax.jit
def _dst(values):
    """DST-I on the interior radial grid."""
    return 2 * hnc.sinft(jnp.pad(values, (1, 0)))[1:]


@jax.jit
def _forward(values, r, k):
    return 2 * jnp.pi * (r[1] - r[0]) * _dst(r * values) / k


@jax.jit
def _inverse(values, r, k):
    return (k[1] - k[0]) * _dst(k * values) / (4 * jnp.pi**2 * r)


@jax.jit
def _py_shells(eta, x):
    """Wertheim--Thiele inverse Laplace series for 1 <= x < 6.

    Expand s L(s)^m / S(s)^m around each of the three simple roots of S.
    Taylor coefficients give the residues at its repeated poles.
    """
    a = (1 - eta) ** 2
    b = 6 * eta / (1 - eta)
    c = 18 * eta**2 / a
    d = -12 * eta * (1 + 2 * eta) / a
    p = c - b**2 / 3
    q = 2 * b**3 / 27 - b * c / 3 + d
    disc = jnp.sqrt((q / 2) ** 2 + (p / 3) ** 3)
    u = jnp.cbrt(-q / 2 + disc)
    v = jnp.cbrt(-q / 2 - disc)
    roots = jnp.array(
        [
            u + v - b / 3,
            -(u + v) / 2 - b / 3 + 0.5j * jnp.sqrt(3.0) * (u - v),
            -(u + v) / 2 - b / 3 - 0.5j * jnp.sqrt(3.0) * (u - v),
        ]
    )
    ell = 1 + eta / 2
    g = jnp.zeros_like(x)
    for shell in range(1, 6):
        offset = jnp.clip(x - shell, 0, 6 - shell)
        inverse = jnp.zeros_like(x, dtype=roots.dtype)
        for index in range(3):
            root = roots[index]
            lroot = ell * root + 1 + 2 * eta
            coeff = jnp.stack(
                [
                    comb(shell, j) * lroot ** (shell - j) * ell**j
                    for j in range(shell)
                ]
            )
            coeff = root * coeff + jnp.concatenate(
                (jnp.zeros_like(coeff[:1]), coeff[:-1])
            )
            for other in range(3):
                if other != index:
                    diff = root - roots[other]
                    factor = jnp.stack(
                        [
                            (-1) ** j
                            * comb(shell + j - 1, j)
                            / diff ** (shell + j)
                            for j in range(shell)
                        ]
                    )
                    coeff = jnp.convolve(coeff, factor)[:shell]
            coeff = coeff / a**shell
            polynomial = sum(
                coeff[j] * offset ** (shell - j - 1) / factorial(shell - j - 1)
                for j in range(shell)
            )
            inverse += polynomial * jnp.exp(root * offset)
        g += jnp.where(
            (x >= shell) & (x < 6),
            (-12 * eta) ** (shell - 1) * inverse.real / x,
            0,
        )
    return g


@partial(jax.jit, static_argnames=("points_per_diameter",))
def _hard_sphere_py(eta, points_per_diameter=256):
    """Return ``(x, B_PY, g_PY)`` on ``x=r/sigma`` out to 64 diameters.

    Uses an analytic core/contact and a C2 shell-to-FFT blend.
    ``eta`` must lie strictly between zero and 0.5. Invalid values produce
    NaNs. ``points_per_diameter`` must be an integer of at least 32.
    """
    if not isinstance(points_per_diameter, int) or points_per_diameter < 32:
        raise ValueError("points_per_diameter must be an integer >= 32.")
    x = jnp.arange(1, 64 * points_per_diameter) / points_per_diameter
    k = jnp.arange(1, x.size + 1) * jnp.pi / 64
    lam1 = (1 + 2 * eta) ** 2 / (1 - eta) ** 4
    lam2 = -((1 + eta / 2) ** 2) / (1 - eta) ** 4
    c = jnp.where(
        x <= 1, -lam1 - 6 * eta * lam2 * x - eta * lam1 * x**3 / 2, 0
    )
    ck = _forward(c, x, k)
    g = 1 + _inverse(ck / (1 - 6 * eta / jnp.pi * ck), x, k)
    exact = _py_shells(eta, x)
    weight = jnp.clip((x - 5.5) / 0.5, 0, 1)
    weight = weight**3 * (10 + weight * (-15 + 6 * weight))
    g = jnp.where((x >= 1) & (x < 6), (1 - weight) * exact + weight * g, g)
    g = jnp.where(x < 1, 0, jnp.maximum(g, 1e-12))
    bridge = jnp.where(
        x < 1,
        jnp.log(jnp.maximum(-c, 1e-300)) + c + 1,
        jnp.log(jnp.maximum(g, 1e-300)) - g + 1,
    )
    valid = (eta > 0) & (eta < 0.5)
    return x, jnp.where(valid, bridge, jnp.nan), jnp.where(valid, g, jnp.nan)


@partial(jax.jit, static_argnames=("cramer_solve_max_size",))
def pair_distribution_function_with_bridge(
    V_s,
    V_l_k,
    r,
    Ti,
    ni,
    bridge_r,
    mix=0.0,
    tmult=None,
    cramer_solve_max_size=4,
):
    """Calculate pair distributions with a dimensionless bridge ``B_ab(r)``.

    ``bridge_r`` has the same shape as ``V_s``. The effective short-range
    potential is ``V_s - k_B * Ti * bridge_r``. All other arguments and the
    returned ``(g, niter)`` follow
    :py:func:`~jaxrts.hypernetted_chain.pair_distribution_function_HNC`.
    """
    if bridge_r.shape != V_s.shape:
        raise ValueError("bridge_r must have the same shape as V_s.")
    return hnc.pair_distribution_function_HNC(
        V_s - ureg.k_B * Ti * bridge_r,
        V_l_k,
        r,
        Ti,
        ni,
        mix,
        tmult,
        cramer_solve_max_size,
    )


def _hnc_correlations(vs, vlk, bridge, r, k, density, mix, tmult):
    radius = r * ureg.a_0
    number_density = density * (1 / ureg.a_0**3)
    g, iterations = pair_distribution_function_with_bridge(
        vs * ureg.eV,
        vlk * ureg.eV * ureg.a_0**3,
        radius,
        1 * ureg.eV / ureg.k_B,
        number_density,
        bridge,
        mix,
        tmult,
    )
    s = hnc.S_ii_HNC(k * (1 / ureg.a_0), g, number_density, radius)
    g, s = g.magnitude, s.magnitude
    # The native solver stops at a squared nodal-step norm of 1e-6 and
    # allows 2000 iterations per temperature stage. A total below 2000
    # guarantees no stage exhausted its budget; also reject NaN branches.
    ok = (
        (iterations < 2000)
        & jnp.all(jnp.isfinite(g))
        & jnp.all(jnp.isfinite(s))
        & jnp.all(g >= 0)
        & jnp.all(s > 0)
    )
    return g, s, iterations, ok


@partial(jax.jit, static_argnames=("points_per_diameter",))
def _solve_eta(eta, vs, vlk, r, k, density, mix, tmult, points_per_diameter):
    x, bridge_x, g_hs = _hard_sphere_py(eta, points_per_diameter)
    sigma = jnp.cbrt(6 * eta / (jnp.pi * density[0]))
    bridge = jnp.interp(r / sigma, x, bridge_x, right=0)[None, None, :]
    g, s, iterations, ok = _hnc_correlations(
        vs, vlk, bridge, r, k, density, mix, tmult
    )
    step = jnp.minimum(
        jnp.maximum(eta * 1e-3, 1e-6), 0.49 * jnp.minimum(eta, 0.5 - eta)
    )
    _, plus, _ = _hard_sphere_py(eta + step, points_per_diameter)
    _, minus, _ = _hard_sphere_py(eta - step, points_per_diameter)
    # The diameter changes with eta, so differentiate at fixed physical r.
    gradient = jnp.gradient(bridge_x, x)
    gradient = gradient.at[0].set(
        (-3 * bridge_x[0] + 4 * bridge_x[1] - bridge_x[2])
        * points_per_diameter
        / 2
    )
    gradient = gradient.at[-1].set(
        (3 * bridge_x[-1] - 4 * bridge_x[-2] + bridge_x[-3])
        * points_per_diameter
        / 2
    )
    derivative = (plus - minus) / (2 * step) - x * gradient / (3 * eta)
    g_x = jnp.interp(sigma * x, r, g[0, 0], right=1)
    integrand = x**2 * (g_x - g_hs) * derivative
    contact = points_per_diameter - 1
    inner = (
        integrand[: contact + 1].at[-1].set(g_x[contact] * derivative[contact])
    )
    outer = integrand[contact:]
    mask = sigma * x[contact + 1 :] <= r[-1]
    integral = (
        jnp.trapezoid(inner, x[: contact + 1])
        + jnp.sum(jnp.where(mask, (outer[:-1] + outer[1:]) / 2, 0))
        / points_per_diameter
    )
    residual = 2 * eta**2 / (1 - eta) ** 3 - 12 * eta * integral
    ok &= (sigma * x[contact + 1] < r[-1]) & jnp.isfinite(residual)
    return residual, g, s, bridge, sigma, iterations, ok


@partial(
    jax.jit,
    static_argnames=(
        "points_per_diameter",
        "eta_bounds",
        "eta_scan_points",
        "eta_max_iter",
    ),
)
def solve_vmhnc(
    V_s,
    V_l_k,
    r,
    Ti,
    ni,
    *,
    mix=0.5,
    tmult=(10.0, 3.0),
    points_per_diameter=256,
    eta_bounds=(0.0, 0.49),
    eta_scan_points=9,
    eta_tol=1e-4,
    eta_max_iter=20,
    bridge_tol=1e-7,
):
    """Calculate one-component VMHNC correlations for split potentials.

    ``r`` contains an even number of positive, uniformly spaced radii.
    ``bridge_tol`` sets the bridge amplitude below which HNC is recovered;
    ``eta_tol`` bounds the packing-fraction bracket width.
    Return :py:class:`VMHNCResult`, including convergence information.
    """
    if r.ndim != 1 or r.size < 4 or r.size % 2:
        raise ValueError(
            "r must be a one-dimensional even grid of >= 4 points."
        )
    if V_s.shape != (1, 1, r.size) or V_l_k.shape != V_s.shape:
        raise ValueError("VMHNC requires one-component potentials (1, 1, N).")
    if ni.shape != (1,) or jnp.size(Ti.magnitude) != 1:
        raise ValueError("VMHNC requires one density and one temperature.")
    low, high = eta_bounds
    if not 0 <= low < high < 0.5:
        raise ValueError("eta_bounds must satisfy 0 <= low < high < 0.5.")
    if eta_scan_points < 2 or eta_max_iter < 1:
        raise ValueError(
            "Iteration limit must be positive; scan needs >= 2 points."
        )
    radius = r.m_as(ureg.a_0)
    dr = radius[1] - radius[0]
    k = jnp.pi / radius[-1] + jnp.arange(r.size) * jnp.pi / (r.size * dr)
    density = ni.m_as(ureg.a_0**-3)
    temperature = (ureg.k_B * Ti).m_as(ureg.eV).reshape(())
    vs = V_s.m_as(ureg.eV) / temperature
    vlk = V_l_k.m_as(ureg.eV * ureg.a_0**3) / temperature
    valid = (
        jnp.all(density > 0)
        & jnp.all(jnp.isfinite(density))
        & (temperature > 0)
        & jnp.isfinite(temperature)
        & (radius[0] > 0)
        & (dr > 0)
        & jnp.all(jnp.isfinite(radius))
        & jnp.allclose(jnp.diff(radius), dr, rtol=1e-7, atol=0)
        & jnp.all(jnp.isfinite(vs))
        & jnp.all(jnp.isfinite(vlk))
        & (mix >= 0)
        & (mix < 1)
        & (bridge_tol > 0)
        & jnp.isfinite(bridge_tol)
        & (eta_tol > 0)
        & jnp.isfinite(eta_tol)
    )

    def evaluate(eta):
        return _solve_eta(
            eta, vs, vlk, radius, k, density, mix, tmult, points_per_diameter
        )

    anchor = min(0.05, high) if low == 0 else low
    first = evaluate(jnp.asarray(anchor))

    def lower_condition(state):
        i, _, _, sol, _ = state
        return (
            (low == 0)
            & (i < eta_max_iter)
            & (sol[0] > 0)
            & (jnp.max(jnp.abs(sol[3])) > bridge_tol)
            & sol[-1]
            & valid
        )

    def lower_step(state):
        i, eta, _, sol, count = state
        next_sol = evaluate(eta / 4)
        return i + 1, eta / 4, eta, next_sol, count + next_sol[5]

    _, lower, upper, first, count = jax.lax.while_loop(
        lower_condition,
        lower_step,
        (0, jnp.asarray(anchor), jnp.asarray(anchor), first, first[5]),
    )
    hnc_limit = (
        (low == 0)
        & (first[0] > 0)
        & (jnp.max(jnp.abs(first[3])) <= bridge_tol)
        & first[-1]
    )

    def scan_condition(state):
        i, _, _, fl, fr, _, ok = state
        return (i < eta_scan_points) & (fl * fr > 0) & ok & valid & ~hnc_limit

    def scan_step(state):
        i, _, right, _, fr, count, ok = state
        eta = lower + (high - lower) * i / (eta_scan_points - 1)
        sol = evaluate(eta)
        return i + 1, right, eta, fr, sol[0], count + sol[5], ok & sol[6]

    _, lo, hi, fl, fr, count, ok = jax.lax.while_loop(
        scan_condition,
        scan_step,
        (
            1,
            lower,
            upper,
            first[0],
            jnp.where(lower < upper, 1.0, first[0]),
            count,
            first[6],
        ),
    )
    bracketed = fl * fr <= 0

    def bisect_condition(state):
        i, lo, hi, _, _, ok = state
        return (
            (i < eta_max_iter) & (hi - lo > eta_tol) & ok & bracketed & valid
        )

    def bisect_step(state):
        i, lo, hi, fl, count, ok = state
        mid = (lo + hi) / 2
        sol = evaluate(mid)
        left = fl * sol[0] <= 0
        return (
            i + 1,
            jnp.where(left, lo, mid),
            jnp.where(left, mid, hi),
            jnp.where(left, fl, sol[0]),
            count + sol[5],
            ok & sol[6],
        )

    _, lo, hi, _, count, ok = jax.lax.while_loop(
        bisect_condition, bisect_step, (0, lo, hi, fl, count, ok)
    )
    eta = jnp.where(hnc_limit, 0.0, (lo + hi) / 2)

    def zero_bridge(_):
        bridge = jnp.zeros_like(vs)
        g, s, iterations, ok = _hnc_correlations(
            vs, vlk, bridge, radius, k, density, mix, tmult
        )
        zero = jnp.zeros_like(eta)
        return zero, g, s, bridge, zero, iterations, ok

    final = jax.lax.cond(
        hnc_limit, zero_bridge, lambda _: evaluate(eta), operand=None
    )
    residual, g, s, bridge, sigma, niter, inner_ok = final
    status = jnp.where(
        ~valid,
        1,
        jnp.where(
            ~(ok & inner_ok),
            2,
            jnp.where(
                hnc_limit,
                0,
                jnp.where(~bracketed, 3, jnp.where(hi - lo > eta_tol, 4, 0)),
            ),
        ),
    )
    converged = status == 0
    return VMHNCResult(
        jnp.where(converged, g, jnp.nan),
        jnp.where(converged, s, jnp.nan),
        bridge,
        eta,
        sigma * ureg.a_0,
        residual,
        count + niter,
        converged,
        status,
    )
