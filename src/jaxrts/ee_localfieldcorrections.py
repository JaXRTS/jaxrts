"""
This submodule is dedicated to the calculation of static and dynamic local
field corrections. structure.
"""

import logging

import jax
from jax import numpy as jnp
from jpu import numpy as jnpu
from quadax import quadts as quad

from .ee_localfieldcorrections_dornheim_2021 import G_analytical as dornheim_G
from .plasma_physics import (
    coupling_param,
    fermi_energy,
    fermi_wavenumber,
    interparticle_spacing,
    wiegner_seitz_radius,
)
from .units import Quantity, ureg

logger = logging.getLogger(__name__)


@jax.jit
def xi_lfc_corrected(
    xi: Quantity | jnp.ndarray, v: Quantity, lfc: Quantity | jnp.ndarray
):
    """
    This function corrects the susceptibility according to the local field
    correction formalism.
    """
    return xi / (1 - v * (1 - lfc) * xi)


# Static local-field corrections
# ===============================================
#


@jax.jit
def eelfc_hubbard(k: Quantity, T_e: Quantity, n_e: Quantity) -> Quantity:
    """
    Static local field correction introduced and based on :cite:`Hubbard.1957`.

    ..math::

      G(k) = \\frac{1}{2}\\frac{k^2}{k^2 + k_F^2}

    While improving on RPA results for degenerate systems, neither long- nor
    short wavelength limits are correct.
    """

    k_F = fermi_wavenumber(n_e)
    return k**2 / (2 * (k**2 + k_F**2))


@jax.jit
def eelfc_geldartvosko(k: Quantity, T_e: Quantity, n_e: Quantity) -> Quantity:
    """
    Static local field correction introduced and based on :cite:`Geldart.1966`.
    Yields reasonable results at high temperatures.
    """

    k_F = fermi_wavenumber(n_e)
    T_F = fermi_energy(n_e) / (1 * ureg.boltzmann_constant)
    # Effective temperature (Gregori)
    T_q = T_F / (
        1.3251
        - 0.1779 * jnpu.sqrt(interparticle_spacing(1, 1, n_e) / (1 * ureg.a0))
    )
    T_ee = (T_e**2 + T_q**2) ** (1 / 2)
    Gamma_ee = coupling_param(1, 1, n_e, T_ee)

    gamma_T = (12.0 * jnp.pi**2) ** (1 / 3) * (
        0.0999305
        + 0.0187058 / Gamma_ee
        + (0.0013240 / Gamma_ee ** (4 / 3))
        - 0.0479236 / (Gamma_ee ** (2 / 3))
    )

    # Fitting formula for H_0, see 'Gregori & Ravasio et al:2007'
    C_sc = 1.0754
    H_0 = (C_sc * Gamma_ee ** (3 / 2)) / (
        (C_sc / jnp.sqrt(3)) ** 4 + Gamma_ee**4
    ) ** (1 / 4)

    xi = (
        (1 * ureg.electron_mass * ureg.elementary_charge**4)
        / (
            (4 * jnp.pi * 1 * ureg.epsilon_0) ** 2
            * 1
            * ureg.boltzmann_constant
            * T_ee
            * 1
            * ureg.hbar**2
        )
    ).m_as(ureg.dimensionless)

    prefactor = jnp.sqrt(2 * jnp.pi) * xi ** (3 / 2)

    def integrand(u):
        return u * jnpu.exp(-xi * u**2) / (jnpu.exp(jnp.pi / u) - 1.0)

    g_bin_0 = (
        prefactor
        * quad(integrand, jnp.array([0, jnp.inf]), epsabs=1e-16, epsrel=1e-16)[
            0
        ]
    )

    gT_ee_0 = g_bin_0 * jnpu.exp(H_0)

    return k**2 / (1 / gamma_T * k_F**2 + (1 - gT_ee_0) ** (-1) * k**2)


@jax.jit
def eelfc_utsumiichimaru(
    k: Quantity, T_e: Quantity, n_e: Quantity
) -> Quantity:
    """
    Static local field correction introduced and based on
    :cite:`UtsumiIchimaru.1982`. This is a result for zero temperature.
    The argument `T_e` is only included to have signatures comparable with
    other static LFC models.
    """

    rs = interparticle_spacing(1, 1, n_e) / (1 * ureg.a0)

    # Expression of g0_ee (T = 0) by Yasuhara
    z = 4 * (4 / (9 * jnp.pi)) ** (1 / 6) * (rs / jnp.pi) ** (1 / 2)
    g0_ee = 1 / 8 * (z / jax.scipy.special.i1(z.m_as(ureg.dimensionless))) ** 2

    rs2dEc_drs = (0.0621814 + 0.61024 * rs ** (1 / 2)) / (
        1 + 9.81379 * rs ** (1 / 2) + 2.82224 * rs + 0.736411 * rs ** (3 / 2)
    )
    rs3d2Ec_d2rs = (
        -8.23505 * rs ** (3 / 2)
        - 2.486 * rs**2
        - 23.0573 * rs
        - 4.50109 * rs ** (1 / 2)
        - 0.229324
    ) / (rs ** (3 / 2) + 3.83243 * rs + 13.3265 * rs ** (1 / 2) + 1.35794) ** 2

    gamma_0 = 1 / 4 - (jnp.pi * (4 / (9 * jnp.pi)) ** (1 / 3)) / (24) * (
        rs3d2Ec_d2rs - 2 * rs2dEc_drs
    )

    A = 0.029
    B = 9 / 16 * gamma_0 - 3 / 64 * (1 - g0_ee) - 16 / 15 * A
    C = -3 / 4 * gamma_0 + 9 / 16 * (1 - g0_ee) - 16 / 5 * A

    k_F = fermi_wavenumber(n_e)
    Q = (k / k_F).m_as(ureg.dimensionless)

    return (
        A * Q**4
        + B * Q**2
        + C
        + (A * Q**4 + (B + (8 / 3) * A) * Q**2 - C)
        * ((4 - Q**2) / (4 * Q))
        * jnp.log(jnp.abs((2 + Q) / (2 - Q)))
    )


@jax.jit
def eelfc_farid(k: Quantity, T_e: Quantity, n_e: Quantity) -> Quantity:
    """
    Improved version of Utsumi and Ichimaru (:py:func:`~.eelfc_utsumiichimaru`,
    based on QMC results (:cite:`Farid.1993`). This is a result for zero
    temperature. The argument `T_e` is only included to have signatures
    comparable with other static LFC models.
    """
    rs = (interparticle_spacing(1, 1, n_e) / (1 * ureg.a0)).m_as(
        ureg.dimensionless
    )

    lamb = (4 / (9 * jnp.pi)) ** (1 / 3)

    k_F = 1 / (lamb * rs)
    E_F = k_F**2 / 2
    w_p = (3 / rs**3) ** (1 / 2)
    # w_p = plasma_frequency(n_e)
    # E_F = fermi_energy(n_e)
    # k_F = fermi_wavenumber(n_e)
    Q = (k / fermi_wavenumber(n_e)).m_as(ureg.dimensionless)

    rs2dEc_drs = (0.0621814 + 0.61024 * rs ** (1 / 2)) / (
        1 + 9.81379 * rs ** (1 / 2) + 2.82224 * rs + 0.736411 * rs ** (3 / 2)
    )
    rs3d2Ec_d2rs = (
        -8.23505 * rs ** (3 / 2)
        - 2.486 * rs**2
        - 23.0573 * rs
        - 4.50109 * rs ** (1 / 2)
        - 0.229324
    ) / (rs ** (3 / 2) + 3.83243 * rs + 13.3265 * rs ** (1 / 2) + 1.35794) ** 2

    gamma_0 = 1 / 4 - (jnp.pi * (4 / (9 * jnp.pi)) ** (1 / 3)) / (24) * (
        rs3d2Ec_d2rs - 2 * rs2dEc_drs
    )

    x = rs ** (1 / 2)

    delta_2 = (
        -2.2963827e-3 * x
        + 5.6991691e-2 * x**2
        + -0.8533622 * x**3
        + (-8.7736539) * x**4
        + 0.7881997 * x**5
        + (-1.2707788e-2 * x**6)
    ) / (
        x**4
        + -79.9684540
        + (-140.5268938) * x
        + (-35.2575566) * x**2
        + (-10.6331769) * x**3
    )

    delta_4 = delta_2 * (
        (
            23.0118890
            + (-64.8378723) * x
            + 63.5105927 * x**2
            + (-13.9457829) * x**3
            + (-12.6252782) * x**4
            + 13.8524989 * x**5
            + (-5.2740937) * x**6
            + 1.0156885 * x**7
            + (-1.1039532e-2) * x**8
        )
        / (
            x**6
            + 9.5753544
            + (-32.9770151) * x
            + 48.2528870 * x**2
            + (-38.7189788) * x**3
            + 20.5595956 * x**4
            + (-6.3066750) * x**5
        )
    )
    a = 0.029

    # Expression of g0_ee (T = 0) by Yasuhara
    z = 4 * (4 / (9 * jnp.pi)) ** (1 / 6) * (rs / jnp.pi) ** (1 / 2)
    g0_ee = 1 / 8 * (z / jax.scipy.special.i1(z)) ** 2

    b0A = 2 / 3 * (1 - g0_ee)
    b0B = 48 * E_F**2 / (35 * w_p**2) * delta_4
    b0C = -16 / 25 * (E_F**2 / w_p**2) * (2 * delta_2 + delta_2**2)
    bm2 = 4 / 5 * E_F**2 / w_p**2 * delta_2
    b0 = b0A + b0B + b0C

    A = 63 / 64 * a + 15 / 4096 * (b0A - 2 * (b0B + b0C) - 16 * bm2)
    B = 9 / 16 * gamma_0 + 7 / 16 * bm2 - 3 / 64 * b0 - 16 / 15 * A
    C = -3 / 4 * gamma_0 + 3 / 4 * bm2 + 9 / 16 * b0 - 16 / 5 * A
    D = 9 / 16 * gamma_0 - 9 / 16 * bm2 - 3 / 64 * b0 + 8 / 5 * A

    return (
        A * Q**4
        + B * Q**2
        + C
        + (A * Q**4 + D * Q**2 - C)
        * (4 - Q**2)
        / (4 * Q)
        * jnp.log(jnp.abs((2 + Q) / (2 - Q)))
    )


@jax.jit
def eelfc_interpolationgregori2007(
    k: Quantity, T_e: Quantity, n_e: Quantity
) -> Quantity:
    """
    Interpolation function between the UtsumiIchimaru result
    :py:func:`~.eelfc_utsumiichimaru` (zero T) for the local field correction
    and the GeldartVosko result :py:func:`eelfc_geldartvosko` (high T).
    :cite:`Gregori.2007`
    """

    Theta = (T_e / (fermi_energy(n_e) / (1 * ureg.boltzmann_constant))).m_as(
        ureg.dimensionless
    )

    return (
        eelfc_utsumiichimaru(k, T_e, n_e)
        + Theta * eelfc_geldartvosko(k, T_e, n_e)
    ) / (1 + Theta)


@jax.jit
def eelfc_interpolationgregori_farid(
    k: Quantity, T_e: Quantity, n_e: Quantity
) -> Quantity:
    """
    Interpolation function between the Farid result :py:func:`~.eelfc_farid`
    (zero T) for the local field correction and the GeldartVosko result
    :py:func:`eelfc_geldartvosko` (high T). :cite:`Fortmann.2010`
    """

    Theta = (T_e / (fermi_energy(n_e) / (1 * ureg.boltzmann_constant))).m_as(
        ureg.dimensionless
    )

    return (
        eelfc_farid(k, T_e, n_e) + Theta * eelfc_geldartvosko(k, T_e, n_e)
    ) / (1 + Theta)


@jax.jit
def eelfc_dornheim2021(k: Quantity, T_e: Quantity, n_e: Quantity) -> Quantity:
    """
    Use the analytical interpolation of the effective static approximation for
    the LFC given by :cite:`Dornheim.2021`. Interpolating path-integral Monte
    Carlo simulations while preserving correct limits.

    Valid for 0.7 <= r_s <= 20, and theta <=4.
    """

    Theta = (T_e / (fermi_energy(n_e) / (1 * ureg.boltzmann_constant))).m_as(
        ureg.dimensionless
    )
    k_over_k_f = (k / fermi_wavenumber(n_e)).m_as(ureg.dimensionless)
    rs = (wiegner_seitz_radius(n_e) / (1 * ureg.a0)).m_as(ureg.dimensionless)

    return dornheim_G(k_over_k_f, rs, Theta)
