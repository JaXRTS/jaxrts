from .units import ureg, Quantity

from typing import List

from quadax import quadts as quad

import jax
from jax import jit
from jax import numpy as jnp
from jpu import numpy as jnpu
import logging

logger = logging.getLogger(__name__)

from .math import fermi_neg12_rational_approximation_antia
from .plasma_physics import (
    coupling_param,
    fermi_wavenumber,
    fermi_energy,
    interparticle_spacing,
)


@jax.jit
def eelfc_geldartvosko(k: Quantity, T_e: Quantity, n_e: Quantity) -> Quantity:

    Gamma_ee = coupling_param(1, 1, n_e, T_e)

    gamma_T = (12.0 * jnp.pi**2) ** (1 / 3) * (
        0.0999305
        + 0.0187058 / Gamma_ee
        + (0.0013240 / Gamma_ee ** (4 / 3))
        - 0.0479236 / (Gamma_ee ** (2 / 3))
    )

    k_F = fermi_wavenumber(n_e)
    T_F = fermi_energy / (1 * ureg.boltzmann_constant)

    # Fitting formula for H_0, see 'Gregori & Ravasio et al:2007'
    C_sc = 1.0754
    H_0 = (C_sc * Gamma_ee ** (3 / 2)) / (
        (C_sc / jnp.sqrt(3)) ** 4 + Gamma_ee**4
    ) ** (1 / 4)

    # Effective temperature (Gregori)
    T_q = T_F / (
        1.3251
        - 0.1779 * jnp.sqrt(interparticle_spacing(1, 1, n_e) / (1 * ureg.a0))
    )
    T_ee = (T_e**2 + T_q**2) ** (1 / 2)

    xi = (1 * ureg.electron_mass * ureg.elementary_charge**4) / (
        (4 * jnp.pi * 1 * ureg.epsilon_0) ** 2
        * 1
        * ureg.boltzmann_constant
        * T_ee
        * 1
        * ureg.hbar**2
    )

    prefactor = jnp.sqrt(2 * jnp.pi) * xi ** (3 / 2)

    def integrand(u):
        return u * jnp.exp(-xi * u**2) / (jnp.exp(jnp.pi / u) - 1.0)

    g_bin_0 = prefactor * quad(
        integrand, [0, jnp.inf], epsabs=1e-16, epsrel=1e-16
    )

    gT_ee_0 = g_bin_0 * jnp.exp(H_0)

    return k**2 / (1 / gamma_T * k_F**2 + (1 - gT_ee_0) ** (-1) * k**2)


@jax.jit
def eelfc_utsumiichimaru(
    k: Quantity, T_e: Quantity, n_e: Quantity
) -> Quantity:

    rs = interparticle_spacing(1, 1, n_e)
    A = 0.029

    # Expression of g^0_ee (T = 0) by Yasuhara
    z = 4 * (4 / (9 * jnp.pi)) ** (1 / 6) * (rs / jnp.pi) ** (1 / 2)

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
    g0_ee = 1 / 8 * (z / jax.scipy.special.i1(z)) ** 2

    B = 9 / 16 * gamma_0 - 3 / 64 * (1 - g0_ee) - 16 / 15 * A
    C = -3 / 4 * gamma_0 + 9 / 16 * (1 - g0_ee) - 16 / 5 * A

    k_F = fermi_wavenumber(n_e)
    Q = k / k_F

    return (
        A * Q**4
        + B * Q**2
        + C
        + (A * Q**4 + (B + (8 / 3) * A) * Q**2 - C)
        * ((4 - Q**2) / (4 * Q))
        * jnp.log(jnp.abs((2 + Q) / (2 - Q)))
    )


@jax.jit
def eelfc_interpolationgregori2007(
    k: Quantity, T_e: Quantity, n_e: Quantity
) -> Quantity:

    Theta = (T_e / (fermi_energy(n_e) / (1 * ureg.boltzmann_constant))).m_as(
        ureg.dimensionless
    )

    return (
        eelfc_utsumiichimaru(k, T_e, n_e)
        + Theta * eelfc_geldartvosko(k, T_e, n_e)
    ) / (1 + Theta)
