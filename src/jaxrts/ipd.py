"""
Module to account for Ionization Potential Depression (IPD)

All these implementations and formulas are taken from 'Modifications of Bound
States in Dense Plasma Environments' by Rory A. Baggott. :cite:`Baggott.2017`
"""

import logging

import jax
from jax import numpy as jnp
from jpu import numpy as jnpu
from quadax import quadts as quad

from .math import fermi_neg12_rational_approximation_antia
from .units import Quantity, ureg
from .plasma_physics import fermi_energy
from .plasma_physics import (
    chem_pot_interpolationIchimaru as chem_pot_interpolation,
)

logger = logging.getLogger(__name__)


@jax.jit
def inverse_screening_length_e(ne: Quantity, Te: Quantity):
    """ """

    chem_pot = chem_pot_interpolation(Te, ne)
    beta = 1 / (1 * ureg.boltzmann_constant * Te)

    fermi_integral_neg1_2 = fermi_neg12_rational_approximation_antia(
        (chem_pot * beta).m_as(ureg.dimensionless)
    )

    E_F = fermi_energy(ne)

    k_sq = (
        12
        * jnp.pi ** (5 / 2)
        * (1 * ureg.elementary_charge**2 / (4 * jnp.pi * ureg.epsilon_0))
        * ne
        * beta
        * fermi_integral_neg1_2
        / (beta * E_F) ** (3 / 2)
    )

    return jnpu.sqrt(k_sq).to(1 / ureg.angstrom)


@jax.jit
def ipd_debye_hueckel(
    Zi: float, ne: Quantity, ni: Quantity, Te: Quantity, Ti: Quantity
) -> Quantity:
    """
    The correction to the ionization potential for the m-th ionization stage in
    Debye-Hueckel approximation.

    .. note::

       The Debye-Hueckel approximation is physically meaningful only when
       the coupling parameter << 1, such that Coulomb forces are weak
       perturbations.

    Parameters
    ----------
    Z_i
        The (mean) charge state of the ions.
    n_e
        Electron density. Units of 1/[length]**3.
    n_i
        Ion density. Units of 1/[length]**3.
    T_e
        The electron temperature.
    T_i
        The ion temperature.

    Returns
    -------
    Quantity
        The ipd shift in units of electronvolt.
    """

    kappa_class = jnpu.sqrt(
        (
            1
            * ureg.elementary_charge**2
            / (1 * ureg.epsilon_0 * ureg.boltzmann_constant * Te)
        )
        * (jnpu.sum(Zi**2 * ni) + ne)
    )

    # The ionization potential depression energy shift
    ipd_shift = kappa_class * (
        -(Zi + 1) * ureg.elementary_charge**2 / (4 * jnp.pi * ureg.epsilon_0)
    )

    return ipd_shift.to(ureg.electron_volt)


@jax.jit
def ipd_ion_sphere(Zi: Quantity, ne: Quantity, ni: Quantity) -> Quantity:
    """
    The correction to the ionization potential for the m-th ionization stage in
    the ion-sphere model. The ion-sphere model considers the ions to be
    strongly correlated. (see also :cite:`Zimmermann.1980`)

    Parameters
    ----------
    Z_i
        The (mean) charge state of the ions.
    n_e
        Electron density. Units of 1/[length]**3.
    n_i
        Ion density. Units of 1/[length]**3.

    Returns
    -------
    Quantity
        The ipd shift in units of electronvolt.
    """
    # This function is not well-defined for Zi==0:
    Zi = jnp.clip(Zi, 1e-6)

    # The ion-sphere radius, determined by the ion density n_i such that the
    # average distance to the nearest neighbor ion is
    # approximately 2 R_0.
    R_0 = (3 * Zi / (4 * jnp.pi * ne)) ** (1 / 3)

    ipd_shift = (
        -3
        * (Zi + 1)
        * 1
        * ureg.elementary_charge**2
        / (R_0 * 8 * jnp.pi * 1 * ureg.epsilon_0)
    )

    return ipd_shift.to(ureg.electron_volt)


@jax.jit
def ipd_stewart_pyatt(
    Zi: float,
    ne: Quantity,
    ni: Quantity,
    Te: Quantity,
    Ti: Quantity,
    Z_and_pop=None,
) -> Quantity:
    """
    The correction to the ionization potential in the Stewart-Pyatt model using
    the small bound state approximation. This model is founded on the
    Thomas-Fermi Model for the electrons and extends it to include ions in the
    vicinity of a given nucleus. Taken from :cite:`Ropke.2019` Eq. (2).

    .. note::

       The Stewart-Pyatt value is always below both the Debye and ion sphere
       results.

    Parameters
    ----------
    Z_i
        The (mean) charge state of the ions.
    n_e
        Electron density. Units of 1/[length]**3.
    n_i
        Ion density. Units of 1/[length]**3.
    T_e
        The electron temperature.
    T_i
        The ion temperature.

    Returns
    -------
    Quantity
        The ipd shift in units of electronvolt.
    """

    if Z_and_pop is None:
        Zp = Zi
        Zbar = Zi
    else:
        Z, pop = Z_and_pop
        Zbar = jnpu.mean(Z * pop)
        Zp = jnpu.mean((Z * pop) ** 2) / Zbar

    R_i = (3 / (4 * jnp.pi * ni)) ** (1 / 3)

    Gamma_i = (
        Zi
        * Zp
        * 1
        * ureg.elementary_charge**2
        / (4 * jnp.pi * ureg.epsilon_0 * R_i * ureg.boltzmann_constant * Ti)
    ).m_as(ureg.dimensionless)

    Lambda_i = (3 * Gamma_i) ** (3/2)

    ipd_shift = -(1 * ureg.boltzmann_constant * Ti) / (2 * (Zp)) * ((1 + Lambda_i)**(2/3) - 1)

    return ipd_shift.to(ureg.electron_volt)


@jax.jit
def ipd_ecker_kroell(
    Zi: float, ne: Quantity, ni: Quantity, Te: Quantity, Ti: Quantity
) -> Quantity:
    """
    The correction to the ionization potential for the m-th ionization stage in
    the Ecker-Kroell model.
    This model is similar to the model of Stewart-Pyatt and divided the radial
    dimension into three regions. For details see :cite:`EckerKroell.1963`.

    Parameters
    ----------
    Z_i
        The (mean) charge state of the ions.
    n_e
        Electron density. Units of 1/[length]**3.
    n_i
        Ion density. Units of 1/[length]**3.
    T_e
        The electron temperature.
    T_i
        The ion temperature.

    Returns
    -------
    Quantity
        The ipd shift in units of electronvolt.

    """

    lambda_Di = jnpu.sqrt(
        ureg.epsilon_0
        * ureg.boltzmann_constant
        * Ti
        / (ne * ureg.elementary_charge**2)
    )

    R_0 = (3 / (4 * jnp.pi * ni)) ** (1 / 3)

    # The critical density in the model of Ecker-Kroell

    n_c = (3 / (4 * jnp.pi)) * (
        4
        * jnp.pi
        * 1
        * ureg.epsilon_0
        * ureg.boltzmann_constant
        * Te
        / ureg.elementary_charge**2
    ) ** 3

    # The constant in Ecker-Kroells model, which is determined from the
    # continuity of the potential across the critical density.

    C = (
        2.2
        * jnpu.sqrt(
            ureg.elementary_charge**2
            / (ureg.boltzmann_constant * Te)
            / (4 * jnp.pi * ureg.epsilon_0)
        )
        * n_c ** (1 / 6)
    ).m_as(ureg.dimensionless)

    ipd_c1 = -1 * ureg.elementary_charge**2 / (ureg.epsilon_0 * lambda_Di) * Zi
    ipd_c2 = -C * ureg.elementary_charge**2 / (ureg.epsilon_0 * R_0) * Zi

    # The ionization potential depression energy shift
    ipd_shift = jnpu.where(ni <= n_c, ipd_c1, ipd_c2)

    return ipd_shift.to(ureg.electron_volt)


@jax.jit
def ipd_pauli_blocking(
    Zi: float, ne: Quantity, ni: Quantity, Te: Quantity, Ti: Quantity
) -> Quantity:
    """
    The correction to the ionization potential due to Pauli blocking, as
    described in :cite:`Ropke.2019`.

    Parameters
    ----------
    Z_i
        The (mean) charge state of the ions.
    n_e
        Electron density. Units of 1/[length]**3.
    n_i
        Ion density. Units of 1/[length]**3.
    T_e
        The electron temperature.
    T_i
        The ion temperature.

    Returns
    -------
    Quantity
        The ipd shift in units of electronvolt.
    """
    # This function is not well-defined for Zi==0:
    Zi = jnp.clip(Zi, 1e-6)

    chem_pot = chem_pot_interpolation(Te, ne)

    # Reduced Bohr radius
    a_Z = ureg.a0 / Zi

    @jax.jit
    def integrand(p):
        p /= 1 * ureg.angstrom

        res = (
            p**2
            / (1 + a_Z**2 * p**2) ** 3
            * (
                1
                / (
                    jnpu.exp(
                        (
                            (1 * ureg.hbar * p) ** 2 / (2 * ureg.electron_mass)
                            - chem_pot
                        )
                        / (1 * ureg.boltzmann_constant * Te)
                    )
                    + 1
                )
            )
        )

        return res.m_as(1 / ureg.angstrom**2)

    integral, errl = quad(integrand, [0, jnp.inf], epsabs=1e-15, epsrel=1e-15)
    integral /= 1 * ureg.angstrom**3
    # The ionization potential depression energy shift
    ipd_shift = -(
        Zi
        * ureg.elementary_charge**2
        / (4 * jnp.pi * ureg.epsilon_0)
        * (16 * a_Z**2)
        / jnp.pi
        * integral
    )

    return ipd_shift.to(ureg.electron_volt)
