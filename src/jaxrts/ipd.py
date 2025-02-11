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

logger = logging.getLogger(__name__)


@jax.jit
def chem_pot_interpolation(T: Quantity, n_e: Quantity) -> Quantity:
    """
    Interpolation function for the chemical potential between the classical and
    quantum region, given in :cite:`Gregori.2003`, eqn. (19).

    Parameters
    ----------
    T
        The plasma temperature in Kelvin.
    n_e
        Electron density. Units of 1/[length]**3.

    Returns
    -------
    Quantity
        Chemical potential
    """
    A = 0.25945
    B = 0.072
    b = 0.858

    E_f = ureg.hbar**2 / (2 * ureg.m_e) * (3 * jnp.pi**2 * n_e) ** (2 / 3)
    Theta = (ureg.k_B * T / E_f).to_base_units()
    f = (
        (-3 / 2 * jnpu.log(Theta))
        + (jnpu.log(4 / (3 * jnp.sqrt(jnp.pi))))
        + (
            (A * Theta ** (-b - 1) + B * Theta ** (-(b + 1) / 2))
            / (1 + A * Theta ** (-b))
        )
    )
    return f * ureg.k_B * T


@jax.jit
def inverse_screening_length_e(q: Quantity, ne: Quantity, Te: Quantity):
    """
    Inverse screening length for arbitrary degeneracy as needed for WDM
    applications. (Taken from :cite:`Baggott.2017`)
    """

    chem_pot = chem_pot_interpolation(Te, ne)
    beta = 1 / (1 * ureg.boltzmann_constant * Te)

    fermi_integral_neg1_2 = fermi_neg12_rational_approximation_antia(
        (chem_pot * beta).m_as(ureg.dimensionless)
    )

    therm_wv = jnpu.sqrt(
        (2 * jnp.pi * 1 * ureg.hbar**2)
        / ((1 * ureg.electron_mass) * 1 * ureg.boltzmann_constant * Te)
    )

    k_sq = (
        (q**2)
        / (ureg.epsilon_0 * ureg.boltzmann_constant * Te)
        * (2.0 / therm_wv**3)
        * fermi_integral_neg1_2
    )

    # k_sq = pref * fermi_integral_neg1_2

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
    # The Debye (screening) wavenumber for the electrons for arbitrary
    # degeneracy
    # K_De = inverse_screening_length_e(ne, Te)
    # The Debye wavenumber for the ions
    # K_Di_squared = Zi ** 2 * 1 * ureg.elementary_charge ** 2  * ni /
    # (ureg.epsilon_0 * ureg.boltzmann_constant * Ti)

    # kappa = jnpu.sqrt(K_Di_squared + K_De**2)

    kappa_class = jnpu.sqrt(
        Zi**2
        * ni
        * 1
        * ureg.elementary_charge**2
        / (1 * ureg.epsilon_0 * ureg.boltzmann_constant * Ti)
    )
    # The ionization potential depression energy shift
    ipd_shift = (
        -(Zi + 1)
        * ureg.elementary_charge**2
        * kappa_class
        / (4 * jnp.pi * ureg.epsilon_0)
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

    pref = 9 / 5  # Zimmermann & More 1980
    # pref = 3/2 # Standard prefactor

    # The ion-sphere radius, determined by the ion density n_i such that the
    # average distance to the nearest neighbor ion is
    # approximately 2 R_0.
    R_0 = (3 * Zi / (4 * jnp.pi * ne)) ** (1 / 3)

    ipd_shift = (
        -pref
        * Zi**2
        * 1
        * ureg.elementary_charge**2
        / (R_0 * 4 * jnp.pi * 1 * ureg.epsilon_0)
    )

    return ipd_shift.to(ureg.electron_volt)


@jax.jit
def ipd_stewart_pyatt(
    Zi: float, ne: Quantity, ni: Quantity, Te: Quantity, Ti: Quantity
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

    R_0 = (3 * Zi / (4 * jnp.pi * ne)) ** (1 / 3)

    # The Debye (screening) wavenumber for the electrons for arbitrary
    # degeneracy
    K_De = inverse_screening_length_e(-1 * ureg.elementary_charge, ne, Te)
    # The Debye wavenumber for the ions
    K_Di = jnpu.sqrt(
        Zi**2
        * 1
        * ureg.elementary_charge**2
        * ni
        / (1 * ureg.epsilon_0 * ureg.boltzmann_constant * Ti)
    )
    kappa = jnpu.sqrt(K_De**2 + K_Di**2).to(1 / ureg.angstrom)

    # The ionization potential depression energy shift
    s = 1 / (kappa * R_0)
    ipd_shift = -(
        (
            3
            * (Zi + 1)
            * ureg.elementary_charge**2
            / (2 * 4 * jnp.pi * 1 * ureg.epsilon_0 * R_0)
        )
        * ((1 + s**3) ** (2 / 3) - s**2)
    )

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
        * jnpu.sqrt(ureg.elementary_charge**2 / (ureg.boltzmann_constant * Te))
        * n_c ** (1 / 6)
    )

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
                            (
                                (1 * ureg.hbar * p) ** 2
                                / (2 * ureg.electron_mass)
                                - chem_pot
                            )
                            / (1 * ureg.boltzmann_constant * Te)
                        )
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
