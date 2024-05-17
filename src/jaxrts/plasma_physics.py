"""
This submodule contains basic formulas used in plasma physics.
"""

from .units import ureg, Quantity
from .math import fermi_integral

import jax
from jax import numpy as jnp
from jpu import numpy as jnpu


def plasma_frequency(electron_density: Quantity) -> Quantity:
    """
    Calculate the plasma frequency :math:`\\omega_\\text{pe}`

    .. math::
       \\omega_\\text{pe} = \\sqrt{\\frac{e^2 n_e}{\\epsilon_0 m_e}}

    where :math:`e` is the elementary charge,
    where :math:`n_e` is the electron density,
    where :math:`\\epsilon_0` is the vacuum permittivity,
    where :math:`m_e` is the electron's mass.

    Parameters
    ----------
    electron_density
        The electron density in units of 1/volume.

    Returns
    -------
    Quantity
        The plasma frequency in units of Hz.
    """
    return jnpu.sqrt(
        (ureg.elementary_charge**2 * electron_density)
        / (ureg.vacuum_permittivity * ureg.electron_mass)
    ).to(ureg.Hz)


def thomson_momentum_transfer(energy: Quantity, angle: Quantity):
    """
    Momentum transfer :math:`k = \\mid\\vec{k}\\mid`, assuming that the
    absolute value of the momentum for incoming and scattered light is only
    slightly changed.
    """
    return (2 * energy) / (ureg.hbar * ureg.c) * jnpu.sin(angle / 2)


def coulomb_potential_fourier(
    Z1: Quantity | float, Z2: Quantity | float, k: Quantity
) -> Quantity:
    """
    The Fourier transform of the Coloumb potential.

    Parameters
    ----------
    Z1, Z2 : Quantity | float
        Charge 1 and Charge 2
    k : Quantity
        Scattering vector length

    Returns
    -------
    Quantity
        The Fourier transform of the Coloub potential.
    """
    return (
        (1 / ureg.vacuum_permittivity)
        * (Z1 * Z2 * ureg.elementary_charge**2)
        / k**2
    )


@jax.jit
def kin_energy(k: Quantity) -> Quantity:
    """
    Kinetic energy of a free electron with wavevector ``k``.

    Parameters
    ----------
    k : Quantity
        Wavevector in units of 1/[length]

    Returns
    -------
    Quantity
        Kinetic energy
    """
    return (ureg.hbar**2 * k**2) / (2 * ureg.electron_mass)


@jax.jit
def fermi_dirac(k: Quantity, chem_pot: Quantity, T: Quantity) -> Quantity:
    """
    Return the Fermi-Dirac distribution.

    ..math ::

        f=\\frac{1}{\\exp \\left((E-\\mu) / k_{\\mathrm{B}} T\\right)+1}

    Parameters
    ----------
    k : Quantity
        Length of the scattering number (given by the scattering angle and the
        energies of the incident photons (unit: 1 / [length]).
    chem_pot : Quantity
        The chemical potential in units of energy.
    T : Quantity
        The plasma temperature in Kelvin.

    Returns
    -------
    Quantity
        The Fermi-Dirac distribution
    """
    energy = kin_energy(k)
    exponent = (energy - chem_pot) / (ureg.k_B * T)
    return 1 / (jnpu.exp(exponent) + 1)


@jax.jit
def fermi_energy(n_e: Quantity) -> Quantity:
    """
    Calculate the Fermi energy of an ideal fermi gas from a given electron
    densiy.

    .. math::

        E_F=\\frac{{\\hbar}^2}{2m_e}\\left(3{\\pi}^2 n_e\\right)^{\\frac{2}{3}}

    Parameters
    ----------
    n_e : Quantity
        Electron density. Units of 1/[length]**3.

    Returns
    -------
    E_F : Quantity
        Fermi energy
    """
    factor1 = ureg.hbar**2 / (2 * ureg.m_e)
    factor2 = (3 * jnp.pi**2 * n_e) ** (2 / 3)
    E_F = factor1.to(
        ureg.centimeter**4 * ureg.gram / ureg.second**2
    ) * factor2.to(1 / ureg.centimeter**2)
    return E_F.to(ureg.electron_volt)


def wiegner_seitz_radius(n_e: Quantity) -> Quantity:
    """
    Return the Wiegner Seitz Radius :math:`r_s`.

    .. math::

        r_s = \\sqrt[3]{\\frac{3}{4\\pi n_e}}

    .. note::

        Some authors use the Wiegner Seits radius as a dimensionless unit by
        dividing by the Bohr radius. This is not done here, rather :math:`r_s`
        has the dimensionality of a length.

    Parameters
    ----------
    n_e
        Electron density. Units of 1/[length]**3.

    Returns
    -------
    Quantity
        The Wiegner Seitz Radius :math:`r_s` in units of a length.
    """
    return (3 / (4 * jnp.pi * n_e)) ** (1 / 3)


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

    Ef = fermi_energy(n_e)
    Theta = (ureg.k_B * T / Ef).to_base_units()
    f = (
        (-3 / 2 * jnpu.log(Theta))
        + (jnpu.log(4 / (3 * jnp.sqrt(jnp.pi))))
        + (
            (A * Theta ** (-b - 1) + B * Theta ** (-(b + 1) / 2))
            / (1 + A * Theta ** (-b))
        )
    )
    return f * ureg.k_B * T


def degeneracy_param(n_e: Quantity, T_e: Quantity) -> Quantity:
    return n_e * therm_de_broglie_wl(T_e)**3


def therm_de_broglie_wl(T):
    return ureg.hbar * jnpu.sqrt(
        (2 * jnp.pi) / (ureg.electron_mass * ureg.k_B * T)
    )
