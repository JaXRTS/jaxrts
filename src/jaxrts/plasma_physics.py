"""
This submodule contains basic formulas used in plasma physics.
"""

from typing import List

from .units import ureg, Quantity, to_array
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
def fermi_wavenumber(n_e: Quantity) -> Quantity:

    return (3 * jnp.pi**2 * n_e) ** (1 / 3)


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
    k_fsquared = (3 * jnp.pi**2 * n_e) ** (2 / 3)
    E_F = factor1 * k_fsquared
    return E_F.to(ureg.electron_volt)


def wiegner_seitz_radius(n_e: Quantity) -> Quantity:
    """
    Return the Wiegner Seitz Radius :math:`r_s`.

    .. math::

        r_s = \\sqrt[3]{\\frac{3}{4\\pi n_e}}

    .. note::

        Some authors use the Wiegner-Seitz radius as a dimensionless unit by
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


def chem_pot_interpolationIchimaru(T: Quantity, n_e: Quantity) -> Quantity:
    """
    Interpolation function for the chemical potential between the classical and
    quantum region, originally from :cite:`Ichimaru.2018`, eqn.(3.147). The
    same formula (with a number flip in parameter A) is  in
    :cite:`Gregori.2003`, eqn. (19).

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
    A = 0.25954
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


@jax.jit
def Debye_Hueckel_screening_length(
    n: Quantity | List, T: Quantity, Z: float | List | jnp.ndarray = 1.0
) -> Quantity:
    """
    Calculate the Debye-Hückel screening length. Use the general formula using
    a sum over n, Z.

    Parameters
    ----------
    n : Quantity or List
        Electron density in 1/[length]**3
    T : Quantity
        The temperature in [K]. Many authors, e.g., :cite:`Gregori.2010`
        suggest an effective temperature which interpolates between the
        system's temperature and the fermi temperature.
        See :py:func:`temperature_interpolation`
    Z : float, list or np.ndarray, defaults to 1
        Ionization. The default value of 1 corresponds to electrons
    """
    n = to_array(n)
    Z = to_array(Z)
    num = ureg.epsilon_0 * ureg.k_B * T
    denom = jnpu.sum(n * (Z * ureg.elementary_charge) ** 2)
    return jnpu.sqrt(num / denom)


@jax.jit
def temperature_interpolation(
    n_e: Quantity, T_e: Quantity, power: int = 2
) -> Quantity:
    """
    Interpolate between electron temperature and Fermi temperature. (See, e.g.,
    :cite:`Gericke.2010`, who propose a `power` of 4)

    .. math::

        T = \\left( T_e^p + T_F^p\\right)^(1/p)

    Parameters
    ----------
    n_e : Quantity
        Electron density in 1/[length]**3
    T_e : Quantity
        Electron temperature in [K]
    power : int, optional
        Power to use for interpolation (default: 2)

    Returns
    -------
    Quantity
        Interpolated temperature
    """
    # Calculate the fermi temperature
    T_f = fermi_energy(n_e) / ureg.k_B
    return (
        (T_e.m_as(ureg.kelvin) ** power + T_f.m_as(ureg.kelvin) ** power)
        ** (1 / power)
    ) * ureg.kelvin


@jax.jit
def degeneracy_param(n_e: Quantity, T_e: Quantity) -> Quantity:
    """
    Calculate the plasma degeneracy parameter.

    The plasma degeneracy parameter is given by the product of the electron
    density `n_e` and the cube of the thermal de Broglie wavelength.
    For values about unity, the probability clouds of the electron wave
    functions overlap.

    A classical treatment of the plasma is allowed for values << 1, while a
    fully degenerate electron gas is appropriate for values > 10. See
    :cite:`Kraus.2012`.

    Parameters
    ----------
    n_e : Quantity
        Electron density in 1/[length]**3
    T_e : Quantity
        Electron temperature in [K]

    Returns
    -------
    degeneracy_param : Quantity
        Plasma degeneracy parameter

    """
    return n_e * therm_de_broglie_wl(T_e) ** 3


def interparticle_spacing(Z1: float, Z2: float, n_e: Quantity):

    return (3 * (Z1 * Z2) ** (1 / 2) / (4 * jnp.pi * n_e)) ** (1 / 3)


def coupling_param(Z1: float, Z2: float, n_e: Quantity, T_e: Quantity):
    """
    Returns the degree of interparticle coupling with corresponding charge
    numbers Z1 and Z2 at temperature T_e and density n_e.
    """

    intspac = interparticle_spacing(Z1, Z2, n_e)

    return (
        Z1
        * Z2
        * 1
        * ureg.elementary_charge**2
        / (
            4
            * jnp.pi
            * ureg.epsilon_0
            * intspac
            * 1
            * ureg.boltzmann_constant
            * T_e
        )
    ).m_as(ureg.dimensionless)


def therm_de_broglie_wl(T):
    return ureg.hbar * jnpu.sqrt(
        (2 * jnp.pi) / (ureg.electron_mass * ureg.k_B * T)
    )


def compton_energy(probe_energy, scattering_angle):

    shift = (
        probe_energy
        * (
            1
            - (1)
            / (
                1
                + probe_energy
                / (1 * ureg.electron_mass * ureg.speed_of_light**2)
                * (1 - jnpu.cos(scattering_angle))
            )
        )
    ).to(ureg.electron_volt)
    return shift


@jax.jit
def susceptibility_from_epsilon(epsilon: Quantity, k: Quantity) -> Quantity:
    """
    Calculate the full susceptilibily from a given dielectric function epsilon
    by inverting

    ..math::

        \\varepsilon^{-1} = 1 + V_{ee} xi_{ee}

    Where :math:`V_{ee}` is the Coulomb potential in k space.

    See, e.g., :cite:`Dandrea.1986`.
    """
    Vee = coulomb_potential_fourier(-1, -1, k)
    return (epsilon ** (-1) - 1) / Vee


@jax.jit
def epsilon_from_susceptibility(xi: Quantity, k: Quantity) -> Quantity:
    """
    Calculate the dielectric function from a full susceptibility xi

    ..math::

        \\varepsilon^{-1} = 1 + V_{ee} xi_{ee}

    Where :math:`V_{ee}` is the Coulomb potential in k space.

    See, e.g., :cite:`Dandrea.1986`.
    """
    Vee = coulomb_potential_fourier(-1, -1, k)
    return ((1 + Vee * xi) ** -1).m_as(ureg.dimensionless)


@jax.jit
def noninteracting_susceptibility_from_eps_RPA(
    epsilon: Quantity, k: Quantity
) -> Quantity:
    """
    Calculates the non-interacting susceptilibily from a given dielectric
    function epsilon in RPA.

    See, e.g., :cite:`Fortmann.2010`.

    ..math::

        \\xi^{(0)}_{e} = \\frac{1 - \\varepsilon^{RPA}}{V_{ee}(k)}

    Where :math:`V_{ee}` is the Coulomb potential in k space.
    """
    Vee = coulomb_potential_fourier(-1, -1, k)
    return (1 - epsilon) / Vee
