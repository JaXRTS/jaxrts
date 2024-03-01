"""
This submodule is dedicated to the calculation of the free electron dynamic
structure.
"""

from .units import ureg, Quantity
from .plasma_physics import coulomb_potential_fourier, kin_energy, fermi_dirac
from .static_structure_factors import S_ii_AD
from .math import inverse_fermi_12_fukushima_single_prec

import pint
from typing import List

import jax
from jax import jit
from jax import numpy as jnp
from jpu import numpy as jnpu
import numpy as onp

from quadax import quadgk

import logging

logger = logging.getLogger(__name__)

import jpu

jax.config.update("jax_enable_x64", True)


@jit
def _W_salpeter(x: jnp.ndarray | float) -> jnp.ndarray:
    """
    Convenience function for the electron dielectric response function as
    defined in :cite:`Gregori.2003`.

    Parameters
    ----------
    x : jnp.ndarray | float

    Returns
    -------
    W:  jnp.ndarray
        The value of the convenience function.

    """
    
    def integrand(_x):
        return jnpu.exp(_x**2).m_as(ureg.dimensionless)
    
    integral, errl = quadgk(integrand, [0, x.m_as(ureg.dimensionless)], epsabs = 1E-20, epsrel = 1E-20)
    
    # x_v = jnp.linspace(0, x.magnitude, 3000).T
    # y_v = jnpu.exp(x_v**2)
    # integral = jax.scipy.integrate.trapezoid(y_v, x_v, axis=1)

    res = (
        1
        - 2
        * x
        * jnpu.exp(-(x**2))
        * integral
        + (jnpu.sqrt(jnp.pi) * x * jnpu.exp(-(x**2))) * 1j
    ).to_base_units()

    return res


@jit
def dielectric_function_salpeter(
    k: Quantity, T_e: Quantity, n_e: Quantity, E: Quantity | List
) -> jnp.ndarray:
    """
    Implementation of the quantum corrected Salpeter approximation of the
    electron dielectric response function.

    Parameters
    ----------
    k : Quantity
        Length of the scattering number (given by the scattering angle and the
        energies of the incident photons (unit: 1 / [length]).
    T_e: Quantity
        The electron temperature.
    n_e: Quantity
        The electron number density.
    E : Quantity | List
        The energy shift for which the free electron dynamic structure is
        calculated.
        Can be an interval of values.

    Returns
    -------
    eps: jnp.ndarray
         The electron dielectric response function.
    """

    omega = (E / (1 * ureg.planck_constant / (2 * jnp.pi))).to_base_units()

    v_t = jpu.numpy.sqrt(
        ((1 * ureg.boltzmann_constant) * T_e) / (1 * ureg.electron_mass)
    ).to_base_units()

    x_e = (omega / (jpu.numpy.sqrt(2) * k * v_t)).to_base_units()

    kappa = (
        (1 * ureg.planck_constant / (2 * jnp.pi))
        * k
        / (2 * jpu.numpy.sqrt(2) * (1 * ureg.electron_mass) * v_t)
    ).to_base_units()

    # The electron plasma frequency
    w_p_sq = (
        n_e
        * (1 * ureg.elementary_charge) ** 2
        / ((1 * ureg.electron_mass) * (1 * ureg.vacuum_permittivity))
    )

    eps = 1 + ((w_p_sq) / (k**2 * v_t**2)) * (1 / (4 * kappa)) * (
        (1 - _W_salpeter((x_e + kappa).to_base_units())) / (x_e + kappa)
        - (1 - _W_salpeter((x_e - kappa).to_base_units())) / (x_e - kappa)
    )

    return eps.m_as(ureg.dimensionless)


def S0ee_from_dielectric_func_FDT(
    k: Quantity,
    T_e: Quantity,
    n_e: Quantity,
    E: Quantity | List,
    dielectric_function: Quantity | List,
) -> Quantity:
    """
    Links the dielectric function to S0_ee via the fluctuation dissipation
    theorem, see, e.g., eqn (9) in :cite:`Gregori.2004`.

    .. math::

        S_{\\mathrm{ee}}^{0}(k,\\omega) =
        -\\frac{\\hbar}{1-\\exp(-\\hbar\\omega/k_{B}T_{e})}
        \\frac{\\epsilon_{0}k^{2}}{\\pi e^{2}n_{e}}
        \\mathrm{Im}\\left[\\frac{1}{\\epsilon(k,\\omega)}\\right]


    Parameters
    ----------
    k :  Quantity
        Length of the scattering number (given by the scattering angle and the
        energies of the incident photons (unit: 1 / [length]).
    T_e: Quantity
        The electron temperature.
    n_e: Quantity
        The electron number density.
    E : Quantity | List
        The energy shift for which the free electron dynamic structure is
        calculated.
        Can be an interval of values.
    dielectric_function: List:
        The dielectric function, normally dependent on :math:`k` and :math:`E`.
        Several aproximations for this are given in this submodule, e.g.
        :py:func:`~.dielectric_function_salpeter`.

    Returns
    -------
    S0_ee: jnp.ndarray
         The free electron dynamic structure.
    """
    return -(
        (1 * ureg.planck_constant / (2 * jnp.pi))
        / (1 - jpu.numpy.exp(-(E / (1 * ureg.boltzmann_constant * T_e))))
        * (
            ((1 * ureg.vacuum_permittivity) * k**2)
            / (jnp.pi * (1 * ureg.elementary_charge) ** 2 * n_e)
        )
        * jnp.imag(1 / dielectric_function)
    ).to_base_units()


@jit
def S0_ee_Salpeter(
    k: Quantity, T_e: Quantity, n_e: Quantity, E: Quantity | List
) -> jnp.ndarray:
    """
    Calculates the free electron dynamics structure using the quantum corrected
    Salpeter approximation of the electron dielectric response function.

    Parameters
    ----------
    k : Quantity
        Length of the scattering number (given by the scattering angle and the
        energies of the incident photons (unit: 1 / [length]).
    T_e : Quantity
        The electron temperature.
    n_e : Quantity
        The electron number density.
    E : Quantity | List
        The energy shift for which the free electron dynamic structure is
        calculated.
        Can be an interval of values.

    Returns
    -------
    S0_ee: jnp.ndarray
           The free electron dynamic structure.
    """
    # Perform the sign flip
    E = -E
    eps = dielectric_function_salpeter(k, T_e, n_e, E)
    return S0ee_from_dielectric_func_FDT(k, T_e, n_e, E, eps)


def _imag_diel_func_RPA_no_damping(
    k: Quantity, E: Quantity, chem_pot: Quantity, T: Quantity
) -> Quantity:
    """
    The imaginary part of the dielectric function without damping (i.e., in the
    limit nu → 0) from Bonitz's 'dielectric theory' script, Eqn 1.122,
    assuming a Fermi function for the particles' velocity distribution.

    Parameters
    ----------
    k : Quantity
        Length of the scattering number (given by the scattering angle and the
        energies of the incident photons (unit: 1 / [length]).
    E: Quantity
        The energy shift for which the free electron dynamic structure is
        calculated.
    chem_pot : Quantity
        The chemical potential in units of energy.
    T : Quantity
        The plasma temperature in Kelvin.

    Returns
    -------
    Quantity
        The imaginary part of the dielectric function
    """
    # Calculate the frequency shift
    w = E / ureg.hbar
    kappa = w * ureg.m_e / (ureg.hbar * k)
    prefactor = (
        ureg.k_B * T * ureg.m_e**2 * ureg.elementary_charge**2
    ) / (2 * jnp.pi * ureg.epsilon_0 * ureg.hbar**4 * k**3)
    exponent1 = (chem_pot - kin_energy(-k / 2 - kappa)) / (ureg.k_B * T)
    exponent2 = (chem_pot - kin_energy(+k / 2 - kappa)) / (ureg.k_B * T)
    return -prefactor * jnpu.log(
        (1 + jnpu.exp(exponent1)) / (1 + jnpu.exp(exponent2))
    )


def _real_diel_func_RPA_no_damping(
    k: Quantity, E: Quantity, chem_pot: Quantity, T: Quantity
) -> Quantity:
    """
    The real part of the dielectric function without damping (i.e., in the
    limit nu → 0) from Bonitz's 'dielectric theory' script, Eqn 1.120,
    assuming a Fermi function for the particles' velocity distribution.

    Parameters
    ----------
    k : Quantity
        Length of the scattering number (given by the scattering angle and the
        energies of the incident photons (unit: 1 / [length]).
    E: Quantity
        The energy shift for which the free electron dynamic structure is
        calculated.
    chem_pot : Quantity
        The chemical potential in units of energy.
    T : Quantity
        The plasma temperature in Kelvin.

    Returns
    -------
    Quantity
        The real part of the dielectric function
    """
    # Calculate the frequency shift
    w = E / ureg.hbar

    kappa = w * ureg.m_e / (ureg.hbar * k)
    prefactor = (
        ureg.e**2
        * ureg.m_e
        / (2 * jnp.pi**2 * ureg.epsilon_0 * ureg.hbar**2 * k**3)
    )
    # note from MAX: previous line may be missing a factor 2

    def integrand(Q):
        Q /= (1 * ureg.meter)
        alph_min_min = kappa - k / 2 - Q
        alph_min_plu = kappa - k / 2 + Q
        alph_plu_min = kappa + k / 2 - Q
        alph_plu_plu = kappa + k / 2 + Q
        numerator = jnpu.absolute(alph_min_min) * jnpu.absolute(alph_plu_plu)
        denominator = jnpu.absolute(alph_min_plu) * jnpu.absolute(alph_plu_min)
        ln_arg = numerator / denominator
        f_0 = fermi_dirac(Q, chem_pot, T)
        res = Q * f_0 * jnpu.log(ln_arg)
        return res.m_as(1 / ureg.meter)

    integral, errl = quadgk(integrand, [0, jnp.inf], epsabs = 1E-20, epsrel = 1E-20)
    integral /= (1 * ureg.meter)**2

    return 1 + (prefactor * integral).to_base_units()


def dielectric_function_RPA_no_damping(
    k: Quantity, E: Quantity, chem_pot: Quantity, T: Quantity
) -> Quantity:
    """
    The the dielectric function without damping (i.e., in the limit nu → 0)
    from Bonitz's 'dielectric theory' script, Eqn 1.120 and Eqn 1.122,
    assuming a Fermi function for the particles' velocity distribution.

    Parameters
    ----------
    k : Quantity
        Length of the scattering number (given by the scattering angle and the
        energies of the incident photons (unit: 1 / [length]).
    E: Quantity
        The energy shift for which the free electron dynamic structure is
        calculated.
    chem_pot : Quantity
        The chemical potential in units of energy.
    T : Quantity
        The plasma temperature in Kelvin.

    Returns
    -------
    Quantity
        The full dielectric function (complex number)
    """
    real = _real_diel_func_RPA_no_damping(k, E, chem_pot, T)
    imag = _imag_diel_func_RPA_no_damping(k, E, chem_pot, T)
    return real.m_as(ureg.dimensionless) + 1j * imag.m_as(ureg.dimensionless)


@jit
def S0_ee_RPA_no_damping(
    k: Quantity,
    T_e: Quantity,
    n_e: Quantity,
    E: Quantity | List,
    chem_pot: Quantity,
) -> jnp.ndarray:
    """
    Calculates the free electron dynamics structure using the quantum corrected
    Salpeter approximation of the electron dielectric response function.

    Parameters
    ----------
    k : Quantity
        Length of the scattering number (given by the scattering angle and the
        energies of the incident photons (unit: 1 / [length]).
    T_e : Quantity
        The electron temperature.
    n_e : Quantity
        The electron number density.
    E : Quantity | List
        The energy shift for which the free electron dynamic structure is
        calculated.
        Can be an interval of values.
    chem_pot : Quantity
        The chemical potential in units of energy.

    Returns
    -------
    S0_ee: jnp.ndarray
           The free electron dynamic structure.
    """
    E = -E
    eps = dielectric_function_RPA_no_damping(k, E, chem_pot, T_e)
    return S0ee_from_dielectric_func_FDT(k, T_e, n_e, E, eps)

def zeta_e_squared(k : Quantity, n_e : Quantity, T_e : Quantity) -> jnp.ndarray:
    
    """
    Calculates the k-dependent inverse screening length.
    """
    
    # k = k[:, jnp.newaxis]

    w_pe = jpu.numpy.sqrt((4 * jnp.pi * n_e * ureg.elementary_charge ** 2) / (ureg.electron_mass))
    
    p_e = jpu.numpy.sqrt(ureg.electron_mass * ureg.boltzmann_constant * T_e)
    v_e = ureg.hbar * (k / 2) / jnp.sqrt(2) * p_e
    
    kappa_De = w_pe / v_e
    kappa_e = ureg.hbar * (k / 2) / (jnp.sqrt(2) * p_e)    
    gamma_e = jnp.sqrt(2 * jnp.pi) * ureg.hbar / p_e
    D_e = (n_e * gamma_e ** 3).to_base_units()
    
    eta_e = inverse_fermi_12_fukushima_single_prec(D_e / 2)
    
    prefactor = kappa_De ** 2 / (jnp.sqrt(jnp.pi) * kappa_e * D_e)

    def integrand(_w):
        # w_e = ureg.electron_mass * (_w / k) / (jnp.sqrt(2) * p_e)
        return (1 / _w) * jnp.log((1 + jnp.exp(eta_e - (_w - kappa_e) ** 2))/(1 + jnp.exp(eta_e - (_w + kappa_e) ** 2)))
    
    # integrand = (1 / w_intervall) * jnp.log((1 + jnp.exp(eta_e - (w_intervall - kappa_e) ** 2))/(1 + jnp.exp(eta_e - (w_intervall + kappa_e) ** 2)))
    # integral = jax.scipy.integrate.trapezoid(w_intervall.T, integrand, axis=1)
    
    integral, errl = quadgk(integrand, [0, jnp.inf], epsabs = 1E-20, epsrel = 1E-20)

    return prefactor * integral


def collision_frequency_BA(
    E: Quantity,
    T: Quantity,
    m_ion: Quantity,
    n_e: Quantity,
    chem_pot: Quantity,
    Zf: float,
):

    w = (E / ureg.hbar)

    T_e = T

    # k_intervall = jnp.linspace(0, 1000, 2000)

    def integrand(k):
        return (
        k*4
        * S_ii_AD(k, T_e, n_e, m_ion, Zf)
        * (zeta_e_squared(k, n_e, T_e) - k**2 * (dielectric_function_RPA_no_damping(k, E, chem_pot, T) - 1))
        / (k**2 + zeta_e_squared(k, n_e, T_e)) ** 2
    ).to_base_units().magnitude

    integral, errl = quadgk(integrand, [0, jnp.inf], epsabs = 1E-20, epsrel = 1E-20)
    # integrand = (
    #     k_intervall**4
    #     * S_ii_AD(k_intervall, T_e, n_e, m_ion, Zf)
    #     * (zeta_e_squared(k_intervall) - k_intervall**2 * (eps_ee_f - 1))
    #     / (k_intervall**2 + zeta_e_squared) ** 2
    # )

    # integral = jax.scipy.integrate.trapezoid(k_intervall.T, integrand, axis=1)

    # Calculate ion density and the ionic plasma frequency
    n_i = n_e / Zf
    w_pi = jpu.numpy.sqrt((4 * jnp.pi * n_i * Zf ** 2 * ureg.elementary_charge ** 2) / (m_ion))

    prefactor = (
        1j
        * (2 * jnp.pi / 3)
        * (w_pi**2 / w)
        * (1 / n_e)
        * (m_ion / ureg.electron_mass)
    )

    return prefactor * integral


def dielectric_function_BMA(
    k: Quantity,
    E: Quantity | List,
    chem_pot: Quantity,
    T: Quantity,
    n_e: Quantity,
    m_ion: Quantity,
    Zf: float
) -> jnp.ndarray:
    """
    Calculates the Born-Mermin Approximation for the dielectric function, which takes collisions
    into account.

    """
    w = E / ureg.hbar
    coll_freq = collision_frequency_BA(E, T, m_ion, n_e, Zf)

    numerator = (
        (w + 1j * coll_freq)
        * dielectric_function_RPA_no_damping(k, 0, chem_pot, T)
        * dielectric_function_RPA_no_damping(
            k, E + 1j * ureg.hbar * coll_freq, chem_pot, T
        )
    )
    denumerator = w * dielectric_function_RPA_no_damping(
        k, E, chem_pot, T
    ) + 1j * coll_freq * dielectric_function_RPA_no_damping(
        k, E + 1j * ureg.hbar * coll_freq, chem_pot, T
    )

    return numerator / denumerator

def S0_ee_BMA(k: Quantity, T: Quantity, chem_pot : Quantity, m_ion : Quantity, n_e: Quantity, Zf : float, E: Quantity | List) -> jnp.ndarray:
    
    E = -E
    eps = dielectric_function_BMA(k, E, chem_pot, T, n_e, m_ion, Zf)
    return S0ee_from_dielectric_func_FDT(k, T, n_e, E, eps)


# def ret_diel_func_DPA(k: Quantity, Z: jnp.ndarray) -> Quantity:
#     """
#     Retarded dielectric funciton in diagonalised polarization approximation DPA
#     See :cite:`Chapman.2015`, (Eqn. 2.80), by inserting a Kronecker delta.
#     """

#     # The electron part
#     eps = 1 - PiR_e_e * coulomb_potential_fourier(
#         -1, -1, k
#     )
#     # The ion part
#     eps -= jnpu.sum(PiR_ion_ion(Z, ) * coulomb_potential_fourier(Z, Z, k))
#     return eps
