"""
This submodule is dedicated to the calculation of the free electron dynamic
structure.
"""

from .units import ureg, Quantity
from .plasma_physics import coulomb_potential_fourier, kin_energy, fermi_dirac
from .static_structure_factors import S_ii_AD
from .math import inverse_fermi_12_fukushima_single_prec

import pint

import time
from typing import List

import jax
from jax import jit
from jax import numpy as jnp
from jpu import numpy as jnpu
import numpy as onp

from quadax import *

import logging

logger = logging.getLogger(__name__)

import jpu

jax.config.update("jax_enable_x64", True)

from .helpers import timer


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
        return jnp.exp(_x**2)

    @jax.vmap
    def integ(xi):
        integral, errl = quadgk(
            integrand,
            [0, xi.m_as(ureg.dimensionless)],
            epsabs=1e-20,
            epsrel=1e-20,
        )
        return integral

    integral = integ(x)

    res = (
        1
        - 2 * x * jnpu.exp(-(x**2)) * integral
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

@jit
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
   
    res =  -(
        (1 * ureg.planck_constant / (2 * jnp.pi))
        / (1 - jpu.numpy.exp(-(E / (1 * ureg.boltzmann_constant * T_e))))
        * (
            ((1 * ureg.vacuum_permittivity) * k**2)
            / (jnp.pi * (1 * ureg.elementary_charge) ** 2 * n_e)
        )
        * jnp.imag((1 / dielectric_function))
    ).to_base_units()
    
    return res

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


@jit
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


@jit
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
        Q /= 1 * ureg.meter
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

    integral, errl = quadgk(
        integrand, [0, jnp.inf], epsabs=1e-20, epsrel=1e-20
    )
    integral /= (1 * ureg.meter) ** 2

    return 1 + (prefactor * integral).to_base_units()


@jit
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


@jit
def inverse_screening_length_exact(T: Quantity, chem_pot: Quantity):
    prefactor_debye = (
        ureg.elementary_charge**2 * ureg.electron_mass ** (3 / 2)
    ) / (jnp.sqrt(2) * jnp.pi**2 * ureg.epsilon_0 * ureg.hbar**3)

    def integrand_debye(Ep):

        Ep *= 1 * ureg.electron_volt

        res = Ep ** (-1 / 2) * (
            jnp.exp(
                ((Ep - chem_pot) / (ureg.boltzmann_constant * T)).m_as(
                    ureg.dimensionless
                )
            )
            + 1
        ) ** (-1)

        return res.m_as(ureg.electron_volt ** (-1 / 2))

    integral_debye, errl_debye = quadts(
        integrand_debye, [0, jnp.inf], epsabs=1e-20, epsrel=1e-20
    )
    integral_debye *= (1 * ureg.electron_volt) ** (1 / 2)

    return jpu.numpy.sqrt(prefactor_debye * integral_debye)

@jit
def inverse_screening_length_non_degenerate(n_e: Quantity, T: Quantity):

    return jpu.numpy.sqrt(
        n_e
        * ureg.elementary_charge**2
        / (ureg.epsilon_0 * ureg.boltzmann_constant * T)
    )

@jit
def statically_screened_ie_debye_potential(
    q: Quantity, kappa: Quantity, Zf: float
):
    return (
        -Zf * ureg.elementary_charge**2 / (ureg.epsilon_0 * (q**2 + kappa**2))
    )

@jit
def collision_frequency_BA(
    E: Quantity,
    T: Quantity,
    m_ion: Quantity,
    n_e: Quantity,
    chem_pot: Quantity,
    Zf: float,
):

    w = E / (1 * ureg.hbar)
    T_e = T
    kappa = inverse_screening_length_non_degenerate(n_e, T)

    prefactor = (
        -1j
        * (ureg.epsilon_0)
        / (6 * jnp.pi**2 * Zf * ureg.elementary_charge**2 * ureg.electron_mass)
        * (1 / w)
    )

    def integrand(q):

        q /= 1 * ureg.angstrom
        
        # t1 = time.time()
        eps_zero = dielectric_function_RPA_no_damping(q, 0 * ureg.electron_volt, chem_pot, T)
        eps_part = jnp.conjugate(
                dielectric_function_RPA_no_damping(q, E, chem_pot, T) - eps_zero
        ) / jnp.abs(eps_zero) ** 2
        
        #jax.debug.print("x : {} " + str(time.time() - t1), eps_part)
        
        
        # t1 = time.time()
        res = (
            q**6
            * statically_screened_ie_debye_potential(q, kappa, Zf) ** 2
            * S_ii_AD(q, T_e, n_e, m_ion, Zf)
            * eps_part
        ).m_as(ureg.kilogram**2 * ureg.angstrom**4 / ureg.second**4
                    )
        # jax.debug.print("rest : {} " + str(time.time() - t1), res)
        return jnp.array(
            [
                jnp.real(res
                ),
                jnp.imag(res
                ),
            ]
        )

    integral, errl = quadts(integrand, [0, jnp.inf], epsabs = 1E-20, epsrel = 1E-20)

    integral_real, integral_imag = integral

    integral = integral_real + 1j * integral_imag

    integral *= 1 * ureg.kilogram**2 * ureg.angstrom**3 / ureg.second**4

    return (prefactor * integral).to(1 / ureg.second)

@jit
def dielectric_function_BMA(
    k: Quantity,
    E: Quantity | List,
    chem_pot: Quantity,
    T: Quantity,
    n_e: Quantity,
    m_ion: Quantity,
    Zf: float,
) -> jnp.ndarray:
    """
    Calculates the Born-Mermin Approximation for the dielectric function, which takes collisions
    into account.

    """
    
    w = E / (1 * ureg.hbar)
    coll_freq = collision_frequency_BA(E, T, m_ion, n_e, chem_pot, Zf)
    
    numerator = (1 + 1j * coll_freq / w) * (
        dielectric_function_RPA_no_damping(
            k, E + 1j * ureg.hbar * coll_freq, chem_pot, T
        )
        - 1
    )
    
    denumerator = 1 + 1j * (coll_freq / w) * (
        (
            dielectric_function_RPA_no_damping(
                k, E + 1j * ureg.hbar * coll_freq, chem_pot, T
            )
            - 1
        )
    ) / (
        dielectric_function_RPA_no_damping(
            k, 0 * ureg.electron_volt, chem_pot, T
        )
        - 1
    )
    
    return (1 + numerator / denumerator).m_as(ureg.dimensionless)

@jit
def S0_ee_BMA(
    k: Quantity,
    T: Quantity,
    chem_pot: Quantity,
    m_ion: Quantity,
    n_e: Quantity,
    Zf: float,
    E: Quantity | List,
) -> jnp.ndarray:

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
