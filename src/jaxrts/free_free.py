"""
This submodule is dedicated to the calculation of the free electron dynamic
structure.
"""

from .units import ureg, Quantity
from .plasma_physics import (
    coulomb_potential_fourier,
    kin_energy,
    fermi_dirac,
    plasma_frequency,
)
from .static_structure_factors import S_ii_AD
from .math import inverse_fermi_12_fukushima_single_prec

import time
from typing import List
from functools import partial

import jax
from jax import jit
from jax import numpy as jnp
from jpu import numpy as jnpu
import numpy as onp

from quadax import quadgk, quadts, romberg, STATUS

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

    res = -(
        (1 * ureg.hbar)
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
    w = E / (1 * ureg.hbar)
    kappa = w * ureg.m_e / (ureg.hbar * k)
    prefactor = (
        ureg.k_B * T * ureg.m_e**2 * ureg.elementary_charge**2
    ) / (2 * jnp.pi * ureg.epsilon_0 * ureg.hbar**4 * k**3)
    exponent1 = (chem_pot - kin_energy(-k / 2 - kappa)) / (ureg.k_B * T)
    exponent2 = (chem_pot - kin_energy(+k / 2 - kappa)) / (ureg.k_B * T)
    return -prefactor * (
        jnpu.logaddexp(0.0, exponent1) - jnpu.logaddexp(0.0, exponent2)
    )


@jit
def _imag_diel_func_RPA(
    k: Quantity, E: Quantity, chem_pot: Quantity, T: Quantity
) -> Quantity:
    """
    The imaginary part of the dielectric function in Random Phase
    Approximation.
    See :cite:`Schorner.2023`, eqn (A8).


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
    w = E / (1 * ureg.hbar)

    kappa = (
        (jnp.real(w.m_as(1 / ureg.second)) * ureg.m_e)
        / (ureg.hbar * k)
        * (1 / ureg.second)
    )
    delta = (
        (jnp.imag(w.m_as(1 / ureg.second)) * ureg.m_e)
        / (ureg.hbar * k)
        * (1 / ureg.second)
    )

    prefactor = -(1 * ureg.e**2 * ureg.m_e) / (
        2 * jnp.pi**2 * ureg.epsilon_0 * ureg.hbar**2 * k**3
    )

    unit = jnpu.sqrt(ureg.hbar**2 / (2 * ureg.electron_mass) / (ureg.k_B * T))

    def integrand(Q):
        Q /= unit
        alph_min_min = kappa - k / 2 - Q
        alph_min_plu = kappa - k / 2 + Q
        alph_plu_min = kappa + k / 2 - Q
        alph_plu_plu = kappa + k / 2 + Q
        f_0 = fermi_dirac(Q, chem_pot, T)
        res = (Q * f_0) * (
            jnpu.arctan(alph_min_min / delta)
            + jnpu.arctan(alph_plu_plu / delta)
            - jnpu.arctan(alph_min_plu / delta)
            - jnpu.arctan(alph_plu_min / delta)
        )
        return (res * unit).m_as(ureg.dimensionless)

    integral, errl = quadgk(
        integrand,
        [0, jnp.inf],
        epsabs=1e-20,
        epsrel=1e-20,
        max_ninter=150,
    )
    integral *= 1 / unit**2

    full = (prefactor * integral).to_base_units()

    # Default to the non-damped function if delta is zero
    del0 = _imag_diel_func_RPA_no_damping(k, E, chem_pot, T)
    return jnpu.where(delta.m_as(1 / ureg.angstrom) == 0, del0, full)


@jit
def _real_diel_func_RPA(
    k: Quantity,
    E: Quantity,
    chem_pot: Quantity,
    T: Quantity,
) -> Quantity:
    """
    The real part of the dielectric function in Random Phase
    Approximation.
    See :cite:`Schorner.2023`, eqn (A7).

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
    w = E / (1 * ureg.hbar)

    kappa = (
        jnp.real(w.m_as(1 / ureg.second))
        * ureg.m_e
        / (ureg.hbar * k)
        * (1 / ureg.second)
    )
    delta = (
        jnp.imag(w.m_as(1 / ureg.second))
        * ureg.m_e
        / (ureg.hbar * k)
        * (1 / ureg.second)
    )
    prefactor = (1 * ureg.e**2 * ureg.m_e) / (
        4 * jnp.pi**2 * ureg.epsilon_0 * ureg.hbar**2 * k**3
    )
    unit = jnpu.sqrt(ureg.hbar**2 / (2 * ureg.electron_mass) / (ureg.k_B * T))

    def integrand(Q):
        Q /= unit
        alph_min_min = kappa - k / 2 - Q
        alph_min_plu = kappa - k / 2 + Q
        alph_plu_min = kappa + k / 2 - Q
        alph_plu_plu = kappa + k / 2 + Q
        numerator = (delta**2 + alph_min_min**2) * (delta**2 + alph_plu_plu**2)
        denominat = (delta**2 + alph_min_plu**2) * (delta**2 + alph_plu_min**2)
        ln_arg = numerator / denominat
        f_0 = fermi_dirac(Q, chem_pot, T)
        res = Q * f_0 * jnpu.log(ln_arg)
        return (res * unit).m_as(ureg.dimensionless)

    integral, errl = quadgk(
        integrand,
        [0, jnp.inf],
        epsabs=1e-20,
        epsrel=1e-20,
        max_ninter=150,
    )
    integral *= 1 / unit**2

    return 1 + (prefactor * integral).to_base_units()


@partial(jit, static_argnames=("unsave"))
def _real_diel_func_RPA_no_damping(
    k: Quantity,
    E: Quantity,
    chem_pot: Quantity,
    T: Quantity,
    unsave=False,
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
    w = E / (1 * ureg.hbar)

    kappa = w * ureg.m_e / (ureg.hbar * k)
    prefactor = (
        ureg.e**2
        * ureg.m_e
        / (2 * jnp.pi**2 * ureg.epsilon_0 * ureg.hbar**2 * k**3)
    )
    # note there is no factor two missing, it is absorbed in using absolute
    # values instead of squared values in the logarithm in the integrand

    if not unsave:
        diverg_q = jnp.min(
            jnp.array(
                [
                    (kappa - k / 2).m_as(1 / ureg.angstrom),
                    (-kappa - k / 2).m_as(1 / ureg.angstrom),
                ]
            ),
            axis=0,
        )
        unit = unit = 1 * ureg.angstrom / jnpu.absolute(diverg_q)

        def integrand(Q):
            Q /= unit
            alph_min_min = kappa - k / 2 - Q
            alph_min_plu = kappa - k / 2 + Q
            alph_plu_min = kappa + k / 2 - Q
            alph_plu_plu = kappa + k / 2 + Q
            numerator = jnpu.absolute(alph_min_min) * jnpu.absolute(
                alph_plu_plu
            )
            denominator = jnpu.absolute(alph_min_plu) * jnpu.absolute(
                alph_plu_min
            )
            ln_arg = numerator / denominator
            f_0 = fermi_dirac(Q, chem_pot, T)
            res = Q * f_0 * jnpu.log(ln_arg)
            return (res * unit).m_as(ureg.dimensionless)

        integral1, errl = quadgk(
            integrand,
            [0, 1],
            epsabs=1e-20,
            epsrel=1e-20,
            max_ninter=150,
        )
        integral2, errl = quadgk(
            integrand,
            [1, jnp.inf],
            epsabs=1e-20,
            epsrel=1e-20,
        )
        integral = integral1 + integral2

    else:
        # Do the old thing
        unit = jnpu.sqrt(
            ureg.hbar**2 / (2 * ureg.electron_mass) / (ureg.k_B * T)
        )

        def integrand(Q):
            Q /= unit
            alph_min_min = kappa - k / 2 - Q
            alph_min_plu = kappa - k / 2 + Q
            alph_plu_min = kappa + k / 2 - Q
            alph_plu_plu = kappa + k / 2 + Q
            numerator = jnpu.absolute(alph_min_min) * jnpu.absolute(
                alph_plu_plu
            )
            denominat = jnpu.absolute(alph_min_plu) * jnpu.absolute(
                alph_plu_min
            )
            ln_arg = numerator / denominat
            f_0 = fermi_dirac(Q, chem_pot, T)
            res = Q * f_0 * jnpu.log(ln_arg)
            return (res * unit).m_as(ureg.dimensionless)

        integral, errl = quadgk(
            integrand,
            [0, jnp.inf],
            epsabs=1e-20,
            epsrel=1e-20,
            max_ninter=150,
        )

    integral *= 1 / unit**2
    return 1 + (prefactor * integral).to_base_units()


@partial(jit, static_argnames=("unsave"))
def dielectric_function_RPA_no_damping(
    k: Quantity,
    E: Quantity,
    chem_pot: Quantity,
    T: Quantity,
    unsave: bool = False,
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
    real = _real_diel_func_RPA_no_damping(k, E, chem_pot, T, unsave)
    imag = _imag_diel_func_RPA_no_damping(k, E, chem_pot, T)
    return real.m_as(ureg.dimensionless) + 1j * imag.m_as(ureg.dimensionless)


@jit
def dielectric_function_RPA(
    k: Quantity, E: Quantity, chem_pot: Quantity, T: Quantity
) -> Quantity:
    """
    The the dielectric function including potentially a complex argument for E.
    See, e.g., :cite:`Schorner.2023`, eqn (A7) and (A8).

    In comparsion to :py:func:`~dielectric_function_RPA_no_damping`, this
    function should be slower to evaluate, as one cannot find an analytical
    expression for the imaginary component, but the result is used when damping
    is required, e.g., in the Born-Mermin approximation.

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
    real = _real_diel_func_RPA(k, E, chem_pot, T)
    imag = _imag_diel_func_RPA(k, E, chem_pot, T)
    return real.m_as(ureg.dimensionless) + 1j * imag.m_as(ureg.dimensionless)


@partial(jit, static_argnames=("unsave"))
def S0_ee_RPA_no_damping(
    k: Quantity,
    T_e: Quantity,
    n_e: Quantity,
    E: Quantity | List,
    chem_pot: Quantity,
    unsave: bool = False,
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
    eps = dielectric_function_RPA_no_damping(k, E, chem_pot, T_e, unsave)
    return S0ee_from_dielectric_func_FDT(k, T_e, n_e, E, eps)


@jit
def S0_ee_RPA(
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
    eps = dielectric_function_RPA(k, E, chem_pot, T_e)
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

    integral_debye, errl_debye = quadgk(
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
        -Zf
        * ureg.elementary_charge**2
        / (ureg.epsilon_0 * jnp.sqrt(4 * jnp.pi) * (q**2 + kappa**2))
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
    """
    Calculate the Born electron-ion collision frequency. See
    :cite:`Schorner.2023`, eqn (B1). (note: There might be **(-1)'s missing in
    the cited equation).
    """

    w = E / (1 * ureg.hbar)
    T_e = T
    T_i = T
    kappa = inverse_screening_length_non_degenerate(n_e, T)

    prefactor = (
        -1j
        * (ureg.epsilon_0 * jnp.pi * 4)
        / (6 * jnp.pi**2 * Zf * ureg.elementary_charge**2 * ureg.electron_mass)
    )

    def integrand(q):

        q /= 1 * ureg.angstrom

        # t1 = time.time()
        eps_zero = dielectric_function_RPA_no_damping(
            q, 0 * ureg.electron_volt, chem_pot, T, unsave=True
        )
        # This is a trick to calculate [1/eps(q, E) - 1/eps(q, 0)], faster
        # eps_part = (
        #     jnp.conjugate(
        #         dielectric_function_RPA_no_damping(q, E, chem_pot, T) - eps_zero
        #     )
        #     / jnp.abs(eps_zero) ** 2
        # )
        eps_part = (
            dielectric_function_RPA_no_damping(q, E, chem_pot, T, unsave=True)
            - eps_zero
        )

        # jax.debug.print("x : {} " + str(time.time() - t1), eps_part)

        # t1 = time.time()
        res = (
            q**6
            * statically_screened_ie_debye_potential(q, kappa, Zf) ** 2
            * S_ii_AD(q, T_e, T_i, n_e, m_ion, Zf)
            * eps_part
            * (1 / w)
        ).m_as(ureg.kilogram**2 * ureg.angstrom**4 / ureg.second**3)
        # jax.debug.print("rest : {} " + str(time.time() - t1), res)
        return jnp.array(
            [
                jnp.real(res),
                jnp.imag(res),
            ]
        )

    integral, errl = quadgk(
        integrand, [0, jnp.inf], epsabs=1e-10, epsrel=1e-10
    )

    integral_real, integral_imag = integral

    integral = integral_real + 1j * integral_imag

    integral *= 1 * ureg.kilogram**2 * ureg.angstrom**3 / ureg.second**3

    return (prefactor * integral).to(1 / ureg.second)


@partial(jit, static_argnames=("no_of_points"))
def collision_frequency_BA_Chapman_interp(
    E: Quantity,
    T: Quantity,
    m_ion: Quantity,
    n_e: Quantity,
    chem_pot: Quantity,
    Zf: float,
    no_of_points: int = 20,
    E_cutoff: None | Quantity = None,
):
    """
    Calculate the electron-ion collision frequency for the Born approximation,
    at it is done in :py:func:`~collision_frequency_BA`, but instead of using a
    quadrature, we evaluate only at a `no_of_point` points between the
    frequencies :math:`10^-8 \\omega_{pe}` and :math:`1.1
    \\max(\\mid \\omega \\mid)`.

    """

    E_pe = plasma_frequency(n_e) * (1 * ureg.hbar)

    if E_cutoff is None:
        E_cutoff = 1.1 * jnpu.max(jnpu.max(E))

    interp_E = jnpu.linspace(
        0.1 * (jnpu.min(jnpu.max(E)) + 1e-6 * E_pe),
        E_cutoff,
        no_of_points,
    )
    interp_w = interp_E / (1 * ureg.hbar)
    T_e = T
    T_i = T
    kappa = inverse_screening_length_non_degenerate(n_e, T)

    prefactor = (
        -1j
        * (ureg.epsilon_0 * jnp.pi * 4)
        / (6 * jnp.pi**2 * Zf * ureg.elementary_charge**2 * ureg.electron_mass)
    )

    def integrand(q):

        q /= 1 * ureg.angstrom

        # t1 = time.time()
        eps_zero = dielectric_function_RPA_no_damping(
            q, 0 * ureg.electron_volt, chem_pot, T, unsave=True
        )
        # eps_part = (
        #     jnp.conjugate(
        #         dielectric_function_RPA_no_damping(q, interp_E, chem_pot, T) - eps_zero
        #     )
        #     / jnp.abs(eps_zero) ** 2
        # )
        eps_part = (
            dielectric_function_RPA_no_damping(
                q, interp_E, chem_pot, T, unsave=True
            )
            - eps_zero
        )

        # jax.debug.print("x : {} " + str(time.time() - t1), eps_part)

        # t1 = time.time()
        res = (
            q**6
            * statically_screened_ie_debye_potential(q, kappa, Zf) ** 2
            * S_ii_AD(q, T_e, T_i, n_e, m_ion, Zf)
            * eps_part
            * (1 / interp_w)
        ).m_as(ureg.kilogram**2 * ureg.angstrom**4 / ureg.second**3)
        # jax.debug.print("rest : {} " + str(time.time() - t1), res)
        return jnp.array(
            [
                jnp.real(res),
                jnp.imag(res),
            ]
        )

    integral, errl = quadgk(
        integrand, [0, jnp.inf], epsabs=1e-10, epsrel=1e-10
    )

    integral *= 1 * ureg.kilogram**2 * ureg.angstrom**3 / ureg.second**3
    integral *= prefactor

    integral_real, integral_imag = integral

    extended_interp_E = (
        jnp.array(
            [
                *(-1 * interp_E[::-1].m_as(ureg.electron_volt)),
                *interp_E.m_as(ureg.electron_volt),
            ]
        )
        * ureg.electron_volt
    )
    extended_integral_real = jnp.array(
        [
            *(-1 * integral_real[::-1]).m_as(1 / ureg.second),
            *integral_real.m_as(1 / ureg.second),
        ]
    ) / (1 * ureg.second)
    extended_integral_imag = jnp.array(
        [
            *(1 * integral_imag[::-1]).m_as(1 / ureg.second),
            *integral_imag.m_as(1 / ureg.second),
        ]
    ) / (1 * ureg.second)

    interpolated_real = jnpu.interp(
        E, extended_interp_E, extended_integral_real
    )
    interpolated_imag = jnpu.interp(
        E, extended_interp_E, extended_integral_imag
    )

    interpolated_integral = interpolated_real + 1j * interpolated_imag

    return interpolated_integral.to(1 / ureg.second)


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

    See, e.g., :cite:`Redmer.2005`, eqn (20) and :cite:`Mermin.1970`, (eqn 8).
    """
    w = E / (1 * ureg.hbar)
    coll_freq = collision_frequency_BA(E, T, m_ion, n_e, chem_pot, Zf)

    numerator = (1 + 1j * coll_freq / w) * (
        dielectric_function_RPA(k, E + 1j * ureg.hbar * coll_freq, chem_pot, T)
        - 1
    )

    denumerator = 1 + 1j * (coll_freq / w) * (
        (
            dielectric_function_RPA(
                k, E + 1j * ureg.hbar * coll_freq, chem_pot, T
            )
            - 1
        )
    ) / (
        dielectric_function_RPA_no_damping(
            k, 0 * ureg.electron_volt, chem_pot, T, unsave=True
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


@partial(jit, static_argnames=("no_of_points"))
def dielectric_function_BMA_chapman_interp(
    k: Quantity,
    E: Quantity | List,
    chem_pot: Quantity,
    T: Quantity,
    n_e: Quantity,
    m_ion: Quantity,
    Zf: float,
    no_of_points: int = 20,
) -> jnp.ndarray:
    """
    Calculates the Born-Mermin Approximation for the dielectric function, which takes collisions
    into account.

    """
    w = E / (1 * ureg.hbar)

    # Calculate the cut-off energy from the RPA

    See_RPA = S0_ee_RPA_no_damping(k, T, n_e, E, chem_pot, unsave=True)
    E_cutoff = (
        jnpu.min(
            jnpu.where(See_RPA > jnpu.max(See_RPA * 0.001), E, jnpu.max(E))
        )
        * 1.5
    )
    E_cutoff = jnpu.absolute(E_cutoff)

    coll_freq = collision_frequency_BA_Chapman_interp(
        E, T, m_ion, n_e, chem_pot, Zf, no_of_points, E_cutoff
    )

    numerator = (1 + 1j * coll_freq / w) * (
        dielectric_function_RPA(k, E + 1j * ureg.hbar * coll_freq, chem_pot, T)
        - 1
    )

    denumerator = 1 + 1j * (coll_freq / w) * (
        (
            dielectric_function_RPA(
                k, E + 1j * ureg.hbar * coll_freq, chem_pot, T
            )
            - 1
        )
    ) / (
        dielectric_function_RPA_no_damping(
            k, 0 * ureg.electron_volt, chem_pot, T, unsave=True
        )
        - 1
    )

    return (1 + numerator / denumerator).m_as(ureg.dimensionless)


@partial(jit, static_argnames=("no_of_points"))
def S0_ee_BMA_chapman_interp(
    k: Quantity,
    T: Quantity,
    chem_pot: Quantity,
    m_ion: Quantity,
    n_e: Quantity,
    Zf: float,
    E: Quantity | List,
    no_of_points: int = 20,
) -> jnp.ndarray:

    E = -E

    eps = dielectric_function_BMA_chapman_interp(
        k, E, chem_pot, T, n_e, m_ion, Zf, no_of_points
    )

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
