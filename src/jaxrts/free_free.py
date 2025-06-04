"""
This submodule is dedicated to the calculation of the free electron dynamic
structure.
"""

import logging
from functools import partial
from typing import List

import jax
from jax import jit
from jax import numpy as jnp
from jpu import numpy as jnpu
from quadax import quadgk

from . import math
from .ee_localfieldcorrections import xi_lfc_corrected
from .plasma_physics import (
    coulomb_potential_fourier,
    epsilon_from_susceptibility,
    fermi_dirac,
    fermi_energy,
    fermi_wavenumber,
    kin_energy,
    noninteracting_susceptibility_from_eps_RPA,
    plasma_frequency,
    wiegner_seitz_radius,
)
from .units import Quantity, ureg

logger = logging.getLogger(__name__)


jax.config.update("jax_enable_x64", True)


@jit
def _KKT_single_point(Edash, E_vals, I_vals, eps):
    integrand = I_vals / ((E_vals - Edash))
    mask_low = E_vals < (Edash - eps)
    mask_high = E_vals > (Edash + eps)
    part1 = jnp.trapezoid(jnp.where(mask_low, integrand, 0.0), E_vals)
    part2 = jnp.trapezoid(jnp.where(mask_high, integrand, 0.0), E_vals)
    return part1 + part2


@jit
def KramersKronigTransform(
    E_vals,
    I_vals,
    eps=1e-3 * ureg.electron_volt,
    no_of_points: int | None = 1000,
):
    if no_of_points is None:
        no_of_points = len(E_vals)
    E_max = jnpu.max(jnpu.absolute(E_vals))
    E_interp = jnpu.linspace(1 * -E_max, 1 * E_max, no_of_points)
    I_interp = jnpu.interp(E_interp, E_vals, I_vals.m_as(1 / ureg.second)) * (
        1 / ureg.second
    )
    # The "-1" factor comes from the definition of energy transfer
    I_KK = jax.vmap(_KKT_single_point, in_axes=(0, None, None, None))(
        E_interp.m_as(ureg.electron_volt),
        E_interp.m_as(ureg.electron_volt),
        (I_interp * ureg.hbar).m_as(ureg.electron_volt),
        eps.m_as(ureg.electron_volt),
    ) * (-1 * ureg.electron_volt / ureg.hbar / jnp.pi).to(1 / ureg.second)
    return jnpu.interp(E_vals, E_interp, I_KK)


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

    v_t = jnpu.sqrt(
        ((1 * ureg.boltzmann_constant) * T_e) / (1 * ureg.electron_mass)
    ).to_base_units()

    x_e = (omega / (jnpu.sqrt(2) * k * v_t)).to_base_units()

    kappa = (
        (1 * ureg.planck_constant / (2 * jnp.pi))
        * k
        / (2 * jnpu.sqrt(2) * (1 * ureg.electron_mass) * v_t)
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
        Several approximations for this are given in this submodule, e.g.
        :py:func:`~.dielectric_function_salpeter`.

    Returns
    -------
    S0_ee: jnp.ndarray
         The free electron dynamic structure.
    """

    res = -(
        (1 * ureg.hbar)
        / (1 - jnpu.exp(-(E / (1 * ureg.boltzmann_constant * T_e))))
        * (
            ((1 * ureg.vacuum_permittivity) * k**2)
            / (jnp.pi * (1 * ureg.elementary_charge) ** 2 * n_e)
        )
        * jnp.imag((1 / dielectric_function))
    ).to_base_units()

    return res


@jit
def S0ee_from_susceptibility_FDT(
    k: Quantity,
    T_e: Quantity,
    n_e: Quantity,
    E: Quantity | List,
    susceptibility: Quantity | List,
) -> Quantity:
    """
    Links the dielectric function to S0_ee via the fluctuation dissipation
    theorem, see, e.g., :cite:`Fortmann.2010`, Eqn (2).

    .. math::

        S_{\\mathrm{ee}}^{0}(k,\\omega) =
        -\\frac{\\hbar}{1-\\exp(-\\hbar\\omega/k_{B}T_{e})}
        \\frac{1}{\\pi n_{e}}
        \\mathrm{Im}\\left[\\xi(k,\\omega)\\right]


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
        Several approximations for this are given in this submodule, e.g.
        :py:func:`~.dielectric_function_salpeter`.

    Returns
    -------
    S0_ee: jnp.ndarray
         The free electron dynamic structure.
    """
    Vee = coulomb_potential_fourier(-1, -1, k)
    res = -(
        ((1 * ureg.hbar) / (jnp.pi * n_e * Vee))
        / (1 - jnpu.exp(-(E / (1 * ureg.boltzmann_constant * T_e))))
        * jnp.imag((susceptibility * Vee).m_as(ureg.dimensionless))
    ).to_base_units()

    return res


@jit
def S0_ee_Salpeter(
    k: Quantity,
    T_e: Quantity,
    n_e: Quantity,
    E: Quantity | List,
    lfc: Quantity = 0.0,
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
    xi0 = noninteracting_susceptibility_from_eps_RPA(eps, k)
    v_k = (1 * ureg.elementary_charge**2) / ureg.vacuum_permittivity / k**2
    xi = xi_lfc_corrected(xi0, v_k, lfc)
    return S0ee_from_susceptibility_FDT(k, T_e, n_e, E, xi)


@jit
def _gFortmann(x):
    """
    See :cite:`Fortmann.2010`, Eqn 6.
    """
    return x + 1 / 2 * (1 - x**2) * jnpu.log((x + 1) / (x - 1))


@jit
def noninteracting_susceptibility_0K(
    k: Quantity, E: Quantity, n_e: Quantity
) -> Quantity:
    """
    Calculates the non-interacting susceptilibily for the limiting case of 0
    Kelivn.
    The purpose of this rather educational, but can also be used for testing

    See, e.g., :cite:`Fortmann.2010`, eqn 5.
    """
    Vee = coulomb_potential_fourier(-1, -1, k)

    w = E / (1 * ureg.hbar)
    kf = fermi_wavenumber(n_e)
    z = k / (2 * kf)
    u = ureg.m_e * w / (ureg.hbar * k * kf)
    chi0square = 1 / (jnp.pi * ureg.a_0 * kf)

    pref = -chi0square / (4 * z**3 * Vee)
    brackets = _gFortmann(u + z + 0j) - _gFortmann(u - z + 0j)
    return pref * brackets


@jit
def dielectric_function_RPA_0K(k: Quantity, E: Quantity, n_e: Quantity):
    """
    Calculates the Dielektric function for the limiting case of 0
    Kelivn.
    The purpose of this rather educational, but can also be used for testing

    See, e.g., :cite:`Fortmann.2010`, eqn 4.
    """
    Vee = coulomb_potential_fourier(-1, -1, k)
    chi0 = noninteracting_susceptibility_0K(k, E, n_e)

    return 1 - Vee * chi0


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


def _phi_Dandrea(x, theta, eta0):
    # Note: Dandrea does not norm the Fermi integrals!
    I_min05_eta0 = math.fermi_neg12_rational_approximation_antia(
        eta0
    ) * jax.scipy.special.gamma(-1 / 2 + 1)
    I_plu15_eta0 = math.fermi_32_rational_approximation_antia(
        eta0
    ) * jax.scipy.special.gamma(3 / 2 + 1)
    I_plu25_eta0 = math.fermi_52_rational_approximation_antia(
        eta0
    ) * jax.scipy.special.gamma(5 / 2 + 1)
    # A21, ff
    a2 = (-0.2280 + theta) / (
        0.4222 + -0.6466 * theta**0.70572 + 5.8820 * theta**2
    )
    a4 = (1 - 3.0375 * theta + 64.646 * theta**2) / (
        19.608
        - 96.978 * theta
        + 423.66 * theta**2
        - 331.01 * theta**3
        + 20.833 * 64.646 * theta**4
    )
    a6 = (-0.1900 + theta) / (
        0.36538
        - 2.2575 * theta
        + 22.942 * theta**2
        - 43.492 * theta**3
        + 106.40 * theta**4
    )
    a8 = (0.91 - 6.4453 * theta + 12.2324 * theta**2) / (
        1
        - 7.1316 * theta
        + 22.725 * theta**2
        + 58.092 * theta**3
        - 436.02 * theta**4
        - 826.51 * theta**5
        + 4912.9 * theta**6
    )

    # A19
    J = (1 + 3248.8 * theta**2 - 691.47 * theta**4 - 3202700 * theta**7) / (
        1
        + (3248.8 - jnp.pi**2 / 6) * theta**2
        - 4535.6 * theta**4
        - 462400 * theta**6
        + (
            (
                3
                * jnp.sqrt(2)
                * -3202700
                / (4 * jnp.sqrt(jnp.pi))
                * theta ** (15 / 2)
            )
            + (3 * -3202700 / 4) * theta**9
        )
    )
    Kn = 1 - 4.8780 * theta**2 + 473.25 * theta**4 - 2337.5 * theta**7
    Kd = (
        1
        + (-4.8780 - 3 * jnp.pi**2 / 4) * theta**2
        + 348.31 * theta**4
        + 1517.3 * theta**7
        - (
            7
            * jnp.sqrt(2)
            * -2337.5
            / (8 * jnp.sqrt(jnp.pi))
            * theta ** (17 / 2)
        )
        - ((3 * -2337.5 / 8) * theta**10)
    )
    K = Kn / Kd

    b2 = a2 + 2 * J / (3 * jnpu.sqrt(theta) * I_min05_eta0)
    b4 = b2**2 - a2 * b2 + a4 + 2 * K / (15 * jnpu.sqrt(theta) * I_min05_eta0)
    b10 = 3 / 2 * jnpu.sqrt(theta) * I_min05_eta0 * a8
    b8 = (
        3 / 2 * jnpu.sqrt(theta) * I_min05_eta0 * a6
        - 1 / 2 * theta ** (5 / 2) * I_plu15_eta0 * b10
    )
    b6 = (
        3 / 2 * jnpu.sqrt(theta) * I_min05_eta0 * a4
        - 1 / 2 * theta ** (5 / 2) * I_plu15_eta0 * b8
        - 3 / 10 * theta ** (7 / 2) * I_plu25_eta0 * b10
    )

    # Dandrea, eqn 4.8b
    numerator = 1 + a2 * x**2 + a4 * x**4 + a6 * x**6 + a8 * x**8
    denum = 1 + b2 * x**2 + b4 * x**4 + b6 * x**6 + b8 * x**8 + b10 * x**10
    phi_tilde = numerator / denum
    # Dandrea, eqn 4.8a
    phi = jnpu.sqrt(theta) * I_min05_eta0 * x * phi_tilde
    return phi


@jit
def _real_susceptibility_func_RPA_Dandrea(
    k: Quantity,
    theta: float,
    Q: float,
    rs: float,
    z: float,
    eta0: float,
    alpha: float,
) -> Quantity:
    """
    Eqn 4.6 of :cite:`Dandrea.1986`.

    The name of this function might be misleading, the 1/V factor is applied in
    :py:func:`~.susceptibility_RPA_Dandrea1986`.
    """

    x_plu = z / Q + Q
    x_min = z / Q - Q

    # Minus was found by tests, seems like eqn 4.7 in :cite:`Dandrea.1986` is
    # sign-flipped to (10a) in :cite:`Arista.1984`.
    pref = -alpha * rs / (4 * jnp.pi * Q**3)

    return pref * (
        _phi_Dandrea(x_plu, theta, eta0) - _phi_Dandrea(x_min, theta, eta0)
    )


@jit
def _imag_susceptibility_func_RPA_Dandrea(
    k: Quantity,
    theta: float,
    Q: float,
    rs: float,
    z: float,
    eta0: float,
    alpha: float,
) -> Quantity:
    """
    Eqn 4.5 of :cite:`Dandrea.1986`.

    The name of this function might be misleading, the 1/V factor is applied in
    :py:func:`~.susceptibility_RPA_Dandrea1986`.
    """
    x_plu = z / Q + Q
    x_min = z / Q - Q

    pref = -alpha * rs * theta / (8 * Q**3)
    return pref * jnpu.log(
        (1 + jnpu.exp(eta0 - 1 / theta * x_min**2))
        / (1 + jnpu.exp(eta0 - 1 / theta * x_plu**2))
    )


@jit
def noninteracting_susceptibility_Dandrea1986(
    k: Quantity,
    E: Quantity,
    T: Quantity,
    n_e: Quantity,
):
    Ef = fermi_energy(n_e)
    kf = fermi_wavenumber(n_e)
    theta = (T * ureg.k_B / Ef).m_as(ureg.dimensionless)
    alpha = (4 / (9 * jnp.pi)) ** (1 / 3)
    rs = (wiegner_seitz_radius(n_e) / ureg.a_0).m_as(ureg.dimensionless)
    Q = (k / (2 * kf)).m_as(ureg.dimensionless)
    # hbar missing in comparison do Dandrea?
    z = (E / (4 * Ef)).m_as(ureg.dimensionless)
    # Dendrea, Eqn 2.2
    eta0 = math.inverse_fermi_12_fukushima_single_prec(
        (2 / 3 * theta ** (-3 / 2)) / jax.scipy.special.gamma(1 / 2 + 1)
    )
    real = _real_susceptibility_func_RPA_Dandrea(
        k, theta, Q, rs, z, eta0, alpha
    )
    imag = _imag_susceptibility_func_RPA_Dandrea(
        k, theta, Q, rs, z, eta0, alpha
    )
    return 1 / coulomb_potential_fourier(-1, -1, k) * (real + 1j * imag)


@jit
def dielectric_function_RPA_Dandrea1986(
    k: Quantity,
    E: Quantity,
    T: Quantity,
    n_e: Quantity,
):
    """
    Calculate the dielectric function in random phase approximation by using
    the fitts given by :cite:`Dandrea.1986`, which should give a notable
    increase in the calculation time over solving the integrals, numerically.

    This function does not require a chemical potential, and rather a electron
    number density `n_e`. It seems to be included in the fitting functions.

    Parameters
    ----------
    k : Quantity
        Length of the scattering number (given by the scattering angle and the
        energies of the incident photons (unit: 1 / [length]).
    E: Quantity
        The energy shift for which the free electron dynamic structure is
        calculated.
    T : Quantity
        The plasma temperature in Kelvin.
    n_e : Quantity
        The electron number_density in 1 / [length]**3.

    Returns
    -------
    Quantity
        The full dielectric function (complex number)

    See Also
    --------
    jaxrts.free_free.susceptibility_RPA_Dandrea1986
        The function used to calculate xi0, the noninteracting susceptibility
    """
    xi0 = noninteracting_susceptibility_Dandrea1986(k, E, T, n_e)
    return 1 - (coulomb_potential_fourier(-1, -1, k) * xi0).m_as(
        ureg.dimensionless
    )


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

    In comparison to :py:func:`~dielectric_function_RPA_no_damping`, this
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
    lfc: Quantity = 0.0,
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
    xi0 = noninteracting_susceptibility_from_eps_RPA(eps, k)
    v_k = coulomb_potential_fourier(-1, -1, k)
    xi = xi_lfc_corrected(xi0, v_k, lfc)
    return S0ee_from_susceptibility_FDT(k, T_e, n_e, E, xi)


@jit
def S0_ee_RPA(
    k: Quantity,
    T_e: Quantity,
    n_e: Quantity,
    E: Quantity | List,
    chem_pot: Quantity,
    lfc: Quantity = 0.0,
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
    xi0 = noninteracting_susceptibility_from_eps_RPA(eps, k)
    v_k = (1 * ureg.elementary_charge**2) / ureg.vacuum_permittivity / k**2
    xi = xi_lfc_corrected(xi0, v_k, lfc)
    return S0ee_from_susceptibility_FDT(k, T_e, n_e, E, xi)


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

    return jnpu.sqrt(prefactor_debye * integral_debye)


@jit
def inverse_screening_length_non_degenerate(n_e: Quantity, T: Quantity):

    return jnpu.sqrt(
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
def collision_frequency_BA_full(
    E: Quantity,
    T: Quantity,
    S_ii: callable,
    V_eiS: callable,
    n_e: Quantity,
    chem_pot: Quantity,
    Zf: float,
):
    """
    Calculate the Born electron-ion collision frequency. See
    :cite:`Schorner.2023`, eqn (B1).
    """

    w = E / (1 * ureg.hbar)

    prefactor = (
        1
        * (ureg.epsilon_0)
        / (6 * jnp.pi**2 * Zf * ureg.elementary_charge**2 * ureg.electron_mass)
    )

    def integrand(q):

        q /= 1 * ureg.angstrom

        eps_zero = dielectric_function_RPA_no_damping(
            q, 0 * ureg.electron_volt, chem_pot, T, unsave=True
        )
        eps_part = (
            dielectric_function_RPA_no_damping(q, E, chem_pot, T, unsave=True)
            - eps_zero
        )
        res = (q**6 * V_eiS(q) ** 2 * S_ii(q) * eps_part * (1 / w)).m_as(
            ureg.kilogram**2 * ureg.angstrom**4 / ureg.second**3
        )
        res *= -1j
        return jnp.real(res)

    integral_real, errl = quadgk(
        integrand, [0, jnp.inf], epsabs=1e-10, epsrel=1e-10
    )
    integral_real *= 1 * ureg.kilogram**2 * ureg.angstrom**3 / ureg.second**3
    integral_real *= prefactor

    integral_imag = KramersKronigTransform(E, integral_real)

    return (integral_real + 1j * integral_imag).to(1 / ureg.second)


@jit
def collision_frequency_BA_0K(
    E: Quantity,
    S_ii: callable,
    V_eiS: callable,
    n_e: Quantity,
    Zf: float,
):
    """
    Calculate the Born electron-ion collision frequency at 0 K. See
    :cite:`Schorner.2023`, eqn (B1) and :cite:`Fortmann.2010` for the 0K
    dielectric function.
    """

    w = E / (1 * ureg.hbar)

    prefactor = (
        1
        * (ureg.epsilon_0)
        / (6 * jnp.pi**2 * Zf * ureg.elementary_charge**2 * ureg.electron_mass)
    )

    def integrand(q):

        q /= 1 * ureg.angstrom

        eps_zero = dielectric_function_RPA_0K(q, 0 * ureg.electron_volt, n_e)
        eps_part = dielectric_function_RPA_0K(q, E, n_e) - eps_zero
        res = (q**6 * V_eiS(q) ** 2 * S_ii(q) * eps_part * (1 / w)).m_as(
            ureg.kilogram**2 * ureg.angstrom**4 / ureg.second**3
        )
        res *= -1j
        return jnp.real(res)

    integral_real, errl = quadgk(
        integrand, [0, jnp.inf], epsabs=1e-10, epsrel=1e-10
    )
    integral_real *= 1 * ureg.kilogram**2 * ureg.angstrom**3 / ureg.second**3
    integral_real *= prefactor

    integral_imag = KramersKronigTransform(E, integral_real)

    return (integral_real + 1j * integral_imag).to(1 / ureg.second)


@jit
def collision_frequency_BA_fullFit(
    E: Quantity,
    T: Quantity,
    S_ii: callable,
    V_eiS: callable,
    n_e: Quantity,
    Zf: float,
):
    """
    Calculate the Born electron-ion collision frequency. See
    :cite:`Schorner.2023`, eqn (B1).

    Uses the :cite:`Dandrea.1986` interpolation for the RPA.
    """

    w = E / (1 * ureg.hbar)

    prefactor = (
        -1j
        * (ureg.epsilon_0)
        / (6 * jnp.pi**2 * Zf * ureg.elementary_charge**2 * ureg.electron_mass)
    )

    def integrand(q):

        q /= 1 * ureg.angstrom

        eps_zero = dielectric_function_RPA_Dandrea1986(
            q, 0 * ureg.electron_volt, T, n_e
        )
        eps_part = dielectric_function_RPA_Dandrea1986(q, E, T, n_e) - eps_zero
        res = (q**6 * V_eiS(q) ** 2 * S_ii(q) * eps_part * (1 / w)).m_as(
            ureg.kilogram**2 * ureg.angstrom**4 / ureg.second**3
        )
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
    S_ii: callable,
    V_eiS: callable,
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
        E_cutoff = 1.1 * jnpu.max(E)

    interp_E = jnpu.linspace(
        1e-6 * E_pe,
        E_cutoff,
        no_of_points,
    )
    interp_w = interp_E / (1 * ureg.hbar)

    prefactor = (
        1
        * (ureg.epsilon_0)
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
        #         dielectric_function_RPA_no_damping(q, interp_E, chem_pot, T)
        #         - eps_zero
        #     )
        #     / jnp.abs(eps_zero) ** 2
        # )
        eps_part = (
            dielectric_function_RPA_no_damping(
                q, interp_E, chem_pot, T, unsave=True
            )
            - eps_zero
        )
        res = (
            q**6 * V_eiS(q) ** 2 * S_ii(q) * eps_part * (1 / interp_w)
        ).m_as(ureg.kilogram**2 * ureg.angstrom**4 / ureg.second**3)
        res *= -1j
        return jnp.real(res)

    integral_real, errl = quadgk(
        integrand, [0, jnp.inf], epsabs=1e-10, epsrel=1e-10
    )

    integral_real *= 1 * ureg.kilogram**2 * ureg.angstrom**3 / ureg.second**3
    integral_real *= prefactor

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
            *(1 * integral_real[::-1]).m_as(1 / ureg.second),
            *integral_real.m_as(1 / ureg.second),
        ]
    ) / (1 * ureg.second)

    interpolated_real = jnpu.interp(
        E, extended_interp_E, extended_integral_real
    )
    extended_integral_imag = KramersKronigTransform(
        extended_interp_E, extended_integral_real
    )
    interpolated_imag = jnpu.interp(
        E, extended_interp_E, extended_integral_imag
    )

    interpolated_integral = interpolated_real + 1j * interpolated_imag

    return interpolated_integral.to(1 / ureg.second)


@partial(jit, static_argnames=("no_of_points"))
def collision_frequency_BA_Chapman_interpFit(
    E: Quantity,
    T: Quantity,
    S_ii: callable,
    V_eiS: callable,
    n_e: Quantity,
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

    Uses the :cite:`Dandrea.1986` interpolation for the RPA.
    """

    E_pe = plasma_frequency(n_e) * (1 * ureg.hbar)

    if E_cutoff is None:
        E_cutoff = 1.1 * jnpu.max(E)

    interp_E = jnpu.linspace(
        0.1 * (jnpu.min(E) + 1e-6 * E_pe),
        E_cutoff,
        no_of_points,
    )
    interp_w = interp_E / (1 * ureg.hbar)

    prefactor = (
        1
        * (ureg.epsilon_0)
        / (6 * jnp.pi**2 * Zf * ureg.elementary_charge**2 * ureg.electron_mass)
    )

    def integrand(q):

        q /= 1 * ureg.angstrom

        # t1 = time.time()
        eps_zero = dielectric_function_RPA_Dandrea1986(
            q, 0 * ureg.electron_volt, T, n_e
        )
        eps_part = (
            dielectric_function_RPA_Dandrea1986(q, interp_E, T, n_e) - eps_zero
        )
        res = (
            q**6 * V_eiS(q) ** 2 * S_ii(q) * eps_part * (1 / interp_w)
        ).m_as(ureg.kilogram**2 * ureg.angstrom**4 / ureg.second**3)
        res *= -1j
        return jnp.real(res)

    integral_real, errl = quadgk(
        integrand, [0, jnp.inf], epsabs=1e-10, epsrel=1e-10
    )

    integral_real *= 1 * ureg.kilogram**2 * ureg.angstrom**3 / ureg.second**3
    integral_real *= prefactor

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
            *(1 * integral_real[::-1]).m_as(1 / ureg.second),
            *integral_real.m_as(1 / ureg.second),
        ]
    ) / (1 * ureg.second)

    interpolated_real = jnpu.interp(
        E, extended_interp_E, extended_integral_real
    )

    extended_integral_imag = KramersKronigTransform(
        extended_interp_E, extended_integral_real
    )
    interpolated_imag = jnpu.interp(
        E, extended_interp_E, extended_integral_imag
    )

    interpolated_integral = interpolated_real + 1j * interpolated_imag

    return interpolated_integral.to(1 / ureg.second)


@jit
def dielectric_function_BMA_full(
    k: Quantity,
    E: Quantity | List,
    chem_pot: Quantity,
    T: Quantity,
    n_e: Quantity,
    S_ii: callable,
    V_eiS: callable,
    Zf: float,
) -> jnp.ndarray:
    """
    Calculates the Born-Mermin Approximation for the dielectric function, which
    takes collisions into account.

    See, e.g., :cite:`Redmer.2005`, eqn (20) and :cite:`Mermin.1970`, (eqn 8).
    """
    w = E / (1 * ureg.hbar)
    coll_freq = collision_frequency_BA_full(
        jnpu.linspace(10 * jnpu.min(E), 10 * jnpu.max(E), 1000),
        T,
        S_ii,
        V_eiS,
        n_e,
        chem_pot,
        Zf,
    )
    coll_freq = jnpu.interp(
        E, jnpu.linspace(10 * jnpu.min(E), 10 * jnpu.max(E), 1000), coll_freq
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


@jit
def S0_ee_RPA_Dandrea(
    k: Quantity,
    T: Quantity,
    n_e: Quantity,
    E: Quantity | List,
    lfc: Quantity = 0.0,
) -> jnp.ndarray:

    E = -E

    xi0 = noninteracting_susceptibility_Dandrea1986(k, E, T, n_e)
    v_k = (1 * ureg.elementary_charge**2) / ureg.vacuum_permittivity / k**2
    xi = xi_lfc_corrected(xi0, v_k, lfc)
    return S0ee_from_susceptibility_FDT(k, T, n_e, E, xi)


@jit
def S0_ee_BMA(
    k: Quantity,
    T: Quantity,
    chem_pot: Quantity,
    S_ii: callable,
    V_eiS: callable,
    n_e: Quantity,
    Zf: float,
    E: Quantity | List,
    lfc: Quantity = 0.0,
) -> jnp.ndarray:

    E = -E

    eps = dielectric_function_BMA_full(k, E, chem_pot, T, n_e, S_ii, V_eiS, Zf)

    xi0 = noninteracting_susceptibility_from_eps_RPA(eps, k)
    v_k = (1 * ureg.elementary_charge**2) / ureg.vacuum_permittivity / k**2
    xi = xi_lfc_corrected(xi0, v_k, lfc)
    return S0ee_from_susceptibility_FDT(k, T, n_e, E, xi)


@partial(jit, static_argnames=("no_of_points"))
def dielectric_function_BMA_chapman_interpFit(
    k: Quantity,
    E: Quantity | List,
    chem_pot: Quantity,
    T: Quantity,
    n_e: Quantity,
    S_ii: callable,
    V_eiS: callable,
    Zf: float,
    no_of_points: int = 20,
) -> jnp.ndarray:
    """
    Calculates the Born-Mermin Approximation for the dielectric function, which
    takes collisions into account.

    """
    w = E / (1 * ureg.hbar)

    # Calculate the cut-off energy from the RPA

    E_RPA = jnpu.linspace(0 * ureg.electron_volt, 10 * jnpu.max(E), 100)
    See_RPA = S0_ee_RPA_Dandrea(jnpu.mean(k), T, n_e, E_RPA)
    E_cutoff = (
        jnpu.min(
            jnpu.where(
                See_RPA > jnpu.max(See_RPA * 0.01), E_RPA, jnpu.max(E_RPA)
            )
        )
        * 1.5
    )
    E_cutoff = jnpu.absolute(E_cutoff)

    coll_freq = collision_frequency_BA_Chapman_interpFit(
        E, T, S_ii, V_eiS, n_e, Zf, no_of_points, E_cutoff
    )

    p0 = (
        dielectric_function_RPA_Dandrea1986(k, 0 * ureg.electron_volt, T, n_e)
        - 1
    )
    pnu = (
        dielectric_function_RPA(k, E + 1j * ureg.hbar * coll_freq, chem_pot, T)
        - 1
    )

    numerator = (w + 1j * coll_freq) * p0 * pnu
    denumerator = p0 * w + 1j * coll_freq * pnu

    return (1 + numerator / denumerator).m_as(ureg.dimensionless)


@partial(jit, static_argnames=("no_of_points"))
def dielectric_function_BMA_Fortmann(
    k: Quantity,
    E: Quantity | List,
    chem_pot: Quantity,
    T: Quantity,
    n_e: Quantity,
    S_ii: callable,
    V_eiS: callable,
    Zf: float,
    lfc: float = 0,
    no_of_points: int = 20,
) -> jnp.ndarray:

    xi = susceptibility_BMA_Fortmann(
        k, E, chem_pot, T, n_e, S_ii, V_eiS, Zf, lfc, no_of_points
    )
    return epsilon_from_susceptibility(xi, k)


@partial(jit, static_argnames=("no_of_points"))
def susceptibility_BMA_Fortmann(
    k: Quantity,
    E: Quantity | List,
    chem_pot: Quantity,
    T: Quantity,
    n_e: Quantity,
    S_ii: callable,
    V_eiS: callable,
    Zf: float,
    lfc: float = 0,
    no_of_points: int = 20,
) -> jnp.ndarray:
    """
    Calculates the Born-Mermin Approximation for the dielectric function, which
    takes collisions into account. The collision frequency is evaluated at
    ``no_of_points`` frequencies, and interpolated to save computation time.
    (See the MCSS user guide :cite:`Chapman.2016`).

    Local field corrections (``lfc``) are included as described by
    :cite:`Fortmann.2010`, i.e., by modifying the susceptibilities before the
    Mermin approach is calculated (See eqn. (7) and (8), therein).
    The collision frequency, however, is not using the LFC corrected xi's
    """
    w = E / (1 * ureg.hbar)

    # Calculate the cut-off energy from the RPA

    E_RPA = jnpu.linspace(0 * ureg.electron_volt, 10 * jnpu.max(E), 100)
    See_RPA = S0_ee_RPA_Dandrea(k, T, n_e, E_RPA)
    E_cutoff = (
        jnpu.min(
            jnpu.where(
                See_RPA > jnpu.max(See_RPA * 0.01), E_RPA, jnpu.max(E_RPA)
            )
        )
        * 1.5
    )
    E_cutoff = jnpu.absolute(E_cutoff)

    coll_freq = collision_frequency_BA_Chapman_interpFit(
        E, T, S_ii, V_eiS, n_e, Zf, no_of_points, E_cutoff
    )

    V_ee = coulomb_potential_fourier(-1, -1, k)

    # The dynamic (and therefore dumped) part
    xi_0_dyn = noninteracting_susceptibility_from_eps_RPA(
        dielectric_function_RPA(
            k, E + 1j * ureg.hbar * coll_freq, chem_pot, T
        ),
        k,
    )
    xi_OCP_dyn = xi_0_dyn / (1 - V_ee * (1 - lfc) * xi_0_dyn)
    # The static part, were E = 0eV)
    xi_0_stat = noninteracting_susceptibility_Dandrea1986(
        k, 0 * ureg.electron_volt, T, n_e
    )
    xi_OCP_stat = xi_0_stat / (1 - V_ee * (1 - lfc) * xi_0_stat)

    pref = 1 - 1j * w / coll_freq

    numerator = xi_OCP_dyn * xi_OCP_stat
    denominator = xi_OCP_dyn - (1j * w / coll_freq) * xi_OCP_stat

    return pref * numerator / denominator


@partial(jit, static_argnames=("no_of_points"))
def dielectric_function_BMA_chapman_interp(
    k: Quantity,
    E: Quantity | List,
    chem_pot: Quantity,
    T: Quantity,
    n_e: Quantity,
    S_ii: callable,
    V_eiS: callable,
    Zf: float,
    no_of_points: int = 20,
) -> jnp.ndarray:
    """
    Calculates the Born-Mermin Approximation for the dielectric function, which
    takes collisions into account.
    """
    w = E / (1 * ureg.hbar)

    # Calculate the cut-off energy from the RPA
    E_RPA = jnpu.linspace(0 * ureg.electron_volt, 10 * jnpu.max(E), 100)
    See_RPA = S0_ee_RPA_no_damping(
        jnpu.mean(k), T, n_e, E_RPA, chem_pot, unsave=True
    )
    E_cutoff = (
        jnpu.min(
            jnpu.where(
                See_RPA > jnpu.max(See_RPA * 0.01), E_RPA, jnpu.max(E_RPA)
            )
        )
        * 1.5
    )
    E_cutoff = jnpu.absolute(E_cutoff)

    coll_freq = collision_frequency_BA_Chapman_interp(
        E, T, S_ii, V_eiS, n_e, chem_pot, Zf, no_of_points, E_cutoff
    )

    p0 = (
        dielectric_function_RPA_no_damping(
            k, 0 * ureg.electron_volt, chem_pot, T, unsave=True
        )
        - 1
    )
    pnu = (
        dielectric_function_RPA(
            k + 0j, E + 1j * ureg.hbar * coll_freq, chem_pot, T
        )
        - 1
    )

    numerator = w * p0 * pnu + 1j * coll_freq * p0 * pnu
    denumerator = p0 * w + 1j * coll_freq * pnu


    return (1 + numerator / denumerator).m_as(ureg.dimensionless)


@partial(jit, static_argnames=("no_of_points"))
def S0_ee_BMA_chapman_interp(
    k: Quantity,
    T: Quantity,
    chem_pot: Quantity,
    S_ii: callable,
    V_eiS: callable,
    n_e: Quantity,
    Zf: float,
    E: Quantity | List,
    lfc: Quantity = 0.0,
    no_of_points: int = 20,
) -> jnp.ndarray:

    E = -E

    eps = dielectric_function_BMA_chapman_interp(
        k, E, chem_pot, T, n_e, S_ii, V_eiS, Zf, no_of_points
    )

    xi0 = noninteracting_susceptibility_from_eps_RPA(eps, k)
    v_k = (1 * ureg.elementary_charge**2) / ureg.vacuum_permittivity / k**2
    xi = xi_lfc_corrected(xi0, v_k, lfc)

    return S0ee_from_susceptibility_FDT(k, T, n_e, E, xi)


@partial(jit, static_argnames=("no_of_points"))
def S0_ee_BMA_chapman_interpFit(
    k: Quantity,
    T: Quantity,
    chem_pot: Quantity,
    S_ii: callable,
    V_eiS: callable,
    n_e: Quantity,
    Zf: float,
    E: Quantity | List,
    lfc: Quantity = 0.0,
    no_of_points: int = 20,
) -> jnp.ndarray:

    E = -E

    eps = dielectric_function_BMA_chapman_interpFit(
        k, E, chem_pot, T, n_e, S_ii, V_eiS, Zf, no_of_points
    )

    xi0 = noninteracting_susceptibility_from_eps_RPA(eps, k)
    v_k = (1 * ureg.elementary_charge**2) / ureg.vacuum_permittivity / k**2
    xi = xi_lfc_corrected(xi0, v_k, lfc)

    return S0ee_from_susceptibility_FDT(k, T, n_e, E, xi)


@partial(jit, static_argnames=("no_of_points"))
def S0_ee_BMA_Fortmann(
    k: Quantity,
    T: Quantity,
    chem_pot: Quantity,
    S_ii: callable,
    V_eiS: callable,
    n_e: Quantity,
    Zf: float,
    E: Quantity | List,
    lfc: Quantity = 0.0,
    no_of_points: int = 20,
) -> jnp.ndarray:

    E = -E

    xi = susceptibility_BMA_Fortmann(
        k, E, chem_pot, T, n_e, S_ii, V_eiS, Zf, lfc, no_of_points
    )
    return S0ee_from_susceptibility_FDT(k, T, n_e, E, xi)
