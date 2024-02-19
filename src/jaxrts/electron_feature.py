"""
This submodule is dedicated to the calculation of the free electron dynamic
structure.
"""

from .units import ureg, Quantity
import pint
from typing import List

import jax
from jax import jit
import jax.numpy as jnp
import numpy as onp

import logging

logger = logging.getLogger(__name__)

import jpu

jax.config.update("jax_enable_x64", True)


@jit
def W(x: jnp.ndarray | float) -> jnp.ndarray:
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

    x_v = jnp.linspace(0, x.magnitude, 3000).T

    y_v = jpu.numpy.exp(x_v**2)

    res = (
        1
        - 2
        * x
        * jpu.numpy.exp(-(x**2))
        * jax.scipy.integrate.trapezoid(y_v, x_v, axis=1)
        + (jpu.numpy.sqrt(jnp.pi) * x * jpu.numpy.exp(-(x**2))) * 1j
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
        (1 - W((x_e + kappa).to_base_units())) / (x_e + kappa)
        - (1 - W((x_e - kappa).to_base_units())) / (x_e - kappa)
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
    S0_ee: jnp.ndarray
           The free electron dynamic structure.
    """
    # Perform the sign flip
    E = -E
    eps = dielectric_function_salpeter(k, T_e, n_e, E)
    return S0ee_from_dielectric_func_FDT(k, T_e, n_e, E, eps)
