"""
This submodule is dedicated to the calculation of the ion-feature.
"""

from .units import ureg, Quantity
from .free_free import dielectric_function_salpeter
from .static_structure_factors import S_ee_AD, S_ei_AD, S_ii_AD
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
def q_Gregori2004(
    k: Quantity,
    m_ion: Quantity,
    n_e: Quantity,
    T_e: Quantity,
    T_i: Quantity,
    Z_f: float,
) -> Quantity:
    """
    Calculates the screening charge by the Function given by
    :cite:`Gregori.2004`.

    Parameters
    ----------
    k : Quantity
        Length of the scattering number (given by the scattering angle and the
        energies of the incident photons (unit: 1 / [length]).
    m_ion : Quantity
        The mass of the ion for which the ion feature is calculated.
    n_e: Quantity
        The electron number density.
    T_e : Quantity
        The electron temperature.
    T_i : Quantity
        The ion temperature.
    Z_f : float
        The number of electrons not tightly bound to the atom = valence
        electrons

    Returns
    -------
    q(k):  Quantity
        The screening charge.
    """

    # Way to calculate it given by Gregori.2004:
    S_ei = S_ei_AD(k, T_e, T_i, n_e, m_ion, Z_f)
    S_ee = S_ee_AD(k, T_e, T_i, n_e, m_ion, Z_f)
    S_ii = S_ii_AD(k, T_e, T_i, n_e, m_ion, Z_f)

    C_ei = (jpu.numpy.sqrt(Z_f) * S_ei) / (S_ee * S_ii - S_ei**2)

    # This would be the q given by Glenzer.2009, instead
    # return jpu.numpy.sqrt(Z_f) * S_ei / S_ii

    return (
        C_ei
        / (
            dielectric_function_salpeter(
                k, T_e=T_e, n_e=n_e, E=0 * ureg.electron_volts
            )
        )
    ).to_base_units()


@jit
def q_Glenzer2009(
    S_ei: Quantity,
    S_ii: Quantity,
    Z_f: float,
) -> Quantity:
    """
    Calculates the screening charge by the Function given by
    :cite:`Glenzer.2009`.

    Parameters
    ----------
    S_ei : Quantity
        The static electron-ion structure factor.
    S_ii : Quantity
        The static ion-ion structure factor.
    Z_f : float
        The number of electrons not tightly bound to the atom = valence
        electrons

    Returns
    -------
    q(k):  Quantity
        The screening charge.
    """
    return jpu.numpy.sqrt(Z_f) * S_ei / S_ii


@jit
def free_electron_susceptilibily_RPA(
    k: Quantity,
    kappa: Quantity,
):
    """
    Return the free electron susceptilibily given by :cite:`Gregori.2010` eqn 4

    .. math::

        \\xi_{ee}^\\text{RPA} =
        \\frac{\\kappa^2 \\epsilon_0} {e^2 \\varepsilon^\\text{RPA}}

    where :math:`\\varepsilon^\\text{RPA} = \\frac{k^2 + \\kappa^2}{k^2}`

    Parameters
    ----------
    k : Quantity
        Length of the scattering number (given by the scattering angle and the
        energies of the incident photons (unit: 1 / [length]).
    kappa : Quantity
        Inverse screening length.

    Returns
    -------
    xi(k) : Quantity
        The free electron susceptilibily.
    """
    xi0 = kappa**2 * ureg.epsilon_0 / ((1 * ureg.elementary_charge) **2)
    varepsilon = (k**2 + kappa**2)/(k**2)
    return xi0 / varepsilon
