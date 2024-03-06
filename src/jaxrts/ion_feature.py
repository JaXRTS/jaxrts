"""
This submodule is dedicated to the calculation of the ion-feature.
"""

from .units import ureg, Quantity
from .electron_feature import dielectric_function_salpeter
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
def q(
    k: Quantity, m_ion: Quantity, n_e: Quantity, T_e: Quantity, Z_f: float
) -> Quantity:
    """
    Calculates the screening charge.

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
    Z_f : float
        The number of electrons not tightly bound to the atom = valence
        electrons

    Returns
    -------
    q(k):  Quantity
        The screening charge.
    """

    # Way to calculate it given by Gregori.2004:
    S_ei = S_ei_AD(k, T_e, n_e, m_ion, Z_f)
    S_ee = S_ee_AD(k, T_e, n_e, m_ion, Z_f)
    S_ii = S_ii_AD(k, T_e, n_e, m_ion, Z_f)

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
