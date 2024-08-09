"""
This submodule is dedicated to the calculation of the ion-feature.
"""

import logging

import jax
import jax.numpy as jnp
import jpu
from jax import jit

from .free_free import (
    dielectric_function_salpeter,
    noninteracting_susceptibility_Dandrea1986,
)
from .plasma_physics import coulomb_potential_fourier
from .static_structure_factors import S_ee_AD, S_ei_AD, S_ii_AD
from .units import Quantity, ureg

logger = logging.getLogger(__name__)


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
def q_FiniteWLChapman2015(
    k: Quantity,
    V_ei: Quantity,
    T: Quantity,
    n_e: Quantity,
    lfc: float = 0,
) -> Quantity:
    """
    Calculates the Finite wavelength screening presented in
    :cite:`Chapman.2015`, eqn (3). This function relies on the Dandrea
    interpolation fit of the non-interacting susceptibility for a faster
    computation (see
    :py:func:`jaxrts.free_free.noninteracting_susceptibility_Dandrea1986`).

    We furthermore allow for a local field correction which might deviate from
    the default of zero, which goes beyond the RPA.

    Parameters
    ----------
    k : Quantity
        Length of the scattering number (given by the scattering angle and the
        energies of the incident photons (unit: 1 / [length]).
    V_ei : Quantity
        The Potential between electrons and ions. Should have the the shape
        ``(n, m)``, where n is the numer of ion species considered, and ``m =
        len(k)``.
    T : Quantity
        The electron temperature in units of [temperature].
    n_e : Quantity
        The electron number density in units of 1/[length]**3.
    lfc: float
        The local field correction.

    Returns
    -------
    q(k):  Quantity
        The screening charge. Has the shape ``(n, m)``
    """
    if k.shape == ():
        k = k[jnp.newaxis]
    k = k[jnp.newaxis, :]
    chi0 = noninteracting_susceptibility_Dandrea1986(
        k, 0 * ureg.electron_volts, T, n_e
    )
    V_ee = coulomb_potential_fourier(-1, -1, k)
    eps_ee = 1 - V_ee * (1 - lfc) * chi0
    return chi0 * V_ei / eps_ee


@jit
def q_DebyeHueckelChapman2015(
    k: Quantity,
    kappa: Quantity,
    Z_f: Quantity | jnp.ndarray,
) -> Quantity:
    """
    Calculates the Debye Hückel screening presented in :cite:`Chapman.2015`,
    eqn (5).

    Parameters
    ----------
    k : Quantity
        Length of the scattering number (given by the scattering angle and the
        energies of the incident photons (unit: 1 / [length]).
    kappa : Quantity
        The inverse sceening lenght, units of 1 / [length]
    Z_f : Quantity, jnp.ndarray
        The ionization / mean charge state of the ions. Should be one entry per
        ion considered.

    Returns
    -------
    q(k):  Quantity
        The screening charge. Has the shape ``(n, m)``, where ``n = len(Z_f)``
        is the number of ions, and ``m = len(k)``.
    """
    return Z_f[:, jnp.newaxis] * kappa**2 / (k[jnp.newaxis, :] ** 2 + kappa**2)


@jit
def free_electron_susceptilibily_RPA(
    k: Quantity,
    kappa: Quantity,
):
    """
    Return the free electron susceptilibily given by :cite:`Gericke.2010` eqn 4

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
    xi0 = kappa**2 * ureg.epsilon_0 / ((1 * ureg.elementary_charge) ** 2)
    varepsilon = (k**2 + kappa**2) / (k**2)
    return xi0 / varepsilon
