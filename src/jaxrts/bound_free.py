"""
This submodule is dedicated to calculate the contribution of tightly bound electrons to the dynamic structure factor.
"""

from .units import ureg, Quantity
from typing import List

import jax
from jax import jit
import jax.numpy as jnp
import jpu.numpy as jnpu
from jax.scipy.special import factorial
import numpy as onp

import logging

logger = logging.getLogger(__name__)

from .form_factors import pauling_all_ff
from .plasma_physics import thomson_momentum_transfer
import jpu


def _xi(n: int, Zeff: Quantity, omega: Quantity, k: Quantity):
    omega_c = (ureg.hbar * k**2) / (2 * ureg.m_e)
    q = (omega - omega_c) / (ureg.c * k)
    return (n * q) / (Zeff * ureg.alpha)


def _y(n: int, Zeff: Quantity, omega: Quantity, k: Quantity):
    """
    This is the argument to the phi in Eqn (28) of :cite:`Schumacher.1975`.
    """
    return 1 + (_xi(n, Zeff, omega, k)) ** 2


def _J10_BM(omega: Quantity, k: Quantity, Zeff: Quantity) -> Quantity:
    """
    See :cite:`Bloch.1975`
    """
    xi = _xi(1, Zeff, omega, k)
    return 8 / (3 * jnp.pi * Zeff * ureg.alpha * (1 + xi**2) ** 3)


def _J20_BM(omega: Quantity, k: Quantity, Zeff: Quantity) -> Quantity:
    """
    See :cite:`Bloch.1975`
    """
    xi = _xi(2, Zeff, omega, k)
    return (64 / (jnp.pi * Zeff * ureg.alpha)) * (
        (1 / (3 * (1 + xi**2) ** 3))
        - (1 / (1 + xi**2) ** 4)
        + (4 / (5 * (1 + xi**2) ** 5))
    )


def _J21_BM(omega: Quantity, k: Quantity, Zeff: Quantity) -> Quantity:
    """
    See :cite:`Bloch.1975`
    """
    xi = _xi(2, Zeff, omega, k)
    return (64 / (15 * jnp.pi * Zeff * ureg.alpha)) * (
        (1 + 5 * xi**2) / (1 + xi**2) ** 5
    )


def _pref_Schumacher_1975(n: int, l: int, Zeff: Quantity) -> Quantity:
    """
    This is the prefactor in Eqn (28) of :cite:`Schumacher.1975`.
    """
    return (
        (2 ** (4 * l + 3) / (jnp.pi))
        * (factorial(n - l - 1) / factorial(n + l))
        * ((n**2 * factorial(l) ** 2) / (Zeff * ureg.alpha))
    )


def _J10_Schum75(omega: Quantity, k: Quantity, Zeff: Quantity) -> Quantity:
    """
    See :cite:`Schumacher.1975`.
    """
    y = _y(1, Zeff, omega, k)
    pref = _pref_Schumacher_1975(1, 0, Zeff)
    # Note: Schumacher goes with y**2!
    phi = 1 / (3 * y**3)
    return pref * phi


def _J20_Schum75(omega: Quantity, k: Quantity, Zeff: Quantity) -> Quantity:
    """
    See :cite:`Schumacher.1975`.
    """
    y = _y(2, Zeff, omega, k)
    pref = _pref_Schumacher_1975(2, 0, Zeff)
    phi = 4 * (1 / (3 * y**3) - 1 / y**4 + 4 / (5 * y**5))
    return pref * phi


def _J21_Schum75(omega: Quantity, k: Quantity, Zeff: Quantity) -> Quantity:
    """
    See :cite:`Schumacher.1975`.
    """
    y = _y(2, Zeff, omega, k)
    pref = _pref_Schumacher_1975(2, 1, Zeff)
    phi = 1 / (4 * y**4) - 1 / (5 * y**5)
    return pref * phi


def _J10_HR(omega: Quantity, k: Quantity, Zeff: Quantity) -> Quantity:
    """
    :cite:`Gregori.2004`, eqn (16)
    """
    xi = _xi(1, Zeff, omega, k)
    J10Schum75 = _J10_Schum75(omega, k, Zeff)
    return (
        J10Schum75
        * (Zeff * ureg.alpha / (k * ureg.a_0))
        * (3 / 2 * xi - 2 * jnpu.arctan(xi))
    )


def _J20_HR(omega: Quantity, k: Quantity, Zeff: Quantity) -> Quantity:
    """
    :cite:`Gregori.2004`, eqn (17)
    """
    xi = _xi(2, Zeff, omega, k)
    J20Schum75 = _J20_Schum75(omega, k, Zeff)
    return (
        J20Schum75
        * (Zeff * ureg.alpha / (k * ureg.a_0))
        * (
            5 * xi * (1 + 3 * xi**4) / (1 - 2.5 * xi**2 + 2.5 * xi**4) / 8
            - 2 * jnpu.arctan(xi)
        )
    )


def _J21_HR(omega: Quantity, k: Quantity, Zeff: Quantity) -> Quantity:
    """
    :cite:`Gregori.2004`, eqn (18)
    """
    xi = _xi(2, Zeff, omega, k)
    J21Schum75 = _J21_Schum75(omega, k, Zeff)
    return (
        J21Schum75
        * (Zeff * ureg.alpha / (k * ureg.a_0))
        * (
            (1 / 3) * ((10 + 15 * xi**2) / (1 + 5 * xi**2)) * xi
            - jnpu.arctan(xi)
        )
    )


def bm_bound_wavefunction(
    n: int,
    l: int,  # noqa E741
    omega: Quantity,
    k: Quantity,
    Zeff: Quantity,
    HR_Correction: bool = True,
) -> Quantity:
    """
    This set of hydrogenic wave functions for bound electrons taken from
    :cite:`Gregori.2004`.

    Parameters
    ----------
    n : int
        Principal quantum number
    l : int
        Azimuthal quantum number
    omega : Quantity
        Frequency shift of the scattering (unit: 1 / [time])
    k : Quantity
        Length of the scattering number (given by the scattering angle and the
        energies of the incident photons (unit: 1 / [length]).
    Zeff : Quantity
        Effective charge (unit: dimensionless)
    HR_Correction : bool, default=True
        If ``True`` the first order asymmetric correction to the impulse
        approximation will be applied.

    Returns
    -------
    J: Quantity
        Contribution of one electron in the given state to the dynamic
        bound-free structure factor (without the correction for elastic
        scattering which reduces the contribution [James.1962]).
    """
    # Find the correct _Jxx_Schum75 function and execute it
    Jxx0 = globals()["_J{:1d}{:1d}_Schum75".format(n, l)](omega, k, Zeff)
    if HR_Correction:
        Jxx1 = globals()["_J{:1d}{:1d}_HR".format(n, l)](omega, k, Zeff)
        return Jxx0 + Jxx1
    return Jxx0

@jit
def all_J_BM(
    omega: Quantity, k: Quantity, Zeff: Quantity | jnp.ndarray
) -> Quantity:
    return jnp.array(
        [
            _J10_BM(omega, k, Zeff[0, :]).m_as(ureg.dimensionless),
            _J20_BM(omega, k, Zeff[1, :]).m_as(ureg.dimensionless),
            _J21_BM(omega, k, Zeff[2, :]).m_as(ureg.dimensionless),
            jnp.zeros_like(
                omega
            ),  # _J30_Schum75(omega, k, Zeff[3]).m_as(ureg.dimensionless),
            jnp.zeros_like(
                omega
            ),  # _J31_Schum75(omega, k, Zeff[4]).m_as(ureg.dimensionless),
            jnp.zeros_like(
                omega
            ),  # _J32_Schum75(omega, k, Zeff[5]).m_as(ureg.dimensionless),
            jnp.zeros_like(
                omega
            ),  # _J40_Schum75(omega, k, Zeff[6]).m_as(ureg.dimensionless),
            jnp.zeros_like(
                omega
            ),  # _J41_Schum75(omega, k, Zeff[7]).m_as(ureg.dimensionless),
            jnp.zeros_like(
                omega
            ),  # _J42_Schum75(omega, k, Zeff[8]).m_as(ureg.dimensionless),
            jnp.zeros_like(
                omega
            ),  # _J43_Schum75(omega, k, Zeff[9]).m_as(ureg.dimensionless),
        ]
    )


@jit
def all_J_Schum75(
    omega: Quantity, k: Quantity, Zeff: Quantity | jnp.ndarray
) -> Quantity:
    return jnp.array(
        [
            _J10_Schum75(omega, k, Zeff[0, :]).m_as(ureg.dimensionless),
            _J20_Schum75(omega, k, Zeff[1, :]).m_as(ureg.dimensionless),
            _J21_Schum75(omega, k, Zeff[2, :]).m_as(ureg.dimensionless),
            jnp.zeros_like(
                omega
            ),  # _J30_Schum75(omega, k, Zeff[3]).m_as(ureg.dimensionless),
            jnp.zeros_like(
                omega
            ),  # _J31_Schum75(omega, k, Zeff[4]).m_as(ureg.dimensionless),
            jnp.zeros_like(
                omega
            ),  # _J32_Schum75(omega, k, Zeff[5]).m_as(ureg.dimensionless),
            jnp.zeros_like(
                omega
            ),  # _J40_Schum75(omega, k, Zeff[6]).m_as(ureg.dimensionless),
            jnp.zeros_like(
                omega
            ),  # _J41_Schum75(omega, k, Zeff[7]).m_as(ureg.dimensionless),
            jnp.zeros_like(
                omega
            ),  # _J42_Schum75(omega, k, Zeff[8]).m_as(ureg.dimensionless),
            jnp.zeros_like(
                omega
            ),  # _J43_Schum75(omega, k, Zeff[9]).m_as(ureg.dimensionless),
        ]
    )


@jit
def all_J_HR(
    omega: Quantity, k: Quantity, Zeff: Quantity | jnp.ndarray
) -> Quantity:
    return jnp.array(
        [
            _J10_HR(omega, k, Zeff[0, :]).m_as(ureg.dimensionless),
            _J20_HR(omega, k, Zeff[1, :]).m_as(ureg.dimensionless),
            _J21_HR(omega, k, Zeff[2, :]).m_as(ureg.dimensionless),
            jnp.zeros_like(
                omega
            ),  # _J30_HR(omega, k, Zeff[3, :]).m_as(ureg.dimensionless),
            jnp.zeros_like(
                omega
            ),  # _J31_HR(omega, k, Zeff[4, :]).m_as(ureg.dimensionless),
            jnp.zeros_like(
                omega
            ),  # _J31_HR(omega, k, Zeff[5, :]).m_as(ureg.dimensionless),
            jnp.zeros_like(
                omega
            ),  # _J40_HR(omega, k, Zeff[6, :]).m_as(ureg.dimensionless),
            jnp.zeros_like(
                omega
            ),  # _J41_HR(omega, k, Zeff[7, :]).m_as(ureg.dimensionless),
            jnp.zeros_like(
                omega
            ),  # _J42_HR(omega, k, Zeff[8, :]).m_as(ureg.dimensionless),
            jnp.zeros_like(
                omega
            ),  # _J43_HR(omega, k, Zeff[9, :]).m_as(ureg.dimensionless),
        ]
    )


def J_BM_HR_approx(
    omega: Quantity,
    k: Quantity,
    pop: jnp.ndarray,
    Zeff: jnp.ndarray,
    E_b: Quantity,
) -> Quantity:

    intensity = (
        pop[:, jnp.newaxis]
        * (
            all_J_BM(omega, k, Zeff[:, jnp.newaxis])
            + all_J_HR(omega, k, Zeff[:, jnp.newaxis])
        )
        * jnp.heaviside(
            (omega[jnp.newaxis, :] * ureg.hbar - E_b[:, jnp.newaxis]).m_as(
                ureg.electron_volt
            ),
            0.5,
        )
    ) / (1 * ureg.c * k)
    return jnpu.sum(intensity, axis=0)


def J_BM_approx(
    omega: Quantity,
    k: Quantity,
    pop: jnp.ndarray,
    Zeff: jnp.ndarray,
    E_b: Quantity,
) -> Quantity:

    intensity = (
        pop[:, jnp.newaxis]
        * (all_J_BM(omega, k, Zeff[:, jnp.newaxis]))
        * jnp.heaviside(
            (omega[jnp.newaxis, :] * ureg.hbar - E_b[:, jnp.newaxis]).m_as(
                ureg.electron_volt
            ),
            0.5,
        )
    ) / (1 * ureg.c * k)
    return jnpu.sum(intensity, axis=0)


def J_impulse_approx(
    omega: Quantity,
    k: Quantity,
    pop: jnp.ndarray,
    Zeff: jnp.ndarray,
    E_b: Quantity,
) -> Quantity:

    intensity = (
        pop[:, jnp.newaxis]
        * (
            all_J_Schum75(omega, k, Zeff[:, jnp.newaxis])
            + all_J_HR(omega, k, Zeff[:, jnp.newaxis])
        )
        * jnp.heaviside(
            (omega[jnp.newaxis, :] * ureg.hbar - E_b[:, jnp.newaxis]).m_as(
                ureg.electron_volt
            ),
            0.5,
        )
    ) / (1 * ureg.c * k)

    return jnpu.sum(intensity, axis=0)
