"""
This submodule is dedicated to calculate the contribution of tightly bound
electrons to the dynamic structure factor.
"""

import logging

import jax.numpy as jnp
import jpu.numpy as jnpu
from jax import jit
from jax.scipy.special import factorial

from .units import Quantity, ureg

logger = logging.getLogger(__name__)


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


def _phi10_Schum75(y):
    """
    See :cite:`Schumacher.1975`.
    """
    return 1 / (3 * y**3)


def _phi20_Schum75(y):
    """
    See :cite:`Schumacher.1975`.
    """
    return 4 * (1 / (3 * y**3) - 1 / y**4 + 4 / (5 * y**5))


def _phi21_Schum75(y):
    """
    See :cite:`Schumacher.1975`.
    """
    return 1 / (4 * y**4) - 1 / (5 * y**5)


def _phi30_Schum75(y):
    """
    See :cite:`Schumacher.1975`.
    """
    return (
        3 * y ** (-3)
        - 24 * y ** (-4)
        + (352 / 5) * y ** (-5)
        - (256 / 3) * y ** (-6)
        + (256 / 7) * y ** (-7)
    )


def _phi31_Schum75(y):
    """
    See :cite:`Schumacher.1975`.
    """
    return (
        16 * (4 / 3) * y ** (-4)
        - y ** (-5)
        + (4 / 3) * y ** (-6)
        - (4 / 7) * y ** (-7)
    )


def _phi32_Schum75(y):
    """
    See :cite:`Schumacher.1975`.
    """
    return (3 / 5) * y ** (-5) - (3 / 5) * y ** (-6) + (1 / 7) * y ** (-7)


def _phi40_Schum75(y):
    """
    See :cite:`Schumacher.1975`.
    """
    return (
        16 * (3 / 5) * y ** (-3)
        - 5 * y ** (-4)
        + (148 / 5) * y ** (-5)
        - (256 / 3) * y ** (-6)
        + 128 * y ** (-7)
        - 96 * y ** (-8)
        + (256 / 9) * y ** (-9)
    )


def _phi41_Schum75(y):
    """
    See :cite:`Schumacher.1975`.
    """
    return (
        4 * (25 / 4) * y ** (-4)
        - 53 * y ** (-5)
        + 176 * y ** (-6)
        - (1968 / 7) * y ** (-7)
        + 216 * y ** (-8)
        - 64 * y ** (-9)
    )


def _phi42_Schum75(y):
    """
    See :cite:`Schumacher.1975`.
    """
    return (
        36 * (15 / 7) * y ** (-5)
        - y ** (-6)
        + (13 / 7) * y ** (-7)
        - (3 / 2) * y ** (-8)
        + (4 / 9) * y ** (-9)
    )


def _phi43_Schum75(y):
    """
    See :cite:`Schumacher.1975`.
    """
    return (
        (6 / 7) * y ** (-6)
        - (3 / 7) * y ** (-7)
        + (3 / 8) * y ** (-8)
        - (1 / 9) * y ** (-9)
    )


def _phi50_Schum75(y):
    """
    See :cite:`Schumacher.1975`.
    """
    return (
        25 / 3 * y ** (-3)
        - 200 * y ** (-4)
        + 1952 * y ** (-5)
        - (29440 / 3) * y ** (-6)
        + (197376 / 7) * y ** (-7)
        - 48 * 128 * y ** (-8)
        + (434176 / 9) * y ** (-9)
        - (131072 / 5) * y ** (-10)
        + (65536 / 11) * y ** (-11)
    )


def _phi51_Schum75(y):
    """
    See :cite:`Schumacher.1975`.
    """
    return (
        100 * y ** (-4)
        - 1424 * y ** (-5)
        + 8384 * y ** (-6)
        - (182848 / 7) * y ** (-7)
        + 46 * 592 * y ** (-8)
        - (143360 / 3) * y ** (-9)
        + (131072 / 5) * y ** (-10)
        - (65536 / 11) * y ** (-11)
    )


def _phi52_Schum75(y):
    """
    See :cite:`Schumacher.1975`.
    """
    return (
        9 * (49 / 5) * y ** (-5)
        - 91 * y ** (-6)
        + (2417 / 7) * y ** (-7)
        - 680 * y ** (-8)
        + (6592 / 9) * y ** (-9)
        - (2048 / 5) * y ** (-10)
        + (1024 / 11) * y ** (-11)
    )


def _J10_Schum75(omega: Quantity, k: Quantity, Zeff: Quantity) -> Quantity:
    """
    See :cite:`Schumacher.1975`.
    """
    y = _y(1, Zeff, omega, k)
    pref = _pref_Schumacher_1975(1, 0, Zeff)
    # Note: Schumacher goes with y**2!
    phi = _phi10_Schum75(y)
    return pref * phi


def _J20_Schum75(omega: Quantity, k: Quantity, Zeff: Quantity) -> Quantity:
    """
    See :cite:`Schumacher.1975`.
    """
    y = _y(2, Zeff, omega, k)
    pref = _pref_Schumacher_1975(2, 0, Zeff)
    phi = _phi20_Schum75(y)
    return pref * phi


def _J21_Schum75(omega: Quantity, k: Quantity, Zeff: Quantity) -> Quantity:
    """
    See :cite:`Schumacher.1975`.
    """
    y = _y(2, Zeff, omega, k)
    pref = _pref_Schumacher_1975(2, 1, Zeff)
    phi = _phi21_Schum75(y)
    return pref * phi


def _J30_Schum75(omega: Quantity, k: Quantity, Zeff: Quantity) -> Quantity:
    """
    See :cite:`Schumacher.1975`.
    """
    y = _y(3, Zeff, omega, k)
    pref = _pref_Schumacher_1975(3, 0, Zeff)
    phi = _phi30_Schum75(y)
    return pref * phi


def _J31_Schum75(omega: Quantity, k: Quantity, Zeff: Quantity) -> Quantity:
    """
    See :cite:`Schumacher.1975`.
    """
    y = _y(3, Zeff, omega, k)
    pref = _pref_Schumacher_1975(3, 1, Zeff)
    phi = _phi31_Schum75(y)
    return pref * phi


def _J32_Schum75(omega: Quantity, k: Quantity, Zeff: Quantity) -> Quantity:
    """
    See :cite:`Schumacher.1975`.
    """
    y = _y(3, Zeff, omega, k)
    pref = _pref_Schumacher_1975(3, 2, Zeff)
    phi = _phi32_Schum75(y)
    return pref * phi


def _J40_Schum75(omega: Quantity, k: Quantity, Zeff: Quantity) -> Quantity:
    """
    See :cite:`Schumacher.1975`.
    """
    y = _y(4, Zeff, omega, k)
    pref = _pref_Schumacher_1975(4, 0, Zeff)
    phi = _phi40_Schum75(y)
    return pref * phi


def _J41_Schum75(omega: Quantity, k: Quantity, Zeff: Quantity) -> Quantity:
    """
    See :cite:`Schumacher.1975`.
    """
    y = _y(4, Zeff, omega, k)
    pref = _pref_Schumacher_1975(4, 1, Zeff)
    phi = _phi41_Schum75(y)
    return pref * phi


def _J42_Schum75(omega: Quantity, k: Quantity, Zeff: Quantity) -> Quantity:
    """
    See :cite:`Schumacher.1975`.
    """
    y = _y(4, Zeff, omega, k)
    pref = _pref_Schumacher_1975(4, 2, Zeff)
    phi = _phi42_Schum75(y)
    return pref * phi


def _J43_Schum75(omega: Quantity, k: Quantity, Zeff: Quantity) -> Quantity:
    """
    See :cite:`Schumacher.1975`.
    """
    y = _y(4, Zeff, omega, k)
    pref = _pref_Schumacher_1975(4, 3, Zeff)
    phi = _phi43_Schum75(y)
    return pref * phi


def _J50_Schum75(omega: Quantity, k: Quantity, Zeff: Quantity) -> Quantity:
    """
    See :cite:`Schumacher.1975`.
    """
    y = _y(5, Zeff, omega, k)
    pref = _pref_Schumacher_1975(5, 0, Zeff)
    phi = _phi50_Schum75(y)
    return pref * phi


def _J51_Schum75(omega: Quantity, k: Quantity, Zeff: Quantity) -> Quantity:
    """
    See :cite:`Schumacher.1975`.
    """
    y = _y(5, Zeff, omega, k)
    pref = _pref_Schumacher_1975(5, 1, Zeff)
    phi = _phi51_Schum75(y)
    return pref * phi


def _J52_Schum75(omega: Quantity, k: Quantity, Zeff: Quantity) -> Quantity:
    """
    See :cite:`Schumacher.1975`.
    """
    y = _y(5, Zeff, omega, k)
    pref = _pref_Schumacher_1975(5, 2, Zeff)
    phi = _phi52_Schum75(y)
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
    # Find the correct _Jxx_BM function and execute it
    Jxx0 = globals()["_J{:1d}{:1d}_BM".format(n, l)](omega, k, Zeff)
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
            _J30_Schum75(omega, k, Zeff[3, :]).m_as(ureg.dimensionless),
            _J31_Schum75(omega, k, Zeff[4, :]).m_as(ureg.dimensionless),
            _J32_Schum75(omega, k, Zeff[5, :]).m_as(ureg.dimensionless),
            _J40_Schum75(omega, k, Zeff[6, :]).m_as(ureg.dimensionless),
            _J41_Schum75(omega, k, Zeff[7, :]).m_as(ureg.dimensionless),
            _J42_Schum75(omega, k, Zeff[8, :]).m_as(ureg.dimensionless),
            _J43_Schum75(omega, k, Zeff[9, :]).m_as(ureg.dimensionless),
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
