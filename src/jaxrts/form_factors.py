"""
This submodule is dedicated to form factors.
"""

import logging

import numpy as onp
from jax import jit
from jax import numpy as jnp
from jpu import numpy as jnpu

from .units import Quantity, ureg

logger = logging.getLogger(__name__)


def pauling_size_screening_constants(Z: int | Quantity) -> jnp.array:
    """
    See Table I in :cite:`Pauling.1932`.
    """
    S1s = jnpu.interp(Z, onp.array([1, 2]), onp.array([0, 0.19]))
    S2s = jnpu.interp(Z, onp.array([3, 10]), onp.array([1.25, 3.10]))
    S2p = jnpu.interp(Z, onp.array([5, 10]), onp.array([2.50, 4.57]))
    S3s = jnpu.interp(
        Z, onp.array([11, 18, 20, 30]), onp.array([6.6, 9.1, 9.1, 10.9])
    )
    S3p = jnpu.interp(
        Z, onp.array([13, 18, 20, 30]), onp.array([8.7, 10.9, 10.9, 13.2])
    )
    S3d = jnpu.interp(Z, onp.array([21, 30]), onp.array([14.7, 17.7]))
    S4s = jnpu.interp(
        Z,
        onp.array([19, 20, 30, 26, 38, 48, 57, 71]),
        onp.array([13.4, 13.9, 21.5, 24.2, 24.2, 25.6, 25.6, 29.4]),
    )
    S4p = jnpu.interp(
        Z,
        onp.array([31, 36, 38, 48, 57, 71]),
        onp.array([24.4, 26.6, 26.6, 28.4, 28.4, 32.8]),
    )
    S4d = jnpu.interp(
        Z, onp.array([39, 48, 57, 71]), onp.array([31.8, 34.0, 34.0, 39.6])
    )
    S4f = jnpu.interp(Z, onp.array([58, 71]), onp.array([43, 49.8]))
    # S5s = jnpu.interp(
    #     Z,
    #     onp.array([37, 38, 39, 48, 54, 56, 71, 80]),
    #     onp.array([30.4, 30.8, 31.3, 37.0, 38.8, 38.8, 47.8, 50]),
    # )
    # S5p = jnpu.interp(
    #     Z,
    #     onp.array([49, 54, 56, 71, 80]),
    #     onp.array([39.4, 41.8, 41.8, 51.4, 54]),
    # )
    # S5d = jnpu.interp(Z, onp.array([57, 71, 80]), onp.array([48.6, 59, 62]))
    # S6s = jnpu.interp(
    #     Z,
    #     onp.array([54, 55, 56, 71, 80, 86, 92]),
    #     onp.array([47, 47.4, 48, 62, 66, 67, 68]),
    # )
    # S6p = jnpu.interp(Z, onp.array([81, 86, 92]), onp.array([71.0, 72, 73]))
    # S6d = jnpu.interp(Z, onp.array([89, 92]), onp.array([81.0, 82]))
    # S7s = jnpu.interp(Z, onp.array([87, 92]), onp.array([80, 82.4]))
    return jnp.array(
        [
            S1s,
            S2s,
            S2p,
            S3s,
            S3p,
            S3d,
            S4s,
            S4p,
            S4d,
            S4f,
            # S5s,
            # S5p,
            # S5d,
            # S6s,
            # S6p,
            # S6d,
            # S7s,
        ]
    )


def pauling_effective_charge(Z: int | Quantity) -> jnp.array:
    return Z - pauling_size_screening_constants(Z)


@jit
def pauling_xf(k: Quantity, Zeff: Quantity | float) -> Quantity:
    return (k * ureg.a_0) / (2 * Zeff)


@jit
def pauling_f10(k: Quantity, Zeff: Quantity | float) -> Quantity:
    x = pauling_xf(k, Zeff)
    return 1 / (1 + x**2) ** 2


@jit
def pauling_f21(k: Quantity, Zeff: Quantity | float) -> Quantity:
    x = 2 * pauling_xf(k, Zeff)
    return (1 - x**2) / ((1 + x**2) ** 4)


@jit
def pauling_f20(k: Quantity, Zeff: Quantity | float) -> Quantity:
    x = 2 * pauling_xf(k, Zeff)
    return (1 - 2 * x**2) * pauling_f21(k, Zeff)


@jit
def pauling_f32(k: Quantity, Zeff: Quantity | float) -> Quantity:
    x = 3 * pauling_xf(k, Zeff)
    return ((1 - 3 * x**2) * (3 - x**2)) / (3 * (1 + x**2) ** 6)


@jit
def pauling_f31(k: Quantity, Zeff: Quantity | float) -> Quantity:
    x = 3 * pauling_xf(k, Zeff)
    return (1 - 4 * x**2) * pauling_f32(k, Zeff)


@jit
def pauling_f30(k: Quantity, Zeff: Quantity | float) -> Quantity:
    x = 3 * pauling_xf(k, Zeff)
    return (1 - 6 * x**2 + 3 * x**4) * pauling_f32(k, Zeff)


@jit
def pauling_f43(k: Quantity, Zeff: Quantity | float) -> Quantity:
    x = 4 * pauling_xf(k, Zeff)
    return ((1 - x**2) * (1 - 6 * x**2 + x**4)) / (1 + x**2) ** 8


@jit
def pauling_f42(k: Quantity, Zeff: Quantity | float) -> Quantity:
    x = 4 * pauling_xf(k, Zeff)
    return (1 - 6 * x**2) * pauling_f43(k, Zeff)


@jit
def pauling_f41(k: Quantity, Zeff: Quantity | float) -> Quantity:
    x = 4 * pauling_xf(k, Zeff)
    return (1 - 10 * x**2 + 10 * x**4) * pauling_f43(k, Zeff)


@jit
def pauling_f40(k: Quantity, Zeff: Quantity | float) -> Quantity:
    x = 4 * pauling_xf(k, Zeff)
    return (1 - 12 * x**2 + 18 * x**4 - 4 * x**6) * pauling_f43(k, Zeff)


def pauling_atomic_ff(
    n: int, l: int, k: Quantity, Zeff: Quantity | float  # noqa: 741
) -> Quantity:
    """
    Atomic formfactor of a hydrogen-like atom taken from :cite:`Pauling.1932`.

    .. note::

       This function is only intended as a wrapper for more convenient
       access. It it not possible to compile this function, using
       :py:func:`jax.jit`.

    Parameters
    ----------
    n : int
        Principal quantum number
    l : int
        Azimuthal quantum number
    k : Quantity
        Length of the scattering number (given by the scattering angle and the
        energies of the incident photons (unit: 1 / [length]).
    Zeff : Quantity
        Effective charge (unit: dimensionless)

    Returns
    -------
    f: Quantity
        Contribution of one electron in the given state to the atomic form
        factor.
    """
    # Find the correct _fxx function and execute it
    form_factor = globals()["pauling_f{:1d}{:1d}".format(n, l)](k, Zeff)
    return form_factor


@jit
def pauling_all_ff(k: Quantity, Zeff: Quantity | jnp.ndarray) -> Quantity:
    return jnp.array(
        [
            pauling_f10(k, Zeff[0]).m_as(ureg.dimensionless),
            pauling_f20(k, Zeff[1]).m_as(ureg.dimensionless),
            pauling_f21(k, Zeff[2]).m_as(ureg.dimensionless),
            pauling_f30(k, Zeff[3]).m_as(ureg.dimensionless),
            pauling_f31(k, Zeff[4]).m_as(ureg.dimensionless),
            pauling_f32(k, Zeff[5]).m_as(ureg.dimensionless),
            pauling_f40(k, Zeff[6]).m_as(ureg.dimensionless),
            pauling_f41(k, Zeff[7]).m_as(ureg.dimensionless),
            pauling_f42(k, Zeff[8]).m_as(ureg.dimensionless),
            pauling_f43(k, Zeff[9]).m_as(ureg.dimensionless),
        ]
    )
