"""
This submodule contains is dedicated to form factors.
"""

import logging
from jax import jit

from .units import ureg, Quantity

logger = logging.getLogger(__name__)


@jit
def pauling_xf(k: Quantity, Zeff: Quantity) -> Quantity:
    return (k * ureg.a_0) / (2 * Zeff)


@jit
def pauling_f10(k: Quantity, Zeff: Quantity) -> Quantity:
    x = pauling_xf(k, Zeff)
    return 1 / (1 + x**2) ** 2


@jit
def pauling_f21(k: Quantity, Zeff: Quantity) -> Quantity:
    x = 2 * pauling_xf(k, Zeff)
    return (1 - x**2) / ((1 + x**2) ** 4)


@jit
def pauling_f20(k: Quantity, Zeff: Quantity) -> Quantity:
    x = 2 * pauling_xf(k, Zeff)
    return (1 - 2 * x**2) * pauling_f21(k, Zeff)


@jit
def pauling_f32(k: Quantity, Zeff: Quantity) -> Quantity:
    x = 3 * pauling_xf(k, Zeff)
    return ((1 - 3 * x**2) * (3 - x**2)) / (3 * (1 + x**2) ** 6)


@jit
def pauling_f31(k: Quantity, Zeff: Quantity) -> Quantity:
    x = 3 * pauling_xf(k, Zeff)
    return (1 - 4 * x**2) * pauling_f32(k, Zeff)


@jit
def pauling_f30(k: Quantity, Zeff: Quantity) -> Quantity:
    x = 3 * pauling_xf(k, Zeff)
    return (1 - 6 * x**2 + 3 * x**4) * pauling_f32(k, Zeff)


@jit
def pauling_f43(k: Quantity, Zeff: Quantity) -> Quantity:
    x = 4 * pauling_xf(k, Zeff)
    return ((1 - x**2) * (1 - 6 * x**2 + x**4)) / (1 + x**2) ** 8


@jit
def pauling_f42(k: Quantity, Zeff: Quantity) -> Quantity:
    x = 4 * pauling_xf(k, Zeff)
    return (1 - 6 * x**2) * pauling_f43(k, Zeff)


@jit
def pauling_f41(k: Quantity, Zeff: Quantity) -> Quantity:
    x = 4 * pauling_xf(k, Zeff)
    return (1 - 10 * x**2 + 10 * x**4) * pauling_f43(k, Zeff)


@jit
def pauling_f40(k: Quantity, Zeff: Quantity) -> Quantity:
    x = 4 * pauling_xf(k, Zeff)
    return (1 - 12 * x**2 + 18 * x**4 - 4 * x**6) * pauling_f43(k, Zeff)


def pauling_atomic_ff(
    n: int, l: int, k: Quantity, Zeff: Quantity  # noqa: 741
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
