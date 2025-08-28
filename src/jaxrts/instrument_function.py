"""
This submodule is dedicated to the modelling and handling of instrument
functions.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import jpu.numpy as jnpu
import numpy as onp

from .units import Quantity, ureg

logger = logging.getLogger(__name__)


@jax.jit
def instrument_gaussian(
    x: jnp.ndarray | float, sigma: Quantity
) -> jnp.ndarray:
    """

    Gaussian model for the instrument function.

    Parameters
    ----------
    x :     jnp.ndarray | float
            The energy shift.
    sigma:  Quantity
            The standard deviation of the gaussian.

    """

    return (1.0 / (sigma * jnp.sqrt(2 * jnp.pi))) * jnpu.exp(
        -0.5 * x**2 / sigma**2
    )


@jax.jit
def instrument_supergaussian(
    x: jnp.ndarray | float, sigma: Quantity, power: float
) -> jnp.ndarray:
    """

    Super-gaussian model for the instrument function.

    Parameters
    ----------
    x :     jnp.ndarray | float
            The energy shift.
    sigma:  Quantity
            The standard deviation of the gaussian.
    power:  float
            The power of the super-gaussian.

    """

    _x = (
        jnp.linspace(-10 * sigma.magnitude, 10 * sigma.magnitude, 600)
        / sigma.magnitude
    )
    _y = jnp.exp(-((0.5 * _x**2) ** power))

    # Normalize the super-gaussian
    norm = jax.scipy.integrate.trapezoid(_y, _x)

    return jnp.exp(-(0.5 * x**2 / sigma**2).magnitude ** power) / (
        norm * sigma.magnitude
    )


@jax.jit
def instrument_lorentzian(
    x: jnp.ndarray | float, gamma: Quantity
) -> jnp.ndarray:
    """

    Lorentzian model for the instrument function.

    Parameters
    ----------
    x :     jnp.ndarray | float
            The energy shift.
    gamma:  Quantity
            The scale parameter of the lorentzian.

    """
    return 1.0 / (jnp.pi * gamma * (1 + (x / gamma) ** 2))


def instrument_from_file(filename: Path) -> Callable[[Quantity], Quantity]:
    """
    Loads instrument function data from a given file.

    Parameters
    ----------
    filename:   Path
                The filename.
    Returns
    -------
    E:      jnp.ndarray
            The energy shifts.
    ints:   jnp.ndarray
            The intensities of the instrument function
    """

    data = onp.genfromtxt(filename, delimiter=",", skip_header=0)

    E, ints = data[:, 0], data[:, 1]

    ints /= jnp.trapezoid(
        y=ints, x=(E * ureg.electron_volt / ureg.hbar).m_as(1 / ureg.second)
    )

    @jax.jit
    def inst_func_fxrts(w):
        _E = w * ureg.hbar
        Emag = _E.to(ureg.electron_volt).magnitude
        ints_func = jnp.interp(Emag, E, ints, left=0, right=0)
        return ints_func * ureg.second

    return jax.tree_util.Partial(inst_func_fxrts)


def instrument_from_array(
    x: Quantity, ints: jnp.ndarray | Quantity
) -> Callable[[Quantity], Quantity]:
    """
    Set instrument function from an array input.
    The intensities have to have the same length as the energy shift.

    Parameters
    ----------
    x :     Quantity
            :math:`\\omega` or energy shift. Should be given in units of 1/time
            or in energy units.
    ints:   jnp.ndarray | float | Quantity
            Intensity values of Instrument function.
    """
    # Handle units

    # if x is given in energy units, convert to 1/time
    if x.check("[energy]"):
        x = x / (1 * ureg.hbar)
    w = x.m_as(ureg.electron_volt / ureg.hbar)
    if isinstance(ints, Quantity):
        ints = ints.to_base_units().magnitude

    # Assert normalization
    ints /= jnp.trapezoid(y=ints, x=w)

    @jax.jit
    def inst_func_fxrts(_w):
        w_mag = _w.m_as(ureg.electron_volt / ureg.hbar)
        ints_func = jnp.interp(w_mag, w, ints, left=0, right=0)
        return ints_func * (1 * ureg.hbar / ureg.electron_volt)

    return jax.tree_util.Partial(inst_func_fxrts)
