"""
This submodule is dedicated to the modelling and handling of instrument
functions.
"""

import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import jpu.numpy as jnpu
import numpy as onp

from .units import Quantity, ureg

logger = logging.getLogger(__name__)


@jax.jit
def instrument_gaussian(x: jnp.ndarray | float, sigma: Quantity) -> jnp.ndarray:
    """

    Gaussian model for the instrument function.

    Parameters
    ----------
    x :     jnp.ndarray | float
            The energy shift.
    sigma:  Quantity
            The standard deviation of the gaussian.

    """

    return (1.0 / (sigma * jnp.sqrt(2 * jnp.pi))) * jnpu.exp(-0.5 * x**2 / sigma**2)


@jax.jit
def instrument_supergaussian(x: jnp.ndarray | float, sigma: Quantity, power: float) -> jnp.ndarray:
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

    _x = jnp.linspace(-10 * sigma.magnitude, 10 * sigma.magnitude, 600) / sigma.magnitude
    _y = jnp.exp(-((0.5 * _x**2) ** power))

    # Normalize the super-gaussian
    norm = jax.scipy.integrate.trapezoid(_y, _x)

    return jnp.exp(-(0.5 * x**2 / sigma**2).magnitude ** power) / (norm * sigma.magnitude)


@jax.jit
def instrument_lorentzian(x: jnp.ndarray | float, gamma: Quantity) -> jnp.ndarray:
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


def instrument_from_file(filename: Path) -> jnp.ndarray:
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

    ints /= jnp.trapezoid(y=ints, x=(E * ureg.electron_volt / ureg.hbar).m_as(1 / ureg.second))

    @jax.jit
    def inst_func_fxrts(w):
        _E = w * ureg.hbar
        Emag = _E.to(ureg.electron_volt).magnitude
        ints_func = jnp.interp(Emag, E, ints, left=0, right=0)
        return ints_func * ureg.second

    return inst_func_fxrts


def instrument_from_array(x: jnp.ndarray, ints: jnp.ndarray) -> jnp.ndarray:
    """

    Set instrument function from an array input. the Intensities have to be with respect to the energy shift .

    Parameters
    ----------
    x :     jnp.ndarray | float
            The energy shift.
    ints:   jnp.ndarray | float
            Intensity values of Instrument function.

    """
    ints /= jnp.trapezoid(y=ints, x=(x * ureg.electron_volt / ureg.hbar).m_as(1 / ureg.second))

    @jax.jit
    def inst_func_fxrts(w):
        _E = w * ureg.hbar
        Emag = _E.to(ureg.electron_volt).magnitude
        ints_func = jnp.interp(Emag, x, ints, left=0, right=0)
        return ints_func * ureg.second

    return inst_func_fxrts
