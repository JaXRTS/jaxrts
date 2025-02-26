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

from .units import Quantity

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

    data = onp.genfromtxt(filename, delimiter=",", skiprows=0)

    E, ints = data[:, 0], data[:, 1]

    return jnp.array([E, ints])
