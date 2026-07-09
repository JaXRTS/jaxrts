"""
This submodule is dedicated to the modelling and handling of instrument
functions.
"""

import abc
import functools

import logging
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import jpu.numpy as jnpu
import numpy as onp

from .units import Quantity, Unit, ureg

logger = logging.getLogger(__name__)


class InstrumentFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, omega: Quantity) -> Quantity: ...


class Gaussian(InstrumentFunction):
    def __init__(
        self, sigma: Quantity | None = None, fwhm: Quantity | None = None
    ):
        """
        Gaussian instrument function

        .. math::

           \\frac{1}{\\sigma \\sqrt{2\\pi}}
           \\exp\\left(-\\frac{x^2}{{2\\sigma}^2}\\right)


        Parameters
        ----------
        sigma: Quantity, optional
            The parameter sigma, the standard deviation of the gaussian. must
            be given in units of energy or frequency.
        fwhm: Quantity, optional
            The full width half maximum of the distribution.

        Raises
        ------
        TypeError if neither ``sigma`` nor ``fwhm`` are given.
        """
        if sigma is None and fwhm is None:
            raise TypeError("Either sigma or fwhm have to be supplied.")
        if fwhm is not None:
            sigma = fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
        if sigma.check("[energy]"):
            sigma = sigma / (1 * ureg.hbar)
        self.sigma = sigma

    @property
    def fwhm(self):
        return self.sigma * (2 * jnp.sqrt(2 * jnp.log(2)))

    def __call__(self, omega):
        return instrument_gaussian(omega, self.sigma)

    _children_labels = ("sigma",)

    def _tree_flatten(self):
        children = (self.sigma,)
        aux_data = ()  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        (obj.sigma,) = children

        return obj


class SuperGaussian(InstrumentFunction):
    def __init__(
        self,
        power,
        sigma: Quantity | None = None,
        fwhm: Quantity | None = None,
    ):
        """
        Supergaussian instrument function

        .. math::

           \\exp\\left(-\\left(\\frac{x^2}{2\\sigma^2}\\right)^p\\right)

        .. note::

           The function is normalized by numerical integration

        Parameters
        ----------
        power: float
            The exponent p in above equation.
        sigma: Quantity, optional
            The parameter sigma (must be given in units of energy or
            frequency.
        fwhm: Quantity, optional
            The full width half maximum of the distribution.

        Raises
        ------
        TypeError if neither ``sigma`` nor ``fwhm`` are given.

        """
        if sigma is None and fwhm is None:
            raise TypeError("Either sigma or fwhm have to be supplied.")
        if fwhm is not None:
            sigma = fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
        if sigma.check("[energy]"):
            sigma = sigma / (1 * ureg.hbar)
        self.sigma = sigma
        self.power = power

    @property
    def fwhm(self):
        return self.sigma * (2 * jnp.sqrt(2 * jnp.log(2)))

    def __call__(self, omega):
        return instrument_supergaussian(omega, self.power, self.sigma)

    _children_labels = ("sigma", "power")

    def _tree_flatten(self):
        children = self.sigma, self.power
        aux_data = ()  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        (obj.sigma, obj.power) = children
        return obj


class Lorentzian(InstrumentFunction):
    def __init__(self, gamma):
        """
        Lorentzian model for the instrument function.

        .. math::

           \\frac{1}{\\pi}
           \\frac{1}
           {\\gamma \\left(1+\\left(\\frac{x}{\\gamma}\\right)^2\\right)}


        Parameters
        ----------
        gamma: Quantity
            The scale parameter of the lorentzian.
        """
        if gamma.check("[energy]"):
            gamma = gamma / (1 * ureg.hbar)
        self.gamma = gamma

    def __call__(self, omega):
        return instrument_lorentzian(omega, self.gamma)

    def _tree_flatten(self):
        children = (self.gamma,)
        aux_data = ()  # static values
        return (children, aux_data)

    _children_labels = ("gamma",)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        (obj.gamma,) = children
        return obj


class FromCallable(InstrumentFunction):
    def __init__(self, function):
        """
        Create an :py:class`~.InstrumentFunction` from a callable.

        If ``function`` is a ``functools.partial``, its bound positional and
        keyword arguments are unpacked and re-flattened as genuine pytree
        leaves.
        """
        if isinstance(function, functools.partial):
            self.function = function.func
            self.partial_args = function.args
            self.partial_kwargs = function.keywords
        else:
            self.function = function
            self.partial_args = ()
            self.partial_kwargs = {}

    def __call__(self, omega):
        # Bound positional args come before the call-time argument.
        return self.function(*self.partial_args, omega, **self.partial_kwargs)

    def _tree_flatten(self):
        args_leaves, args_treedef = jax.tree_util.tree_flatten(
            self.partial_args
        )
        kwargs_leaves, kwargs_treedef = jax.tree_util.tree_flatten(
            self.partial_kwargs
        )
        children = tuple(args_leaves) + tuple(kwargs_leaves)
        aux_data = (
            self.function,
            len(args_leaves),
            args_treedef,
            kwargs_treedef,
        )
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        func, n_args_leaves, args_treedef, kwargs_treedef = aux_data
        args_leaves = children[:n_args_leaves]
        kwargs_leaves = children[n_args_leaves:]
        obj = object.__new__(cls)
        obj.function = func
        obj.partial_args = jax.tree_util.tree_unflatten(
            args_treedef, args_leaves
        )
        obj.partial_kwargs = jax.tree_util.tree_unflatten(
            kwargs_treedef, kwargs_leaves
        )
        return obj


class FromArray(InstrumentFunction):
    def __init__(self, x, ints):
        """
        Generate an InstrumentFunction from an array input.
        The intensities ``I`` have to have the same length as the energy shift
        ``x``.

        Parameters
        ----------
        x: Quantity
            :math:`\\omega` or energy shift. Should be given in units of 1/time
            or in energy units.
        ints: jnp.ndarray | float | Quantity
            Intensity values of Instrument function.
        """
        if x.check("[energy]"):
            x = x / (1 * ureg.hbar)
        self.omega = x
        w = x.m_as(ureg.electron_volt / ureg.hbar)
        if isinstance(ints, Quantity):
            ints = ints.to_base_units().magnitude

        # Assert normalization
        ints /= jnp.trapezoid(y=ints, x=w)
        self.ints = ints * (1 * ureg.hbar / ureg.electron_volt)

    @jax.jit
    def __call__(self, omega):
        omega_mag = omega.m_as(ureg.electron_volt / ureg.hbar)
        x_mag = self.omega.m_as(ureg.electron_volt / ureg.hbar)
        ints_mag = self.ints.m_as(ureg.hbar / ureg.electron_volt)
        ints_func = jnp.interp(omega_mag, x_mag, ints_mag, left=0, right=0)
        return ints_func * (1 * ureg.hbar / ureg.electron_volt)

    _children_labels = ("omega", "intensity")

    def _tree_flatten(self):
        children = (self.omega, self.ints)
        aux_data = ()  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        (obj.omega, obj.ints) = children
        return obj


@jax.jit
def instrument_gaussian(x: Quantity, sigma: Quantity) -> Quantity:
    """
    Gaussian model for the instrument function.

    .. math::

       \\frac{1}{\\sigma \\sqrt{2\\pi}}
       \\exp\\left(-\\frac{x^2}{{2\\sigma}^2}\\right)

    Parameters
    ----------
    x: Quantity
        The energy shift.
    sigma: Quantity
        The standard deviation of the gaussian.

    Return
    ------
    Quantity
        The Gaussian function, evaluated at positions x. Normed to 1.
    """

    return (1.0 / (sigma * jnp.sqrt(2 * jnp.pi))) * jnpu.exp(
        -0.5 * x**2 / sigma**2
    )


@jax.jit
def instrument_supergaussian(
    x: Quantity, sigma: Quantity, power: float
) -> Quantity:
    """

    Super-gaussian model for the instrument function.


    .. math::

       \\exp\\left(-\\left(\\frac{x^2}{2\\sigma^2}\\right)^p\\right)

    .. note::

       The function is normalized by numerical integration

    Parameters
    ----------
    x: Quantity
        The energy shift.
    sigma:  Quantity
        The standard deviation of the gaussian.
    power:  float
        The power of the super-gaussian.

    Return
    ------
    Quantity
        The super-Gaussian function, evaluated at positions x. Normed to 1.
    """

    _x = (
        jnp.linspace(-10 * sigma.magnitude, 10 * sigma.magnitude, 600)
        / sigma.magnitude
    )
    _y = jnp.exp(-((0.5 * _x**2) ** power))

    # Normalize the super-gaussian
    norm = jax.scipy.integrate.trapezoid(_y, _x)

    return jnp.exp(
        -((0.5 * x**2 / sigma**2).m_as(ureg.dimensionless) ** power)
    ) / (norm * sigma)


@jax.jit
def instrument_lorentzian(x: Quantity, gamma: Quantity) -> Quantity:
    """
    Lorentzian model for the instrument function.

    .. math::

       \\frac{1}{\\pi}
       \\frac{1}{\\gamma \\left(1+\\left(\\frac{x}{\\gamma}\\right)^2\\right)}


    Parameters
    ----------
    x: Quantity
        The frequency shift.
    gamma: Quantity
        The scale parameter of the lorentzian.

    Return
    ------
    Quantity
        The Lorentzian function, evaluated at positions x. Normed to 1.
    """
    return 1.0 / (jnp.pi * gamma * (1 + (x / gamma) ** 2))


def instrument_from_file(
    filename: Path, xunit: Unit = ureg.electron_volt
) -> Callable[[Quantity], Quantity]:
    """
    Loads instrument function data from a given file.

    Parameters
    ----------
    filename: Path
        The path to the file which should be loaded. We assume a
        comma-separated list of energy or frequency shifts in the first, and
        intensity in the second column.
    xunit: Unit, default ``ureg.electron_volt``
        The unit of the x-axis. We allow for energy and frequency units.

    Returns
    -------
    Callable
        The instrument function, which takes one singular argument (the
        frequency shift), and returns the weight of this shift. The function is
        normalized to one.
    """
    data = onp.genfromtxt(filename, delimiter=",", skip_header=0)

    x, ints = data[:, 0], data[:, 1]
    x *= xunit

    return FromArray(x, ints)
