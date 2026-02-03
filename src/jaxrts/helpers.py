"""
Miscellaneous helper functions.
"""

import logging
from functools import partialmethod, wraps
from time import time

import jax
from jax import numpy as jnp

from .units import Quantity, ureg

#: Typically, we return quantities that differ per orbital in an
#: :py:class:`jax.numpy.ndarray` with 10 entries, the orbitals with n<=4.
#: This dictionary contains the orbitals as keys and the corresponding indices
#: in such arrays as values.
orbital_map = {
    "1s": 0,
    "2s": 1,
    "2p": 2,
    "3s": 3,
    "3p": 4,
    "3d": 5,
    "4s": 6,
    "4p": 7,
    "4d": 8,
    "4f": 9,
    # "5s": 10,
    # "5p": 11,
    # "5d": 12,
    # "6s": 13,
    # "6d": 14,
    # "7s": 15,
}


def orbital_array(
    n1s: int | float | Quantity = 0,
    n2s: int | float | Quantity = 0,
    n2p: int | float | Quantity = 0,
    n3s: int | float | Quantity = 0,
    n3p: int | float | Quantity = 0,
    n3d: int | float | Quantity = 0,
    n4s: int | float | Quantity = 0,
    n4p: int | float | Quantity = 0,
    n4d: int | float | Quantity = 0,
    n4f: int | float | Quantity = 0,
) -> jnp.ndarray:
    """
    Create an array with entries for each orbital.

    Parameters
    ----------
    n1s, n2s, n2p, ... int | float | Quantity, default = 0
        The values for the individual orbitals.

    Returns
    -------
    jnp.ndarray
        An array containing the provided entries, sorted so that the index of a
        specific orbital can be obtained by :py:data:`~.orbital_map`.

    Examples
    --------
    >>> carbon_occupancy = orbital_array(n1s=2, n2s=2, n2p=2)
    """
    return jnp.array(
        [
            n1s,
            n2s,
            n2p,
            n3s,
            n3p,
            n3d,
            n4s,
            n4p,
            n4d,
            n4f,
        ]
    )


def invert_dict(dictionary: dict) -> dict:
    """
    Invert a dictionary, so that it's keys become values, and the values are
    the keys of the returned dict.
    """
    out_dir = {v: k for k, v in dictionary.items()}
    if len(dictionary) != len(out_dir):
        raise ValueError(
            f"Dict {dictionary} cannot be inverted because it contains non-unique entries."  # noqa: E501
        )
    return out_dir


def timer(func, custom_prefix=None, loglevel=logging.INFO):
    """
    Simple timer wrapper.
    """

    print("Starting ", func.__name__, "...\n")

    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Executed {func.__name__!r} ...", end="")
        print(f"in {(t2 - t1):.4f} s\n")

        return result

    return wrapper


def mass_from_number_fraction(number_fractions, elements):
    """
    Calculate the mass fraction of a mixture.

    Parameters
    ----------
    number_fractions : array_like
        The number fractions of each chemical element.
    elements : list
        The masses of the respective chemical elements.

    Returns
    -------
    ndarray
        The mass fractions of the chemical elements in the mixture.

    Raises
    ------
    ValueError
        If the lengths of `number_fractions` and `elements` are not the same.

    Examples
    --------
    >>> number_fractions = [1/3, 2/3]
    >>> elements = [jaxrts.Element("C"), jaxrts.Element("H")]
    >>> calculate_mass_fraction(number_fractions, elements)
    Array([0.85627718, 0.14372282], dtype=float64)
    """
    number_fractions = jnp.asarray(number_fractions)
    masses = jnp.array([e.atomic_mass.m_as(ureg.gram) for e in elements])

    if number_fractions.shape != masses.shape:
        raise ValueError("number_fractions and elements must have the same length")

    # Calculate the total mass of the mixture
    total_mass = jnp.sum(number_fractions * masses)

    # Calculate the mass fraction for each element
    mass_fractions = (number_fractions * masses) / total_mass

    return mass_fractions


def mass_density_from_electron_density(n_e, Z, number_fractions, elements):
    """
    Calculate the mass density of a mixture from electron density.

    Parameters
    ----------
    n_e : scalar
        electron density of mixture
    Z : array_like
        The charge of each chemical element in the plasma.
    number_fractions : array_like
        The number fractions of each chemical element.
    elements : list
        The masses of the respective chemical elements.

    Returns
    -------
    array_like, scalar
        The full mass density of the mixture. Can be split into the partial mass
        densities for each component by multiplying it by the result of
        `py:func:~mass_from_number_fraction`.

    Raises
    ------
    ValueError
        If the lengths of `Z`, `number_fractions` and `elements` are not the same.

    Examples
    --------
    >>> n_e = 0.8e24 / ureg.cm**3
    >>> number_fractions = [1/2, 2/2]
    >>> elements = [jaxrts.Element("C"), jaxrts.Element("H")]
    >>> Z_free = jnp.array([4.0, 1.0])
    >>> mass_density_from_electron_density(n_e, Z_free, number_fraction, elements)
    Array(3.45897 dtype=float64) #g/cc
    """

    if not (number_fractions.shape[0] == Z.shape[0] == len(elements)):
        raise ValueError("Z, number_fractions and elements must have the same length")

    m = [x.atomic_mass for x in elements]

    # model avarage atom in the mixture
    nom = sum(x_i * m_i for x_i, m_i in zip(number_fractions, m, strict=False))
    denom = sum(x_i * Z_i for Z_i, x_i in zip(Z, number_fractions, strict=False))

    rho = n_e * nom / denom
    return rho.to(ureg.gram / ureg.cm**3)


class JittableDict(dict):
    # The following is required to jit a state
    def _tree_flatten(self):
        children = self.values()
        aux_data = self.keys()
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = JittableDict.__new__(cls)
        for key, val in zip(aux_data, children, strict=False):
            obj[key] = val
        return obj


@jax.jit
def secant_extrema_finding(func, xmin, xmax, tol=1e-7, max_iter=1e5):
    """
    Use the secant method to find the extrema of a function within specified
    bounds. This is achieved by calling :py:func:`jax.grad` on the function
    func.

    Parameters
    ----------
    func : callable
        The function to minimize. It should take a single input and return a
        scalar output.
    xmin : float
        The minimum bound for the variable x.
    xmax : float
        The maximum bound for the variable x.
    tol : float, optional
        The tolerance for the stopping criteria. The default is 1e-7.
    max_iter : int, optional
        The maximum number of iterations to perform. The default is 100000.

    Returns
    -------
    float
        The x value that minimizes the function within the specified bounds.

    Examples
    --------
    >>> def example_func(x):
    ...     return (x - 2) ** 2
    >>> minimum, iter = secant_minimum_finding(example_func, 0, 4)
    >>> print(minimum)
    2.0
    """

    f = jax.grad(func)

    x0 = (xmin + xmax) / 2
    x1 = xmax

    def body_fun(state):
        x0, x1, i = state
        f0 = f(x0)
        f1 = f(x1)

        # Secant method update
        x_next = x1 - f1 * (x1 - x0) / (f1 - f0)
        x_next = jnp.clip(x_next, xmin, xmax)

        # Update the state
        return x1, x_next, i + 1

    def cond_fun(state):
        x0, x1, i = state
        f0 = f(x0)
        f1 = f(x1)
        return (jnp.abs(f0 - f1) >= tol) & (i < max_iter)

    # Initialize the state
    state = (x0, x1, 0)
    final_state = jax.lax.while_loop(cond_fun, body_fun, state)

    return final_state[1], final_state[2]


def partialclass(cls, *args, **kwds):
    """
    This is an equivalent to functools.partial, but for Classes.

    See https://stackoverflow.com/a/38911383
    """

    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwds)

    return NewCls


jax.tree_util.register_pytree_node(
    JittableDict,
    JittableDict._tree_flatten,
    JittableDict._tree_unflatten,
)
