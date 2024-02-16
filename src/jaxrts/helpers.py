"""
Miscellaneous helper functions.
"""

from jax import numpy as jnp
from .units import Quantity

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
