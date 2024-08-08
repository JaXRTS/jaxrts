"""
This submodule defines the unit registry for calculations involving
dimensionful quantities.
"""

import jax.numpy as jnp
import jpu
from pint import Quantity, Unit  # noqa: F401

ureg = jpu.UnitRegistry()


def to_array(obj) -> jnp.ndarray:
    """
    Create an array of an object. This gets around the annoying fact that a jax
    array cannot easily be created from a list of Quantities. Rather, one has
    to strip every entry of it's unit and multiply it, after.
    """
    # This will happen if obj was a Quantity, already: If if is also a jnp
    # array, return it, elsewise make a jnp array from the object
    if jpu.core.is_quantity(obj):
        if isinstance(obj.magnitude, jnp.ndarray):
            return obj
        else:
            unit = obj.units
            return jnp.array(obj.m_as(unit)) * unit
    # Try to unpack a list or tuple, here
    elif isinstance(obj, list) or isinstance(obj, tuple):
        try:
            unit = obj[0].units
            array = jnp.array([o.m_as(unit) for o in obj]) * unit
            return array
        except AttributeError:
            return jnp.array(obj)
    else:
        return jnp.array(obj)
