import jpu
import jax.numpy as jnp
from pint import Quantity, Unit

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
    # If we had no Quantity, just try the straigt-forward conversion to an
    # array.
    try:
        array = jnp.array(obj)
    # But if this fails because we had a list of Quantities in the first place,
    # we end up here
    except AttributeError:
        unit = obj[0].units
        array = jnp.array([o.m_as(unit) for o in obj]) * unit
    return array
