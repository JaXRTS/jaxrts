import pytest

from jax import numpy as jnp
import jaxrts
import copy

import logging

ureg = jaxrts.ureg

test_state = jaxrts.PlasmaState(
    ions=[jaxrts.Element("C")],
    Z_free=jnp.array([2]),
    mass_density=jnp.array([3.5]) * ureg.gram / ureg.centimeter**3,
    T_e=jnp.array([80]) * ureg.electron_volt / ureg.k_B,
)

def test_PlasmaStateEquality():
    logging.warning(test_state.models.keys())
    # Test comparison with some random type
    assert test_state != 6
    # Test comparison with it's copy
    state_copy = copy.deepcopy(test_state)
    assert test_state == state_copy
    # Now change the copy
    state_copy.T_e *= 2
    assert test_state != state_copy
    # Add a new model to the state
    state_copy = copy.deepcopy(test_state)
    assert test_state == state_copy
    state_copy["form-factors"] = jaxrts.models.PaulingFormFactors()
    assert test_state != state_copy
