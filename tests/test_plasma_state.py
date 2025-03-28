import copy

from jax import numpy as jnp

import jaxrts

ureg = jaxrts.ureg

one_comp_test_state = jaxrts.PlasmaState(
    ions=[jaxrts.Element("C")],
    Z_free=jnp.array([2]),
    mass_density=jnp.array([3.5]) * ureg.gram / ureg.centimeter**3,
    T_e=jnp.array([80]) * ureg.electron_volt / ureg.k_B,
)

mult_comp_test_state = jaxrts.PlasmaState(
    ions=[jaxrts.Element("C"), jaxrts.Element("Cl")],
    Z_free=jnp.array([2, 1.3]),
    mass_density=jnp.array([3.5, 1]) * ureg.gram / ureg.centimeter**3,
    T_e=jnp.array([80]) * ureg.electron_volt / ureg.k_B,
)

def plasmaStateEquality(state):
    logging.warning(state.models.keys())

    # Test comparison with some random type
    assert state != 6
    # Test comparison with it's copy
    state_copy = copy.deepcopy(state)
    assert state == state_copy
    # Now change the copy
    state_copy.T_e *= 2
    assert state != state_copy
    # Add a new model to the state
    state_copy = copy.deepcopy(state)
    assert state == state_copy
    state_copy["form-factors"] = jaxrts.models.PaulingFormFactors()
    assert state != state_copy


def test_OneComponentPlasmaStateEquality():
    plasmaStateEquality(one_comp_test_state)


def test_MultComponentPlasmaStateEquality():
    plasmaStateEquality(mult_comp_test_state)
