import jax.numpy as jnp
import pytest
import jaxrts
import jpu.numpy as jnpu

ureg = jaxrts.ureg


def test_dictionary_inversion():
    test_dict = {"a": 1, "b": 9, 2: "c"}
    assert jaxrts.helpers.invert_dict(test_dict) == {1: "a", 9: "b", "c": 2}


def test_dictionary_inversion_error_with_identical_keys():
    with pytest.raises(ValueError) as context:
        test_dict = {"a": 1, "b": 1, 2: "c"}
        jaxrts.helpers.invert_dict(test_dict)

    assert str(test_dict) in str(context.value)
    assert "cannot be inverted" in str(context.value)


def test_number_to_mass_density():
    ratio = jnp.array([0.2, 0.3, 0.5])
    elements = [jaxrts.Element("C"), jaxrts.Element("H"), jaxrts.Element("F")]
    mass_ratio = jaxrts.helpers.mass_from_number_fraction(ratio, elements)

    # The actual state is not really relevant...
    state = jaxrts.PlasmaState(
        ions=elements,
        Z_free=jnp.array([1, 1, 1]),
        mass_density=3.5
        * jaxrts.ureg.gram
        / jaxrts.ureg.centimeter**3
        * mass_ratio,
        T_e=jnp.array([80]) * jaxrts.ureg.electron_volt / jaxrts.ureg.k_B,
    )

    assert jnp.isclose(ratio, state.number_fraction).all()


def test_mass_density_from_electron_density():
    n_e = 1e24 / ureg.cm**3
    elements = [jaxrts.Element("C"), jaxrts.Element("H"), jaxrts.Element("F")]
    Z_free = jnp.array([1, 1, 1])
    ratio = jnp.array([0.2, 0.3, 0.5])
    rho = jaxrts.helpers.mass_density_from_electron_density(
        n_e, Z_free, ratio, elements, partial=True
    )

    # The actual state is not really relevant...
    state = jaxrts.PlasmaState(
        ions=elements,
        Z_free=Z_free,
        mass_density=rho,
        T_e=jnp.array([80]) * jaxrts.ureg.electron_volt / jaxrts.ureg.k_B,
    )
    assert jnpu.isclose(n_e, state.n_e)
