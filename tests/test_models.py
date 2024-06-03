import pytest

from jax import numpy as jnp
import jaxrts

ureg = jaxrts.ureg

test_state = jaxrts.PlasmaState(
    ions=[jaxrts.Element("C")],
    Z_free=jnp.array([2]),
    mass_density=jnp.array([3.5]) * ureg.gram / ureg.centimeter**3,
    T_e=jnp.array([80]) * ureg.electron_volt / ureg.k_B,
)

test_setup = jaxrts.setup.Setup(
    ureg("145°"),
    ureg("5keV"),
    jnp.linspace(4.5, 5.5) * ureg.kiloelectron_volts,
    lambda x: jaxrts.instrument_function.instrument_gaussian(
        x, 1 / ureg.second
    ),
)


def test_NeglectModel():
    NeglectInstance1 = jaxrts.models.Neglect(
        test_state, "free-free scattering"
    )
    res1 = NeglectInstance1.evaluate(test_setup).m_as(ureg.second)

    NeglectInstance2 = jaxrts.models.Neglect(test_state, "ipd")
    res2 = NeglectInstance2.evaluate(test_setup).m_as(ureg.electron_volt)

    assert jnp.all(res1 == 0)
    assert jnp.all(res2 == 0)
    assert res1.shape == test_setup.measured_energy.shape
    assert res2.shape == test_state.ions[0].binding_energies.shape


def test_KeyError_on_not_allowed_model_key():
    # This shoud work and return no error
    test_state["ionic scattering"] = jaxrts.models.Gregori2003IonFeat

    # But now we should get an error
    with pytest.raises(KeyError) as context:
        test_state["free-free scattering"] = jaxrts.models.Gregori2003IonFeat

    assert "free-free scattering" in str(context.value)
    assert "Gregori2003IonFeat" in str(context.value)
