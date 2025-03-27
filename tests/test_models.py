import copy
import logging
from collections import defaultdict

import pytest
from jax import numpy as jnp

import jaxrts

ureg = jaxrts.ureg


# Get a list of all models defined within jaxrts
all_models = defaultdict(list)

for module in [jaxrts.models]:
    for obj_name in dir(module):
        if "__class__" in dir(obj_name):
            attributes = getattr(module, obj_name)
            if "allowed_keys" in dir(attributes):
                keys = getattr(attributes, "allowed_keys")
                if (
                    ("Model" not in obj_name)
                    & ("model" not in obj_name)
                    & (not obj_name.startswith("_"))
                ):
                    for k in keys:
                        all_models[k].append(getattr(module, obj_name))

available_model_keys = all_models.keys()


def _identity(x):
    return (x / (1 / ureg.angstrom)).m_as(ureg.dimensionless)


# Some models require additional parameters. Set them.
def additional_model_parameters(
    model: jaxrts.models.Model,
    no_of_ions: int,
) -> tuple:
    """
    Define possible additional parameters for Models
    """
    if model == jaxrts.models.ConstantChemPotential:
        return (123.45 * ureg.electron_volt,)
    if model == jaxrts.models.ConstantDebyeTemp:
        return (314.1 * ureg.kelvin,)
    if model == jaxrts.models.ConstantScreeningLength:
        return (12.1 * ureg.angstrom,)
    if model == jaxrts.models.ConstantIPD:
        return (23.42 * ureg.electron_volt,)
    if model == jaxrts.models.ElectronicLFCConstant:
        return (1.2,)
    if model == jaxrts.models.FixedSii:
        return (jnp.ones((no_of_ions, no_of_ions)) * 1.23,)

    if model == jaxrts.models.PeakCollection:
        return (
            jnp.array([1, 2]) / (1 * ureg.angstrom),
            jnp.array([1, 1]),
            _identity,
        )
    if model == jaxrts.models.DebyeWallerSolid:
        PowderModel = jaxrts.models.PeakCollection(
            jnp.array([1, 2]) / (1 * ureg.angstrom),
            jnp.array([1, 1]),
            _identity,
        )
        S_plasmaModel = jaxrts.models.FixedSii(jnp.array([[1]]))
        return (
            S_plasmaModel,
            PowderModel,
        )
    return ()


# You will encounter many warnings about setting defaults. This is fine, here.
logging.getLogger("jaxrts").setLevel(logging.ERROR)

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
    NeglectInstance1 = jaxrts.models.Neglect()
    NeglectInstance1.model_key = "free-free scattering"
    res1 = NeglectInstance1.evaluate(test_state, test_setup).m_as(ureg.second)

    NeglectInstance2 = jaxrts.models.Neglect()
    NeglectInstance2.model_key = "ipd"
    res2 = NeglectInstance2.evaluate(test_state, test_setup).m_as(
        ureg.electron_volt
    )

    assert jnp.all(res1 == 0)
    assert jnp.all(res2 == 0)
    assert res1.shape == test_setup.measured_energy.shape
    assert res2.shape == (len(test_state.ions),)


def test_KeyError_on_not_allowed_model_key():
    # This should work and return no error
    test_state["ionic scattering"] = jaxrts.models.Gregori2003IonFeat()

    # But now we should get an error
    with pytest.raises(KeyError) as context:
        test_state["free-free scattering"] = jaxrts.models.Gregori2003IonFeat()

    assert "free-free scattering" in str(context.value)
    assert "Gregori2003IonFeat" in str(context.value)


def test_ModelEquality():
    test_state["free-free scattering"] = (
        jaxrts.models.QCSalpeterApproximation()
    )
    # Test comparison with some random type
    assert test_state["free-free scattering"] != 6
    model = copy.deepcopy(test_state["free-free scattering"])
    # Test comparison with it's copy
    assert test_state["free-free scattering"] == model
    # Now change the copy
    model.sample_points = 8
    assert test_state["free-free scattering"] != model


def test_all_models_can_be_evaluated_one_component():
    for key in available_model_keys:
        for model in all_models[key]:
            one_comp_test_state = jaxrts.PlasmaState(
                ions=[jaxrts.Element("C")],
                Z_free=jnp.array([2]),
                mass_density=jnp.array([3.5]) * ureg.gram / ureg.centimeter**3,
                T_e=jnp.array([80]) * ureg.electron_volt / ureg.k_B,
            )
            one_comp_test_state[key] = model(
                *additional_model_parameters(model, 1)
            )
            out = one_comp_test_state.evaluate(key, test_setup)
            assert out is not None
