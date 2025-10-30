import pathlib
import os
import tempfile
from copy import deepcopy

import jax
import jax.numpy as jnp

import jaxrts
import jaxrts.saving as saving

ureg = jaxrts.ureg
save_dir = pathlib.Path(__file__).parent / "saves/"

ions = [jaxrts.Element("H"), jaxrts.Element("O")]

test_state = jaxrts.PlasmaState(
    ions=ions,
    Z_free=jnp.array([0.9, 5.2]),
    mass_density=jnp.array([0.11, 0.89]) * (ureg.gram / ureg.centimeter**3),
    T_e=40 * ureg.electron_volt / ureg.k_B,
)

test_state["screening length"] = (
    jaxrts.models.ArbitraryDegeneracyScreeningLength()
)
test_state["screening"] = jaxrts.models.Gregori2004Screening()
test_state["free-free scattering"] = jaxrts.models.Neglect()
test_state["bound-free scattering"] = jaxrts.models.Neglect()
test_state["free-bound scattering"] = jaxrts.models.Neglect()
test_state["ion-ion Potential"] = jaxrts.hnc_potentials.DebyeHueckelPotential()
test_state["form-factors"] = jaxrts.models.PaulingFormFactors()
test_state["ionic scattering"] = jaxrts.models.OnePotentialHNCIonFeat(mix=0.8)


def test_dump_state():
    fd, tmp = tempfile.mkstemp()
    # with open(save_dir / "state.json", "w") as f:
    with open(tmp, "w") as f:
        saving.dump(test_state, f, indent=2)

    with open(tmp) as f:
        lines_created = f.readlines()
    with open(save_dir / "state.json") as f:
        lines_saved = f.readlines()

    os.close(fd)
    os.remove(tmp)
    assert lines_created == lines_saved


def test_load_state():
    with open(save_dir / "state.json") as f:
        loaded_state = saving.load(f, ureg)
    assert loaded_state == test_state


def test_load_model():
    with open(save_dir / "model.json") as f:
        loaded_model = saving.load(f, ureg)
    assert isinstance(
        loaded_model, jaxrts.models.ArbitraryDegeneracyScreeningLength
    )


def test_load_hnc_potential():
    with open(save_dir / "hnc_pot.json") as f:
        loaded_hnc_pot = saving.load(f, ureg)
    assert loaded_hnc_pot.model_key == ""
    assert len(loaded_hnc_pot._transform_r) == 200


def test_save_and_load_setup():
    """
    Just loading a setup might lead to issues, when they come from a different
    system. Therefore, save and load a setup.
    """
    test_setup = jaxrts.Setup(
        ureg("45 deg"),
        4500 * ureg.electron_volt,
        jnp.linspace(4000, 5000) * ureg.electron_volt,
        lambda x: 1 / x,
    )
    fd, tmp = tempfile.mkstemp()
    with open(tmp, "w") as f:
        saving.dump(test_setup, f)
    with open(tmp) as f:
        loaded_setup = saving.load(f, jaxrts.ureg)
    os.close(fd)
    os.remove(tmp)
    assert (
        jnp.abs(loaded_setup.instrument(ureg("5/s")).m_as(ureg.second) - 0.2)
        < 1e-6
    )


def test_function_saving_and_loading():
    test_function = jax.tree_util.Partial(
        jaxrts.instrument_function.instrument_gaussian
    )
    fd, tmp = tempfile.mkstemp()
    with open(tmp, "w") as f:
        saving.dump(test_function, f)
    with open(tmp) as f:
        loaded_function = saving.load(f, jaxrts.ureg)
    os.close(fd)
    os.remove(tmp)
    assert test_function(3, 5) == loaded_function(3, 5)


class AlwaysPiModel(jaxrts.models.Model):
    allowed_keys = ["test"]
    __name__ = "AlwaysPiModel"

    def evaluate(self, plasma_state, setup) -> jnp.ndarray:
        return jnp.array([jnp.pi])


def test_saving_and_restoring_custom_model():
    state = deepcopy(test_state)
    state["test"] = AlwaysPiModel()

    fd, tmp = tempfile.mkstemp()
    with open(tmp, "w") as f:
        saving.dump(state, f)
    with open(tmp) as f:
        loaded_state = saving.load(
            f,
            jaxrts.ureg,
            additional_mappings={"AlwaysPiModel": AlwaysPiModel},
        )
    os.close(fd)
    os.remove(tmp)
    assert loaded_state.evaluate("test", None) == jnp.pi
