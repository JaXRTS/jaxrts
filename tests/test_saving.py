import pathlib
import hashlib
import tempfile

import jax
import jaxrts
import jaxrts.saving as saving
import jax.numpy as jnp

ureg = jaxrts.ureg
save_dir = pathlib.Path(__file__).parent / "saves/"


def hash_file(path):
    with open(path, "rb") as f:
        sha256sum_hash = hashlib.sha256(f.read()).hexdigest()
    return sha256sum_hash


ions = [jaxrts.Element("H"), jaxrts.Element("O")]
number_fraction = jnp.array([2 / 3, 1 / 3])
mass_fraction = jaxrts.helpers.mass_from_number_fraction(number_fraction, ions)

test_state = jaxrts.PlasmaState(
    ions=ions,
    Z_free=jnp.array([0.9, 5.2]),
    mass_density=mass_fraction
    * jnp.array([1.0])
    * ureg.gram
    / ureg.centimeter**3,
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
    with tempfile.NamedTemporaryFile() as tmp:
        # with open(save_dir / "state.json", "w") as f:
        with open(tmp.name, "w") as f:
            saving.dump(test_state, f, indent=2)

        hash_temp_file = hash_file(tmp.name)
    hash_stored_file = hash_file(save_dir / "state.json")

    assert hash_temp_file == hash_stored_file


def test_load_state():
    with open(save_dir / "state.json", "r") as f:
        loaded_state = saving.load(f, ureg)
    assert loaded_state == test_state


def test_load_model():
    with open(save_dir / "model.json", "r") as f:
        loaded_model = saving.load(f, ureg)
    assert isinstance(
        loaded_model, jaxrts.models.ArbitraryDegeneracyScreeningLength
    )


def test_load_hnc_potential():
    with open(save_dir / "hnc_pot.json", "r") as f:
        loaded_hnc_pot = saving.load(f, ureg)
    assert loaded_hnc_pot.model_key == ""
    assert len(loaded_hnc_pot._transform_r) == 200


def test_load_setup():
    with open(save_dir / "setup.json", "r") as f:
        loaded_setup = saving.load(f, ureg)
    assert (
        jnp.abs(
            loaded_setup.instrument(ureg("5/s")).m_as(ureg.second)
            - 1.4867195147e-6
        )
        < 1e-9
    )


def test_function_saving_and_loading():
    test_function = jax.tree_util.Partial(
        jaxrts.instrument_function.instrument_gaussian
    )
    with tempfile.NamedTemporaryFile() as tmp:
        with open(tmp.name, "w") as f:
            saving.dump(test_function, f)
        with open(tmp.name, "r") as f:
            loaded_function = saving.load(f, jaxrts.ureg)
    assert test_function(3, 5) == loaded_function(3, 5)


class AlwaysPiModel(jaxrts.models.Model):
    allowed_keys = ["test"]
    __name__ = "AlwaysPiModel"

    def evaluate(self, plasma_state, setup) -> jnp.ndarray:
        return jnp.array([jnp.pi])


def test_saving_and_restoring_custom_model():
    state = jaxrts.PlasmaState(
        ions=ions,
        Z_free=jnp.array([0.9, 5.2]),
        mass_density=mass_fraction
        * jnp.array([1.0])
        * ureg.gram
        / ureg.centimeter**3,
        T_e=40 * ureg.electron_volt / ureg.k_B,
    )
    state["test"] = AlwaysPiModel()

    with tempfile.NamedTemporaryFile() as tmp:
        with open(tmp.name, "w") as f:
            saving.dump(state, f)
        with open(tmp.name, "r") as f:
            loaded_state = saving.load(
                f,
                jaxrts.ureg,
                additional_mappings={"AlwaysPiModel": AlwaysPiModel},
            )

    assert loaded_state.evaluate("test", None) == jnp.pi
