import pytest

import jaxrts
import jax.numpy as jnp

ureg = jaxrts.ureg


def test_error_on_unknown_cite_key():
    undefined_key = "UndefinedKey"
    with pytest.raises(KeyError) as context:
        jaxrts.literature.get_bibtex_ref_string(undefined_key)
    assert undefined_key.lower() in str(context.value)


def test_model_citation():
    model = jaxrts.models.ArkhipovIonFeat()
    assert "Arkhipov" in model.citation("bibtex")


def test_PlasmaState_citation():
    state = jaxrts.PlasmaState(
        ions=[jaxrts.Element("H")],
        Z_free=jnp.array([0.99]),
        mass_density=jnp.array([0.35]) * ureg.gram / ureg.centimeter**3,
        T_e=jnp.array([80]) * ureg.electron_volt / ureg.k_B,
    )
    state["ionic scattering"] = jaxrts.models.ArkhipovIonFeat()
    assert "Arkhipov" in state.citation()

    # No free free model was set, hence we don't expect it in the citation
    # string.
    assert "free-free scattering" not in state.citation()

    # The Neglect model does not have a citation key. Hence, "free-free
    # scattering: should not turn up as a heading the citation string.
    state["free-free scattering"] = jaxrts.models.Neglect()
    assert "free-free scattering" not in state.citation()

    # The Dandrea model has a citation string.
    state["free-free scattering"] = jaxrts.models.RPA_DandreaFit()
    assert "free-free scattering" in state.citation()
