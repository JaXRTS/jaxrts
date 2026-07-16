"""
Tests checking whether the JIT cache grows (i.e. XLA recompiles) at different
operations.
"""

from functools import partial

import jax
import pytest
from jax import numpy as jnp

import jaxrts

ureg = jaxrts.ureg

_PROBE_FN = jaxrts.PlasmaState.probe


def _cache_size() -> int:
    """
    Number of distinct compiled versions cached for
    :py:func:`jaxts.plasmastate.PlasmaState.probe`.
    """
    return _PROBE_FN._cache_size()


@pytest.fixture(autouse=True)
def _clean_jit_cache():
    """
    Ensure every test starts (and leaves) probe's jit cache empty.

    Without this, cache growth from one test would leak into the next
    and make the assertions order-dependent.
    """
    _PROBE_FN.clear_cache()
    yield
    _PROBE_FN.clear_cache()


def make_state(Z_free=2.0) -> jaxrts.PlasmaState:
    state = jaxrts.PlasmaState(
        ions=[jaxrts.Element("Be")],
        Z_free=jnp.array([Z_free]),
        mass_density=jnp.array([1]) * ureg.gram / ureg.centimeter**3,
        T_e=2 * ureg.electron_volt / ureg.k_B,
    )
    state["ionic scattering"] = jaxrts.models.Neglect()
    state["free-free scattering"] = jaxrts.models.RPA_DandreaFit()
    state["bound-free scattering"] = jaxrts.models.SchumacherImpulse()
    state["free-bound scattering"] = jaxrts.models.DetailedBalance()
    return state


def make_setup(
    angle=ureg("60deg"),
    instrument=jaxrts.instrument_function.Gaussian(ureg("5eV")),
) -> jaxrts.Setup:
    return jaxrts.Setup(
        scattering_angle=angle,
        energy=ureg("4700 eV"),
        measured_energy=ureg("4700 eV")
        + jnp.linspace(-100, 40, 50) * ureg.electron_volt,
        instrument=instrument,
    )


def run_probe(state: jaxrts.PlasmaState, setup: jaxrts.Setup):
    result = state.probe(setup)
    jax.block_until_ready(result)
    return result


class TestProbeJitCacheGrowth:
    def test_new_state_and_setup_does_not_recompile(self):
        """
        Creating new, identical objects does not recompile.
        """

        def f(x):
            return 1 / x

        state = make_state()
        setup = make_setup(instrument=f)

        run_probe(state, setup)
        size_after_first = _cache_size()

        state = make_state()
        setup = make_setup(instrument=f)

        run_probe(state, setup)
        size_after_second = _cache_size()

        assert size_after_first == 1
        assert size_after_second == size_after_first

    def test_new_instrument_function_recompiles(self):
        """
        When defining a new instrument function we have to recompile, even if
        the functions definition is the same.
        """
        state = make_state()
        setup = make_setup(instrument=lambda x: 1 / x)

        run_probe(state, setup)
        size_after_first = _cache_size()

        state = make_state()
        setup = make_setup(instrument=lambda x: 1 / x)

        run_probe(state, setup)
        size_after_second = _cache_size()

        assert size_after_first == 1
        assert size_after_second == size_after_first + 1

    def test_partial_as_instrument_function_does_not_recompile(self):
        """
        The construction of :py:class:`jaxrts.instrument.FromCallable` allows
        to trace trough partial objects.
        """

        def f(x, a):
            return a / x

        state = make_state()
        setup = make_setup(instrument=partial(f, a=1))

        run_probe(state, setup)
        size_after_first = _cache_size()

        state = make_state()
        setup = make_setup(instrument=partial(f, a=2))

        run_probe(state, setup)
        size_after_second = _cache_size()

        assert size_after_first == 1
        assert size_after_second == size_after_first

    def test_traced_values_dont_recompile(self):
        """
        Creating new objects where only traced values are changed will not
        trigger a recompile.
        """

        state = make_state(Z_free=1.0)
        setup = make_setup(angle=ureg("40deg"))

        run_probe(state, setup)
        size_after_first = _cache_size()

        state = make_state(Z_free=1.4)
        setup = make_setup(angle=ureg("80deg"))

        run_probe(state, setup)
        size_after_second = _cache_size()

        assert size_after_first == 1
        assert size_after_second == size_after_first
