import pytest
import jaxrts

from jax import numpy as jnp


def test_antia_inversion():
    x = jnp.linspace(0, 200, 400)
    y = jaxrts.math.fermi_integral(x, 0.5)
    jnp.max(
        jnp.abs(
            jaxrts.math.inverse_fermi_12_rational_approximation_antia(y) - x
        )
    ) < 0.002


def test_fukushima_inversion():
    x = jnp.linspace(0, 200, 400)
    y = jaxrts.math.fermi_integral(x, 0.5)
    assert (
        jnp.max(
            jnp.abs(jaxrts.math.inverse_fermi_12_fukushima_single_prec(y) - x)
        )
        < 0.0002
    )
