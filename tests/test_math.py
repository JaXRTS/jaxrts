from jax import numpy as jnp

import jaxrts


def test_antia_inversion():
    x = jnp.linspace(0, 200, 400)
    y = jaxrts.math.fermi_integral(x, 0.5)[:, 0]
    # Neglect the first values, these is off quite notably
    assert (
        jnp.max(
            jnp.abs(
                jaxrts.math.inverse_fermi_12_rational_approximation_antia(y)
                - x
            )[10:]
        )
        < 0.002
    )


def test_fukushima_inversion():
    x = jnp.linspace(0, 200, 400)
    y = jaxrts.math.fermi_integral(x, 0.5)[:, 0]
    assert (
        jnp.max(
            jnp.abs(jaxrts.math.inverse_fermi_12_fukushima_single_prec(y) - x)
        )
        < 0.001
    )
