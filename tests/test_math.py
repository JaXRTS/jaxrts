import jax
from jax import numpy as jnp
from scipy.fft import dst as scipy_dst

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


def test_cramer_solve():
    Ngrid = 4096

    key = jax.random.PRNGKey(42534252356534)
    for M in [1, 2, 3, 4]:
        for _ in range(50):
            key, loopkey1, loopkey2 = jax.random.split(key, 3)
            # start with diagonal matrix to avoid ill-posed problems
            A = jnp.eye(M) * jax.random.normal(loopkey1, (Ngrid, M, M))
            b = jax.random.normal(loopkey2, (Ngrid, M))

            ref = jax.vmap(lambda Ai, bi: jnp.linalg.solve(Ai, bi))(A, b)
            got = jax.vmap(jaxrts.helpers.cramer_solve)(A, b)
            err = float(jnp.max(jnp.abs(ref - got)))
            assert err < 1e-8, f"Error for inverting {M}"


def test_dst_against_scipy():
    Ngrid = 4096

    key = jax.random.PRNGKey(894148965434567)
    for _ in range(500):
        (
            key,
            loopkey1,
        ) = jax.random.split(key, 2)
        f = jax.random.normal(loopkey1, (Ngrid,))

        # Our norm is slightly different
        ref = scipy_dst(f, type=4) / 2
        got = jaxrts.hypernetted_chain.dst4(f)
        err = float(jnp.mean(jnp.abs(ref - got)))
        assert err < 1e-12
