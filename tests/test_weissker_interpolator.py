import jax.numpy as jnp
import jaxrts

ureg = jaxrts.ureg


def test_1d_interpolation():
    x = jnp.linspace(0, 125, 500)
    curve1 = jaxrts.instrument_function.instrument_gaussian(x - 30, 5)
    curve2 = jaxrts.instrument_function.instrument_gaussian(x - 70, 10)
    curve_true = jaxrts.instrument_function.instrument_gaussian(x - 50, 7.5)

    grid = jnp.array([curve1, curve2]).T
    interpolator = jaxrts.weissker_interpolator.AutoNormInterpolator(
        grid * ureg.dimensionless,
        x / (1 * ureg.angstrom),
        jnp.array([0, 1]) * ureg.dimensionless,
    )

    interp = interpolator([0.5 * ureg.dimensionless]).m_as(ureg.dimensionless)
    assert jnp.max(jnp.absolute(curve_true - interp)) < 0.02


def test_2d_interpolation():
    x = jnp.linspace(0, 125, 500)
    curve1 = jaxrts.instrument_function.instrument_gaussian(x - 30, 5)
    curve2 = jaxrts.instrument_function.instrument_gaussian(x - 30, 10)
    curve3 = jaxrts.instrument_function.instrument_gaussian(x - 70, 10)
    curve4 = jaxrts.instrument_function.instrument_gaussian(x - 70, 5)
    curve_true = jaxrts.instrument_function.instrument_gaussian(x - 50, 7.5)

    grid = jnp.array([[curve1, curve2], [curve3, curve4]]).T
    interpolator = jaxrts.weissker_interpolator.AutoNormInterpolator(
        grid,
        x / (1 * ureg.angstrom),
        jnp.array([0, 1]) * ureg.dimensionless,
        jnp.array([0, 1]) * ureg.dimensionless,
    )

    interp = interpolator([0.5 * ureg.dimensionless, 0.5 * ureg.dimensionless])

    assert jnp.max(jnp.absolute(curve_true - interp)) < 0.02


def test_interpolation_with_vdims():
    x = jnp.linspace(0, 125, 500)
    curve1 = jaxrts.instrument_function.instrument_gaussian(x - 30, 5)
    curve2 = jaxrts.instrument_function.instrument_gaussian(x - 70, 10)
    curve_true = jaxrts.instrument_function.instrument_gaussian(x - 50, 7.5)

    grid = jnp.array([curve1, curve2]).T[jnp.newaxis, jnp.newaxis, :]
    interpolator = jaxrts.weissker_interpolator.AutoNormInterpolator(
        grid * ureg.meter,
        x / (1 * ureg.angstrom),
        jnp.array([0, 1]) * ureg.dimensionless,
    )

    interp = interpolator([0.5 * ureg.dimensionless]).m_as(ureg.meter)[0, 0, :]

    assert jnp.max(jnp.absolute(curve_true - interp)) < 0.02
