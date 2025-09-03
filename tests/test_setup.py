from pathlib import Path

import jpu.numpy as jnpu
import pytest
from jax import numpy as jnp
from pint.errors import DimensionalityError

import jaxrts

ureg = jaxrts.ureg
cwd = Path(__file__).parent


def assert_peak_position_stability(setup) -> None:
    center = ureg("4.9keV")
    test_S = jaxrts.instrument_function.instrument_gaussian(
        (setup.measured_energy - center) / ureg.hbar, ureg("1eV") / ureg.hbar
    )
    conv_S = jaxrts.setup.convolve_stucture_factor_with_instrument(
        test_S, setup
    )

    # Get the center of the curve:
    convolution_center = jnp.average(
        setup.measured_energy.m_as(ureg.electron_volt),
        weights=conv_S.m_as(ureg.second),
    )

    grid_dist = (setup.measured_energy[1] - setup.measured_energy[0]).m_as(
        ureg.electron_volt
    )
    return (
        jnp.absolute(center.m_as(ureg.electron_volt) - convolution_center)
        < grid_dist / 4
    )


def test_peak_position_stability_with_convolution() -> None:
    for energy_grid in [
        jnp.linspace(4.8, 5.189, 1001),
        jnp.linspace(4.8, 5.189, 1000),
        jnp.linspace(4.8, 5.189, 100),
        jnp.linspace(4.84, 5.9, 4242),
    ]:
        test_setup = jaxrts.setup.Setup(
            ureg("145°"),
            ureg("5keV"),
            energy_grid * ureg.kiloelectron_volts,
            lambda x: jaxrts.instrument_function.instrument_gaussian(
                x, ureg("1eV") / ureg.hbar
            ),
        )
        assert assert_peak_position_stability(test_setup)


def test_instrumentfunction_from_array() -> None:
    def psf(x):
        return jaxrts.instrument_function.instrument_gaussian(
            x, ureg("1eV") / ureg.hbar
        )

    # w has to be quite large, because we normalize the instrument function
    # automatically.
    E = jnp.linspace(-10, 10, 400) * ureg.electron_volt
    w = E / ureg.hbar

    # Have intensities that are not normed
    intensities = psf(w) * 42

    # with units on intensities
    array_psf_1 = jaxrts.instrument_function.instrument_from_array(
        w, intensities
    )
    # without units on intensities
    array_psf_2 = jaxrts.instrument_function.instrument_from_array(
        w, intensities.m_as(ureg.picosecond)
    )
    # define the PSF over the energy and not frequencies
    array_psf_3 = jaxrts.instrument_function.instrument_from_array(
        E, intensities
    )

    assert jnp.all(jnpu.isclose(psf(w), array_psf_1(w)))
    assert jnp.all(jnpu.isclose(psf(w), array_psf_2(w)))
    assert jnp.all(jnpu.isclose(psf(w), array_psf_3(w)))


def test_instrumentfunction_from_file() -> None:
    def psf(x):
        return jaxrts.instrument_function.instrument_gaussian(
            x, ureg("2eV/hbar")
        )

    w = jnp.linspace(-30, 30, 300) * ureg.electron_volt / ureg.hbar
    file_psf_E = jaxrts.instrument_function.instrument_from_file(
        cwd / "saves/gaussian_over_E.csv"
    )
    file_psf_w = jaxrts.instrument_function.instrument_from_file(
        cwd / "saves/gaussian_over_w.csv", 1 / ureg.second
    )

    assert jnp.all(jnpu.isclose(psf(w), file_psf_E(w)))
    assert jnp.all(jnpu.isclose(psf(w), file_psf_w(w)))


def test_instrumentfunction_from_file_raises_error_on_wrong_unit() -> None:
    with pytest.raises(DimensionalityError):
        jaxrts.instrument_function.instrument_from_file(
            cwd / "saves/gaussian_over_w.csv", ureg.meter
        )
