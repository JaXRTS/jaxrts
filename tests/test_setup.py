import pytest
from jax import numpy as jnp

import jaxrts

ureg = jaxrts.ureg


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
        print(assert_peak_position_stability(test_setup))


test_peak_position_stability_with_convolution()
