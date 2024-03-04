from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp

from jpu import numpy as jnpu

from .units import ureg, Quantity


class Setup:

    def __init__(
        self,
        scattering_angle: Quantity,
        energy: Quantity,
        measured_energy: Quantity,
        instrument: Callable,
    ):

        self.scattering_angle = scattering_angle
        self.energy = energy
        self.measured_energy = measured_energy
        self.instrument = instrument

    def k(self) -> Quantity:
        return (4 * jnp.pi / self.lambda0()) * jnpu.sin(
            jnpu.deg2rad(self.scattering_angle) / 2.0
        )

    def lambda0(self) -> Quantity:
        return ureg.planck_constant * ureg.c / self.energy
