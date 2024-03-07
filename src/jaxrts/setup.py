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

        #: The scattering angle of the experiment
        self.scattering_angle: Quantity = scattering_angle
        #: The base-energy at which we probe. This should be the central
        #: energy, for the instrument function, see :py:attr:`~.instrument`.
        self.energy: Quantity = energy
        #: The energies at which the scatting is recorded. This is an array of
        #: absolute energies and not an energy shift.
        self.measured_energy: Quantity = measured_energy
        #: A callable function over energy shifts that gives the total
        #: instrument spread. The curve should be normed so that the integral
        #: from `-inf` to `inf` should be one.
        self.instrument: Callable = instrument

    @property
    def k(self) -> Quantity:
        """
        The scattering vector length probed in the experiment.
        """
        return (4 * jnp.pi / self.lambda0) * jnpu.sin(
            jnpu.deg2rad(self.scattering_angle) / 2.0
        )

    @property
    def lambda0(self) -> Quantity:
        """
        The wavelength of the probing light, i.e., the wavelength
        corresponding to :py:attr:`~energy`.
        """
        return ureg.planck_constant * ureg.c / self.energy


def convolve_stucture_factor_with_instrument(
    Sfac: Quantity, setup: Setup
) -> Quantity:
    """
    Colvolve a dynamic structure factor with the instrument function, given by
    the ``setup``.

    .. note::
       The convolution grid is automatically determined from the ``setup``.
       This step requires :py:attr:`jaxrts.setup.Setup.measured_energy` to be
       spaced out equidistantly.

    Parameters
    ----------
    Sfac: Quantity
        A dynamic structure factor. Should have units of [time].
    setup: Setup
        The Setup object containing the instrument function

    Returns
    -------
    Quantity
        The convolution of the structure factor with the instrument function.
    """
    conv_grid = (
        setup.measured_energy - jnpu.mean(setup.measured_energy)
    ) / ureg.hbar
    return (
        jnp.convolve(
            Sfac.m_as(ureg.second),
            setup.instrument(conv_grid).m_as(ureg.second),
            mode="same",
        )
        * (1 * ureg.second**2)
        * (jnpu.diff(setup.measured_energy)[0] / ureg.hbar)
    )
