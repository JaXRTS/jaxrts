from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp

from jpu import numpy as jnpu

from .units import ureg, Quantity
from .plasma_physics import plasma_frequency


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
        self.instrument: Callable = jax.tree_util.Partial(instrument)

    @property
    def k(self) -> Quantity:
        """
        The scattering vector length probed in the experiment.
        """
        return (4 * jnp.pi / self.lambda0) * jnpu.sin(
            jnpu.deg2rad(self.scattering_angle) / 2.0
        )

    @property
    def full_k(self) -> Quantity:
        """
        The scattering vector length probed in the experiment at each energy
        chanel.
        """
        k_in = self.energy / ureg.hbar / ureg.c
        k_out = self.measured_energy / ureg.hbar / ureg.c

        k = jnpu.sqrt(
            (k_out**2 + k_in**2)
            - (
                (2 * k_out * k_in)
                * jnpu.cos(jnpu.deg2rad(self.scattering_angle))
            )
        )
        return k

    @property
    def lambda0(self) -> Quantity:
        """
        The wavelength of the probing light, i.e., the wavelength
        corresponding to :py:attr:`~energy`.
        """
        return ureg.planck_constant * ureg.c / self.energy

    # The following is required to jit a setup
    def _tree_flatten(self):
        children = (
            self.scattering_angle,
            self.energy,
            self.measured_energy,
            self.instrument,
        )
        aux_data = ()  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(Setup)
        (
            obj.scattering_angle,
            obj.energy,
            obj.measured_energy,
            obj.instrument,
        ) = children
        return obj


@jax.jit
def dispersion_corrected_k(setup: Setup, n_e: Quantity) -> Quantity:
    omega_in = setup.energy / ureg.hbar
    omega_out = setup.measured_energy / ureg.hbar
    omega_pl = plasma_frequency(n_e)
    k_in = omega_in / ureg.c
    k_out = omega_out / ureg.c

    # Do the dispersion correction:
    k_in *= jnpu.sqrt(1 - omega_pl**2 / omega_in**2)
    k_out *= jnpu.sqrt(1 - omega_pl**2 / omega_out**2)

    k = jnpu.sqrt(
        (k_out**2 + k_in**2)
        - (2 * k_out * k_in * jnpu.cos(setup.scattering_angle))
    )
    return k


@jax.jit
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


@jax.jit
def get_probe_setup(k: Quantity, setup: Setup) -> Setup:
    """
    Returns a :py:class:`~.Setup`, which has the same properties of setup, but
    a different `k`. This is realized by modifying the
    :py:attr:`~.measured_energy` attribute of setup.

    This helper function can be useful when evaluating e.g., `S_ii` Models at a
    different `k` than realized in an experiment, as it is required, e.g., for
    the Born collision frequency.
    """
    E = (
        (1 * ureg.hbar)
        * k
        * (1 * ureg.c)
        / (2 * jnpu.sin(setup.scattering_angle / 2))
    )
    return Setup(
        setup.scattering_angle, E, setup.measured_energy, setup.instrument
    )


jax.tree_util.register_pytree_node(
    Setup,
    Setup._tree_flatten,
    Setup._tree_unflatten,
)
