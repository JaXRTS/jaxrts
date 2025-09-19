"""
This submodule is dedicated to calculate the contribution of the free bound
contributions to the dynamic structure factor.
Currently the free-bound contribution is calculated using the bound-free
contribution and assuming the principle of detailed balance.
"""

from collections.abc import Callable

import jax
from jpu import numpy as jnpu

from .setup import Setup, plasma_frequency
from .units import Quantity, ureg


class FreeBoundFlippedSetup(Setup):
    """
    An auxiliary setup, which flips the measured energy, but not k. This object
    can be used to :py:meth:`jaxrts.models.Model.evaluate_raw` a bound-free
    :py:class:`jaxrts.models.Model`

    .. warning ::

       Due to the nature of this object,
       :py:meth:`~FreeBoundFlippedSetup.full_k` and
       :py:meth:`~FreeBoundFlippedSetup.dispersion_corrected_k` are not
       monotonically rising with
       :py:attr:`~FreeBoundFlippedSetup.measured_energy`, anymore, but rather
       falling. In other words, the k values do not correspond to the probing
       energy.

    """

    def __init__(
        self,
        setup: Setup,
    ):
        self.scattering_angle: Quantity = setup.scattering_angle
        self.energy: Quantity = setup.energy
        self.measured_energy: Quantity = (
            2 * setup.energy - setup.measured_energy
        )
        self.instrument: Callable = setup.instrument
        self.correct_k_dispersion: bool = setup.correct_k_dispersion

    @property
    def full_k(self) -> Quantity:
        """
        The scattering vector length probed in the experiment at each energy
        channel.
        """
        if self.correct_k_dispersion:
            original_measured_energy = 2 * self.energy - self.measured_energy
            k_in = self.energy / ureg.hbar / ureg.c
            k_out = original_measured_energy / ureg.hbar / ureg.c

            k = jnpu.sqrt(
                (k_out**2 + k_in**2)
                - (
                    (2 * k_out * k_in)
                    * jnpu.cos(jnpu.deg2rad(self.scattering_angle))
                )
            )
        else:
            k = self.k * jnpu.ones_like(self.measured_energy)
        return k

    @jax.jit
    def dispersion_corrected_k(self, n_e: Quantity) -> Quantity:
        """
        Returns the dispersion corrected wavenumber.
        """

        if self.correct_k_dispersion:
            original_measured_energy = 2 * self.energy - self.measured_energy
            omega_in = self.energy / ureg.hbar
            omega_out = original_measured_energy / ureg.hbar
            omega_pl = plasma_frequency(n_e)
            k_in = omega_in / ureg.c
            k_out = omega_out / ureg.c

            # Do the dispersion correction:
            k_in *= jnpu.sqrt(1 - omega_pl**2 / omega_in**2)
            k_out *= jnpu.sqrt(1 - omega_pl**2 / omega_out**2)

            k = jnpu.sqrt(
                (k_out**2 + k_in**2)
                - (2 * k_out * k_in * jnpu.cos(self.scattering_angle))
            )
        else:
            k = self.k * jnpu.ones_like(self.measured_energy)
        return k

    # The following is required to jit a setup
    def _tree_flatten(self):
        children = (
            self.scattering_angle,
            self.energy,
            self.measured_energy,
            self.instrument,
        )
        aux_data = (self.correct_k_dispersion,)  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(FreeBoundFlippedSetup)
        (
            obj.scattering_angle,
            obj.energy,
            obj.measured_energy,
            obj.instrument,
        ) = children
        (obj.correct_k_dispersion,) = aux_data
        return obj


jax.tree_util.register_pytree_node(
    FreeBoundFlippedSetup,
    FreeBoundFlippedSetup._tree_flatten,
    FreeBoundFlippedSetup._tree_unflatten,
)
