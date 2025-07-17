"""
This submodule is dedicated to handling the setup configuration for the
calculation of the synthetic spectra.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from jpu import numpy as jnpu

from .plasma_physics import plasma_frequency
from .units import Quantity, ureg


class Setup:

    def __init__(
        self,
        scattering_angle: Quantity,
        energy: Quantity,
        measured_energy: Quantity,
        instrument: Callable,
        correct_k_dispersion: bool = True,
        frc_exponent: float = 0.0,
    ):

        #: The scattering angle of the experiment
        self.scattering_angle: Quantity = scattering_angle
        #: The base-energy at which we probe. This should be the central
        #: energy, for the instrument function, see :py:attr:`~.instrument`.
        self.energy: Quantity = energy
        #: The energies at which the scattering is recorded. This is an array
        #: of absolute energies and not an energy shift.
        self.measured_energy: Quantity = measured_energy
        #: A callable function over energy shifts that gives the total
        #: instrument spread. The curve should be normed so that the integral
        #: from `-inf` to `inf` should be one.
        self.instrument: Callable = jax.tree_util.Partial(instrument)
        #: In an experiment, the value of k is dependent on the measured energy
        #: at this position on the detector. When including this effect, the
        #: returned spectrum will, however, violate detailed balance. It might
        #: therefore be instructive  to set this value (which is ``True`` be
        #: default) to ``False`` to #: check, e.g. temperature differences when
        #: applying ITCF analysis methods.
        self.correct_k_dispersion: bool = correct_k_dispersion
        #: Exponent of the frequency redistribiton correction in the
        #: differential crossection.
        #:
        #: .. math::
        #: 
        #:    \left(\frac{\omega_\text{out}}{\omega_\text{in}}\right)^\nu
        #:    S(\omega, k)
        #:
        #: See :cite:`Crowley.2013`.
        #:
        #: Defaults to 0.
        #: 
        #: * If 0, the methods :py:meth:`~.PlasmaState.probe` and
        #:   :py:meth:`~.PlasmaState.evaluate` will return dynamic structure
        #:   factors (this is the default behavior of jaxrts).
        #: * However, when comparing to experimental data in which the signal
        #:   is proportional to the number of photons on the detector, the apt
        #:   comparison is to the differential crosssection, which is
        #:   proportional to
        #:   :math:`\frac{\omega_\text{out}}{\omega_\text{in}} S(\omega, k)`
        #:   i.e., :math:`\nu = 1`.
        #: * If the detector-signal is proportional to the deposited energy
        #:   instead (as is should be the case for CCD detectors if no further
        #:   processing occurs), an additional scaling with energy is
        #:   introduced and ``frc_exponent`` should be set to 2.
        #: 
        self.frc_exponent: float = frc_exponent

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
        channel.
        """
        if self.correct_k_dispersion:
            k_in = self.energy / ureg.hbar / ureg.c
            k_out = self.measured_energy / ureg.hbar / ureg.c

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

    @property
    @jax.jit
    def frequency_redistribution_correction(self) -> jnp.ndarray:
        """
        Returns the frequency redistribition correction

        .. math::
            \\left(\\frac{\\omega_\\text{out}}{\\omega_\\text{in}}\\right)^\\nu

        where :math:`\\nu` is :py:attr:`~.frc_exponent`.
        Will return an array of ones if :py:attr:`~.frc_exponent` ``==0``,
        i.e., this setup will produce dynamic structure factors.
        Otherwise, this correction yields the correction to have outputs
        proportional to the adequate differential cross-sections for
        single-photon counting measurements
        (:py:attr:`~.frc_exponent` ``==1``),
        or measurements proportional to the energy deposited
        (:py:attr:`~.frc_exponent` ``==2``).
        See :cite:`Crowley.2013`, for details.
        """
        return (self.measured_energy / self.energy).m_as(
            ureg.dimensionless
        ) ** (self.frc_exponent)

    @jax.jit
    def dispersion_corrected_k(self, n_e: Quantity) -> Quantity:
        """
        Returns the dispersion corrected wavenumber.
        """

        if self.correct_k_dispersion:
            omega_in = self.energy / ureg.hbar
            omega_out = self.measured_energy / ureg.hbar
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
            self.frc_exponent,
        )
        aux_data = (self.correct_k_dispersion,)  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(Setup)
        (
            obj.scattering_angle,
            obj.energy,
            obj.measured_energy,
            obj.instrument,
            obj.frc_exponent,
        ) = children
        (obj.correct_k_dispersion,) = aux_data
        return obj


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

    # Shift the grid by the minimal difference to zero.
    # This ensures that the position of a peak will not move during
    # convolution. See the test
    # test_setup.test_peak_position_stability_with_convolution.
    # Even a small shift might be very relevant for ITCF based analysis and
    # tests.
    conv_grid += jnpu.min(jnpu.absolute(conv_grid))
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
    :py:attr:`~.Setup.measured_energy` attribute of setup.

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
        setup.scattering_angle,
        E,
        setup.measured_energy,
        setup.instrument,
        setup.correct_k_dispersion,
        setup.frc_exponent,
    )


jax.tree_util.register_pytree_node(
    Setup,
    Setup._tree_flatten,
    Setup._tree_unflatten,
)
