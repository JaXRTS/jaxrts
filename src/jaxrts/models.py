"""
This submodule contains high-level wrappers for the different Models
implemented.
"""

import abc
import logging

import jax.numpy as jnp
from jpu import numpy as jnpu

from .units import ureg, Quantity
from .setup import Setup, convolve_stucture_factor_with_instrument
from .plasmastate import PlasmaState
from .elements import electron_distribution_ionized_state
from . import (
    free_free,
    bound_free,
    ion_feature,
    static_structure_factors,
    form_factors,
    plasma_physics,
)

logger = logging.getLogger(__name__)


# This defines a Model, abstractly.
class Model(metaclass=abc.ABCMeta):
    #: A list of keywords where this model is adequate for
    allowed_keys: list[str] = []

    def __init__(self, state: PlasmaState, model_key: str):
        """
        As different prerequisites exist for different models, make sure to
        test that all relevant information is given in the PlasmaState, amend
        defaults if necessary in the ``__init__``, rather than the ``evaluate``
        method. Please log assumtpions, properly
        """
        self.plasma_state = state
        self.model_key = model_key
        self.check()

    @abc.abstractmethod
    def evaluate(self, setup: Setup) -> jnp.ndarray: ...

    def check(self) -> None:
        """
        Test if the model is applicable to the PlasmaState. Might raise logged
        messages and errors.
        """


class ScatteringModel(Model):
    """
    A subset of :py:class:`Model`'s, used to provide some extra functionalities
    to (mainly elastic) scattering models.
    These models have a pre-defined :py:meth:`~.evaluate` method, which is
    performing a convolution with the instrument function, and now requires a
    user to define :py:meth:`.evaluate_raw` which is giving the scattered
    energy **without the instrument function**.

    .. note::

       As these extra functionalities are only relevant when re-sampling and
       convolution with an instrument function is reasonable, the
       :py:class:`~.Model` s used to decribe ionic scattering are currently not
       instances of :py:class:`~.ScatteringModel` as the convolution with a
       delta function would just result in numerical issues.

    It furthermore allows a user to set the :py:attr:`~.sample_points`
    attribute, which is initialized as ``None``.
    If set, the model is evaluated only on `sample_points` points, rather than
    all :math:`k` that are probed and is then evaluated. Afterwards, the result
    is extrapolated to match the :py:class:`~.setup.Setup`'s :math:`k`.
    """

    def __init__(self, state: PlasmaState, model_key: str):
        super().__init__(state, model_key)

        #: The number of points for re-sampeling the model. If ``None``, no
        #: resampeling is none and every of the :py:class:`~.setup.Setup`'s
        #: :math:`k` s is evaluated when calling :py:meth:`~.evaluate`.
        #: However, as the computation might be expensive, you can reduce the
        #: number of relevant :math:`k` s by setting this attribute. After the
        #: evaluation, the resulting scatting signal is interpolated to the
        #: relevant :math:`k` s and then convolved with the instument function.
        self.sample_points: int | None = None

    @abc.abstractmethod
    def evaluate_raw(self, setup: Setup) -> jnp.ndarray: ...

    def sample_grid(self, setup) -> Quantity:
        """
        Define the sample-grid if :py:attr:`~.sample_points` is not ``None``.
        By default, we just divide the :py:attr:`~.setup.Setup.measured_energy`
        in :py:attr:`~sample_points` equidistant energies. However, one could
        overwrite this function if the expected signal is within a certain
        range to achieve faster computation time.
        """
        min_E = setup.measured_energy[0]
        max_E = setup.measured_energy[-1]
        return jnpu.linspace(min_E, max_E, self.sample_points)

    def evaluate(self, setup) -> jnp.ndarray:
        """
        If :py:attr:`~.sample_points` is not ``None``, generate a
        low-resulution :py:class`~.setup.Setup`. Calculate the
        instrument-function free scattering intensity with this or the given
        ``setup``, interpolate it, if needed and then convolve it with the
        instument function.
        """
        if self.sample_points is None:
            raw = self.evaluate_raw(setup)
        else:
            low_res_setup = Setup(
                setup.scattering_angle,
                setup.energy,
                self.sample_grid(setup),
                setup.instrument,
            )
            low_res = self.evaluate_raw(low_res_setup)
            raw = jnpu.interp(
                setup.measured_energy,
                low_res_setup.measured_energy,
                low_res,
                left=0,
                right=0,
            )
        return convolve_stucture_factor_with_instrument(raw, setup)


# HERE LIST OF MODELS
# ===================

# Scattering
# ==========
scattering_models = [
    "free-free scattering",
    "bound-free scattering",
    "free-bound scattering",
    "ionic scattering",
]


class Neglect(Model):
    allowed_keys = [*scattering_models, "ipd"]
    """
    A model that returns an empty with zeros in (units of seconds) for every
    energy probed.
    """

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        if self.model_key in scattering_models:
            return jnp.zeros_like(setup.measured_energy) * (1 * ureg.second)
        elif self.model_key == "ipd":
            return jnp.zeros(10) * (1 * ureg.electron_volt)


# ion-feature
# -----------


class ArkhipovIonFeat(Model):
    """
    Model for the ion feature of the scatting, presented in
    :cite:`Arkhipov.1998` and :cite:`Arkhipov.2000`.

    The structure factors are obtained by using an effective potential
    (pseudopotential) model of the particle interaction of semiclassical
    two-component plasmas with a single temperature :math:`T`. We use the
    electron temperature ``state.T_e`` of the PlasmaState modelled. The authors
    take into account both quantum and collective effects.

    Requires a 'form-factors' model (defaults to
    :py:class:`~PaulingFormFactors`).

    See Also
    --------

    jaxrts.static_structure_factors.S_ii_AD
        Calculation of the static ion ion stucture factor given by
        :cite:`Arkhipov.1998`.
    jaxrts.ion_feature.q
        Calculation of the screening by (quasi) free electrons
    jaxrts.models.PaulingFormFactors
        The default model for the atomic form factors
    """

    allowed_keys = ["ionic scattering"]

    def __init__(self, state: PlasmaState, model_key) -> None:
        # Set sane defaults
        state.update_default_model("form-factors", PaulingFormFactors)
        super().__init__(state, model_key)

    def check(self) -> None:
        if self.plasma_state.T_e != self.plasma_state.T_i:
            logger.warning(
                "'ArkhipovIonFeat' can only handle plasmas with T_e == T_i."
                + " The calculation will only consider the electron temperature."  # noqa: E501
                + " Choose another model if ion and electron temperature models should be treated individually."  # noqa: E501
            )
        if len(self.plasma_state) > 1:
            logger.critical(
                "'ArkhipovIonFeat' is only implemented for a one-component plasma"  # noqa: E501
            )

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        fi = self.plasma_state["form-factors"].evaluate(setup)
        population = electron_distribution_ionized_state(
            self.plasma_state.Z_core
        )[:, jnp.newaxis]

        f = jnp.sum(fi * population)
        q = ion_feature.q(
            setup.k[jnp.newaxis],
            self.plasma_state.atomic_masses,
            self.plasma_state.n_e,
            self.plasma_state.T_e,
            self.plasma_state.T_e,
            self.plasma_state.Z_free,
        )
        S_ii = static_structure_factors.S_ii_AD(
            setup.k,
            self.plasma_state.T_e,
            self.plasma_state.T_e,
            self.plasma_state.n_e,
            self.plasma_state.atomic_masses,
            self.plasma_state.Z_free,
        )
        w_R = jnp.abs(f + q.m_as(ureg.dimensionless)) ** 2 * S_ii
        res = w_R * setup.instrument(
            (setup.measured_energy - setup.energy) / ureg.hbar
        )
        return res


class Gregori2003IonFeat(Model):
    """
    Model for the ion feature of the scatting, presented in
    :cite:`Gregori.2003`.

    This model is identical to :py:class:`~ArkhipovIonFeat`, but uses an
    effective temperature ~:py:func:`jaxtrs.static_structure_factors.T_cf_Greg`
    rather than the electron Temperature throughout the calculation.
    """

    allowed_keys = ["ionic scattering"]

    def __init__(self, state: PlasmaState, model_key) -> None:
        state.update_default_model("form-factors", PaulingFormFactors)
        super().__init__(state, model_key)

    def check(self) -> None:
        if self.plasma_state.T_e != self.plasma_state.T_i:
            logger.warning(
                "'Gregori2003IonFeat' can only handle plasmas with T_e == T_i."
                + " The calculation will only consider the electron temperature."  # noqa: E501
                + " Choose another model if ion and electron temperature models should be treated individually."  # noqa: E501
            )
        if len(self.plasma_state) > 1:
            logger.critical(
                "'Gregori2003IonFeat' is only implemented for a one-component plasma"  # noqa: E501
            )

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        fi = self.plasma_state["form-factors"].evaluate(setup)
        population = electron_distribution_ionized_state(
            self.plasma_state.Z_core[0]
        )[:, jnp.newaxis]

        T_eff = static_structure_factors.T_cf_Greg(
            self.plasma_state.T_e, self.plasma_state.n_e
        )
        f = jnp.sum(fi * population)
        q = ion_feature.q(
            setup.k[jnp.newaxis],
            self.plasma_state.atomic_masses,
            self.plasma_state.n_e,
            T_eff,
            T_eff,
            self.plasma_state.Z_free,
        )
        S_ii = ion_feature.S_ii_AD(
            setup.k,
            T_eff,
            T_eff,
            self.plasma_state.n_e,
            self.plasma_state.atomic_masses,
            self.plasma_state.Z_free,
        )
        w_R = jnp.abs(f + q.m_as(ureg.dimensionless)) ** 2 * S_ii
        res = w_R * setup.instrument(
            (setup.measured_energy - setup.energy) / ureg.hbar
        )
        return res


class Gregori2006IonFeat(Model):
    """
    Model for the ion feature of the scatting, presented in
    :cite:`Gregori.2006`.

    This model extends :py:class:`~ArkhipovIonFeat`, to allow for different
    ion-and electron temperatures.

    .. note::

       :cite:`Gregori.2006` uses effective temperatures for the ion and
       electron temperatures to obtain sane limits to :math:`T\\rightarrow 0`.
       This is done by calling
       :py:func:`jaxtrs.static_structure_factors.T_cf_Greg` for the electron
       temperature and :py:func:`jaxrts.static_structure_factors.T_i_eff_Greg`,
       for the ionic temperatures. The latter requires a 'Debye temperature'
       model.

    Requires a 'Debye temperature' model (defaults to :py:class:`~BohmStaver`).
    """

    allowed_keys = ["ionic scattering"]

    def __init__(self, state: PlasmaState, model_key) -> None:
        state.update_default_model("form-factors", PaulingFormFactors)
        state.update_default_model("Debye temperature", BohmStaver)
        super().__init__(state, model_key)

    def check(self) -> None:
        if len(self.plasma_state) > 1:
            logger.critical(
                "'Gregori2006IonFeat' is only implemented for a one-component plasma"  # noqa: E501
            )

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        fi = self.plasma_state["form-factors"].evaluate(setup)
        population = electron_distribution_ionized_state(
            self.plasma_state.Z_core
        )[:, jnp.newaxis]

        T_D = self.plasma_state["Debye temperature"].evaluate(setup)

        T_e_eff = static_structure_factors.T_cf_Greg(
            self.plasma_state.T_e, self.plasma_state.n_e
        )
        T_i_eff = static_structure_factors.T_i_eff_Greg(
            self.plasma_state.T_i, T_D
        )
        f = jnp.sum(fi * population)
        q = ion_feature.q(
            setup.k[jnp.newaxis],
            self.plasma_state.atomic_masses,
            self.plasma_state.n_e,
            T_e_eff,
            T_i_eff,
            self.plasma_state.Z_free,
        )
        S_ii = ion_feature.S_ii_AD(
            setup.k,
            T_e_eff,
            T_i_eff,
            self.plasma_state.n_e,
            self.plasma_state.atomic_masses,
            self.plasma_state.Z_free,
        )
        w_R = jnp.abs(f + q.m_as(ureg.dimensionless)) ** 2 * S_ii
        res = w_R * setup.instrument(
            (setup.measured_energy - setup.energy) / ureg.hbar
        )
        return res


# Free-free models
# ----------------


class QCSalpeterApproximation(ScatteringModel):
    """
    Quantum Corrected Salpeter Approximation for free-free scattering.
    Presented in :cite:`Gregori.2003`, which provide a quantum correction to
    the results by Salpeter (:cite:`Salpeter.1960`), increasing the range of
    applicability.

    However, this model might be rather considered to be educational, as it is
    only valid for small(er) densities and probing energies. Instead, one might
    use :py:class:~RPA_NoDamping`, which should give more accurate results
    (according to, e.g., "cite:`Gregori.2003`) at a comparable computation
    time.

    See Also
    --------
    jaxtrs.free_free.S0_ee_Salpeter(
        Function used to calculate the dynamic free electron-electron structure
        factor.
    """

    allowed_keys = ["free-free scattering"]

    def check(self) -> None:
        if len(self.plasma_state) > 1:
            logger.critical(
                "'QCSalpeterApproximation' is only implemented for a one-component plasma"  # noqa: E501
            )

    def evaluate_raw(self, setup: Setup) -> jnp.ndarray:
        See_0 = free_free.S0_ee_Salpeter(
            setup.k,
            self.plasma_state.T_e,
            self.plasma_state.n_e,
            setup.measured_energy - setup.energy,
        )

        return See_0 * self.plasma_state.Z_free


class RPA_NoDamping(ScatteringModel):
    """
    Model for elastic free-free scattering based on the Random Phase
    Approximation

    Calculates the dielectic function in RPA and obtain a Structure factor via
    the fluctuation dissipation theorem. Based on lecture notes from M. Bonitz.

    Requires a 'chemical potential' model (defaults to
    :py:class:`~GregoriChemPotential`).

    See Also
    --------
    jaxtrs.free_free.S0_ee_RPA_no_damping
        Function used to calculate the dynamic free-free electron structure
        factor.
    """

    allowed_keys = ["free-free scattering"]

    def __init__(self, state: PlasmaState, model_key) -> None:
        state.update_default_model("chemical potential", GregoriChemPotential)
        super().__init__(state, model_key)

    def check(self) -> None:
        if len(self.plasma_state) > 1:
            logger.critical(
                "'RPA_NoDamping' is only implemented for a one-component plasma"  # noqa: E501
            )

    def evaluate_raw(self, setup: Setup) -> jnp.ndarray:
        mu = self.plasma_state["chemical potential"].evaluate(setup)
        See_0 = free_free.S0_ee_RPA_no_damping(
            setup.k,
            self.plasma_state.T_e,
            self.plasma_state.n_e,
            setup.measured_energy - setup.energy,
            mu,
        )

        return See_0 * self.plasma_state.Z_free


class BornMermin(ScatteringModel):
    """
    Model of the free-free scattering, based on the Born Mermin Approximation
    (:cite:`Mermin.1970`).

    Requires a 'chemical potential' model (defaults to
    :py:class:`~GregoriChemPotential`).

    See Also
    --------

    jaxrts.free_free.S0_ee_BMA(
        Function used to calculate the dynamic structure factor
    """

    allowed_keys = ["free-free scattering"]

    def __init__(self, state: PlasmaState, model_key) -> None:
        state.update_default_model("chemical potential", GregoriChemPotential)
        super().__init__(state, model_key)

    def check(self) -> None:
        if len(self.plasma_state) > 1:
            logger.critical(
                "'BornMermin' is only implemented for a one-component plasma"  # noqa: E501
            )

    def evaluate_raw(self, setup: Setup) -> jnp.ndarray:
        mu = self.plasma_state["chemical potential"].evaluate(setup)
        See_0 = free_free.S0_ee_BMA(
            setup.k,
            self.plasma_state.T_e,
            mu,
            self.plasma_state.atomic_masses,
            self.plasma_state.n_e,
            self.plasma_state.Z_free,
            setup.measured_energy - setup.energy,
        )
        return See_0 * self.plasma_state.Z_free


class BornMermin_ChapmanInterp(ScatteringModel):
    """
    Model of the free-free scattering, based on the Born Mermin Approximation
    (:cite:`Mermin.1970`).
    Uses the Chapman Interpolation which allows for a faster computation of the
    free-free scattering compared to :py:class:`~.BornMermin`, by sampleing at
    the probing frequency at :py:attr:`~.no_of_freq` points and interpolating
    between them, after.

    The number of frequencies defaults to 20. To change it, just change the
    attribute of this model after initializing it. i.e.

    >>> state["free-free scattering"] = jaxrts.models.BornMermin_ChapmanInterp
    >>> state["free-free scattering"].no_of_freq = 10

    Requires a 'chemical potential' model (defaults to
    :py:class:`~GregoriChemPotential`).

    See Also
    --------

    jaxrts.free_free.S0_ee_BMA_chapman_interp
        Function used to calculate the dynamic structure factor
    """

    allowed_keys = ["free-free scattering"]

    def __init__(self, state: PlasmaState, model_key) -> None:
        state.update_default_model("chemical potential", GregoriChemPotential)
        super().__init__(state, model_key)
        #: Number of frequencies used for the interpolation of the elastic
        #: scattering. Defaults to 20.
        self.no_of_freq: int = 20

    def check(self) -> None:
        if len(self.plasma_state) > 1:
            logger.critical(
                "'BornMermin_ChapmanInterp' is only implemented for a one-component plasma"  # noqa: E501
            )

    def evaluate_raw(self, setup: Setup) -> jnp.ndarray:
        mu = self.plasma_state["chemical potential"].evaluate(setup)
        See_0 = free_free.S0_ee_BMA_chapman_interp(
            setup.k,
            self.plasma_state.T_e,
            mu,
            self.plasma_state.atomic_masses,
            self.plasma_state.n_e,
            self.plasma_state.Z_free,
            setup.measured_energy - setup.energy,
            self.no_of_freq
        )
        return See_0 * self.plasma_state.Z_free


# bound-free Models
# -----------------


class SchumacherImpulse(ScatteringModel):
    """
    Bound-free scattering based on the Schumacher Impulse Approximation
    :cite:`Schumacher.1975`. The implementation considers the first order
    asymmetric correction to the impulse approximation, as given in the
    aforementioned paper.

    Requires a 'form-factors' model (defaults to
    :py:class:`~PaulingFormFactors`).
    """

    allowed_keys = ["bound-free scattering"]

    def __init__(self, state: PlasmaState, model_key) -> None:
        state.update_default_model("form-factors", PaulingFormFactors)
        super().__init__(state, model_key)

    def check(self) -> None:
        if len(self.plasma_state) > 1:
            logger.critical(
                "'SchumacherImpulse' is only implemented for a one-component plasma"  # noqa: E501
            )

    def evaluate_raw(self, setup: Setup) -> jnp.ndarray:
        k = setup.k
        omega_0 = setup.energy / ureg.hbar
        omega = omega_0 - setup.measured_energy / ureg.hbar
        Z_c = self.plasma_state.Z_core[0]
        E_b = self.plasma_state.ions[0].binding_energies

        Zeff = form_factors.pauling_effective_charge(
            self.plasma_state.ions[0].Z
        )
        population = electron_distribution_ionized_state(Z_c)
        # Gregori.2004, Eqn 20
        fi = self.plasma_state["form-factors"].evaluate(setup)
        r_k = 1 - jnp.sum(population[:, jnp.newaxis] / Z_c * fi**2)
        B = 1 + 1 / omega_0 * (ureg.hbar * k**2) / (2 * ureg.electron_mass)
        sbe = (
            r_k
            / (Z_c * B**3)
            * bound_free.J_impulse_approx(omega, k, population, Zeff, E_b)
        )
        return sbe * Z_c


# free-bound Models
# -----------------


class DetailedBalance(ScatteringModel):
    """
    Calculate the free-bound scattering by mirroring the free-bound scattering
    around the probing energy and applying a detailed balance factor to the
    intensity.
    See :cite:`Bohme.2023`, introducing the idea.

    .. note::

       We would recommend to have an `evaluate_raw` for the bound-free model,
       which should return the bound-free scattering intensity **not
       convolved** with an instrument function.
       While this Model works in any way, a model as described above should be
       numerically more stable.

    """

    allowed_keys = ["free-bound scattering"]

    def evaluate_raw(self, setup: Setup) -> jnp.ndarray:
        energy_shift = setup.measured_energy - setup.energy
        mirrored_setup = Setup(
            setup.scattering_angle,
            setup.energy,
            setup.energy - energy_shift,
            setup.instrument,
        )
        db_factor = jnpu.exp(
            -energy_shift / (self.plasma_state.T_e * ureg.k_B)
        )
        free_bound = self.plasma_state["bound-free scattering"].evaluate_raw(
            mirrored_setup
        )
        return free_bound * db_factor


# Form Factor Models
# ==================


class PaulingFormFactors(Model):
    """
    Analytical functions for each electrons in quantum states defined by the
    quantum numbers `n` and `l`, assuming a hydrogen-like atom. Published in
    :cite:`Pauling.1932`.

    Uses :py:func:`jaxrts.form_factors.pauling_effective_charge` to calculate
    the effective charge of the atom's core and then calculates form factors
    with :py:func:`jaxrts.form_factors.pauling_all_ff`.
    """

    allowed_keys = ["form-factors"]

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        Zstar = form_factors.pauling_effective_charge(self.plasma_state.Z_A)
        ff = form_factors.pauling_all_ff(setup.k, Zstar)
        # population = self.plasma_state.ions[0].electron_distribution
        # return jnp.where(population > 0, ff, 0)
        return ff


# Chemical Potential Models
# =========================


class GregoriChemPotential(Model):
    """
    A fitting formula for the chemical potential of a plasma between the
    classical and the quantum regime, given by :cite:`Gregori.2003`.
    Uses :py:func:`jaxrts.plasma_physics.chem_pot_interpolation`.
    """

    allowed_keys = ["chemical potential"]

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        return plasma_physics.chem_pot_interpolation(
            self.plasma_state.T_e, self.plasma_state.n_e
        )


# Debye Temperature Models
# ========================


class BohmStaver(Model):
    """
    The Bohm-Staver relation for the Debye temperature, valid for 'simple
    metals', as it is presented in Eqn (3) of :cite:`Gregori.2006`.

    See Also
    --------
    jaxrts.static_structure_factors.T_Debye_Bohm_Staver
        The function used for calculating the Debye temperature.
    """

    allowed_keys = ["Debye temperature"]

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        return static_structure_factors.T_Debye_Bohm_Staver(
            self.plasma_state.T_e,
            self.plasma_state.n_e,
            self.plasma_state.atomic_masses,
            self.plasma_state.Z_free,
        )
