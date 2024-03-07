"""
This submodule contains high-level wrappers for the different Models
implemented.
"""

import abc
import logging

import jax.numpy as jnp

from .units import ureg
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
    def __init__(self, state: PlasmaState):
        """
        As not different prerequisites exist for different models, make sure to
        test that all relevant information is given in the PlasmaState, amend
        defaults if necessary in the ``__init__``, rather than the ``evaluate``
        method. Please log assumtpions, properly
        """
        self.plasma_state = state
        self.check()

    @abc.abstractmethod
    def evaluate(self, setup: Setup) -> jnp.ndarray: ...

    def check(self) -> None:
        """
        Test if the model is applicable to the PlasmaState. Might raise logged
        messages and errors.
        """


# HERE LIST OF MODELS
# ===================

# Scattering
# ==========


class Neglect(Model):
    """
    A model that returns an empty with zeros in (units of seconds) for every
    energy probed.
    """

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        return jnp.zeros_like(setup.measured_energy) * (1 * ureg.second)


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

    def __init__(self, state: PlasmaState) -> None:
        # Set sane defaults
        state.update_default_model("form-factors", PaulingFormFactors)
        super().__init__(state)

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
            self.plasma_state.Z_free,
        )
        S_ii = static_structure_factors.S_ii_AD(
            setup.k,
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

    def __init__(self, state: PlasmaState) -> None:
        state.update_default_model("form-factors", PaulingFormFactors)
        super().__init__(state)

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
            self.plasma_state.Z_core
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
            self.plasma_state.Z_free,
        )
        S_ii = ion_feature.S_ii_AD(
            setup.k,
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


# Free-free models
# ----------------


class QCSalpeterApproximation(Model):
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

    def check(self) -> None:
        if len(self.plasma_state) > 1:
            logger.critical(
                "'QCSalpeterApproximation' is only implemented for a one-component plasma"  # noqa: E501
            )

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        See_0 = free_free.S0_ee_Salpeter(
            setup.k,
            self.plasma_state.T_e,
            self.plasma_state.n_e,
            setup.measured_energy - setup.energy,
        )

        ff = See_0 * self.plasma_state.Z_free
        return convolve_stucture_factor_with_instrument(ff, setup)


class RPA_NoDamping(Model):
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
        facotr.
    """

    def __init__(self, state: PlasmaState) -> None:
        state.update_default_model("chemical potential", GregoriChemPotential)
        super().__init__(state)

    def check(self) -> None:
        if len(self.plasma_state) > 1:
            logger.critical(
                "'RPA_NoDamping' is only implemented for a one-component plasma"  # noqa: E501
            )

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        mu = self.plasma_state["chemical potential"].evaluate(setup)
        See_0 = free_free.S0_ee_RPA_no_damping(
            setup.k,
            self.plasma_state.T_e,
            self.plasma_state.n_e,
            setup.measured_energy - setup.energy,
            mu,
        )

        ff = See_0 * self.plasma_state.Z_free
        return convolve_stucture_factor_with_instrument(ff, setup)


class BornMermin(Model):
    """
    Modell of the free-free scattering, based on the Born Mermin Approximation
    (:cite:`Mermin.1970`).

    Requires a 'chemical potential' model (defaults to
    :py:class:`~GregoriChemPotential`).

    See Also
    --------

    jaxrts.free_free.S0_ee_BMA(
        Function used to calculate the dynamic structure factor
    """

    def __init__(self, state: PlasmaState) -> None:
        state.update_default_model("chemical potential", GregoriChemPotential)
        super().__init__(state)

    def check(self) -> None:
        if len(self.plasma_state) > 1:
            logger.critical(
                "'BornMermin' is only implemented for a one-component plasma"  # noqa: E501
            )

    def evaluate(self, setup: Setup) -> jnp.ndarray:
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
        ff = See_0 * self.plasma_state.Z_free
        return convolve_stucture_factor_with_instrument(ff, setup)


# bound-free Models
# -----------------


class SchumacherImpulse(Model):
    """
    Bound-free scattering based on the Schumacher Impulse Approximation
    :cite:`Schumacher.1975`. The implementation considers the first order
    asymmetric correction to the impulse approximation, as given in the
    aforementioned paper.

    Requires a 'form-factors' model (defaults to
    :py:class:`~PaulingFormFactors`).
    """

    def __init__(self, state: PlasmaState) -> None:
        state.update_default_model("form-factors", PaulingFormFactors)
        super().__init__(state)

    def check(self) -> None:
        if len(self.plasma_state) > 1:
            logger.critical(
                "'SchumacherImpulse' is only implemented for a one-component plasma"  # noqa: E501
            )

    def evaluate(self, setup: Setup) -> jnp.ndarray:
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
        return convolve_stucture_factor_with_instrument(sbe * Z_c, setup)


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

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        return plasma_physics.chem_pot_interpolation(
            self.plasma_state.T_e, self.plasma_state.n_e
        )
