"""
This submodule contains high-level wrappers for the different Models
implemented.
"""

import abc
import logging

import jax
import jax.numpy as jnp
from jpu import numpy as jnpu

from .units import ureg, Quantity, to_array
from .setup import Setup, convolve_stucture_factor_with_instrument
from .plasmastate import PlasmaState
from .elements import electron_distribution_ionized_state
from . import (
    bound_free,
    form_factors,
    free_free,
    hnc_potentials,
    hypernetted_chain,
    ion_feature,
    plasma_physics,
    static_structure_factors,
)

logger = logging.getLogger(__name__)


# This defines a Model, abstractly.
class Model(metaclass=abc.ABCMeta):
    #: A list of keywords where this model is adequate for
    allowed_keys: list[str] = []

    def __init__(self):
        """ """
        self.model_key = ""

    @abc.abstractmethod
    def evaluate(
        self, plasma_state: PlasmaState, setup: Setup
    ) -> jnp.ndarray: ...

    def prepare(self, plasma_state: PlasmaState) -> None:
        """
        Modify the plasma_state in place.

        As different prerequisites exist for different models, make sure to
        test that all relevant information is given in the PlasmaState, amend
        defaults if necessary.
        Please log assumtpions, properly
        """
        pass

    def check(self, plasma_state: PlasmaState) -> None:
        """
        Test if the model is applicable to the PlasmaState. Might raise logged
        messages and errors.
        """
        pass

    # The following is required to jit a Model
    def _tree_flatten(self):
        children = ()
        aux_data = (self.model_key,)  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        (obj.model_key,) = aux_data

        return obj

    # One can use the flatten functions quite nicely for __eq__ methods.
    def __eq__(self, other):
        """
        Test the quality of two models.
        """
        if isinstance(other, Model):
            return self._tree_flatten() == other._tree_flatten()
        return NotImplemented


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

    def __init__(self, sample_points: int | None = None) -> None:
        super().__init__()

        #: The number of points for re-sampeling the model. If ``None``, no
        #: resampeling is none and every of the :py:class:`~.setup.Setup`'s
        #: :math:`k` s is evaluated when calling :py:meth:`~.evaluate`.
        #: However, as the computation might be expensive, you can reduce the
        #: number of relevant :math:`k` s by setting this attribute. After the
        #: evaluation, the resulting scatting signal is interpolated to the
        #: relevant :math:`k` s and then convolved with the instument function.
        self.sample_points = sample_points

    @abc.abstractmethod
    def evaluate_raw(
        self, plasma_state: PlasmaState, setup: Setup
    ) -> jnp.ndarray: ...

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

    @jax.jit
    def evaluate(self, plasma_state: PlasmaState, setup: Setup) -> jnp.ndarray:
        """
        If :py:attr:`~.sample_points` is not ``None``, generate a
        low-resulution :py:class`~.setup.Setup`. Calculate the
        instrument-function free scattering intensity with this or the given
        ``setup``, interpolate it, if needed and then convolve it with the
        instument function.
        """
        if self.sample_points is None:
            raw = self.evaluate_raw(plasma_state, setup)
        else:
            low_res_setup = Setup(
                setup.scattering_angle,
                setup.energy,
                self.sample_grid(setup),
                setup.instrument,
            )
            low_res = self.evaluate_raw(plasma_state, low_res_setup)
            raw = jnpu.interp(
                setup.measured_energy,
                low_res_setup.measured_energy,
                low_res,
                left=0,
                right=0,
            )
        return convolve_stucture_factor_with_instrument(raw, setup)

    # The following is required to jit a Model
    def _tree_flatten(self):
        children = ()
        aux_data = (self.model_key, self.sample_points)  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.model_key, obj.sample_points = aux_data
        return obj


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
    """
    A model that returns an empty with zeros in (units of seconds) for every
    energy probed.
    """

    allowed_keys = [*scattering_models, "ipd"]
    __name__ = "Neglect"

    @jax.jit
    def evaluate(self, plasma_state: PlasmaState, setup: Setup) -> jnp.ndarray:
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
    jaxrts.ion_feature.q_Gregori2004
        Calculation of the screening by (quasi) free electrons
    jaxrts.models.PaulingFormFactors
        The default model for the atomic form factors
    """

    allowed_keys = ["ionic scattering"]
    __name__ = "ArkhipovIonFeat"

    def prepare(self, plasma_state: PlasmaState) -> None:
        plasma_state.update_default_model("form-factors", PaulingFormFactors())

    def check(self, plasma_state: PlasmaState) -> None:
        if plasma_state.T_e != plasma_state.T_i:
            logger.warning(
                "'ArkhipovIonFeat' can only handle plasmas with T_e == T_i."
                + " The calculation will only consider the electron temperature."  # noqa: E501
                + " Choose another model if ion and electron temperature models should be treated individually."  # noqa: E501
            )
        if len(plasma_state) > 1:
            logger.critical(
                "'ArkhipovIonFeat' is only implemented for a one-component plasma"  # noqa: E501
            )

    @jax.jit
    def evaluate(self, plasma_state: PlasmaState, setup: Setup) -> jnp.ndarray:
        fi = plasma_state["form-factors"].evaluate(plasma_state, setup)
        population = electron_distribution_ionized_state(plasma_state.Z_core)[
            :, jnp.newaxis
        ]

        f = jnp.sum(fi * population)
        q = ion_feature.q_Gregori2004(
            setup.k[jnp.newaxis],
            plasma_state.atomic_masses,
            plasma_state.n_e,
            plasma_state.T_e,
            plasma_state.T_e,
            plasma_state.Z_free,
        )
        S_ii = static_structure_factors.S_ii_AD(
            setup.k,
            plasma_state.T_e,
            plasma_state.T_e,
            plasma_state.n_e,
            plasma_state.atomic_masses,
            plasma_state.Z_free,
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
    __name__ = "Gregori2003IonFeat"

    def prepare(self, plasma_state: PlasmaState) -> None:
        plasma_state.update_default_model("form-factors", PaulingFormFactors())

    def check(self, plasma_state: PlasmaState) -> None:
        if plasma_state.T_e != plasma_state.T_i:
            logger.warning(
                "'Gregori2003IonFeat' can only handle plasmas with T_e == T_i."
                + " The calculation will only consider the electron temperature."  # noqa: E501
                + " Choose another model if ion and electron temperature models should be treated individually."  # noqa: E501
            )
        if len(plasma_state) > 1:
            logger.critical(
                "'Gregori2003IonFeat' is only implemented for a one-component plasma"  # noqa: E501
            )

    @jax.jit
    def evaluate(self, plasma_state: PlasmaState, setup: Setup) -> jnp.ndarray:
        fi = plasma_state["form-factors"].evaluate(plasma_state, setup)
        population = electron_distribution_ionized_state(
            plasma_state.Z_core[0]
        )[:, jnp.newaxis]

        T_eff = static_structure_factors.T_cf_Greg(
            plasma_state.T_e, plasma_state.n_e
        )
        f = jnp.sum(fi * population)
        q = ion_feature.q_Gregori2004(
            setup.k[jnp.newaxis],
            plasma_state.atomic_masses,
            plasma_state.n_e,
            T_eff,
            T_eff,
            plasma_state.Z_free,
        )
        S_ii = ion_feature.S_ii_AD(
            setup.k,
            T_eff,
            T_eff,
            plasma_state.n_e,
            plasma_state.atomic_masses,
            plasma_state.Z_free,
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
    __name__ = "Gregori2006IonFeat"

    def prepare(self, plasma_state: PlasmaState) -> None:
        plasma_state.update_default_model("form-factors", PaulingFormFactors())
        plasma_state.update_default_model("Debye temperature", BohmStaver())

    def check(self, plasma_state: PlasmaState) -> None:
        if len(plasma_state) > 1:
            logger.critical(
                "'Gregori2006IonFeat' is only implemented for a one-component plasma"  # noqa: E501
            )

    @jax.jit
    def evaluate(self, plasma_state: PlasmaState, setup: Setup) -> jnp.ndarray:
        fi = plasma_state["form-factors"].evaluate(plasma_state, setup)
        population = electron_distribution_ionized_state(plasma_state.Z_core)[
            :, jnp.newaxis
        ]

        T_D = plasma_state["Debye temperature"].evaluate(plasma_state, setup)

        T_e_eff = static_structure_factors.T_cf_Greg(
            plasma_state.T_e, plasma_state.n_e
        )
        T_i_eff = static_structure_factors.T_i_eff_Greg(plasma_state.T_i, T_D)
        f = jnp.sum(fi * population)
        q = ion_feature.q_Gregori2004(
            setup.k[jnp.newaxis],
            plasma_state.atomic_masses,
            plasma_state.n_e,
            T_e_eff,
            T_i_eff,
            plasma_state.Z_free,
        )
        S_ii = ion_feature.S_ii_AD(
            setup.k,
            T_e_eff,
            T_i_eff,
            plasma_state.n_e,
            plasma_state.atomic_masses,
            plasma_state.Z_free,
        )
        w_R = jnp.abs(f + q.m_as(ureg.dimensionless)) ** 2 * S_ii
        res = w_R * setup.instrument(
            (setup.measured_energy - setup.energy) / ureg.hbar
        )
        return res


class LinearResponseHNCIonFeat(Model):
    """
    Model for the ion feature using a calculating all :math:`S_{ab}` in the
    Hypernetted Chain approximation.

    The screening density :math:`q` is calculated using a result from linear
    response, by

    .. math::

       q(k) = \\xi_{ee} V_{ei}(k)


    See :cite:`Wunsch.2011`, Eqn(5.22) and :cite:`Gericke.2010` Eqn(3).


    Requires 2 Potentials:

        - an 'ion-ion' (defaults to
          :py:class:`~DebyeHuckelPotential`).
        - an 'electron-ion'
          (defaults to :py:class:`~KlimontovichKraeftPotential`).

    Further requires a 'form-factors' model (defaults to
    :py:class:`~PaulingFormFactors`).

    See Also
    --------
    jaxtrs.ion_feature.free_electron_susceptilibily
        Function used to calculate :math:`\\xi{ee}`
    """

    allowed_keys = ["ionic scattering"]
    __name__ = "LinearResponseHNCIonFeat"

    def __init__(
        self,
        rmin: Quantity = 0.001 * ureg.a_0,
        rmax: Quantity = 100 * ureg.a_0,
        pot: int = 14,
    ) -> None:
        #: The minmal radius for evaluating the potentials.
        self.r_min: Quantity = rmin
        #: The maximal radius for evaluating the potentials.
        self.r_max: Quantity = rmax
        #: The exponent (``2 ** pot``), setting the number of points in ``r``
        #: or ``k`` to evaluate.
        self.pot: int = pot
        super().__init__()

    def prepare(self, plasma_state: PlasmaState) -> None:
        plasma_state.update_default_model("form-factors", PaulingFormFactors())
        plasma_state.update_default_model(
            "ion-ion Potential", hnc_potentials.DebyeHuckelPotential()
        )
        plasma_state.update_default_model(
            "electron-ion Potential",
            hnc_potentials.KlimontovichKraeftPotential(),
        )
        plasma_state["ion-ion Potential"].include_electrons = False
        plasma_state["electron-ion Potential"].include_electrons = True

    @property
    def r(self):
        return jnpu.linspace(self.r_min, self.r_max, 2**self.pot)

    @property
    def k(self):
        r = self.r
        dr = r[1] - r[0]
        dk = jnp.pi / (len(r) * dr)
        return jnp.pi / r[-1] + jnp.arange(len(r)) * dk

    @jax.jit
    def evaluate(self, plasma_state: PlasmaState, setup: Setup) -> jnp.ndarray:
        # Prepare the Potentials
        # ----------------------

        # Populate the potential with a full ion potential, for starters
        V_s_r = plasma_state["ion-ion Potential"].short_r(plasma_state, self.r)
        V_l_k = plasma_state["ion-ion Potential"].long_k(plasma_state, self.k)

        # Calculate g_ab in the HNC Approach
        # ----------------------------------
        T = plasma_state["ion-ion Potential"].T(plasma_state)
        n = plasma_state.n_i
        g, niter = hypernetted_chain.pair_distribution_function_HNC(
            V_s_r, V_l_k, self.r, T, n
        )
        logger.debug(
            f"{niter} Iterations of the HNC algorithm were required to reach the solution"  # noqa: 501
        )
        # Calculate S_ab by Fourier-transforming g_ab
        # ---------------------------------------------
        S_ab_HNC = hypernetted_chain.S_ii_HNC(self.k, g, n, self.r)

        # Interpolate this to the k given by the setup

        S_ab = hypernetted_chain.hnc_interp(setup.k, self.k, S_ab_HNC)

        # To Calculate the screening, use the S_ii and S_ei contributions
        # ---------------------------------------------------------------
        # Use the Debye screening length for the screening cloud.
        kappa = 1 / plasma_state.DH_screening_length
        xi = ion_feature.free_electron_susceptilibily_RPA(setup.k, kappa)
        Vei = plasma_state["electron-ion Potential"].full_k(
            plasma_state, to_array(setup.k)[jnp.newaxis]
        )
        q = xi * Vei[-1, :-1]

        # The W_R is calculated as a sum over all combinations of a_b
        ion_spec1, ion_spec2 = jnp.meshgrid(
            jnp.arange(plasma_state.nions),
            jnp.arange(plasma_state.nions),
        )
        # Get the formfactor from the plasma state
        fi = plasma_state["form-factors"].evaluate(plasma_state, setup)
        # Calculate the number-fraction per element
        x = plasma_state.n_i / jnpu.sum(plasma_state.n_i)

        # Add the contributions from all pairs
        w_R = 0

        def add_wrt(a, b):
            return (
                jnpu.sqrt(x[a] * x[b])
                * (fi[a] + q[a])
                * (fi[b] + q[b])
                * S_ab[a, b]
            ).m_as(ureg.dimensionless)

        def dont_add_wrt(a, b):
            return jnp.array([0.0])

        for a, b in zip(ion_spec1.flatten(), ion_spec2.flatten()):
            w_R += jax.lax.cond(a <= b, add_wrt, dont_add_wrt, a, b)
        # Scale the instrument function directly with w_R
        res = w_R * setup.instrument(
            (setup.measured_energy - setup.energy) / ureg.hbar
        )
        return res

    # The following is required to jit a Model
    def _tree_flatten(self):
        children = (self.r_min, self.r_max)
        aux_data = (
            self.model_key,
            self.pot,
        )  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.model_key, obj.pot = aux_data
        obj.r_min, obj.r_max = children

        return obj


class ThreePotentialHNCIonFeat(Model):
    """
    Model for the ion feature using a calculating all :math:`S_{ab}` in the
    Hypernetted Chain approximation. This is achieved by treating the electrons
    as an additional ion species, that is amended to the list of ions. See,
    e.g. :cite:`Schwarz.2007`.

    .. note::

        Compared to :py:class:`HNCIonFeat`, the internal Variables, `V_s` and
        `V_l` are now :math:`(n+1 \\times n+1 \\times m)` matrices, where
        :math:`n` is the number of ion species and :math:`m = 2^\\text{pot}`
        which pot being :py:attr:`~.pot`, the exponent for the number of
        scattering vectors to be evaluated in the HNC approach.

    Requires 3 Potentials:

        - an 'ion-ion Potential' The black entries in the picture below
          (defaults to :py:class:`~DebyeHuckelPotential`).
        - an 'electron-ion Potential' The orange entries in the picture below
          (defaults to :py:class:`~KlimontovichKraeftPotential`).
        - an 'electron-electron Potential' The red entries in the picutre below
          (defaults to :py:class:`~KelbgPotental`).

    .. image:: ../../_images/ThreePotentialHNC.svg
       :width: 600

    """

    allowed_keys = ["ionic scattering"]
    __name__ = "ThreePotentialHNC"

    def __init__(
        self,
        rmin: Quantity = 0.001 * ureg.a_0,
        rmax: Quantity = 100 * ureg.a_0,
        pot: int = 14,
    ) -> None:
        #: The minmal radius for evaluating the potentials.
        self.r_min: Quantity = rmin
        #: The maximal radius for evaluating the potentials.
        self.r_max: Quantity = rmax
        #: The exponent (``2 ** pot``), setting the number of points in ``r``
        #: or ``k`` to evaluate.
        self.pot: int = pot
        super().__init__()

    def prepare(self, plasma_state: PlasmaState) -> None:
        plasma_state.update_default_model("form-factors", PaulingFormFactors())
        plasma_state.update_default_model(
            "ion-ion Potential", hnc_potentials.DebyeHuckelPotential()
        )
        plasma_state.update_default_model(
            "electron-ion Potential",
            hnc_potentials.KlimontovichKraeftPotential(),
        )
        plasma_state.update_default_model(
            "electron-electron Potential",
            hnc_potentials.KelbgPotential(),
        )
        for key in [
            "ion-ion Potential",
            "electron-ion Potential",
            "electron-electron Potential",
        ]:
            plasma_state[key].include_electrons = True

    @property
    def r(self):
        return jnpu.linspace(self.r_min, self.r_max, 2**self.pot)

    @property
    def k(self):
        r = self.r
        dr = r[1] - r[0]
        dk = jnp.pi / (len(r) * dr)
        return jnp.pi / r[-1] + jnp.arange(len(r)) * dk

    @jax.jit
    def evaluate(self, plasma_state: PlasmaState, setup: Setup) -> jnp.ndarray:
        # Prepare the Potentials
        # ----------------------

        # Populate the potential with a full ion potential, for starters
        V_s_r = (
            plasma_state["ion-ion Potential"]
            .short_r(plasma_state, self.r)
            .m_as(ureg.electron_volt)
        )
        # Replace the last line and column with the electron-ion potential
        V_s_r = V_s_r.at[-1, :, :].set(
            plasma_state["electron-ion Potential"]
            .short_r(plasma_state, self.r)
            .m_as(ureg.electron_volt)[-1, :, :]
        )
        V_s_r = V_s_r.at[:, -1, :].set(
            plasma_state["electron-ion Potential"]
            .short_r(plasma_state, self.r)
            .m_as(ureg.electron_volt)[:, -1, :]
        )
        # Add the electron-electron Potential
        V_s_r = V_s_r.at[-1, -1, :].set(
            plasma_state["electron-electron Potential"]
            .short_r(plasma_state, self.r)
            .m_as(ureg.electron_volt)[-1, -1, :]
        )
        V_s_r *= ureg.electron_volt

        # Repeat this for the long-range part of the potential in k space.
        unit = ureg.electron_volt * ureg.angstrom**3
        V_l_k = (
            plasma_state["ion-ion Potential"]
            .long_k(plasma_state, self.k)
            .m_as(unit)
        )
        V_l_k = V_l_k.at[-1, :, :].set(
            plasma_state["electron-ion Potential"]
            .long_k(plasma_state, self.k)
            .m_as(unit)[-1, :, :]
        )
        V_l_k = V_l_k.at[:, -1, :].set(
            plasma_state["electron-ion Potential"]
            .long_k(plasma_state, self.k)
            .m_as(unit)[:, -1, :]
        )
        V_l_k = V_l_k.at[-1, -1, :].set(
            plasma_state["electron-electron Potential"]
            .long_k(plasma_state, self.k)
            .m_as(unit)[-1, -1, :]
        )
        V_l_k *= unit

        # Calculate g_ab in the HNC Approach
        # ----------------------------------
        T = plasma_state["ion-ion Potential"].T(plasma_state)
        n = to_array([*plasma_state.n_i, plasma_state.n_e])
        g, niter = hypernetted_chain.pair_distribution_function_HNC(
            V_s_r, V_l_k, self.r, T, n
        )
        logger.debug(
            f"{niter} Iterations of the HNC algorithm were required to reach the solution"  # noqa: 501
        )
        # Calculate S_ab by Fourier-transforming g_ab
        # ---------------------------------------------
        S_ab_HNC = hypernetted_chain.S_ii_HNC(self.k, g, n, self.r)

        # Interpolate this to the k given by the setup

        S_ab = hypernetted_chain.hnc_interp(setup.k, self.k, S_ab_HNC)

        # To Calculate the screening, use the S_ii and S_ei contributions
        # ---------------------------------------------------------------
        S_ii = jnpu.diagonal(S_ab, axis1=0, axis2=1)[:-1]
        S_ei = S_ab[:-1, -1]
        q = ion_feature.q_Glenzer2009(S_ei, S_ii, plasma_state.Z_free)

        # The W_R is calculated as a sum over all combinations of a_b
        ion_spec1, ion_spec2 = jnp.meshgrid(
            jnp.arange(plasma_state.nions),
            jnp.arange(plasma_state.nions),
        )
        # Get the formfactor from the plasma state
        fi = plasma_state["form-factors"].evaluate(plasma_state, setup)
        # Calculate the number-fraction per element
        x = plasma_state.n_i / jnpu.sum(plasma_state.n_i)

        # Add the contributions from all pairs
        w_R = 0

        def add_wrt(a, b):
            return (
                jnpu.sqrt(x[a] * x[b])
                * (fi[a] + q[a])
                * (fi[b] + q[b])
                * S_ab[a, b]
            ).m_as(ureg.dimensionless)

        def dont_add_wrt(a, b):
            return jnp.array([0.0])

        for a, b in zip(ion_spec1.flatten(), ion_spec2.flatten()):
            w_R += jax.lax.cond(a <= b, add_wrt, dont_add_wrt, a, b)
        # Scale the instrument function directly with w_R
        res = w_R * setup.instrument(
            (setup.measured_energy - setup.energy) / ureg.hbar
        )
        return res

    # The following is required to jit a Model
    def _tree_flatten(self):
        children = (self.r_min, self.r_max)
        aux_data = (
            self.model_key,
            self.pot,
        )  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.model_key, obj.pot = aux_data
        obj.r_min, obj.r_max = children

        return obj


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
    __name__ = "QCSalpeterApproximation"

    def check(self, plasma_state: PlasmaState) -> None:
        if len(plasma_state) > 1:
            logger.critical(
                "'QCSalpeterApproximation' is only implemented for a one-component plasma"  # noqa: E501
            )

    @jax.jit
    def evaluate_raw(
        self, plasma_state: PlasmaState, setup: Setup
    ) -> jnp.ndarray:
        See_0 = free_free.S0_ee_Salpeter(
            setup.k,
            plasma_state.T_e,
            plasma_state.n_e,
            setup.measured_energy - setup.energy,
        )

        return See_0 * plasma_state.Z_free


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
    __name__ = "RPA_NoDamping"

    def prepare(self, plasma_state: PlasmaState) -> None:
        plasma_state.update_default_model(
            "chemical potential", GregoriChemPotential()
        )

    def check(self, plasma_state: PlasmaState) -> None:
        if len(plasma_state) > 1:
            logger.critical(
                "'RPA_NoDamping' is only implemented for a one-component plasma"  # noqa: E501
            )

    @jax.jit
    def evaluate_raw(
        self, plasma_state: PlasmaState, setup: Setup
    ) -> jnp.ndarray:
        mu = plasma_state["chemical potential"].evaluate(plasma_state, setup)
        See_0 = free_free.S0_ee_RPA_no_damping(
            setup.k,
            plasma_state.T_e,
            plasma_state.n_e,
            setup.measured_energy - setup.energy,
            mu,
        )

        return See_0 * plasma_state.Z_free


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

    __name__ = "BornMermin"
    allowed_keys = ["free-free scattering"]

    def prepare(self, plasma_state: PlasmaState) -> None:
        plasma_state.update_default_model(
            "chemical potential", GregoriChemPotential()
        )

    def check(self, plasma_state: PlasmaState) -> None:
        if len(plasma_state) > 1:
            logger.critical(
                "'BornMermin' is only implemented for a one-component plasma"  # noqa: E501
            )

    @jax.jit
    def evaluate_raw(
        self, plasma_state: PlasmaState, setup: Setup
    ) -> jnp.ndarray:
        mu = plasma_state["chemical potential"].evaluate(plasma_state, setup)
        See_0 = free_free.S0_ee_BMA(
            setup.k,
            plasma_state.T_e,
            mu,
            plasma_state.atomic_masses,
            plasma_state.n_e,
            plasma_state.Z_free,
            setup.measured_energy - setup.energy,
        )
        return See_0 * plasma_state.Z_free


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
    __name__ = "BornMermin_ChapmanInterp"

    def __init__(self, no_of_freq: int = 20) -> None:
        super().__init__()
        self.no_of_freq: int = no_of_freq

    def prepare(self, plasma_state: PlasmaState) -> None:
        plasma_state.update_default_model(
            "chemical potential", GregoriChemPotential()
        )

    def check(self, plasma_state: PlasmaState) -> None:
        if len(plasma_state) > 1:
            logger.critical(
                "'BornMermin_ChapmanInterp' is only implemented for a one-component plasma"  # noqa: E501
            )

    @jax.jit
    def evaluate_raw(
        self, plasma_state: PlasmaState, setup: Setup
    ) -> jnp.ndarray:
        mu = plasma_state["chemical potential"].evaluate(plasma_state, setup)
        See_0 = free_free.S0_ee_BMA_chapman_interp(
            setup.k,
            plasma_state.T_e,
            mu,
            plasma_state.atomic_masses,
            plasma_state.n_e,
            plasma_state.Z_free,
            setup.measured_energy - setup.energy,
            self.no_of_freq,
        )
        return See_0 * plasma_state.Z_free

    def _tree_flatten(self):
        children = ()
        aux_data = (
            self.model_key,
            self.sample_points,
            self.no_of_freq,
        )  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.model_key, obj.sample_points, obj.no_of_freq = aux_data

        return obj


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
    __name__ = "SchumacherImpulse"

    def prepare(self, plasma_state: PlasmaState) -> None:
        plasma_state.update_default_model("form-factors", PaulingFormFactors())

    def check(self, plasma_state: PlasmaState) -> None:
        if len(plasma_state) > 1:
            logger.critical(
                "'SchumacherImpulse' is only implemented for a one-component plasma"  # noqa: E501
            )

    @jax.jit
    def evaluate_raw(
        self, plasma_state: PlasmaState, setup: Setup
    ) -> jnp.ndarray:
        k = setup.k
        omega_0 = setup.energy / ureg.hbar
        omega = omega_0 - setup.measured_energy / ureg.hbar
        Z_c = plasma_state.Z_core[0]
        E_b = plasma_state.ions[0].binding_energies

        Zeff = form_factors.pauling_effective_charge(plasma_state.ions[0].Z)
        population = electron_distribution_ionized_state(Z_c)
        # Gregori.2004, Eqn 20
        fi = plasma_state["form-factors"].evaluate(plasma_state, setup)
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

    __name__ = "DetailedBalance"
    allowed_keys = ["free-bound scattering"]

    @jax.jit
    def evaluate_raw(
        self, plasma_state: PlasmaState, setup: Setup
    ) -> jnp.ndarray:
        energy_shift = setup.measured_energy - setup.energy
        mirrored_setup = Setup(
            setup.scattering_angle,
            setup.energy,
            setup.energy - energy_shift,
            setup.instrument,
        )
        db_factor = jnpu.exp(-energy_shift / (plasma_state.T_e * ureg.k_B))
        free_bound = plasma_state["bound-free scattering"].evaluate_raw(
            plasma_state, mirrored_setup
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
    __name__ = "PaulingFormFactors"

    @jax.jit
    def evaluate(self, plasma_state: PlasmaState, setup: Setup) -> jnp.ndarray:
        Zstar = form_factors.pauling_effective_charge(plasma_state.Z_A)
        ff = form_factors.pauling_all_ff(setup.k, Zstar)
        # population = plasma_state.ions[0].electron_distribution
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

    __name__ = "GregoriChemPotential"
    allowed_keys = ["chemical potential"]

    @jax.jit
    def evaluate(self, plasma_state: PlasmaState, setup: Setup) -> jnp.ndarray:
        return plasma_physics.chem_pot_interpolation(
            plasma_state.T_e, plasma_state.n_e
        )


# IPD Models
# ==========


class ConstantIPD(Model):
    """
    A model that returns an empty with zeros in (units of seconds) for every
    energy probed.
    """

    allowed_keys = ["ipd"]
    __name__ = "ConstantIPD"

    def __init__(self, value):
        self.value = value
        super().__init__()

    @jax.jit
    def evaluate(self, plasma_state: PlasmaState, setup: Setup) -> jnp.ndarray:
        return self.value

    # The following is required to jit a Model
    def _tree_flatten(self):
        children = (self.value,)
        aux_data = (self.model_key,)  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        (obj.model_key,) = aux_data
        (obj.value,) = children

        return obj


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
    __name__ = "BohmStaver"

    @jax.jit
    def evaluate(self, plasma_state: PlasmaState, setup: Setup) -> jnp.ndarray:
        return static_structure_factors.T_Debye_Bohm_Staver(
            plasma_state.T_e,
            plasma_state.n_e,
            plasma_state.atomic_masses,
            plasma_state.Z_free,
        )


_all_models = [
    ArkhipovIonFeat,
    BohmStaver,
    BornMermin,
    BornMermin_ChapmanInterp,
    ConstantIPD,
    DetailedBalance,
    Gregori2003IonFeat,
    Gregori2006IonFeat,
    GregoriChemPotential,
    LinearResponseHNCIonFeat,
    Model,
    Neglect,
    PaulingFormFactors,
    QCSalpeterApproximation,
    RPA_NoDamping,
    ScatteringModel,
    SchumacherImpulse,
    ThreePotentialHNCIonFeat,
]

for model in _all_models:
    jax.tree_util.register_pytree_node(
        model,
        model._tree_flatten,
        model._tree_unflatten,
    )
