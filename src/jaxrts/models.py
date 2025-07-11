"""
This submodule contains high-level wrappers for the different Models
implemented.
"""

import abc
import logging
from copy import deepcopy
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jpu import numpy as jnpu

from .analysis import ITCF_fsum

from . import (
    bound_free,
    ee_localfieldcorrections,
    form_factors,
    free_bound,
    free_free,
    hnc_potentials,
    hypernetted_chain,
    ion_feature,
    ipd,
    plasma_physics,
    static_structure_factors,
)
from .elements import MixElement, electron_distribution_ionized_state
from .plasma_physics import noninteracting_susceptibility_from_eps_RPA
from .setup import (
    Setup,
    convolve_stucture_factor_with_instrument,
    get_probe_setup,
)
from .units import Quantity, to_array, ureg

if TYPE_CHECKING:
    from .plasmastate import PlasmaState

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
        self, plasma_state: "PlasmaState", setup: Setup
    ) -> jnp.ndarray: ...

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        """
        Modify the plasma_state in place.

        As different prerequisites exist for different models, make sure to
        test that all relevant information is given in the PlasmaState, amend
        defaults if necessary.
        Please log assumptions, properly
        """

    def check(self, plasma_state: "PlasmaState") -> None:
        """
        Test if the model is applicable to the PlasmaState. Might raise logged
        messages and errors.
        """

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
       :py:class:`~.Model` s used to describe ionic scattering are currently
       not instances of :py:class:`~.ScatteringModel` as the convolution with a
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
        #: relevant :math:`k` s and then convolved with the instrument
        #: function.
        self.sample_points = sample_points

    @abc.abstractmethod
    def evaluate_raw(
        self, plasma_state: "PlasmaState", setup: Setup
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
    def evaluate(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """
        If :py:attr:`~.sample_points` is not ``None``, generate a
        low-resulution :py:class`~.setup.Setup`. Calculate the
        instrument-function free scattering intensity with this or the given
        ``setup``, interpolate it, if needed and then convolve it with the
        instrument function.
        """
        if self.sample_points is None:
            raw = self.evaluate_raw(plasma_state, setup, *args, **kwargs)
        else:
            low_res_setup = Setup(
                setup.scattering_angle,
                setup.energy,
                self.sample_grid(setup),
                setup.instrument,
            )
            low_res = self.evaluate_raw(
                plasma_state, low_res_setup, *args, **kwargs
            )
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
    def evaluate(
        self, plasma_state: "PlasmaState", setup: Setup
    ) -> jnp.ndarray:
        if self.model_key in scattering_models:
            return jnp.zeros_like(setup.measured_energy) * (1 * ureg.second)
        elif self.model_key == "ipd":
            return jnp.zeros_like(plasma_state.n_i) * (1 * ureg.electron_volt)

    @jax.jit
    def evaluate_raw(
        self, plasma_state: "PlasmaState", setup: Setup
    ) -> jnp.ndarray:
        if self.model_key in scattering_models:
            return jnp.zeros_like(setup.measured_energy) * (1 * ureg.second)


# ion-feature
# -----------
class IonFeatModel(Model):
    """
    These set of models describe the scattering by electrons tightly bound to
    the ions, causing quasi-elastic scattering. `IonFeatModels` have to define
    a method :py:meth:`~.S_ii` which returns the static ion-ion structure
    factor.
    """

    #: A list of keywords where this model is adequate for
    allowed_keys: list[str] = ["ionic scattering"]

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        plasma_state.update_default_model("form-factors", PaulingFormFactors())
        plasma_state.update_default_model("screening", Gregori2004Screening())

    @abc.abstractmethod
    def S_ii(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
    ) -> jnp.ndarray: ...

    @jax.jit
    def Rayleigh_weight(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
    ) -> jnp.ndarray:
        """
        This is the result from Wünsch, to calculate the Rayleigh weight for
        multiple species.

        .. math::

           W_R = \\sum_{a, b} \\sqrt{x_a x_b}
           (f_a + q_a) (f_b + q_b) S_{ab}


        Where

        * The sum runs over all pairs of atoms :math:`a` and :math`b`.

        * :math:`x` is the number fraction of the ion species :math:`a`, and
          :math:`b`, respectively.

        * :math:`f_x` are the form factors, based on the `plasma_state`'s
          ``'form-factors'`` model.

        * :math:`q_x` describe the screening by free electrons, which can be
          set via the `plasma_state`'s ``'screening'`` model.

        * :math:`S_{ab}` is the static ion-ion structure factor, which has to
          be provided by the :py:meth:`~.S_ii` method.
        """
        S_ab = self.S_ii(plasma_state, setup)
        # Get the formfactor from the plasma state
        fi = plasma_state["form-factors"].evaluate(plasma_state, setup)
        population = electron_distribution_ionized_state(plasma_state.Z_core)
        f = jnp.sum(fi * population, axis=0)
        # Calculate the number-fraction per element
        x = plasma_state.number_fraction

        # Get the screening from the plasma state
        q = plasma_state.evaluate("screening", setup)

        # Add the contributions from all pairs
        w_R = 0

        def add_wrt(a, b):
            return (
                jnpu.sqrt(x[a] * x[b])
                * (f[a] + q[a])
                * (f[b] + q[b])
                * S_ab[a, b]
            ).m_as(ureg.dimensionless)

        # The W_R is calculated as a sum over all combinations of a_b
        ion_spec1, ion_spec2 = jnp.meshgrid(
            jnp.arange(plasma_state.nions),
            jnp.arange(plasma_state.nions),
        )
        for a, b in zip(ion_spec1.flatten(), ion_spec2.flatten()):
            w_R += add_wrt(a, b)
        return w_R

    @jax.jit
    def evaluate(
        self, plasma_state: "PlasmaState", setup: Setup
    ) -> jnp.ndarray:
        w_R = self.Rayleigh_weight(plasma_state, setup)
        res = w_R * setup.instrument(
            (setup.measured_energy - setup.energy) / ureg.hbar
        )
        return res / plasma_state.mean_Z_A


class ArkhipovIonFeat(IonFeatModel):
    """
    Model for the ion feature of the scattering, presented in
    :cite:`Arkhipov.1998` and :cite:`Arkhipov.2000`.

    The structure factors are obtained by using an effective potential
    (pseudopotential) model of the particle interaction of semiclassical
    two-component plasmas with a single temperature :math:`T`. We use the
    electron temperature ``state.T_e`` of the PlasmaState modelled. The authors
    take into account both quantum and collective effects.

    Requires a 'form-factors' model (defaults to
    :py:class:`~PaulingFormFactors`) and a 'screening' model (defaults to
    :py:class:`Gregori2004Screening`).

    See Also
    --------

    jaxrts.static_structure_factors.S_ii_AD
        Calculation of the static ion ion structure factor given by
        :cite:`Arkhipov.1998`.
    jaxrts.models.PaulingFormFactors
        The default model for the atomic form factors
    """

    __name__ = "ArkhipovIonFeat"

    def check(self, plasma_state: "PlasmaState") -> None:
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
    def S_ii(self, plasma_state: "PlasmaState", setup: Setup) -> jnp.ndarray:
        S_ii = static_structure_factors.S_ii_AD(
            setup.k,
            plasma_state.T_e,
            plasma_state.T_e,
            plasma_state.n_e,
            plasma_state.atomic_masses,
            plasma_state.Z_free,
        )
        # Add a dimension, so that the shape is (1x1)
        return S_ii[:, jnp.newaxis]


class Gregori2003IonFeat(IonFeatModel):
    """
    Model for the ion feature of the scattering, presented in
    :cite:`Gregori.2003`.

    This model is identical to :py:class:`~ArkhipovIonFeat`, but uses an
    effective temperature ~:py:func:`jaxtrs.static_structure_factors.T_cf_Greg`
    rather than the electron Temperature throughout the calculation.
    """

    __name__ = "Gregori2003IonFeat"

    def check(self, plasma_state: "PlasmaState") -> None:
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
    def S_ii(self, plasma_state: "PlasmaState", setup: Setup) -> jnp.ndarray:
        T_eff = static_structure_factors.T_cf_Greg(
            plasma_state.T_e, plasma_state.n_e
        )
        S_ii = ion_feature.S_ii_AD(
            setup.k,
            T_eff,
            T_eff,
            plasma_state.n_e,
            plasma_state.atomic_masses,
            plasma_state.Z_free,
        )
        # Add a dimension, so that the shape is (1x1)
        return S_ii[:, jnp.newaxis]


class Gregori2006IonFeat(IonFeatModel):
    """
    Model for the ion feature of the scattering, presented in
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

    __name__ = "Gregori2006IonFeat"

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        super().prepare(plasma_state, key)
        plasma_state.update_default_model("Debye temperature", BohmStaver())

    def check(self, plasma_state: "PlasmaState") -> None:
        if len(plasma_state) > 1:
            logger.critical(
                "'Gregori2006IonFeat' is only implemented for a one-component plasma"  # noqa: E501
            )

    @jax.jit
    def S_ii(self, plasma_state: "PlasmaState", setup: Setup) -> jnp.ndarray:
        T_D = plasma_state["Debye temperature"].evaluate(plasma_state, setup)

        T_e_eff = static_structure_factors.T_cf_Greg(
            plasma_state.T_e, plasma_state.n_e
        )
        T_i_eff = static_structure_factors.T_i_eff_Greg(plasma_state.T_i, T_D)
        S_ii = ion_feature.S_ii_AD(
            setup.k,
            T_e_eff,
            T_i_eff,
            plasma_state.n_e,
            plasma_state.atomic_masses,
            plasma_state.Z_free,
        )
        # Add a dimension, so that the shape is (1x1)
        return S_ii[:, jnp.newaxis]


class FixedSii(IonFeatModel):
    """
    Model for the ion feature with a fixed value for :math:`S_{ii}`.
    """

    __name__ = "FixedSii"

    def __init__(self, Sii) -> None:
        self._S_ii = Sii
        super().__init__()

    @jax.jit
    def S_ii(self, plasma_state: "PlasmaState", setup: Setup) -> jnp.ndarray:
        return self._S_ii

    # The following is required to jit a Model
    def _tree_flatten(self):
        children = (self._S_ii,)
        aux_data = (self.model_key,)  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        (obj.model_key,) = aux_data
        (obj._S_ii,) = children

        return obj


class OnePotentialHNCIonFeat(IonFeatModel):
    """
    Model for the ion feature using a calculating all :math:`S_{ab}` in the
    Hypernetted Chain approximation.

    In contrast to :py:class:`~.ThreePotentialHNCIonFeat`, this models
    calculates only the ion-ion Structure factors, and is neglecting the
    electron-contibutions. Hence, screening is not included, automatically, but
    has to be provided as an additional `screening`.


    Requires an 'ion-ion' (defaults to :py:class:`~DebyeHueckelPotential`) and
    a `screening` model (default:
    :py::class:`~.LinearResponseScreeningGericke2010`. Further requires a
    'form-factors' model (defaults to :py:class:`~PaulingFormFactors`).
    """

    __name__ = "OnePotentialHNCIonFeat"

    def __init__(
        self,
        rmin: Quantity = 0.001 * ureg.a_0,
        rmax: Quantity = 100 * ureg.a_0,
        pot: int = 14,
        mix: float = 0.0,
    ) -> None:
        #: The minimal radius for evaluating the potentials.
        self.r_min: Quantity = rmin
        #: The maximal radius for evaluating the potentials.
        self.r_max: Quantity = rmax
        #: The exponent (``2 ** pot``), setting the number of points in ``r``
        #: or ``k`` to evaluate.
        self.pot: int = pot
        #: Value in [0, 1); desribes how much of the last iterations' nodal
        #: correction term should be added to the newly obtained `N_ab`. A
        #: value of zero corresponds to no parts of the old solution. Can be
        #: increased when HNC becomes numerically unstable due to high coupling
        #: strengths.
        self.mix: float = mix
        super().__init__()

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        super().prepare(plasma_state, key)
        plasma_state.update_default_model(
            "ion-ion Potential", hnc_potentials.DebyeHueckelPotential()
        )
        plasma_state["ion-ion Potential"].include_electrons = "off"

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
    def S_ii(self, plasma_state: "PlasmaState", setup: Setup) -> jnp.ndarray:
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

        return S_ab

    # The following is required to jit a Model
    def _tree_flatten(self):
        children = (self.r_min, self.r_max, self.mix)
        aux_data = (
            self.model_key,
            self.pot,
        )  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.model_key, obj.pot = aux_data
        obj.r_min, obj.r_max, obj.mix = children

        return obj


class ThreePotentialHNCIonFeat(IonFeatModel):
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
          (defaults to :py:class:`~DebyeHueckelPotential`).
        - an 'electron-ion Potential' The orange entries in the picture below
          (defaults to :py:class:`~KlimontovichKraeftPotential`).
        - an 'electron-electron Potential' The red entries in the picutre below
          (defaults to :py:class:`~KelbgPotental`).

    .. image:: ../images/ThreePotentialHNC.svg
       :width: 600

    See Also
    --------

    jaxrts.ion_feature.q_Glenzer2009
        Calculation of the screening, when both S_ei and S_ii are known. As
        this should be accurate, we don't require a 'screening' model with this
        'ionic scttering' model.

    """

    __name__ = "ThreePotentialHNC"

    def __init__(
        self,
        rmin: Quantity = 0.001 * ureg.a_0,
        rmax: Quantity = 100 * ureg.a_0,
        pot: int = 14,
        mix: float = 0.0,
    ) -> None:
        #: The minimal radius for evaluating the potentials.
        self.r_min: Quantity = rmin
        #: The maximal radius for evaluating the potentials.
        self.r_max: Quantity = rmax
        #: The exponent (``2 ** pot``), setting the number of points in ``r``
        #: or ``k`` to evaluate.
        self.pot: int = pot
        #: Value in [0, 1); desribes how much of the last iterations' nodal
        #: correction term should be added to the newly obtained `N_ab`. A
        #: value of zero corresponds to no parts of the old solution. Can be
        #: increased when HNC becomes numerically unstable due to high coupling
        #: strengths.
        self.mix: float = mix
        super().__init__()

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        # Overwrite the old prepare function, here, because we don't need a
        # screening model
        plasma_state.update_default_model("form-factors", PaulingFormFactors())
        plasma_state.update_default_model(
            "ion-ion Potential", hnc_potentials.DebyeHueckelPotential()
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
            plasma_state[key].include_electrons = "SpinAveraged"

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
    def _S_ii_with_electrons(
        self, plasma_state: "PlasmaState", setup: Setup
    ) -> jnp.ndarray:
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
        return S_ab

    def S_ii(self, plasma_state, setup):
        S_ab = self._S_ii_with_electrons(plasma_state, setup)
        return S_ab[:-1, :-1]

    def Rayleigh_weight(self, plasma_state, setup):
        """
        Here, we have to calculate the Rayleigh weight different than the
        default, because we get the screening from the calculated S_ei, rather
        than any model.
        """
        # To Calculate the screening, use the S_ii and S_ei contributions
        # ---------------------------------------------------------------
        S_ab = self._S_ii_with_electrons(plasma_state, setup)
        S_ii = jnpu.diagonal(S_ab, axis1=0, axis2=1)[:-1]
        S_ei = S_ab[:-1, -1]

        # Add a dimension here, to be compatible with the the typical output of
        # plasma_state["screening].evaluate...
        q = ion_feature.q_Glenzer2009(S_ei, S_ii, plasma_state.Z_free)[
            :, jnp.newaxis
        ]

        # The W_R is calculated as a sum over all combinations of a_b
        ion_spec1, ion_spec2 = jnp.meshgrid(
            jnp.arange(plasma_state.nions),
            jnp.arange(plasma_state.nions),
        )
        # Get the formfactor from the plasma state
        fi = plasma_state["form-factors"].evaluate(plasma_state, setup)
        population = electron_distribution_ionized_state(plasma_state.Z_core)
        # Sum up all fi to the full f
        f = jnp.sum(fi * population, axis=0)
        # Calculate the number-fraction per element
        x = plasma_state.number_fraction

        # Add the contributions from all pairs
        w_R = 0

        def add_wrt(a, b):
            return (
                jnpu.sqrt(x[a] * x[b])
                * (f[a] + q[a])
                * (f[b] + q[b])
                * S_ab[a, b]
            ).m_as(ureg.dimensionless)

        for a, b in zip(ion_spec1.flatten(), ion_spec2.flatten()):
            w_R += add_wrt(a, b)
        # Scale the instrument function directly with w_R
        return w_R

    # The following is required to jit a Model
    def _tree_flatten(self):
        children = (self.r_min, self.r_max, self.mix)
        aux_data = (
            self.model_key,
            self.pot,
        )  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.model_key, obj.pot = aux_data
        obj.r_min, obj.r_max, obj.mix = children

        return obj


class PeakCollection(IonFeatModel):
    """
    A model for approximating :py:meth:`~.PeakCollection.S_ii` as a sum of
    peaks. Can be used to model, e.g., ideal powder diffraction, where elastic
    signal would only be expected at certain values of :math:`k`.
    """

    __name__ = "PeakCollection"

    def __init__(
        self,
        k_pos: Quantity,
        intensity: jnp.ndarray,
        peak_function: callable,
    ):
        """
        Parameters
        ----------
        k_pos: Quantity
            The position of the peaks in :math:`k` space.
        intensity: jnp.ndarray
            The intensity of the peaks. Should have the same length as `k_pos`.
        peak_function: callable
            The shape of each peak. This function is expected to be normed over
            the integral over all :math:`k`, should be centered around 0, and
            expects to require exactly 1 argument, the position in k space.
            The `peak_function` should return a shape that is compatible with
            :py:attr:jaxrts.PlasmaState.nions`. I.e., the output should be a
            `(nxn)` matrix where `n` is the number of ion species considered.
        """
        self.k_pos = to_array(k_pos)
        self.intensity = to_array(intensity)
        self.peak_function = jax.tree_util.Partial(peak_function)
        super().__init__()

    @jax.jit
    def S_ii(self, plasma_state: "PlasmaState", setup: Setup) -> jnp.ndarray:
        # Create an array with the correct dimensions
        out = 0 * self.peak_function(1 / ureg.angstrom)
        for center, factor in zip(self.k_pos, self.intensity):
            out += self.peak_function(setup.k - center) * factor
        return out

    # The following is required to jit a Model
    def _tree_flatten(self):
        children = (self.k_pos, self.intensity, self.peak_function)
        aux_data = (self.model_key,)  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        (obj.model_key,) = aux_data
        (obj.k_pos, obj.intensity, obj.peak_function) = children

        return obj


class DebyeWallerSolid(IonFeatModel):
    """
    This model approximates the static structure factor (SSF) of a solid at
    finite temperature as suggested by :cite:`Gregori.2006`. It assumes the SSF
    to consist of a crystall peak feature :math:`b(k)`, which are damped by the
    debye Waller factor :math:`2W`, and an increasing, diffuse scattering part,
    which is decribed py the static structure factor :math:`S_\\text{plasma}`
    of the plasma (or liquid)-seeming contributions.

    .. math::

       S_{ii}(k) = S_\\text{plasma}(k)
       \\left[(1 - \\exp(-2W)) + \\exp(-2W)b(k)\\right]

    .. note::

       For the temperatures this model might be used,
       :math:`S_\\text{plasma}(k) = 1` has been used in the literature, before
       to extract Debye temperatures from DFT-MD Simulations. See
       :cite:`Schuster.2020`.

    This function uses the :py:class:`jaxrts.plasmastate.PlasmaState` ``'Debye
    temperature'`` model to calculate the Debye Waller factor.
    Hence, it requires a 'Debye temperature' model (defaults to
    :py:class:`~BohmStaver`).

    See Also
    --------
    jaxrts.static_structure_factors.debyeWallerFactor
       Function used to calculate the Debye Waller Factor. Note that the
       implementation used based on :cite:`Murphy.2008` does differ from the
       formula in :cite:`Gregori.2006`.
    """

    __name__ = "DebyeWallerSolid"

    def __init__(
        self,
        S_plasma: IonFeatModel,
        b: IonFeatModel,
    ):
        """
        Parameters
        ----------
        S_plasma: IonFeatModel
            A model for the static structure factor of the plasma-like
            scattering contribution.
        b: IonFeatModel
            A model for the lattice bragg-peak structure. Likely a
            :py:class:`~.PeakCollection`.
        """
        self.S_plasma = S_plasma
        self.b = b
        super().__init__()

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        super().prepare(plasma_state, key)
        plasma_state.update_default_model("Debye temperature", BohmStaver())

    @jax.jit
    def S_ii(self, plasma_state: "PlasmaState", setup: Setup) -> jnp.ndarray:
        debyeTemperature = plasma_state.evaluate("Debye temperature", setup)
        DWFactor = static_structure_factors.debyeWallerFactor(
            setup.k,
            jnpu.sum(
                plasma_state.atomic_masses * plasma_state.number_fraction
            ).m_as(ureg.atomic_mass_constant)
            * (1 * ureg.gram / ureg.mol),
            debyeTemperature,
            jnpu.sum(plasma_state.T_i * plasma_state.number_fraction),
        )
        S_plasma = self.S_plasma.S_ii(plasma_state, setup)
        b = self.b.S_ii(plasma_state, setup)

        return S_plasma * ((1 - DWFactor) + (DWFactor * b))

    # The following is required to jit a Model
    def _tree_flatten(self):
        children = (self.S_plasma, self.b)
        aux_data = (self.model_key,)  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        (obj.model_key,) = aux_data
        (obj.S_plasma, obj.b) = children

        return obj


# Free-free models
# ----------------
#
# These models also give a susceptibility method, which might be used by
# screening models, later.


class FreeFreeModel(ScatteringModel):
    """
    A class of models suitable for ``'free-free scattering'``. These models
    have to define a :py:meth:`~.susceptibility` method, which can be used by a
    ``'screening'`` model, later.
    """

    #: A list of keywords where this model is adequate for
    allowed_keys: list[str] = ["free-free scattering"]

    @abc.abstractmethod
    def susceptibility(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        E: Quantity,
        *args,
        **kwargs,
    ) -> jnp.ndarray: ...


class QCSalpeterApproximation(FreeFreeModel):
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

    This model does not provide a straight-forward approach to include local
    field corrections. We have included it, for now, by assuming the behavior
    would be the same as it is for the RPA.

    See Also
    --------
    jaxtrs.free_free.S0_ee_Salpeter(
        Function used to calculate the dynamic free electron-electron structure
        factor.
    """

    __name__ = "QCSalpeterApproximation"

    @jax.jit
    def evaluate_raw(
        self, plasma_state: "PlasmaState", setup: Setup, *args, **kwargs
    ) -> jnp.ndarray:
        k = setup.dispersion_corrected_k(plasma_state.n_e)
        See_0 = free_free.S0_ee_Salpeter(
            k,
            plasma_state.T_e,
            plasma_state.n_e,
            setup.measured_energy - setup.energy,
            plasma_state["ee-lfc"].evaluate(plasma_state, setup),
        )

        ff = See_0 * jnp.sum(
            plasma_state.Z_free * plasma_state.number_fraction
        )
        # Return 0 scattering if there are no free electrons
        return (
            jax.lax.cond(
                jnp.sum(plasma_state.Z_free) == 0,
                lambda: jnp.zeros_like(setup.measured_energy) * ureg.second,
                lambda: ff,
            )
            / plasma_state.mean_Z_A
        )

    @jax.jit
    def susceptibility(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        E: Quantity,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        k = setup.k
        eps = free_free.dielectric_function_salpeter(
            k,
            plasma_state.T_e,
            plasma_state.n_e,
            E,
        )
        xi0 = noninteracting_susceptibility_from_eps_RPA(eps, k)
        lfc = plasma_state["ee-lfc"].evaluate(plasma_state, setup)
        V = plasma_physics.coulomb_potential_fourier(-1, -1, k)
        xi = ee_localfieldcorrections.xi_lfc_corrected(xi0, V, lfc)
        return xi


class RPA_NoDamping(FreeFreeModel):
    """
    Model for elastic free-free scattering based on the Random Phase
    Approximation

    Calculates the dielectic function in RPA and obtain a Structure factor via
    the fluctuation dissipation theorem. Based on lecture notes from M. Bonitz.

    Requires a 'chemical potential' model (defaults to
    :py:class:`~IchimaruChemPotential`).

    See Also
    --------
    jaxtrs.free_free.S0_ee_RPA_no_damping
        Function used to calculate the dynamic free-free electron structure
        factor.
    """

    __name__ = "RPA_NoDamping"

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        plasma_state.update_default_model(
            "chemical potential", IchimaruChemPotential()
        )

    @jax.jit
    def evaluate_raw(
        self, plasma_state: "PlasmaState", setup: Setup, *args, **kwargs
    ) -> jnp.ndarray:
        mu = plasma_state["chemical potential"].evaluate(plasma_state, setup)
        k = setup.dispersion_corrected_k(plasma_state.n_e)
        See_0 = free_free.S0_ee_RPA_no_damping(
            k,
            plasma_state.T_e,
            plasma_state.n_e,
            setup.measured_energy - setup.energy,
            mu,
            plasma_state["ee-lfc"].evaluate_fullk(plasma_state, setup),
        )

        ff = See_0 * jnp.sum(
            plasma_state.Z_free * plasma_state.number_fraction
        )
        # Return 0 scattering if there are no free electrons
        return (
            jax.lax.cond(
                jnp.sum(plasma_state.Z_free) == 0,
                lambda: jnp.zeros_like(setup.measured_energy) * ureg.second,
                lambda: ff,
            )
            / plasma_state.mean_Z_A
        )

    @jax.jit
    def susceptibility(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        E: Quantity,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        mu = plasma_state["chemical potential"].evaluate(plasma_state, setup)
        k = setup.k

        eps = free_free.dielectric_function_RPA_no_damping(
            k,
            E,
            mu,
            plasma_state.T_e,
        )
        xi0 = noninteracting_susceptibility_from_eps_RPA(eps, k)
        lfc = plasma_state["ee-lfc"].evaluate(plasma_state, setup)
        V = plasma_physics.coulomb_potential_fourier(-1, -1, k)
        xi = ee_localfieldcorrections.xi_lfc_corrected(xi0, V, lfc)
        return xi


class RPA_DandreaFit(FreeFreeModel):
    """
    Model for elastic free-free scattering based fitting to the Random Phase
    Approximation, as presented by :cite:`Dandrea.1986`.

    Requires a 'chemical potential' model (defaults to
    :py:class:`~IchimaruChemPotential`).

    See Also
    --------
    jaxtrs.free_free.
        Function used to calculate the dynamic free-free electron structure
        factor.
    """

    __name__ = "RPA_DandreaFit"

    @jax.jit
    def evaluate_raw(
        self, plasma_state: "PlasmaState", setup: Setup, *args, **kwargs
    ) -> jnp.ndarray:
        k = setup.dispersion_corrected_k(plasma_state.n_e)
        See_0 = free_free.S0_ee_RPA_Dandrea(
            k,
            plasma_state.T_e,
            plasma_state.n_e,
            setup.measured_energy - setup.energy,
            plasma_state["ee-lfc"].evaluate_fullk(plasma_state, setup),
        )

        # Interpolate to avoid the nan value for E == 0
        w_pl = plasma_physics.plasma_frequency(plasma_state.n_e)
        interpE = jnp.array([-1e-6, 1e-6]) * (1 * ureg.hbar) * w_pl

        See_interp = free_free.S0_ee_RPA_Dandrea(
            setup.k,
            plasma_state.T_e,
            plasma_state.n_e,
            interpE,
            plasma_state["ee-lfc"].evaluate(plasma_state, setup),
        )

        See_0 = jnpu.where(
            setup.measured_energy - setup.energy == 0,
            jnpu.mean(See_interp),
            See_0,
        )
        ff = See_0 * jnp.sum(
            plasma_state.Z_free * plasma_state.number_fraction
        )
        # Return 0 scattering if there are no free electrons
        return (
            jax.lax.cond(
                jnp.sum(plasma_state.Z_free) == 0,
                lambda: jnp.zeros_like(setup.measured_energy) * ureg.second,
                lambda: ff,
            )
            / plasma_state.mean_Z_A
        )

    @jax.jit
    def susceptibility(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        E: Quantity,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        k = setup.k
        xi0 = free_free.noninteracting_susceptibility_Dandrea1986(
            k, E, plasma_state.T_e, plasma_state.n_e
        )
        lfc = plasma_state["ee-lfc"].evaluate(plasma_state, setup)
        V = plasma_physics.coulomb_potential_fourier(-1, -1, k)
        xi = ee_localfieldcorrections.xi_lfc_corrected(xi0, V, lfc)
        return xi


class BornMerminFull(FreeFreeModel):
    """
    Model of the free-free scattering, based on the Born Mermin Approximation
    (:cite:`Mermin.1970`).

    Has the optional argument ``RPA_rewrite``, which defaults to ``True``. If
    ``True``, the integral RPA integral as formulated by :cite:`Chapman.2015`
    is used, otherwise, use the formulas that are found, e.g., in
    :cite:`Schorner.2023`.

    Furtemore, it has the optional attribute ``KKT``, defaulting to ``False``,
    using :cite:`jaxrts.free_free.KramersKronigTransform`, for the imaginary
    part of the collision frequency, rather than solving the integral for the
    imaginary part, as well.
    We found for edge cases to avoid numerical spikes.

    Requires a 'chemical potential' model (defaults to
    :py:class:`~.IchimaruChemPotential`).
    Requires a 'BM V_eiS' model (defaults to
    :py:class:`~.FiniteWavelength_BM_V`).

    See Also
    --------

    jaxrts.free_free.S0_ee_BMA(
        Function used to calculate the dynamic structure factor
    """

    __name__ = "BornMerminFull"

    def __init__(self, RPA_rewrite: bool = True, KKT: bool = False) -> None:
        super().__init__()
        self.RPA_rewrite: bool = RPA_rewrite
        self.KKT: bool = KKT

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        plasma_state.update_default_model(
            "chemical potential", IchimaruChemPotential()
        )
        plasma_state.update_default_model("BM V_eiS", FiniteWavelength_BM_V())
        if len(plasma_state) == 1:
            plasma_state.update_default_model("BM S_ii", Sum_Sii())
        else:
            plasma_state.update_default_model("BM S_ii", AverageAtom_Sii())

    @jax.jit
    def evaluate_raw(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        *args,
        **kwargs,
    ) -> jnp.ndarray:

        @jax.tree_util.Partial
        def S_ii(k):
            probe_setup = get_probe_setup(k, setup)
            return plasma_state.evaluate("BM S_ii", probe_setup)

        @jax.tree_util.Partial
        def V_eiS(k):
            return plasma_state["BM V_eiS"].V(plasma_state, k)

        mean_Z_free = jnpu.sum(
            plasma_state.Z_free * plasma_state.number_fraction
        )
        mu = plasma_state["chemical potential"].evaluate(plasma_state, setup)
        k = setup.dispersion_corrected_k(plasma_state.n_e)
        See_0 = free_free.S0_ee_BMA(
            k,
            plasma_state.T_e,
            mu,
            S_ii,
            V_eiS,
            plasma_state.n_e,
            mean_Z_free,
            setup.measured_energy - setup.energy,
            plasma_state["ee-lfc"].evaluate(plasma_state, setup),
            rpa_rewrite=self.RPA_rewrite,
            KKT=self.KKT,
        )
        ff = See_0 * mean_Z_free
        # Return 0 scattering if there are no free electrons
        return (
            jax.lax.cond(
                jnp.sum(plasma_state.Z_free) == 0,
                lambda: jnp.zeros_like(setup.measured_energy) * ureg.second,
                lambda: ff,
            )
            / plasma_state.mean_Z_A
        )

    @jax.jit
    def susceptibility(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        E: Quantity,
        *args,
        **kwargs,
    ) -> jnp.ndarray:

        @jax.tree_util.Partial
        def S_ii(k):
            probe_setup = get_probe_setup(k, setup)
            return plasma_state.evaluate("BM S_ii", probe_setup)

        @jax.tree_util.Partial
        def V_eiS(k):
            return plasma_state["BM V_eiS"].V(plasma_state, k)

        mu = plasma_state["chemical potential"].evaluate(plasma_state, setup)
        k = setup.k
        mean_Z_free = jnpu.sum(
            plasma_state.Z_free * plasma_state.number_fraction
        )

        def chi(energy):
            eps = free_free.dielectric_function_BMA_full(
                k,
                energy,
                mu,
                plasma_state.T_e,
                plasma_state.n_e,
                S_ii,
                V_eiS,
                mean_Z_free,
                rpa_rewrite=self.RPA_rewrite,
                KKT=self.KKT,
            )
            xi0 = noninteracting_susceptibility_from_eps_RPA(eps, k)
            lfc = plasma_state["ee-lfc"].evaluate(plasma_state, setup)
            V = plasma_physics.coulomb_potential_fourier(-1, -1, k)
            xi = ee_localfieldcorrections.xi_lfc_corrected(xi0, V, lfc)
            return xi

        # Interpolate for small energy transfers, as it will give nans for zero
        w_pl = plasma_physics.plasma_frequency(plasma_state.n_e)
        interpE = jnp.array([-1e-4, 1e-4]) * (1 * ureg.hbar) * w_pl
        interpchi = chi(interpE)
        return jnpu.where(
            jnpu.absolute(E) > interpE[1],
            chi(E),
            jnpu.interp(E, interpE, interpchi),
        )

    def _tree_flatten(self):
        children = ()
        aux_data = (
            self.model_key,
            self.sample_points,
            self.RPA_rewrite,
            self.KKT,
        )  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.model_key, obj.sample_points, obj.RPA_rewrite, obj.KKT = aux_data

        return obj


class BornMermin(FreeFreeModel):
    """
    Model of the free-free scattering, based on the Born Mermin Approximation
    (:cite:`Mermin.1970`).
    Uses the Chapman interpolation which allows for a faster computation of the
    free-free scattering compared to :py:class:`~.BornMerminFull`, by sampleing
    at the probing frequency at :py:attr:`~.no_of_freq` points and
    interpolating between them, after.

    The number of frequencies defaults to 20 if ``KKT`` is ``False``, and to
    100 otherwise. To change it, just change the attribute of this model after
    initializing it. i.e.

    >>> state["free-free scattering"] = jaxrts.models.BornMermin()
    >>> state["free-free scattering"].no_of_freq = 10

    The boundaries for this interpolation can be set as arguments
    ``E_cutoff_min`` and ``E_cutoff_max``. It should be set to sane defaults
    for most use cases; however, it is recommended to revisit this setting
    carefully. As a minimal good practice, the defaults should be adjusted to
    the setup used. This can be done with the
    :py:meth:`~./set_guessed_E_cutoffs` method:

    >>> state["free-free scattering"].set_guessed_E_cutoffs(state, setup)

    Has the optional argument ``RPA_rewrite``, which defaults to ``True``. If
    ``True``, the integral RPA integral as formulated by :cite:`Chapman.2015`
    is used, otherwise, use the formulas that are found, e.g., in
    :cite:`Schorner.2023`.

    Furtemore, it has the optional attribute ``KKT``, defaulting to ``False``,
    using :cite:`jaxrts.free_free.KramersKronigTransform`, for the imaginary
    part of the collision frequency, rather than solving the integral for the
    imaginary part, as well.
    We found for edge cases to avoid numerical spikes.

    Requires a 'chemical potential' model (defaults to
    :py:class:`~.IchimaruChemPotential`).
    Requires a 'BM V_eiS' model (defaults to
    :py:class:`~.FiniteWavelength_BM_V`).

    See Also
    --------

    jaxrts.free_free.S0_ee_BMA_chapman_interp
        Function used to calculate the dynamic structure factor
    """

    __name__ = "BornMermin"

    def __init__(
        self,
        no_of_freq: int | None = None,
        RPA_rewrite: bool = True,
        KKT: bool = False,
        E_cutoff_min: Quantity = -1.0 * ureg.electron_volt,
        E_cutoff_max: Quantity = -1.0 * ureg.electron_volt,
    ) -> None:
        super().__init__()
        if no_of_freq is not None:
            self.no_of_freq: int = no_of_freq
        else:
            self.no_of_freq: int = 100 if KKT else 20
        self.RPA_rewrite: bool = RPA_rewrite
        self.KKT: bool = KKT
        self.E_cutoff_min: Quantity = E_cutoff_min
        self.E_cutoff_max: Quantity = E_cutoff_max

    def guess_E_cutoffs(
        self,
        plasma_state: "PlasmaState",
        setup: Setup | None = None,
        E_max: Quantity = 500 * ureg.electron_volt,
    ) -> None:
        """
        Guess and set cutoff energies for the collision frequency
        interpolation, based on the plasma_state and setup evaluated.
        """
        if setup is not None:
            E_max = jnpu.max(
                jnpu.absolute(setup.measured_energy - setup.energy)
            )
            k = setup.k
        else:
            lam = ureg.planck_constant * ureg.c / E_max
            k = 4 * jnp.pi / lam
        E_cutoff_min = free_free.guess_E_cutoff_min(plasma_state.n_e, self.KKT)
        E_cutoff_max = free_free.guess_E_cutoff_max(
            k, plasma_state.T_e, plasma_state.n_e, E_max, self.KKT
        )
        return E_cutoff_min, E_cutoff_max

    def set_guessed_E_cutoffs(
        self,
        plasma_state: "PlasmaState",
        setup: Setup | None = None,
        E_max: Quantity = 500 * ureg.electron_volt,
    ) -> None:
        """
        Guess and set cutoff energies for the collision frequency
        interpolation, based on the plasma_state and setup evaluated.
        """
        E_cutoff_min, E_cutoff_max = self.guess_E_cutoffs(
            plasma_state, setup, E_max
        )
        self.E_cutoff_min = E_cutoff_min
        self.E_cutoff_max = E_cutoff_max

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        plasma_state.update_default_model(
            "chemical potential", IchimaruChemPotential()
        )
        plasma_state.update_default_model("BM V_eiS", FiniteWavelength_BM_V())
        if len(plasma_state) == 1:
            plasma_state.update_default_model("BM S_ii", Sum_Sii())
        else:
            plasma_state.update_default_model("BM S_ii", AverageAtom_Sii())
        E_cutoff_min, E_cutoff_max = self.guess_E_cutoffs(plasma_state)
        self.E_cutoff_min = jnpu.where(
            self.E_cutoff_min > 0 * ureg.electron_volt,
            self.E_cutoff_min,
            E_cutoff_min,
        )
        self.E_cutoff_max = jnpu.where(
            self.E_cutoff_max > 0 * ureg.electron_volt,
            self.E_cutoff_max,
            E_cutoff_max,
        )

    @jax.jit
    def evaluate_raw(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        *args,
        **kwargs,
    ) -> jnp.ndarray:

        @jax.tree_util.Partial
        def S_ii(k):
            probe_setup = get_probe_setup(k, setup)
            return plasma_state.evaluate("BM S_ii", probe_setup)

        @jax.tree_util.Partial
        def V_eiS(k):
            return plasma_state["BM V_eiS"].V(plasma_state, k)

        mean_Z_free = jnpu.sum(
            plasma_state.Z_free * plasma_state.number_fraction
        )
        mu = plasma_state["chemical potential"].evaluate(plasma_state, setup)
        k = setup.dispersion_corrected_k(plasma_state.n_e)
        See_0 = free_free.S0_ee_BMA_chapman_interp(
            k,
            plasma_state.T_e,
            mu,
            S_ii,
            V_eiS,
            plasma_state.n_e,
            mean_Z_free,
            self.E_cutoff_min,
            self.E_cutoff_max,
            setup.measured_energy - setup.energy,
            plasma_state["ee-lfc"].evaluate(plasma_state, setup),
            self.no_of_freq,
            rpa_rewrite=self.RPA_rewrite,
            KKT=self.KKT,
        )
        ff = See_0 * mean_Z_free
        # Return 0 scattering if there are no free electrons
        return (
            jax.lax.cond(
                jnp.sum(plasma_state.Z_free) == 0,
                lambda: jnp.zeros_like(setup.measured_energy) * ureg.second,
                lambda: ff,
            )
            / plasma_state.mean_Z_A
        )

    @jax.jit
    def susceptibility(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        E: Quantity,
        *args,
        **kwargs,
    ) -> jnp.ndarray:

        @jax.tree_util.Partial
        def S_ii(k):
            probe_setup = get_probe_setup(k, setup)
            return plasma_state.evaluate("BM S_ii", probe_setup)

        @jax.tree_util.Partial
        def V_eiS(k):
            probe_setup = get_probe_setup(k, setup)
            return plasma_state["BM V_eiS"].evaluate(plasma_state, probe_setup)

        mean_Z_free = jnpu.sum(
            plasma_state.Z_free * plasma_state.number_fraction
        )
        mu = plasma_state["chemical potential"].evaluate(plasma_state, setup)
        k = setup.k

        def chi(energy):
            eps = free_free.dielectric_function_BMA_chapman_interp(
                k,
                energy,
                mu,
                plasma_state.T_e,
                plasma_state.n_e,
                S_ii,
                V_eiS,
                mean_Z_free,
                self.E_cutoff_min,
                self.E_cutoff_max,
                self.no_of_freq,
                rpa_rewrite=self.RPA_rewrite,
                KKT=self.KKT,
            )
            xi0 = noninteracting_susceptibility_from_eps_RPA(eps, k)
            lfc = plasma_state["ee-lfc"].evaluate(plasma_state, setup)
            V = plasma_physics.coulomb_potential_fourier(-1, -1, k)
            xi = ee_localfieldcorrections.xi_lfc_corrected(xi0, V, lfc)
            return xi

        # Interpolate for small energy transfers, as it will give nans for zero
        w_pl = plasma_physics.plasma_frequency(plasma_state.n_e)
        interpE = jnp.array([-1e-4, 1e-4]) * (1 * ureg.hbar) * w_pl
        interpchi = chi(interpE)
        return jnpu.where(
            jnpu.absolute(E) > interpE[1],
            chi(E),
            jnpu.interp(E, interpE, interpchi),
        )

    def _tree_flatten(self):
        children = (
            self.E_cutoff_min,
            self.E_cutoff_max,
        )
        aux_data = (
            self.model_key,
            self.sample_points,
            self.no_of_freq,
            self.RPA_rewrite,
            self.KKT,
        )  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        (
            obj.model_key,
            obj.sample_points,
            obj.no_of_freq,
            obj.RPA_rewrite,
            obj.KKT,
        ) = aux_data
        (
            obj.E_cutoff_min,
            obj.E_cutoff_max,
        ) = children

        return obj


class BornMermin_Fit(FreeFreeModel):
    """
    Model of the free-free scattering, based on the Born Mermin Approximation
    (:cite:`Mermin.1970`).
    Identical to :py:class:`~.BornMermin`, but uses the Dandrea
    fit (:cite:`Dandrea.1986`), rather than numerically calculating the
    un-damped RPA, numerically. However, the damped RPA is still evaluated
    using the integral.

    The number of frequencies defaults to 20 if ``KKT`` is ``False``, and to
    100 otherwise. To change it, just change the attribute of this model after
    initializing it. i.e.

    >>> state["free-free scattering"] = jaxrts.models.BornMermin_Fit()
    >>> state["free-free scattering"].no_of_freq = 10

    The boundaries for this interpolation can be set as arguments
    ``E_cutoff_min`` and ``E_cutoff_max``. It should be set to sane defaults
    for most use cases; however, it is recommended to revisit this setting
    carefully. As a minimal good practice, the defaults should be adjusted to
    the setup used. This can be done with the
    :py:meth:`~./set_guessed_E_cutoffs` method:

    >>> state["free-free scattering"].set_guessed_E_cutoffs(state, setup)

    Has the optional argument ``RPA_rewrite``, which defaults to ``True``. If
    ``True``, the integral RPA integral as formulated by :cite:`Chapman.2015`
    is used, otherwise, use the formulas that are found, e.g., in
    :cite:`Schorner.2023`.

    Furtemore, it has the optional attribute ``KKT``, defaulting to ``False``,
    using :cite:`jaxrts.free_free.KramersKronigTransform`, for the imaginary
    part of the collision frequency, rather than solving the integral for the
    imaginary part, as well.
    We found for edge cases to avoid numerical spikes.

    Requires a 'chemical potential' model (defaults to
    :py:class:`~.IchimaruChemPotential`).
    Requires a 'BM V_eiS' model (defaults to
    :py:class:`~.FiniteWavelength_BM_V`).

    See Also
    --------

    jaxrts.free_free.S0_ee_BMA_chapman_interpFit
        Function used to calculate the dynamic structure factor
    """

    __name__ = "BornMermin_Fit"

    def __init__(
        self,
        no_of_freq: int | None = None,
        RPA_rewrite: bool = True,
        KKT: bool = False,
        E_cutoff_min: Quantity = -1.0 * ureg.electron_volt,
        E_cutoff_max: Quantity = -1.0 * ureg.electron_volt,
    ) -> None:
        super().__init__()
        if no_of_freq is not None:
            self.no_of_freq: int = no_of_freq
        else:
            self.no_of_freq: int = 100 if KKT else 20
        self.RPA_rewrite: bool = RPA_rewrite
        self.KKT: bool = KKT
        self.E_cutoff_min: Quantity = E_cutoff_min
        self.E_cutoff_max: Quantity = E_cutoff_max

    def guess_E_cutoffs(
        self,
        plasma_state: "PlasmaState",
        setup: Setup | None = None,
        E_max: Quantity = 500 * ureg.electron_volt,
    ) -> None:
        """
        Guess and set cutoff energies for the collision frequency
        interpolation, based on the plasma_state and setup evaluated.
        """
        if setup is not None:
            E_max = jnpu.max(
                jnpu.absolute(setup.measured_energy - setup.energy)
            )
            k = setup.k
        else:
            lam = ureg.planck_constant * ureg.c / E_max
            k = 4 * jnp.pi / lam
        E_cutoff_min = free_free.guess_E_cutoff_min(plasma_state.n_e, self.KKT)
        E_cutoff_max = free_free.guess_E_cutoff_max(
            k, plasma_state.T_e, plasma_state.n_e, E_max, self.KKT
        )
        return E_cutoff_min, E_cutoff_max

    def set_guessed_E_cutoffs(
        self,
        plasma_state: "PlasmaState",
        setup: Setup | None = None,
        E_max: Quantity = 500 * ureg.electron_volt,
    ) -> None:
        """
        Guess and set cutoff energies for the collision frequency
        interpolation, based on the plasma_state and setup evaluated.
        """
        E_cutoff_min, E_cutoff_max = self.guess_E_cutoffs(
            plasma_state, setup, E_max
        )
        self.E_cutoff_min = E_cutoff_min
        self.E_cutoff_max = E_cutoff_max

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        plasma_state.update_default_model(
            "chemical potential", IchimaruChemPotential()
        )
        plasma_state.update_default_model("BM V_eiS", FiniteWavelength_BM_V())
        if len(plasma_state) == 1:
            plasma_state.update_default_model("BM S_ii", Sum_Sii())
        else:
            plasma_state.update_default_model("BM S_ii", AverageAtom_Sii())
        E_cutoff_min, E_cutoff_max = self.guess_E_cutoffs(plasma_state)
        self.E_cutoff_min = jnpu.where(
            self.E_cutoff_min > 0 * ureg.electron_volt,
            self.E_cutoff_min,
            E_cutoff_min,
        )
        self.E_cutoff_max = jnpu.where(
            self.E_cutoff_max > 0 * ureg.electron_volt,
            self.E_cutoff_max,
            E_cutoff_max,
        )

    @jax.jit
    def evaluate_raw(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        *args,
        **kwargs,
    ) -> jnp.ndarray:

        @jax.tree_util.Partial
        def S_ii(k):
            probe_setup = get_probe_setup(k, setup)
            return plasma_state.evaluate("BM S_ii", probe_setup)

        @jax.tree_util.Partial
        def V_eiS(k):
            return plasma_state["BM V_eiS"].V(plasma_state, k)

        mean_Z_free = jnpu.sum(
            plasma_state.Z_free * plasma_state.number_fraction
        )
        mu = plasma_state["chemical potential"].evaluate(plasma_state, setup)
        k = setup.dispersion_corrected_k(plasma_state.n_e)
        See_0 = free_free.S0_ee_BMA_chapman_interpFit(
            k,
            plasma_state.T_e,
            mu,
            S_ii,
            V_eiS,
            plasma_state.n_e,
            mean_Z_free,
            self.E_cutoff_min,
            self.E_cutoff_max,
            setup.measured_energy - setup.energy,
            plasma_state["ee-lfc"].evaluate(plasma_state, setup),
            self.no_of_freq,
            rpa_rewrite=self.RPA_rewrite,
            KKT=self.KKT,
        )
        ff = See_0 * mean_Z_free
        # Return 0 scattering if there are no free electrons
        return (
            jax.lax.cond(
                jnp.sum(plasma_state.Z_free) == 0,
                lambda: jnp.zeros_like(setup.measured_energy) * ureg.second,
                lambda: ff,
            )
            / plasma_state.mean_Z_A
        )

    @jax.jit
    def susceptibility(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        E: Quantity,
        *args,
        **kwargs,
    ) -> jnp.ndarray:

        @jax.tree_util.Partial
        def S_ii(k):
            probe_setup = get_probe_setup(k, setup)
            return plasma_state.evaluate("BM S_ii", probe_setup)

        @jax.tree_util.Partial
        def V_eiS(k):
            return plasma_state["BM V_eiS"].V(plasma_state, k)

        mean_Z_free = jnpu.sum(
            plasma_state.Z_free * plasma_state.number_fraction
        )
        mu = plasma_state["chemical potential"].evaluate(plasma_state, setup)
        k = setup.k

        def chi(energy):
            eps = free_free.dielectric_function_BMA_chapman_interpFit(
                k,
                energy,
                mu,
                plasma_state.T_e,
                plasma_state.n_e,
                S_ii,
                V_eiS,
                mean_Z_free,
                self.E_cutoff_min,
                self.E_cutoff_max,
                self.no_of_freq,
                rpa_rewrite=self.RPA_rewrite,
                KKT=self.KKT,
            )
            xi0 = noninteracting_susceptibility_from_eps_RPA(eps, k)
            lfc = plasma_state["ee-lfc"].evaluate(plasma_state, setup)
            V = plasma_physics.coulomb_potential_fourier(-1, -1, k)
            xi = ee_localfieldcorrections.xi_lfc_corrected(xi0, V, lfc)
            return xi

        # Interpolate for small energy transfers, as it will give nans for zero
        w_pl = plasma_physics.plasma_frequency(plasma_state.n_e)
        interpE = jnp.array([-1e-2, 1e-2]) * (1 * ureg.hbar) * w_pl
        interpchi = chi(interpE)
        return jnpu.where(
            jnpu.absolute(E) > interpE[1],
            chi(E),
            jnpu.interp(E, interpE, interpchi),
        )

    def _tree_flatten(self):
        children = (
            self.E_cutoff_min,
            self.E_cutoff_max,
        )
        aux_data = (
            self.model_key,
            self.sample_points,
            self.no_of_freq,
            self.RPA_rewrite,
            self.KKT,
        )  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        (
            obj.model_key,
            obj.sample_points,
            obj.no_of_freq,
            obj.RPA_rewrite,
            obj.KKT,
        ) = aux_data
        (
            obj.E_cutoff_min,
            obj.E_cutoff_max,
        ) = children

        return obj


class BornMermin_Fortmann(FreeFreeModel):
    """
    Model of the free-free scattering, based on the Born Mermin Approximation
    (:cite:`Mermin.1970`).
    Uses the same collision frequency as :py:class:`~.BornMermin_Fit`
    (including the :cite:`Dandrea.1986` fit), but uses a rigorous
    implementation of the local field correction, proposed by
    :cite:`Fortmann.2010`.

    The number of frequencies defaults to 20 if ``KKT`` is ``False``, and to
    100 otherwise. To change it, just change the attribute of this model after
    initializing it. i.e.

    >>> state["free-free scattering"] = jaxrts.models.BornMermin_Fortmann()
    >>> state["free-free scattering"].no_of_freq = 10

    The boundaries for this interpolation can be set as arguments
    ``E_cutoff_min`` and ``E_cutoff_max``. It should be set to sane defaults
    for most use cases; however, it is recommended to revisit this setting
    carefully. As a minimal good practice, the defaults should be adjusted to
    the setup used. This can be done with the
    :py:meth:`~./set_guessed_E_cutoffs` method:

    >>> state["free-free scattering"].set_guessed_E_cutoffs(state, setup)

    Has the optional argument ``RPA_rewrite``, which defaults to ``True``. If
    ``True``, the integral RPA integral as formulated by :cite:`Chapman.2015`
    is used, otherwise, use the formulas that are found, e.g., in
    :cite:`Schorner.2023`.

    Furtemore, it has the optional attribute ``KKT``, defaulting to ``False``,
    using :cite:`jaxrts.free_free.KramersKronigTransform`, for the imaginary
    part of the collision frequency, rather than solving the integral for the
    imaginary part, as well.
    We found for edge cases to avoid numerical spikes.

    Requires a 'chemical potential' model (defaults to
    :py:class:`~.IchimaruChemPotential`).
    Requires a 'BM V_eiS' model (defaults to
    :py:class:`~.FiniteWavelength_BM_V`).

    See Also
    --------

    jaxrts.free_free.S0_ee_BMA_Fortmann
        Function used to calculate the dynamic structure factor
    jaxrts.free_free.susceptibility_BMA_Fortmann
        Function used to calculate the susceptibility
    """

    __name__ = "BornMermin_Fortmann"

    def __init__(
        self,
        no_of_freq: int | None = None,
        RPA_rewrite: bool = True,
        KKT: bool = False,
        E_cutoff_min: Quantity = -1.0 * ureg.electron_volt,
        E_cutoff_max: Quantity = -1.0 * ureg.electron_volt,
    ) -> None:
        super().__init__()
        if no_of_freq is not None:
            self.no_of_freq: int = no_of_freq
        else:
            self.no_of_freq: int = 100 if KKT else 20
        self.RPA_rewrite: bool = RPA_rewrite
        self.KKT: bool = KKT
        self.E_cutoff_min: Quantity = E_cutoff_min
        self.E_cutoff_max: Quantity = E_cutoff_max

    def guess_E_cutoffs(
        self,
        plasma_state: "PlasmaState",
        setup: Setup | None = None,
        E_max: Quantity = 500 * ureg.electron_volt,
    ) -> None:
        """
        Guess and set cutoff energies for the collision frequency
        interpolation, based on the plasma_state and setup evaluated.
        """
        if setup is not None:
            E_max = jnpu.max(
                jnpu.absolute(setup.measured_energy - setup.energy)
            )
            k = setup.k
        else:
            lam = ureg.planck_constant * ureg.c / E_max
            k = 4 * jnp.pi / lam
        E_cutoff_min = free_free.guess_E_cutoff_min(plasma_state.n_e, self.KKT)
        E_cutoff_max = free_free.guess_E_cutoff_max(
            k, plasma_state.T_e, plasma_state.n_e, E_max, self.KKT
        )
        return E_cutoff_min, E_cutoff_max

    def set_guessed_E_cutoffs(
        self,
        plasma_state: "PlasmaState",
        setup: Setup | None = None,
        E_max: Quantity = 500 * ureg.electron_volt,
    ) -> None:
        """
        Guess and set cutoff energies for the collision frequency
        interpolation, based on the plasma_state and setup evaluated.
        """
        E_cutoff_min, E_cutoff_max = self.guess_E_cutoffs(
            plasma_state, setup, E_max
        )
        self.E_cutoff_min = E_cutoff_min
        self.E_cutoff_max = E_cutoff_max

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        plasma_state.update_default_model(
            "chemical potential", IchimaruChemPotential()
        )
        plasma_state.update_default_model("BM V_eiS", FiniteWavelength_BM_V())
        if len(plasma_state) == 1:
            plasma_state.update_default_model("BM S_ii", Sum_Sii())
        else:
            plasma_state.update_default_model("BM S_ii", AverageAtom_Sii())
        E_cutoff_min, E_cutoff_max = self.guess_E_cutoffs(plasma_state)
        self.E_cutoff_min = jnpu.where(
            self.E_cutoff_min > 0 * ureg.electron_volt,
            self.E_cutoff_min,
            E_cutoff_min,
        )
        self.E_cutoff_max = jnpu.where(
            self.E_cutoff_max > 0 * ureg.electron_volt,
            self.E_cutoff_max,
            E_cutoff_max,
        )

    @jax.jit
    def evaluate_raw(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        *args,
        **kwargs,
    ) -> jnp.ndarray:

        @jax.tree_util.Partial
        def S_ii(k):
            probe_setup = get_probe_setup(k, setup)
            return plasma_state.evaluate("BM S_ii", probe_setup)

        @jax.tree_util.Partial
        def V_eiS(k):
            return plasma_state["BM V_eiS"].V(plasma_state, k)

        mu = plasma_state["chemical potential"].evaluate(plasma_state, setup)
        k = setup.dispersion_corrected_k(plasma_state.n_e)
        mean_Z_free = jnpu.sum(
            plasma_state.Z_free * plasma_state.number_fraction
        )
        See_0 = free_free.S0_ee_BMA_Fortmann(
            k,
            plasma_state.T_e,
            mu,
            S_ii,
            V_eiS,
            plasma_state.n_e,
            mean_Z_free,
            self.E_cutoff_min,
            self.E_cutoff_max,
            setup.measured_energy - setup.energy,
            plasma_state["ee-lfc"].evaluate(plasma_state, setup),
            self.no_of_freq,
            rpa_rewrite=self.RPA_rewrite,
            KKT=self.KKT,
        )
        ff = See_0 * mean_Z_free
        # Return 0 scattering if there are no free electrons
        return (
            jax.lax.cond(
                jnp.sum(plasma_state.Z_free) == 0,
                lambda: jnp.zeros_like(setup.measured_energy) * ureg.second,
                lambda: ff,
            )
            / plasma_state.mean_Z_A
        )

    @jax.jit
    def susceptibility(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        E: Quantity,
        *args,
        **kwargs,
    ) -> jnp.ndarray:

        @jax.tree_util.Partial
        def S_ii(k):
            probe_setup = get_probe_setup(k, setup)
            return plasma_state.evaluate("BM S_ii", probe_setup)

        @jax.tree_util.Partial
        def V_eiS(k):
            return plasma_state["BM V_eiS"].V(plasma_state, k)

        mu = plasma_state["chemical potential"].evaluate(plasma_state, setup)
        k = setup.k

        mean_Z_free = jnpu.sum(
            plasma_state.Z_free * plasma_state.number_fraction
        )
        xi = free_free.susceptibility_BMA_Fortmann(
            k,
            E,
            mu,
            plasma_state.T_e,
            plasma_state.n_e,
            S_ii,
            V_eiS,
            mean_Z_free,
            self.E_cutoff_min,
            self.E_cutoff_max,
            plasma_state["ee-lfc"].evaluate(plasma_state, setup),
            self.no_of_freq,
            rpa_rewrite=self.RPA_rewrite,
            KKT=self.KKT,
        )
        return xi

    def _tree_flatten(self):
        children = (
            self.E_cutoff_min,
            self.E_cutoff_max,
        )
        aux_data = (
            self.model_key,
            self.sample_points,
            self.no_of_freq,
            self.RPA_rewrite,
            self.KKT,
        )  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        (
            obj.model_key,
            obj.sample_points,
            obj.no_of_freq,
            obj.RPA_rewrite,
            obj.KKT,
            obj.E_cutoff_min,
            obj.E_cutoff_max,
        ) = aux_data
        (
            obj.E_cutoff_min,
            obj.E_cutoff_max,
        ) = children

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

    Requires an 'ipd' model (defaults to
    :py:class:`~Neglect`).
    """

    allowed_keys = ["bound-free scattering"]
    __name__ = "SchumacherImpulse"

    def __init__(self, r_k: float | None = None) -> None:
        """
        r_k is the correction given in :cite:`Gregori.2004`. If None, or if a
        negative value is given, we use the folula given by
        :cite:`Gregori.2004`. Otherwise, it is just the value provided by the
        user.
        """
        super().__init__()

        #: The value for r_k (see :cite:`Gregori.2004`). A negative value means
        #: to use the calculation given in the aforementioned paper.
        self.r_k = r_k
        if self.r_k is None:
            self.r_k = -1.0
        # This is required to catch a potential int input by a user
        if isinstance(self.r_k, int):
            self.r_k = float(self.r_k)

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        plasma_state.update_default_model("form-factors", PaulingFormFactors())
        plasma_state.update_default_model("ipd", Neglect())

    @jax.jit
    def evaluate_raw(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
    ) -> jnp.ndarray:
        k = setup.dispersion_corrected_k(plasma_state.n_e)
        omega_0 = setup.energy / ureg.hbar
        omega = omega_0 - setup.measured_energy / ureg.hbar
        x = plasma_state.number_fraction

        out = 0 * ureg.second
        for idx in range(plasma_state.nions):
            Z_c = plasma_state.Z_core[idx]
            E_b = (
                plasma_state.ions[idx].binding_energies
                + plasma_state.models["ipd"].evaluate(plasma_state, None)[idx]
            )
            E_b = jnpu.where(
                E_b < 0 * ureg.electron_volt, 0 * ureg.electron_volt, E_b
            )

            Zeff = (
                plasma_state.ions[idx].Z
            ) - form_factors.pauling_size_screening_constants(Z_c)
            population = electron_distribution_ionized_state(Z_c)

            def rk_on(r_k):
                # Gregori.2004, Eqn 20
                fi = plasma_state["form-factors"].evaluate(
                    plasma_state, setup
                )[:, idx]
                new_r_k = 1 - jnp.sum(population * (fi) ** 2) / Z_c
                # Catch the division by zero error
                new_r_k = jax.lax.cond(Z_c == 0, lambda: 1.0, lambda: new_r_k)
                return new_r_k

            def rk_off(r_k):
                """
                Use the rk provided by the user
                """
                return r_k

            r_k = jax.lax.cond(self.r_k < 0, rk_on, rk_off, self.r_k)
            B = 1 + 1 / omega_0 * (ureg.hbar * k**2) / (2 * ureg.electron_mass)
            # B should be close to unity
            # B = 1 * ureg.dimensionless
            factor = r_k / (Z_c * B**3).m_as(ureg.dimensionless)
            sbe = factor * bound_free.J_impulse_approx(
                omega, k, population, Zeff, E_b
            )
            val = sbe * Z_c * x[idx]
            out += jnpu.where(
                jnp.isnan(val.m_as(ureg.second)), 0 * ureg.second, val
            )
        return out / plasma_state.mean_Z_A

    def _tree_flatten(self):
        children = ()
        aux_data = (
            self.model_key,
            self.sample_points,
            self.r_k,
        )  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.model_key, obj.sample_points, obj.r_k = aux_data

        return obj


class SchumacherImpulseFitRk(ScatteringModel):
    """
    Bound-free scattering based on the Schumacher Impulse Approximation
    :cite:`Schumacher.1975`. The implementation considers the first order
    asymmetric correction to the impulse approximation, as given in the
    aforementioned paper. The r_k factor is fitted to fulfill the f-sum rule.
    Note, that this implementation is still experimental.

    Requires a 'form-factors' model (defaults to
    :py:class:`~.PaulingFormFactors`).

    Requires an 'ipd' model (defaults to
    :py:class:`~.Neglect`).
    j
    Requires a 'free-free scattering' model (defaults to
    :py:class:`~.RPA_DandreaFit`).
    """

    allowed_keys = ["bound-free scattering"]
    __name__ = "SchumacherImpulseFitRk"

    def __init__(self) -> None:
        super().__init__()

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        plasma_state.update_default_model("form-factors", PaulingFormFactors())
        plasma_state.update_default_model("ipd", Neglect())
        plasma_state.update_default_model(
            "free-free scattering", RPA_DandreaFit()
        )

    @jax.jit
    def r_k(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
    ) -> jnp.ndarray:

        # Calculate r_k through f-sum rule

        fsum_theory = -1 * ureg.hbar**2 * setup.k**2 / (2 * ureg.electron_mass)

        setup_dispersion_off = Setup(
            setup.scattering_angle,
            setup.energy,
            setup.measured_energy,
            setup.instrument,
            False,
        )

        bf = SchumacherImpulse(r_k=1.0).evaluate_raw(
            plasma_state, setup_dispersion_off
        )

        energy_shift = setup.measured_energy - setup.energy

        mirrored_setup = free_bound.FreeBoundFlippedSetup(setup_dispersion_off)
        db_factor = jnpu.exp(-energy_shift / (plasma_state.T_e * ureg.k_B))
        fb = (
            SchumacherImpulse(r_k=1.0).evaluate_raw(
                plasma_state, mirrored_setup
            )
            * db_factor
        )

        free_free_fsum = ITCF_fsum(
            plasma_state["free-free scattering"].evaluate_raw(
                plasma_state, setup_dispersion_off
            ),
            raw=True,
            setup=setup_dispersion_off,
            E_cut=jnpu.max(setup.measured_energy - setup.energy),
        )

        fb_bf_fsum = ITCF_fsum(
            fb + bf,
            raw=True,
            setup=setup_dispersion_off,
            E_cut=jnpu.max(setup.measured_energy - setup.energy),
        )

        r_k = (fsum_theory - free_free_fsum) / (fb_bf_fsum)
        return r_k.m_as(ureg.dimensionless)

    @jax.jit
    def evaluate_raw(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
    ) -> jnp.ndarray:
        val = SchumacherImpulse(r_k=1.0).evaluate_raw(
            plasma_state, setup
        ) * self.r_k(plasma_state, setup)
        return jnpu.where(
            jnp.isnan(val.m_as(ureg.second)), 0 * ureg.second, val
        )

    def _tree_flatten(self):
        children = ()
        aux_data = (
            self.model_key,
            self.sample_points,
        )  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.model_key, obj.sample_points = aux_data

        return obj


# free-bound Models
# -----------------


class DetailedBalance(ScatteringModel):
    """
    Calculate the free-bound scattering by mirroring the free-bound scattering
    around the probing energy and applying a detailed balance factor to the
    intensity.
    See :cite:`Bohme.2023`, introducing the idea.

    .. note::

       This model requires to have an `evaluate_raw` method for the bound-free
       model, which should return the bound-free scattering intensity **not
       convolved** with an instrument function.

    .. note::

       The typical normalization factor (average ionization) is not required
       here, as the bound-free model should already incorporate this.

    """

    __name__ = "DetailedBalance"
    allowed_keys = ["free-bound scattering"]

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        plasma_state.update_default_model(
            "bound-free scattering", SchumacherImpulse()
        )

    @jax.jit
    def evaluate_raw(
        self, plasma_state: "PlasmaState", setup: Setup
    ) -> jnp.ndarray:
        energy_shift = setup.measured_energy - setup.energy
        mirrored_setup = free_bound.FreeBoundFlippedSetup(setup)
        db_factor = jnpu.exp(-energy_shift / (plasma_state.T_e * ureg.k_B))
        fb = plasma_state["bound-free scattering"].evaluate_raw(
            plasma_state, mirrored_setup
        )
        return fb * db_factor


# Form Factor Models
# ==================


class PaulingFormFactors(Model):
    """
    Analytical functions for each electrons in quantum states defined by the
    quantum numbers `n` and `l`, assuming a hydrogen-like atom. Published in
    :cite:`Pauling.1932`.

    Uses Z - :py:func:`jaxrts.form_factors.pauling_size_screening_constants` to
    calculate the effective charge of the atom's core and then calculates form
    factors with :py:func:`jaxrts.form_factors.pauling_all_ff`.
    """

    allowed_keys = ["form-factors"]
    __name__ = "PaulingFormFactors"

    @jax.jit
    def evaluate(
        self, plasma_state: "PlasmaState", setup: Setup
    ) -> jnp.ndarray:
        Zstar = (
            plasma_state.Z_A
            - form_factors.pauling_size_screening_constants(
                plasma_state.Z_core
            )
        )
        ff = form_factors.pauling_all_ff(setup.k, Zstar)
        # population = plasma_state.ions[0].electron_distribution
        # return jnp.where(population > 0, ff, 0)
        return ff


# Chemical Potential Models
# =========================


class IchimaruChemPotential(Model):
    """
    A fitting formula for the chemical potential of a plasma between the
    classical and the quantum regime, given by :cite:`Gregori.2003`.
    Uses :py:func:`jaxrts.plasma_physics.chem_pot_interpolation`.
    """

    __name__ = "IchimaruChemPotential"
    allowed_keys = ["chemical potential"]

    @jax.jit
    def evaluate(
        self, plasma_state: "PlasmaState", setup: Setup
    ) -> jnp.ndarray:
        return plasma_physics.chem_pot_interpolationIchimaru(
            plasma_state.T_e, plasma_state.n_e
        )


class ConstantChemPotential(Model):
    """
    A model that returns an a constant for each energy probed.
    """

    allowed_keys = ["chemical potential"]
    __name__ = "ConstantChemPotential"

    def __init__(self, value):
        self.value = value
        super().__init__()

    @jax.jit
    def evaluate(
        self, plasma_state: "PlasmaState", setup: Setup
    ) -> jnp.ndarray:
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


class ConstantDebyeTemp(Model):
    """
    A model of constant Debye Temperature.
    """

    allowed_keys = ["Debye temperature"]
    __name__ = "ConstantDebyeTemp"

    def __init__(self, value):
        self.value = value
        super().__init__()

    @jax.jit
    def evaluate(
        self, plasma_state: "PlasmaState", setup: Setup
    ) -> jnp.ndarray:
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
    def evaluate(
        self, plasma_state: "PlasmaState", setup: Setup
    ) -> jnp.ndarray:
        return static_structure_factors.T_Debye_Bohm_Staver(
            plasma_state.T_e,
            plasma_state.n_e,
            plasma_state.atomic_masses,
            plasma_state.Z_free,
        )


# Ionization Potential Depression Models
# ======================================


class ConstantIPD(Model):
    """
    A model that returns constant value for the IPD.
    """

    allowed_keys = ["ipd"]
    __name__ = "ConstantIPD"

    def __init__(self, value):
        self.value = value
        super().__init__()

    @jax.jit
    def evaluate(
        self, plasma_state: "PlasmaState", setup: Setup
    ) -> jnp.ndarray:
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


class DebyeHueckelIPD(Model):
    """
    Debye Hückel IPD Model

    See Also
    --------
    jaxrts.ipd.ipd_debye_hueckel
        Function used to calculate the IPD
    """

    allowed_keys = ["ipd"]
    __name__ = "DebyeHueckel"

    @jax.jit
    def evaluate(self, plasma_state: "PlasmaState", setup: Setup) -> Quantity:
        return ipd.ipd_debye_hueckel(
            plasma_state.Z_free,
            plasma_state.n_e,
            plasma_state.n_i,
            plasma_state.T_e,
            plasma_state.T_i,
        )


class StewartPyattIPD(Model):
    """
    Stewart Pyatt IPD Model.

    See Also
    --------
    jaxrts.ipd.ipd_stewart_pyatt
        Function used to calculate the IPD
    """

    allowed_keys = ["ipd"]
    __name__ = "StewartPyatt"

    @jax.jit
    def evaluate(self, plasma_state: "PlasmaState", setup: Setup) -> Quantity:
        return ipd.ipd_stewart_pyatt(
            plasma_state.Z_free,
            plasma_state.n_e,
            plasma_state.n_i,
            plasma_state.T_e,
            plasma_state.T_i,
        )


class IonSphereIPD(Model):
    """
    Ion Sphere IPD Model.

    See Also
    --------
    jaxrts.ipd.ipd_ion_sphere
        Function used to calculate the IPD
    """

    allowed_keys = ["ipd"]
    __name__ = "IonSphere"

    @jax.jit
    def evaluate(self, plasma_state: "PlasmaState", setup: Setup) -> Quantity:
        return ipd.ipd_ion_sphere(
            plasma_state.Z_free, plasma_state.n_e, plasma_state.n_i
        )


class EckerKroellIPD(Model):
    """
    Ecker Kröll IPD Model.

    See Also
    --------
    jaxrts.ipd.ipd_ecker_kroell
        Function used to calculate the IPD
    """

    allowed_keys = ["ipd"]
    __name__ = "EckerKroell"

    @jax.jit
    def evaluate(self, plasma_state: "PlasmaState", setup: Setup) -> Quantity:
        return ipd.ipd_ecker_kroell(
            plasma_state.Z_free,
            plasma_state.n_e,
            plasma_state.n_i,
            plasma_state.T_e,
            plasma_state.T_i,
        )


class PauliBlockingIPD(Model):
    """
    Pauli Blocking IPD Model.

    See Also
    --------
    jaxrts.ipd.ipd_pauli_blocking
        Function used to calculate the IPD
    """

    allowed_keys = ["ipd"]
    __name__ = "PauliBlocking"

    @jax.jit
    def evaluate(self, plasma_state: "PlasmaState", setup: Setup) -> Quantity:
        return ipd.ipd_pauli_blocking(
            plasma_state.Z_free,
            plasma_state.n_e,
            plasma_state.n_i,
            plasma_state.T_e,
            plasma_state.T_i,
        )


# Screening Length Models
# =======================


class DebyeHueckelScreeningLength(Model):
    """
    This is just the normal Debye Hückel screening length. Use the electron
    temperature for the known formula

    See also
    --------
    jaxrts.plasma_physics.Debye_Hueckel_screening_length
        The function used to calculate the screening length
    """

    allowed_keys = ["screening length"]
    __name__ = "DebyeHueckelScreeningLength"

    @jax.jit
    def evaluate(self, plasma_state: "PlasmaState", setup: Setup) -> Quantity:
        return plasma_physics.Debye_Hueckel_screening_length(
            plasma_state.n_e, plasma_state.T_e
        )


class Gericke2010ScreeningLength(Model):
    """
    Return the Debye-Hückel Debye screening length. Uses a 4th-power
    interpolation between electron and fermi temperature, as proposed by
    :cite:`Gericke.2010`

    See Also
    --------
    jaxrts.plasma_physics.temperature_interpolation:
        The function used for the temperature interpolation
    jaxrts.plasma_physics.Debye_Hueckel_screening_length
        The function used to calculate the screening length
    """

    allowed_keys = ["screening length"]
    __name__ = "Gericke2010ScreeningLength"

    @jax.jit
    def evaluate(self, plasma_state: "PlasmaState", setup: Setup) -> Quantity:
        T = plasma_physics.temperature_interpolation(
            plasma_state.n_e, plasma_state.T_e, 4
        )
        lam_DH = plasma_physics.Debye_Hueckel_screening_length(
            plasma_state.n_e, T
        )
        return lam_DH.to(ureg.angstrom)


class ArbitraryDegeneracyScreeningLength(Model):
    """
    A screening length valid for arbitrary degeneracy.

    See Also
    --------
    ipd.inverse_screening_length_e
        The function used to calculate the inverse of the screening length
    """

    allowed_keys = ["screening length"]
    __name__ = "ArbitraryDegeneracyScreeningLength"

    @jax.jit
    def evaluate(self, plasma_state: "PlasmaState", setup: Setup) -> Quantity:
        inverse_lam = ipd.inverse_screening_length_e(
            (1 * ureg.elementary_charge),
            plasma_state.n_e,
            plasma_state.T_e,
        )
        # inverse_lam will always have exactly one entry. But since we just
        # want the number and not an array, we take the first element.
        return (1 / inverse_lam[0]).to(ureg.angstrom)


class ConstantScreeningLength(Model):
    """
    A model that returns a constant screening length, given by a user.
    """

    allowed_keys = ["screening length"]
    __name__ = "ConstantScreeningLength"

    def __init__(self, value):
        self.value = value
        super().__init__()

    @jax.jit
    def evaluate(
        self, plasma_state: "PlasmaState", setup: Setup
    ) -> jnp.ndarray:
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


# Screening Models
# ================
#
# These models should return a `q`, the screening by free electrons which is
# relevant when calculating the Raighley weight.


class LinearResponseScreeningGericke2010(Model):
    """
    The screening density :math:`q` is calculated using a result from linear
    response, by

    .. math::

       q(k) = \\xi_{ee}^{RPA} V_{ei}(k)


    See :cite:`Wunsch.2011`, Eqn(5.22) and :cite:`Gericke.2010` Eqn(3).

    Requires an 'electron-ion' potential. (defaults to
    :py:class:`~KlimontovichKraeftPotential`).

    See Also
    --------
    jaxtrs.ion_feature.free_electron_susceptilibily
        Function used to calculate :math:`\\xi{ee}`
    """

    allowed_keys = ["screening"]
    __name__ = "LinearResponseScreeningGericke2010"

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        plasma_state.update_default_model(
            "electron-ion Potential",
            hnc_potentials.KlimontovichKraeftPotential(),
        )
        plasma_state["electron-ion Potential"].include_electrons = (
            "SpinAveraged"
        )

    @jax.jit
    def evaluate(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        *args,
        **kwargs,
    ) -> jnp.ndarray:

        # Use the Debye screening length for the screening cloud.
        kappa = 1 / plasma_state.screening_length
        xi = ion_feature.free_electron_susceptilibily_RPA(setup.k, kappa)
        Vei = plasma_state["electron-ion Potential"].full_k(
            plasma_state, to_array(setup.k)[jnp.newaxis]
        )
        q = (xi * Vei[-1, :-1]).to(ureg.dimensionless)
        # Screening vanishes if there are no free electrons
        q = jax.lax.cond(
            jnp.sum(plasma_state.Z_free) == 0,
            lambda: jnp.zeros(len(plasma_state.n_i))[:, jnp.newaxis]
            * ureg.dimensionless,
            lambda: q,
        )
        return q


class FiniteWavelengthScreening(Model):
    """
    Finite wavelenth screening as presented by :cite:`Chapman.2015b`.

    Should be identical to :py:class:`~.LinearResponseScreening`, if the
    free-free model is a RPA model.

    See also
    --------
    jaxrts.ion_feature.q_FiniteWLChapman2015
        The function used to calculate ``q``.
    """

    allowed_keys = ["screening"]
    __name__ = "FiniteWavelengthScreening"

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        plasma_state.update_default_model(
            "electron-ion Potential",
            hnc_potentials.KlimontovichKraeftPotential(),
        )
        plasma_state["electron-ion Potential"].include_electrons = (
            "SpinAveraged"
        )

    @jax.jit
    def evaluate(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        *args,
        **kwargs,
    ) -> jnp.ndarray:

        Vei = plasma_state["electron-ion Potential"].full_k(
            plasma_state, to_array(setup.k)[jnp.newaxis]
        )[-1, :-1]
        lfc = plasma_state["ee-lfc"].evaluate(plasma_state, setup)
        q = ion_feature.q_FiniteWLChapman2015(
            setup.k, Vei, plasma_state.T_e, plasma_state.n_e, lfc
        )
        q = jnp.real(q.m_as(ureg.dimensionless))
        # Screening vanishes if there are no free electrons
        q = jnpu.where(plasma_state.Z_free == 0, 0, q[:, 0])[:, jnp.newaxis]
        return q


class DebyeHueckelScreening(Model):
    """
    Debye Hueckel screening as presented by :cite:`Chapman.2015b`.

    See also
    --------
    jaxrts.ion_feature.q_DebyeHueckelChapman2015
        The function used to calculate ``q``.
    """

    allowed_keys = ["screening"]
    __name__ = "DebyeHueckelScreening"

    @jax.jit
    def evaluate(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        *args,
        **kwargs,
    ) -> jnp.ndarray:

        kappa = 1 / plasma_state.screening_length
        q = ion_feature.q_DebyeHueckelChapman2015(
            setup.k[jnp.newaxis],
            kappa,
            plasma_state.Z_free,
        )
        q = jnp.real(q.m_as(ureg.dimensionless))
        # Screening vanishes if there are no free electrons
        q = jnpu.where(plasma_state.Z_free == 0, 0, q[:, 0])[:, jnp.newaxis]
        return q


class LinearResponseScreening(Model):
    """
    The screening density :math:`q` is calculated using a result from linear
    response, by

    .. math::

       q(k) = \\xi_{ee} V_{ei}(k)


    See :cite:`Wunsch.2011`, Eqn(5.22) and :cite:`Gericke.2010` Eqn(3).

    Requires an 'electron-ion' potential. (defaults to
    :py:class:`~KlimontovichKraeftPotential`).

    Uses the :py:meth:`~.FreeFreeModel.susceptibility` method of the chosen
    Free Free model. If not free-free model is specified, set the default to
    :py:class:`~.RPA_DandreaFit`.
    """

    allowed_keys = ["screening"]
    __name__ = "LinearResponseScreening"

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        plasma_state.update_default_model(
            "electron-ion Potential",
            hnc_potentials.KlimontovichKraeftPotential(),
        )
        plasma_state["electron-ion Potential"].include_electrons = (
            "SpinAveraged"
        )
        plasma_state["free-free scattering"] = RPA_DandreaFit()

    @jax.jit
    def evaluate(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        *args,
        **kwargs,
    ) -> jnp.ndarray:

        xi = plasma_state["free-free scattering"].susceptibility(
            plasma_state, setup, 0 * ureg.electron_volt
        )
        Vei = plasma_state["electron-ion Potential"].full_k(
            plasma_state, to_array(setup.k)[jnp.newaxis]
        )
        q = xi * Vei[-1, :-1]
        q = jnp.real(q.m_as(ureg.dimensionless))
        # Screening vanishes if there are no free electrons
        q = jnpu.where(plasma_state.Z_free == 0, 0, q[:, 0])[:, jnp.newaxis]
        return q


class Gregori2004Screening(Model):
    """
    Calculating the screening from free electrons according to
    :cite:`Gregori.2004`.

    See Also
    --------
    jaxrts.ion_feature.q_Gregori2004
        Calculation of the screening by (quasi) free electrons
    """

    allowed_keys = ["screening"]
    __name__ = "Gregori2004Screening"

    @jax.jit
    def evaluate(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        q = ion_feature.q_Gregori2004(
            setup.k[jnp.newaxis],
            plasma_state.atomic_masses,
            plasma_state.n_e,
            plasma_state.T_e,
            plasma_state.T_e,
            plasma_state.Z_free,
        )[:, jnp.newaxis]
        q = jnp.real(q.m_as(ureg.dimensionless))
        # Screening vanishes if there are no free electrons
        q = jnpu.where(plasma_state.Z_free == 0, 0, q[:, 0])[:, jnp.newaxis]
        return q


# Electron-Electron Local Field Correction Models
# ===============================================
#


class ElectronicLFCGeldartVosko(Model):
    """
    Static local field correction model by Geldart and Vosko
    :cite:`Geldart.1966`

    See Also
    --------
    jaxrts.ee_localfieldcorrections.eelfc_geldartvosko
        Function used to calculate the LFC
    """

    allowed_keys = ["ee-lfc"]
    __name__ = "GeldartVosko Static LFC"

    @jax.jit
    def evaluate(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        return ee_localfieldcorrections.eelfc_geldartvosko(
            setup.k, plasma_state.T_e, plasma_state.n_e
        )

    @jax.jit
    def evaluate_fullk(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        k = setup.dispersion_corrected_k(plasma_state.n_e)
        return ee_localfieldcorrections.eelfc_geldartvosko(
            k, plasma_state.T_e, plasma_state.n_e
        )


class ElectronicLFCUtsumiIchimaru(Model):
    """
    Static local field correction model by Utsumi and Ichimaru
    :cite:`UtsumiIchimaru.1982`.

    See Also
    --------
    jaxrts.ee_localfieldcorrections.eelfc_utsumiichimaru
        Function used to calculate the LFC.
    """

    allowed_keys = ["ee-lfc"]
    __name__ = "UtsumiIchimaru Static LFC"

    @jax.jit
    def evaluate(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        return ee_localfieldcorrections.eelfc_utsumiichimaru(
            setup.k, plasma_state.T_e, plasma_state.n_e
        )

    @jax.jit
    def evaluate_fullk(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        k = setup.dispersion_corrected_k(plasma_state.n_e)
        return ee_localfieldcorrections.eelfc_utsumiichimaru(
            k, plasma_state.T_e, plasma_state.n_e
        )


class ElectronicLFCDornheimAnalyticalInterp(Model):
    """
    Static local field correction model by Dornheim et al
    :cite:`Dornheim.2021`. Their model is an analytical interpolation of
    ab-initio PIMC simulations.

    See Also
    --------
    jaxrts.ee_localfieldcorrections.eelfc_dornheim2021
        Function used to calculate the LFC.
    """

    allowed_keys = ["ee-lfc"]
    __name__ = "ElectronicLFCDornheimAnalyticalInterp"

    @jax.jit
    def evaluate(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        return ee_localfieldcorrections.eelfc_dornheim2021(
            setup.k, plasma_state.T_e, plasma_state.n_e
        )

    @jax.jit
    def evaluate_fullk(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        k = setup.dispersion_corrected_k(plasma_state.n_e)
        return ee_localfieldcorrections.eelfc_dornheim2021(
            k, plasma_state.T_e, plasma_state.n_e
        )


class ElectronicLFCStaticInterpolation(Model):
    """
    Static local field correction model that interpolates between the
    high-degeneracy result by Farid :cite:`Farid.1993` and the Geldart result
    :cite:`Geldart.1966`. See, e.g. :cite:`Fortmann.2010`.

    See Also
    --------
    jaxrts.ee_localfieldcorrections.eelfc_interpolationgregori_farid
        Function used to calculate the LFC.
    """

    allowed_keys = ["ee-lfc"]
    __name__ = "Static Interpolation"

    @jax.jit
    def evaluate(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        return ee_localfieldcorrections.eelfc_interpolationgregori_farid(
            setup.k, plasma_state.T_e, plasma_state.n_e
        )

    @jax.jit
    def evaluate_fullk(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        k = setup.dispersion_corrected_k(plasma_state.n_e)
        return ee_localfieldcorrections.eelfc_interpolationgregori_farid(
            k, plasma_state.T_e, plasma_state.n_e
        )


class ElectronicLFCConstant(Model):
    """
    A constant local field correction which can be defined by the user.
    """

    allowed_keys = ["ee-lfc"]
    __name__ = "ElectronicLFCConstant"

    def __init__(self, value):
        self.value = value
        super().__init__()

    @jax.jit
    def evaluate(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        return self.value

    @jax.jit
    def evaluate_fullk(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
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


# BM S_ii models
# ===============


def averagePlasmaState(state: "PlasmaState") -> "PlasmaState":
    """
    Create an average plasma state that shares the models of the original.
    """
    mean_Z = jnpu.sum(state.Z_A * state.number_fraction)[jnp.newaxis]
    mean_Z_free = jnpu.sum(state.Z_free * state.number_fraction)[jnp.newaxis]
    mean_mass = jnpu.sum(state.atomic_masses * state.number_fraction)

    mean_ion_T = jnpu.sum(state.T_i * state.number_fraction)[jnp.newaxis]
    mean_rho = jnpu.sum(state.mass_density * state.number_fraction)[
        jnp.newaxis
    ]

    mix_element = MixElement(mean_Z, mean_mass)

    newState = deepcopy(state)
    newState.ions = [mix_element]
    newState.Z_free = mean_Z_free
    newState.mass_density = mean_rho
    newState.T_i = mean_ion_T
    return newState


class Sum_Sii(Model):
    """
    This model sums up all S_ab from the HNC and multiplies it with ``sqrt(x_a
    * x_b)``. While it is obviously ok for a single-species plasma, multiple
    species might not be treated correctly.
    """

    allowed_keys = ["BM S_ii"]
    __name__ = "Sum_Sii"

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        plasma_state.update_default_model(
            "ionic scattering", OnePotentialHNCIonFeat()
        )

    def evaluate(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        *args,
        **kwargs,
    ):

        S_ab = plasma_state["ionic scattering"].S_ii(plasma_state, setup)
        x = plasma_state.number_fraction
        # Add the contributions from all pairs
        S_ii = 0

        def add_Sii(a, b):
            return (jnpu.sqrt(x[a] * x[b]) * S_ab[a, b]).m_as(
                ureg.dimensionless
            )[jnp.newaxis]

        # The W_R is calculated as a sum over all combinations of a_b
        ion_spec1, ion_spec2 = jnp.meshgrid(
            jnp.arange(plasma_state.nions),
            jnp.arange(plasma_state.nions),
        )
        for a, b in zip(ion_spec1.flatten(), ion_spec2.flatten()):
            S_ii += add_Sii(a, b)
        return S_ii


class AverageAtom_Sii(Model):
    """
    This model performs a HNC calculation, assuming one average atom with a
    given, average charge state. While it might lead to reasonable results,
    this is not tested and takes some computation time.
    """

    allowed_keys = ["BM S_ii"]
    __name__ = "AverageAtom_Sii"

    def __init__(
        self,
        rmin: Quantity = 0.001 * ureg.a_0,
        rmax: Quantity = 100 * ureg.a_0,
        pot: int = 14,
    ) -> None:
        #: The minimal radius for evaluating the potentials.
        self.r_min: Quantity = rmin
        #: The maximal radius for evaluating the potentials.
        self.r_max: Quantity = rmax
        #: The exponent (``2 ** pot``), setting the number of points in ``r``
        #: or ``k`` to evaluate.
        self.pot: int = pot
        super().__init__()

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        super().prepare(plasma_state, key)
        plasma_state.update_default_model(
            "ion-ion Potential", hnc_potentials.DebyeHueckelPotential()
        )
        plasma_state["ion-ion Potential"].include_electrons = "off"
        plasma_state.update_default_model(
            "ionic scattering", OnePotentialHNCIonFeat()
        )

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
    def evaluate(
        self, plasma_state: "PlasmaState", setup: Setup
    ) -> jnp.ndarray:
        # Average the Plasma State
        aaState = averagePlasmaState(plasma_state)

        # Prepare the Potentials
        # ----------------------

        # Populate the potential with a full ion potential, for starters
        V_s_r = aaState["ion-ion Potential"].short_r(aaState, self.r)
        V_l_k = aaState["ion-ion Potential"].long_k(aaState, self.k)

        # Calculate g_ab in the HNC Approach
        # ----------------------------------
        T = aaState["ion-ion Potential"].T(aaState)
        n = aaState.n_i
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
        # Return the Sii with the correct shape
        return S_ab[0, 0]

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


# BM V_eiS models
# ===============


class BM_V_eiSModel(Model):
    """
    These models implement potentials which can be when calculating the Born
    collision frequencies in :py:class:`~.BornMermin` and derived free-free
    scattering models.
    """

    @abc.abstractmethod
    def V(self, plasma_state: "PlasmaState", k: Quantity) -> jnp.ndarray: ...

    def evaluate(
        self,
        plasma_state: "PlasmaState",
        setup: Setup,
        *args,
        **kwargs,
    ):
        return self.V(plasma_state, setup.k)


class DebyeHueckel_BM_V(BM_V_eiSModel):
    """
    A Debye Hückel potential, using the
    :py:meth:`jaxrts.plasmastate.PlasmaState.screening_length` method to get
    the screening length.

    See Also
    --------

    jaxrts.free_free.statically_screened_ie_debye_potential
        The Potential used
    """

    allowed_keys = ["BM V_eiS"]
    __name__ = "DebyeHueckel_BM_V"

    @jax.jit
    def V(
        self,
        plasma_state: "PlasmaState",
        k: Quantity,
        *args,
        **kwargs,
    ):
        kappa = 1 / plasma_state.screening_length
        return free_free.statically_screened_ie_debye_potential(
            k,
            kappa,
            jnp.sum(plasma_state.number_fraction * plasma_state.Z_free),
        )


class FiniteWavelength_BM_V(BM_V_eiSModel):
    """
    Uses finite wavelength screening to screen the bare Coulomb potential,
    i.e., :math:`V_{s} = \\frac{V_\\mathrm{Coulomb}}{\\vareps_{RPA}(k, E=0)}`

    We use the pure RPA result to calculate the dielectric function, and use
    the :cite:`Dandrea.1986` fitting formula.

    See Also
    --------

    jaxrts.plasma_physics.coulomb_potential_fourier
        The Coulomb potential in k space
    jaxrts.free_free.dielectric_function_RPA_Dandrea1986
        The function used to calculate the dielectric function in random phase
        approximation using numerically inexpensive fitting functions.
    """

    allowed_keys = ["BM V_eiS"]
    __name__ = "FiniteWavelength_BM_V"

    @jax.jit
    def V(
        self,
        plasma_state: "PlasmaState",
        k: Quantity,
    ):
        V = plasma_physics.coulomb_potential_fourier(
            jnpu.sum(plasma_state.number_fraction * plasma_state.Z_free),
            -1,
            k,
        )
        eps0 = free_free.dielectric_function_RPA_Dandrea1986(
            k,
            0 * ureg.electron_volt,
            plasma_state.T_e,
            plasma_state.n_e,
        )
        return V / eps0


_all_models = [
    ArbitraryDegeneracyScreeningLength,
    ArkhipovIonFeat,
    AverageAtom_Sii,
    BohmStaver,
    BornMermin,
    BornMerminFull,
    BornMermin_Fit,
    BornMermin_Fortmann,
    ConstantChemPotential,
    ConstantDebyeTemp,
    ConstantIPD,
    ConstantScreeningLength,
    DebyeHueckelIPD,
    DebyeHueckelScreening,
    DebyeHueckelScreeningLength,
    DebyeHueckel_BM_V,
    DebyeWallerSolid,
    DetailedBalance,
    EckerKroellIPD,
    ElectronicLFCConstant,
    ElectronicLFCDornheimAnalyticalInterp,
    ElectronicLFCGeldartVosko,
    ElectronicLFCStaticInterpolation,
    ElectronicLFCUtsumiIchimaru,
    FiniteWavelengthScreening,
    FiniteWavelength_BM_V,
    FixedSii,
    Gericke2010ScreeningLength,
    Gregori2003IonFeat,
    Gregori2004Screening,
    Gregori2006IonFeat,
    IchimaruChemPotential,
    IonSphereIPD,
    LinearResponseScreening,
    LinearResponseScreeningGericke2010,
    Model,
    Neglect,
    OnePotentialHNCIonFeat,
    PauliBlockingIPD,
    PaulingFormFactors,
    PeakCollection,
    QCSalpeterApproximation,
    RPA_DandreaFit,
    RPA_NoDamping,
    ScatteringModel,
    SchumacherImpulse,
    SchumacherImpulseFitRk,
    Sum_Sii,
    StewartPyattIPD,
    ThreePotentialHNCIonFeat,
]

for _model in _all_models:
    jax.tree_util.register_pytree_node(
        _model,
        _model._tree_flatten,
        _model._tree_unflatten,
    )
