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
from .setup import (
    Setup,
    convolve_stucture_factor_with_instrument,
    dispersion_corrected_k,
    get_probe_setup,
)
from .plasma_physics import noninteracting_susceptibility_from_eps_RPA
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
    ipd,
    ee_localfieldcorrections,
)


from typing import TYPE_CHECKING

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
        Please log assumtpions, properly
        """
        pass

    def check(self, plasma_state: "PlasmaState") -> None:
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
        instument function.
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
            return 0.0 * (1 * ureg.electron_volt)


# ion-feature
# -----------
class IonFeatModel(Model):
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

        def dont_add_wrt(a, b):
            return jnp.array([0.0])

        # The W_R is calculated as a sum over all combinations of a_b
        ion_spec1, ion_spec2 = jnp.meshgrid(
            jnp.arange(plasma_state.nions),
            jnp.arange(plasma_state.nions),
        )
        for a, b in zip(ion_spec1.flatten(), ion_spec2.flatten()):
            w_R += jax.lax.cond(a <= b, add_wrt, dont_add_wrt, a, b)
        return w_R

    @jax.jit
    def evaluate(
        self, plasma_state: "PlasmaState", setup: Setup
    ) -> jnp.ndarray:
        w_R = self.Rayleigh_weight(plasma_state, setup)
        res = w_R * setup.instrument(
            (setup.measured_energy - setup.energy) / ureg.hbar
        )
        return res


class ArkhipovIonFeat(IonFeatModel):
    """
    Model for the ion feature of the scatting, presented in
    :cite:`Arkhipov.1998` and :cite:`Arkhipov.2000`.

    The structure factors are obtained by using an effective potential
    (pseudopotential) model of the particle interaction of semiclassical
    two-component plasmas with a single temperature :math:`T`. We use the
    electron temperature ``state.T_e`` of the PlasmaState modelled. The authors
    take into account both quantum and collective effects.

    Requires a 'form-factors' model (defaults to
    :py:class:`~PaulingFormFactors`) and a 'screening' model (defaults to
    :py:class:`Gregori2004Screenig`).

    See Also
    --------

    jaxrts.static_structure_factors.S_ii_AD
        Calculation of the static ion ion stucture factor given by
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

    # @jax.jit
    # def Rayleigh_weight(
    #     self, plasma_state: "PlasmaState", setup: Setup
    # ) -> jnp.ndarray:
    #     fi = plasma_state["form-factors"].evaluate(plasma_state, setup)
    #     population = electron_distribution_ionized_state(plasma_state.Z_core)[
    #         :, jnp.newaxis
    #     ]
    #     q = plasma_state.evaluate("screening", setup)
    #     f = jnp.sum(fi * population)
    #     S_ii = self.S_ii(plasma_state, setup)
    #     w_R = jnp.abs(f + q.m_as(ureg.dimensionless)) ** 2 * S_ii
    #     return w_R


class Gregori2003IonFeat(IonFeatModel):
    """
    Model for the ion feature of the scatting, presented in
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


class OnePotentialHNCIonFeat(IonFeatModel):
    """
    Model for the ion feature using a calculating all :math:`S_{ab}` in the
    Hypernetted Chain approximation.

    In contrast to :py:class:`~.ThreePotentialHNCIonFeat`, this models
    calculates only the ion-ion Structure factors, and is neglecting the
    electron-contibutions. Hence, screening is not included, automatically, but
    has to be provided as an additional `screening`.


    Requires an 'ion-ion' (defaults to :py:class:`~DebyeHuckelPotential`) and a
    `screening` model (default:
    :py::class:`~.LinearResponseScreeningGericke2010`.
    Further requires a 'form-factors' model (defaults to
    :py:class:`~PaulingFormFactors`).
    """

    __name__ = "OnePotentialHNCIonFeat"

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

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        super().prepare(plasma_state, key)
        plasma_state.update_default_model(
            "ion-ion Potential", hnc_potentials.DebyeHuckelPotential()
        )
        plasma_state["ion-ion Potential"].include_electrons = False

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
          (defaults to :py:class:`~DebyeHuckelPotential`).
        - an 'electron-ion Potential' The orange entries in the picture below
          (defaults to :py:class:`~KlimontovichKraeftPotential`).
        - an 'electron-electron Potential' The red entries in the picutre below
          (defaults to :py:class:`~KelbgPotental`).

    .. image:: ../../_images/ThreePotentialHNC.svg
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
    ) -> None:
        #: The minmal radius for evaluating the potentials.
        self.r_min: Quantity = rmin
        #: The maximal radius for evaluating the potentials.
        self.r_max: Quantity = rmax
        #: The exponent (``2 ** pot``), setting the number of points in ``r``
        #: or ``k`` to evaluate.
        self.pot: int = pot
        super().__init__()

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        # Overwrite the old prepare function, here, because we don't need a
        # screening model
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
        Here, we have to calculate teh Rayleigh weight different than the
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

        def dont_add_wrt(a, b):
            return jnp.array([0.0])

        for a, b in zip(ion_spec1.flatten(), ion_spec2.flatten()):
            w_R += jax.lax.cond(a <= b, add_wrt, dont_add_wrt, a, b)
        # Scale the instrument function directly with w_R
        return w_R

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
#
# These models also give a susceptibility method, which might be used by
# screening models, later.


class FreeFreeModel(ScatteringModel):
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
        k = dispersion_corrected_k(setup, plasma_state.n_e)
        See_0 = free_free.S0_ee_Salpeter(
            k,
            plasma_state.T_e,
            plasma_state.n_e,
            setup.measured_energy - setup.energy,
            plasma_state["ee-lfc"].evaluate(plasma_state, setup),
        )

        return See_0 * jnp.sum(
            plasma_state.Z_free * plasma_state.number_fraction
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
        k = dispersion_corrected_k(setup, plasma_state.n_e)
        See_0 = free_free.S0_ee_RPA_no_damping(
            k,
            plasma_state.T_e,
            plasma_state.n_e,
            setup.measured_energy - setup.energy,
            mu,
            plasma_state["ee-lfc"].evaluate_fullk(plasma_state, setup),
        )

        return See_0 * jnp.sum(
            plasma_state.Z_free * plasma_state.number_fraction
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

    __name__ = "PRA_DandreaFit"

    @jax.jit
    def evaluate_raw(
        self, plasma_state: "PlasmaState", setup: Setup, *args, **kwargs
    ) -> jnp.ndarray:
        k = dispersion_corrected_k(setup, plasma_state.n_e)
        See_0 = free_free.S0_ee_RPA_Dandrea(
            k,
            plasma_state.T_e,
            plasma_state.n_e,
            setup.measured_energy - setup.energy,
            plasma_state["ee-lfc"].evaluate_fullk(plasma_state, setup),
        )

        return See_0 * jnp.sum(
            plasma_state.Z_free * plasma_state.number_fraction
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

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        plasma_state.update_default_model(
            "chemical potential", IchimaruChemPotential()
        )
        plasma_state.update_default_model("BM V_eiS", FiniteWavelength_BM_V())

    def check(self, plasma_state: "PlasmaState") -> None:
        if len(plasma_state) > 1:
            logger.critical(
                f"'{self.__name__}' is only implemented for a one-component plasma"  # noqa: E501
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
            return jnpu.diagonal(
                plasma_state["ionic scattering"].S_ii(
                    plasma_state, probe_setup
                )
            )

        @jax.tree_util.Partial
        def V_eiS(k):
            return plasma_state["BM V_eiS"].V(plasma_state, k)

        mu = plasma_state["chemical potential"].evaluate(plasma_state, setup)
        k = dispersion_corrected_k(setup, plasma_state.n_e)
        See_0 = free_free.S0_ee_BMA(
            k,
            plasma_state.T_e,
            mu,
            S_ii,
            V_eiS,
            plasma_state.n_e,
            plasma_state.Z_free,
            setup.measured_energy - setup.energy,
            plasma_state["ee-lfc"].evaluate(plasma_state, setup),
        )
        return See_0 * plasma_state.Z_free

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
            return jnpu.diagonal(
                plasma_state["ionic scattering"].S_ii(
                    plasma_state, probe_setup
                )
            )

        @jax.tree_util.Partial
        def V_eiS(k):
            return plasma_state["BM V_eiS"].V(plasma_state, k)

        mu = plasma_state["chemical potential"].evaluate(plasma_state, setup)
        k = setup.k

        def chi(energy):
            eps = free_free.dielectric_function_BMA_full(
                k,
                energy,
                mu,
                plasma_state.T_e,
                plasma_state.n_e,
                S_ii,
                V_eiS,
                plasma_state.Z_free,
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


class BornMermin(FreeFreeModel):
    """
    Model of the free-free scattering, based on the Born Mermin Approximation
    (:cite:`Mermin.1970`).
    Uses the Chapman interpolation which allows for a faster computation of the
    free-free scattering compared to :py:class:`~.BornMerminFull`, by sampleing
    at the probing frequency at :py:attr:`~.no_of_freq` points and
    interpolating between them, after.

    The number of frequencies defaults to 20. To change it, just change the
    attribute of this model after initializing it. i.e.

    >>> state["free-free scattering"] = jaxrts.models.BornMermin
    >>> state["free-free scattering"].no_of_freq = 10

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

    def __init__(self, no_of_freq: int = 20) -> None:
        super().__init__()
        self.no_of_freq: int = no_of_freq

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        plasma_state.update_default_model(
            "chemical potential", IchimaruChemPotential()
        )
        plasma_state.update_default_model("BM V_eiS", FiniteWavelength_BM_V())

    def check(self, plasma_state: "PlasmaState") -> None:
        if len(plasma_state) > 1:
            logger.critical(
                f"'{self.__name__}' is only implemented for a one-component plasma"  # noqa: E501
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
            return jnpu.diagonal(
                plasma_state["ionic scattering"].S_ii(
                    plasma_state, probe_setup
                )
            )

        @jax.tree_util.Partial
        def V_eiS(k):
            return plasma_state["BM V_eiS"].V(plasma_state, k)

        mu = plasma_state["chemical potential"].evaluate(plasma_state, setup)
        k = dispersion_corrected_k(setup, plasma_state.n_e)
        See_0 = free_free.S0_ee_BMA_chapman_interp(
            k,
            plasma_state.T_e,
            mu,
            S_ii,
            V_eiS,
            plasma_state.n_e,
            plasma_state.Z_free,
            setup.measured_energy - setup.energy,
            plasma_state["ee-lfc"].evaluate(plasma_state, setup),
            self.no_of_freq,
        )
        return See_0 * plasma_state.Z_free

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
            return jnpu.diagonal(
                plasma_state["ionic scattering"].S_ii(
                    plasma_state, probe_setup
                )
            )

        @jax.tree_util.Partial
        def V_eiS(k):
            probe_setup = get_probe_setup(k, setup)
            return plasma_state["BM V_eiS"].evaluate(plasma_state, probe_setup)

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
                plasma_state.Z_free,
                self.no_of_freq,
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
            self.no_of_freq,
        )  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.model_key, obj.sample_points, obj.no_of_freq = aux_data

        return obj


class BornMermin_Fit(FreeFreeModel):
    """
    Model of the free-free scattering, based on the Born Mermin Approximation
    (:cite:`Mermin.1970`).
    Identical to :py:class:`~.BornMermin`, but uses the Dandrea
    fit (:cite:`Dandrea.1986`), rather than numerically calculating the
    un-damped RPA, numerically. However, the damped RPA is still evaluated
    using the integral.

    The number of frequencies defaults to 20. To change it, just change the
    attribute of this model after initializing it. i.e.

    >>> state["free-free scattering"] = jaxrts.models.BornMermin_Fit
    >>> state["free-free scattering"].no_of_freq = 10

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

    def __init__(self, no_of_freq: int = 20) -> None:
        super().__init__()
        self.no_of_freq: int = no_of_freq

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        plasma_state.update_default_model(
            "chemical potential", IchimaruChemPotential()
        )
        plasma_state.update_default_model("BM V_eiS", FiniteWavelength_BM_V())

    def check(self, plasma_state: "PlasmaState") -> None:
        if len(plasma_state) > 1:
            logger.critical(
                f"'{self.__name__}' is only implemented for a one-component plasma"  # noqa: E501
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
            return jnpu.diagonal(
                plasma_state["ionic scattering"].S_ii(
                    plasma_state, probe_setup
                )
            )

        @jax.tree_util.Partial
        def V_eiS(k):
            return plasma_state["BM V_eiS"].V(plasma_state, k)

        mu = plasma_state["chemical potential"].evaluate(plasma_state, setup)
        k = dispersion_corrected_k(setup, plasma_state.n_e)
        See_0 = free_free.S0_ee_BMA_chapman_interpFit(
            k,
            plasma_state.T_e,
            mu,
            S_ii,
            V_eiS,
            plasma_state.n_e,
            plasma_state.Z_free,
            setup.measured_energy - setup.energy,
            plasma_state["ee-lfc"].evaluate(plasma_state, setup),
            self.no_of_freq,
        )
        return See_0 * plasma_state.Z_free

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
            return jnpu.diagonal(
                plasma_state["ionic scattering"].S_ii(
                    plasma_state, probe_setup
                )
            )

        @jax.tree_util.Partial
        def V_eiS(k):
            return plasma_state["BM V_eiS"].V(plasma_state, k)

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
                plasma_state.Z_free,
                self.no_of_freq,
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
            self.no_of_freq,
        )  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.model_key, obj.sample_points, obj.no_of_freq = aux_data

        return obj


class BornMermin_Fortmann(FreeFreeModel):
    """
    Model of the free-free scattering, based on the Born Mermin Approximation
    (:cite:`Mermin.1970`).
    Uses the same collision frequency as :py:class:`~.BornMermin_Fit`
    (including the :cite:`Dandrea.1986` fit), but uses a rigorous
    implementation of the local field correction, proposed by
    :cite:`Fortmann.2010`.

    The number of frequencies defaults to 20. To change it, just change the
    attribute of this model after initializing it. i.e.

    >>> state["free-free scattering"] = jaxrts.models.BornMermin_Fortmann
    >>> state["free-free scattering"].no_of_freq = 10

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

    def __init__(self, no_of_freq: int = 20) -> None:
        super().__init__()
        self.no_of_freq: int = no_of_freq

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        plasma_state.update_default_model(
            "chemical potential", IchimaruChemPotential()
        )
        plasma_state.update_default_model("BM V_eiS", FiniteWavelength_BM_V())

    def check(self, plasma_state: "PlasmaState") -> None:
        if len(plasma_state) > 1:
            logger.critical(
                f"'{self.__name__}' is only implemented for a one-component plasma"  # noqa: E501
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
            return jnpu.diagonal(
                plasma_state["ionic scattering"].S_ii(
                    plasma_state, probe_setup
                )
            )

        @jax.tree_util.Partial
        def V_eiS(k):
            return plasma_state["BM V_eiS"].V(plasma_state, k)

        mu = plasma_state["chemical potential"].evaluate(plasma_state, setup)
        k = dispersion_corrected_k(setup, plasma_state.n_e)
        See_0 = free_free.S0_ee_BMA_Fortmann(
            k,
            plasma_state.T_e,
            mu,
            S_ii,
            V_eiS,
            plasma_state.n_e,
            plasma_state.Z_free,
            setup.measured_energy - setup.energy,
            plasma_state["ee-lfc"].evaluate(plasma_state, setup),
            self.no_of_freq,
        )
        return See_0 * plasma_state.Z_free

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
            return jnpu.diagonal(
                plasma_state["ionic scattering"].S_ii(
                    plasma_state, probe_setup
                )
            )

        @jax.tree_util.Partial
        def V_eiS(k):
            return plasma_state["BM V_eiS"].V(plasma_state, k)

        mu = plasma_state["chemical potential"].evaluate(plasma_state, setup)
        k = setup.k

        xi = free_free.susceptibility_BMA_Fortmann(
            k,
            E,
            mu,
            plasma_state.T_e,
            plasma_state.n_e,
            S_ii,
            V_eiS,
            plasma_state.Z_free,
            plasma_state["ee-lfc"].evaluate(plasma_state, setup),
            self.no_of_freq,
        )
        return xi

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
        k = dispersion_corrected_k(setup, plasma_state.n_e)
        omega_0 = setup.energy / ureg.hbar
        omega = omega_0 - setup.measured_energy / ureg.hbar
        x = plasma_state.number_fraction

        out = 0 * ureg.second
        for idx in range(plasma_state.nions):
            Z_c = plasma_state.Z_core[idx]
            E_b = plasma_state.ions[
                idx
            ].binding_energies + plasma_state.models["ipd"].evaluate(
                plasma_state, None
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
                fi = plasma_state["form-factors"].evaluate(plasma_state, setup)
                # Because we did restict ourselfs to the first ion, we have to
                # add a dimension to population, here.
                new_r_k = (
                    1 - jnp.sum(population[:, jnp.newaxis] * (fi) ** 2) / Z_c
                )
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
            out += sbe * Z_c * x[idx]
        return jnpu.where(
            jnp.isnan(out.m_as(ureg.second)), 0 * ureg.second, out
        )

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
        self, plasma_state: "PlasmaState", setup: Setup
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
    def evaluate(
        self, plasma_state: "PlasmaState", setup: Setup
    ) -> jnp.ndarray:
        Zstar = form_factors.pauling_effective_charge(plasma_state.Z_A)
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


class IdealElectronChemPotential(Model):
    """
    A fitting formula for the chemical potential of a plasma between the
    classical and the quantum regime, given by :cite:`Gregori.2003`.
    Uses :py:func:`jaxrts.plasma_physics.chem_pot_interpolation`.
    """

    __name__ = "IdealElectronGregoriChemPotential"
    allowed_keys = ["chemical potential"]

    @jax.jit
    def evaluate(
        self, plasma_state: "PlasmaState", setup: Setup
    ) -> jnp.ndarray:
        return


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
    A model that returns an empty with zeros in (units of seconds) for every
    energy probed.
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

    allowed_keys = ["ipd"]
    __name__ = "IonSphere"

    @jax.jit
    def evaluate(self, plasma_state: "PlasmaState", setup: Setup) -> Quantity:
        return ipd.ipd_ion_sphere(
            plasma_state.Z_free, plasma_state.n_e, plasma_state.n_i
        )


class EckerKroellIPD(Model):

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
    jaxrts.plasma_physics.Debye_Huckel_screening_length
        The function used to calculate the screening length
    """

    allowed_keys = ["screening length"]
    __name__ = " DebyeHueckelScreeningLength"

    @jax.jit
    def evaluate(self, plasma_state: "PlasmaState", setup: Setup) -> Quantity:
        return plasma_physics.Debye_Huckel_screening_length(
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
    jaxrts.plasma_physics.Debye_Huckel_screening_length
        The function used to calculate the screening length
    """

    allowed_keys = ["screening length"]
    __name__ = "Gericke2010ScreeningLength"

    @jax.jit
    def evaluate(self, plasma_state: "PlasmaState", setup: Setup) -> Quantity:
        T = plasma_physics.temperature_interpolation(
            plasma_state.n_e, plasma_state.T_e, 4
        )
        lam_DH = plasma_physics.Debye_Huckel_screening_length(
            plasma_state.n_e, T
        )
        return lam_DH.to(ureg.angstrom)


class ArbitraryDegeneracyScreeningLength(Model):
    """
    A screening length valid for arbitraty degeneracy.

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
            jnpu.mean(plasma_state.Z_free * plasma_state.n_i)
            / jnpu.mean(plasma_state.n_i)
            * (1 * ureg.elementary_charge),
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
    allowed_keys = ["screening"]
    __name__ = "LinearResponseScreeningGericke2010"

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

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        plasma_state.update_default_model(
            "electron-ion Potential",
            hnc_potentials.KlimontovichKraeftPotential(),
        )
        plasma_state["electron-ion Potential"].include_electrons = True

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
        q = xi * Vei[-1, :-1]
        return q


class FiniteWavelengthScreening(Model):
    allowed_keys = ["screening"]
    __name__ = "FiniteWavelengthScreening"

    """
    The screening density :math:`q` is calculated using a result from linear
    response, by

    .. math::

       q(k) = \\xi_{ee} V_{ei}(k)


    See :cite:`Wunsch.2011`, Eqn(5.22) and :cite:`Gericke.2010` Eqn(3).

    Requires an 'electron-ion' potential. (defaults to
    :py:class:`~KlimontovichKraeftPotential`).

    Uses the :py:meth:`~.FreeFreeModel.susceptibility` method of the chosen
    Free Free model.
    """

    def prepare(self, plasma_state: "PlasmaState", key: str) -> None:
        plasma_state.update_default_model(
            "electron-ion Potential",
            hnc_potentials.KlimontovichKraeftPotential(),
        )
        plasma_state["electron-ion Potential"].include_electrons = True

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
        return jnp.real(q.m_as(ureg.dimensionless))


class Gregori2004Screening(Model):
    allowed_keys = ["screening"]
    __name__ = "Gregori2004Screenig"
    """
    Calculating the screening from free electrons according to
    :cite:`Gregori.2004`.

    See Also
    --------
    jaxrts.ion_feature.q_Gregori2004
        Calculation of the screening by (quasi) free electrons
    """

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
        return jnp.real(q.m_as(ureg.dimensionless))


# Electron-Electron Local Field Correction Models
# ===============================================
#


class ElectronicLFCGeldartVosko(Model):
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
        k = dispersion_corrected_k(setup, plasma_state.n_e)
        return ee_localfieldcorrections.eelfc_geldartvosko(
            k, plasma_state.T_e, plasma_state.n_e
        )


class ElectronicLFCUtsumiIchimaru(Model):
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
        k = dispersion_corrected_k(setup, plasma_state.n_e)
        return ee_localfieldcorrections.eelfc_utsumiichimaru(
            k, plasma_state.T_e, plasma_state.n_e
        )


class ElectronicLFCStaticInterpolation(Model):
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
        k = dispersion_corrected_k(setup, plasma_state.n_e)
        return ee_localfieldcorrections.eelfc_interpolationgregori_farid(
            k, plasma_state.T_e, plasma_state.n_e
        )


class ElectronicLFCConstant(Model):
    allowed_keys = ["ee-lfc"]
    __name__ = "None"

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


# BM V_eiS models
# ===============


class BM_V_eiSModel(Model):
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
    ElectronicLFCUtsumiIchimaru,
    ElectronicLFCGeldartVosko,
    ElectronicLFCConstant,
    ElectronicLFCStaticInterpolation,
    ArbitraryDegeneracyScreeningLength,
    ArkhipovIonFeat,
    BohmStaver,
    BornMerminFull,
    BornMermin,
    BornMermin_Fit,
    BornMermin_Fortmann,
    ConstantChemPotential,
    ConstantIPD,
    ConstantScreeningLength,
    DebyeHueckel_BM_V,
    DebyeHueckelIPD,
    DebyeHueckelScreeningLength,
    DetailedBalance,
    EckerKroellIPD,
    FiniteWavelengthScreening,
    FiniteWavelength_BM_V,
    Gericke2010ScreeningLength,
    Gregori2003IonFeat,
    Gregori2004Screening,
    Gregori2006IonFeat,
    IchimaruChemPotential,
    IonSphereIPD,
    LinearResponseScreeningGericke2010,
    Model,
    Neglect,
    OnePotentialHNCIonFeat,
    PauliBlockingIPD,
    PaulingFormFactors,
    QCSalpeterApproximation,
    RPA_DandreaFit,
    RPA_NoDamping,
    ScatteringModel,
    SchumacherImpulse,
    StewartPyattIPD,
    ThreePotentialHNCIonFeat,
]

for model in _all_models:
    jax.tree_util.register_pytree_node(
        model,
        model._tree_flatten,
        model._tree_unflatten,
    )
