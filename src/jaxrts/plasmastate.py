import logging
from abc import ABCMeta
from copy import deepcopy

import jax
import jpu.numpy as jnpu
import numpy as np
from jax import numpy as jnp

from .elements import Element
from .helpers import JittableDict
from .models import DebyeHueckelScreeningLength, ElectronicLFCConstant
from .plasma_physics import fermi_energy, wiegner_seitz_radius
from .setup import Setup
from .units import Quantity, to_array, ureg

logger = logging.getLogger(__name__)


class PlasmaState:

    def __init__(
        self,
        ions: list[Element],
        Z_free: list | Quantity,
        mass_density: list | Quantity,
        T_e: Quantity,
        T_i: list | Quantity | None = None,
    ):

        assert (len(ions) == len(Z_free)) and (
            len(ions) == len(mass_density)
        ), "WARNING: Input parameters should be the same shape as <ions>!"
        if T_i is not None:
            assert len(ions) == len(
                T_i
            ), "WARNING: Input parameters should be the same shape as <ions>!"

        self.ions = ions

        # Define charge configuration
        self.Z_free = to_array(Z_free)

        self.mass_density = to_array(mass_density)

        if (
            isinstance(T_e, list)
            or isinstance(T_e.magnitude, jnp.ndarray)
            and len(T_e.shape) == 1
        ):
            self.T_e = T_e[0]
        else:
            self.T_e = T_e

        if T_i is None:
            T_i = T_e * jnp.ones(self.nions)
        self.T_i = to_array(T_i)
        self.models = JittableDict()
        self._overwritten = {
            "ion_core_radius": -1.0
            * jnp.ones_like(self.Z_free)
            * ureg.angstrom,
        }
        # Set a default screening length & LFC
        self["screening length"] = DebyeHueckelScreeningLength()
        self["ee-lfc"] = ElectronicLFCConstant(0.0)

    def __len__(self) -> int:
        return len(self.ions)

    def __getitem__(self, key: str):
        return self.models[key]

    def __setitem__(self, key: str, model) -> None:
        if key not in model.allowed_keys:
            raise KeyError(f"Model {model} not allowed for key {key}.")
        model.prepare(self, key)
        model.check(self)
        model.model_key = key
        self.models[key] = model

    def expand_integer_ionization_states(self) -> "PlasmaState":
        """
        Creates a new PlasmaState with twice the amount of ion species, where
        each of the ions has an integer charge state and the number of ions is
        adjusted so that the mean charge is the (potentially fractional) charge
        of self.
        """
        doub_ion_list = [i for i in self.ions for _ in range(2)]
        doub_Ti = jnpu.repeat(self.T_i, 2)
        doub_Z = jnpu.repeat(self.Z_free, 2)
        new_Z = jnpu.where(
            jnp.arange(len(doub_Z)) % 2 == 0,
            jnp.floor(doub_Z),
            jnp.ceil(doub_Z),
        )
        xi = jnp.where(
            jnp.arange(len(doub_Z)) % 2 == 0,
            1 - (doub_Z - jnp.floor(doub_Z)),
            doub_Z - jnp.floor(doub_Z),
        )
        doub_rho = jnpu.repeat(self.mass_density, 2)
        state = deepcopy(self)
        state.ions = doub_ion_list
        state.Z_free = new_Z
        state.mass_density = xi * doub_rho
        state.T_i = doub_Ti
        # for key, model in state.models.items():
        #    model.prepare(state, key)
        return state

    def update_default_model(
        self, model_name: str, model_class: ABCMeta
    ) -> None:
        """
        Add a model to the ``PlasmaState``, if it does not exist, already.
        If a new Model is appended to this state, issue a warning.

        This function is intended to be used by a
        :py:class:`jaxrts.models.Model` if it relies on having other models
        set. An example would be models for ionic scattering, which normally
        require some notion for the form-factors, for which different models
        exist. To allow a user to get to some reasonable spectra fast, a
        :py:class:`jaxrts.models.Model` can modify the ``PlasmaState`` and set
        defaults. This should, however, be non-destructive, i.e., if a user
        specifically selected a value, this should not be overwritten. Hence,
        this convenience-function only adds the model if the given
        ``model_name`` is not in use, already and will inform a user about the
        automatically selected model.

        Parameters
        ----------
        model_name : str
            Key under that the model should be saved.
        model_class ABCMeta
            The ModelClass that should be a sane default.

        Warns
        -----
        Warning
            If a default was set to inform a user about the "default choice".

        Examples
        --------
        >> s.update_default_model("form-factors", PaulingFormFactors)
        """
        if model_name not in self.models:
            logger.warning(
                f"Setting default '{model_name}' model to '{model_class.__name__}'."  # noqa: E501
                + " You can suppress this warning by setting the model manually."  # noqa: E501
            )
            self[model_name] = model_class

    @property
    def Z_A(self) -> jnp.ndarray:
        """
        The atomic number of the atom-species.
        """
        return jnp.array([i.Z for i in self.ions])

    @property
    def nions(self) -> int:
        return len(self.ions)

    @property
    def mean_Z_A(self) -> float:
        """
        The mean atomic number.
        """
        return jnp.sum(self.Z_A * self.number_fraction)

    @property
    def Z_core(self) -> jnp.ndarray:
        """
        The number of electrons still bound to the ion.
        """
        return self.Z_A - self.Z_free

    @property
    def atomic_masses(self) -> Quantity:
        """
        The atomic weight of the atoms.
        """
        return jnp.array(
            [i.atomic_mass.m_as(ureg.atomic_mass_constant) for i in self.ions]
        ) * (1 * ureg.atomic_mass_constant)

    @property
    def n_i(self):
        return (self.mass_density / self.atomic_masses).to_base_units()

    @property
    def n_e(self):
        return (jnpu.sum(self.n_i * self.Z_free)).to_base_units()

    @property
    def Teff_e(self):
        """
        Return the effective electron temperature. Quantum temperature as used
        by :cite:`Gregori.2003`.
        """
        rs = wiegner_seitz_radius(self.n_e) / ureg.a_0
        # This is another definition of Tq, not from Gregori et al.
        # Tq = (
        #     fermi_energy(self.n_e)
        #     / (1.594 - 0.3160 * jnpu.sqrt(rs) + 0.024 * rs)
        #     / (1 * ureg.boltzmann_constant)
        # )

        Tq = (
            fermi_energy(self.n_e)
            / (1.3251 - 0.1779 * jnpu.sqrt(rs))
            / (1 * ureg.boltzmann_constant)
        )

        return jnpu.sqrt(Tq**2 + self.T_e**2)

    @property
    def ee_coupling(self):
        d = (3 / (4 * np.pi * self.n_e)) ** (1.0 / 3.0)

        return (
            (1 * ureg.elementary_charge) ** 2
            / (
                4
                * np.pi
                * (1 * ureg.vacuum_permittivity)
                * (1 * ureg.boltzmann_constant)
                * self.T_e
                * d
            )
        ).to_base_units()

    @property
    def ii_coupling(self):
        pass

    def _lookup_ion_core_radius(self):
        ioc = [e.atomic_radius_calc for e in self.ions]
        return to_array(ioc)

    @property
    def ion_core_radius(self):
        return jnpu.where(
            self._overwritten["ion_core_radius"] > 0 * ureg.angstrom,
            self._overwritten["ion_core_radius"],
            self._lookup_ion_core_radius(),
        )

    @ion_core_radius.setter
    def ion_core_radius(self, value):
        calc = self._lookup_ion_core_radius()
        logger.warning(
            "The value ion_core_radius was overwritten by a user. "
            + f"The calculated value was {calc.to(ureg.angstrom)}, "
            + f"the new value is {value.to(ureg.angstrom)}."
        )
        self._overwritten["ion_core_radius"] = value

    @property
    def screening_length(self):
        """
        This is a shortcut to just get the screening length, which is used,
        e.g., by the :py:class:`jaxrts.hnc_potentials.DebyeHueckelPotential`.
        """
        return self.evaluate("screening length", None)

    @property
    def number_fraction(self):
        """
        Return the number fraction of the elements.
        """
        x = self.n_i / jnpu.sum(self.n_i)
        return x.m_as(ureg.dimensionless)

    def db_wavelength(self, kind: list | str):

        wavelengths = []

        if isinstance(kind, str):
            kind = [kind]
        for par in kind:
            assert (par == "e-") or (
                par in self.ions
            ), "Kind must be one of the ion species or an electron (e-)!"
            if par == "e-":
                wavelengths.append(
                    (
                        (1 * ureg.planck_constant)
                        / jnpu.sqrt(
                            2.0
                            * jnp.pi
                            * 1
                            * ureg.electron_mass
                            * 1
                            * ureg.boltzmann_constant
                            * self.T_e
                        )
                    ).to_base_units()
                )
            else:
                wavelengths.append(
                    (
                        (1 * ureg.planck_constant)
                        / jnpu.sqrt(
                            2.0
                            * jnp.pi
                            * 1
                            * self.atomic_masses[
                                np.argwhere(np.array(self.ions == par))
                            ]
                            * 1
                            * ureg.boltzmann_constant
                            * self.T_e
                        )
                    ).to_base_units()
                )

    @jax.jit
    def probe(self, setup: Setup) -> Quantity:
        ionic = self["ionic scattering"].evaluate(self, setup)
        free_free = self["free-free scattering"].evaluate(self, setup)
        bound_free = self["bound-free scattering"].evaluate(self, setup)
        free_bound = self["free-bound scattering"].evaluate(self, setup)

        return ionic + free_free + bound_free + free_bound

    def evaluate(self, key: str, setup: Setup) -> Quantity:
        """
        This is just a to avoid the redundancy when one wants to evaluate a
        specific model and would otherwise need to provide the state, again.
        """
        return self[key].evaluate(self, setup)

    # Set labels for a save state that is better readable by humans
    _children_labels = (
        "Z_free",
        "mass_density",
        "T_e",
        "T_i",
        "ion_core_radius",
        "models",
    )
    _aux_labels = ("ions",)

    # The following is required to jit a state
    def _tree_flatten(self):
        children = (
            self.Z_free,
            self.mass_density,
            self.T_e,
            self.T_i,
            self._overwritten["ion_core_radius"],
            self.models,
        )
        aux_data = (self.ions,)  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(PlasmaState)
        (obj.ions,) = aux_data
        (
            obj.Z_free,
            obj.mass_density,
            obj.T_e,
            obj.T_i,
            ion_core_radius,
            obj.models,
        ) = children
        obj._overwritten = {
            "ion_core_radius": ion_core_radius,
        }
        return obj

    # This might be easier, now
    def _eq_characteristic(self):
        children, _ = self._tree_flatten()
        static_model_list = [
            self.models[key]._tree_flatten()[1]
            for key in sorted(self.models.keys())
        ]
        return children[:-1], static_model_list, self.ions, self.models.keys()

    def __eq__(self, other):
        if isinstance(other, PlasmaState):
            # Check that all _eq_characteristics are the same for both
            # plasmastates
            self_c, self_s, self_i, self_m = self._eq_characteristic()
            other_c, other_s, other_i, other_m = other._eq_characteristic()

            # Test the children. If any is not equal, return false
            for s, o in zip(self_c, other_c, strict=False):
                eq = jnpu.equal(s, o)
                if not isinstance(eq, bool):
                    eq = eq.all()
                if not eq:
                    return False
            # If all children are identical, compare the model_list, ions and
            # model_keys
            return (
                (self_s == other_s) & (self_i == other_i) & (self_m == other_m)
            )
        return NotImplemented


jax.tree_util.register_pytree_node(
    PlasmaState,
    PlasmaState._tree_flatten,
    PlasmaState._tree_unflatten,
)
