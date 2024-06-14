from abc import ABCMeta
from typing import List, Dict
import numpy as np
import logging
import jpu
import jax
from jax import numpy as jnp

from .elements import Element
from .units import ureg, Quantity, to_array
from .helpers import JittableDict
from .setup import Setup
from . import plasma_physics

logger = logging.getLogger(__name__)


class PlasmaState:

    def __init__(
        self,
        ions: List[Element],
        Z_free: List | Quantity,
        mass_density: List | Quantity,
        T_e: Quantity,
        T_i: List | Quantity | None = None,
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

        if isinstance(T_e, list):
            self.T_e = T_e[0]
        elif isinstance(T_e.magnitude, jnp.ndarray) and len(T_e.shape) == 1:
            self.T_e = T_e[0]
        else:
            self.T_e = T_e
        T_i = T_i if T_i else T_e * jnp.ones(self.nions)
        self.T_i = to_array(T_i)
        self.models = JittableDict()
        self._overwritten = {
            "DH_screening_length": -1.0 * ureg.angstrom,
            "ion_core_radius": -1.0
            * jnp.ones_like(self.Z_free)
            * ureg.angstrom,
        }

    def __len__(self) -> int:
        return len(self.ions)

    def __getitem__(self, key: str):
        return self.models[key]

    def __setitem__(self, key: str, model) -> None:
        if key not in model.allowed_keys:
            raise KeyError(f"Model {model} not allowed for key {key}.")
        model.prepare(self)
        model.check(self)
        model.model_key = key
        self.models[key] = model

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
        if model_name not in self.models.keys():
            logger.warning(
                f"Setting default '{model_name}' model to '{model_class.__name__}'."
                + " You can suppress this warning by setting the model manually."
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
        return (jpu.numpy.sum(self.n_i * self.Z_free)).to_base_units()

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

    @jax.jit
    def _calc_DH_screening_length(self):
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
        T = plasma_physics.temperature_interpolation(self.n_e, self.T_e, 4)
        lam_DH = plasma_physics.Debye_Huckel_screening_length(self.n_e, T)
        return lam_DH.to(ureg.angstrom)

    def _lookup_ion_core_radius(self):
        ioc = [e.atomic_radius_calc for e in self.ions]
        return to_array(ioc)

    @property
    def ion_core_radius(self):
        return jpu.numpy.where(
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
    def DH_screening_length(self):
        """
        Return the Debye-Hückel Debye screening length. Uses a 4th-power
        interpolation between electron and fermi temperature, as proposed by
        :cite:`Gericke.2010`

        This property can be overwritten by a user.

        See Also
        --------
        jaxrts.plasma_physics.temperature_interpolation:
            The function used for the temperature interpolation
        jaxrts.plasma_physics.Debye_Huckel_screening_length
            The function used to calculate the screening length
        """
        return jax.lax.cond(
            self._overwritten["DH_screening_length"].to(ureg.angstrom)
            >= 0 * ureg.angstrom,
            lambda: self._overwritten["DH_screening_length"].to(ureg.angstrom),
            self._calc_DH_screening_length,
        )

    @DH_screening_length.setter
    def DH_screening_length(self, value):
        calc = self._calc_DH_screening_length()
        logger.warning(
            "The value DH_screening_length was overwritten by a user. "
            + f"The calculated value was {calc:.3f}, "
            + f"the new value is {value.to(ureg.angstrom):.3f}."
        )
        self._overwritten["DH_screening_length"] = value

    def db_wavelength(self, kind: List | str):

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
                        / jpu.numpy.sqrt(
                            2.0
                            * np.pi
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
                        / jpu.numpy.sqrt(
                            2.0
                            * np.pi
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

    def evaluate(self, key, setup) -> Quantity:
        return self[key].evaluate(self, setup)

    @jax.jit
    def probe(self, setup: Setup) -> Quantity:
        ionic = self["ionic scattering"].evaluate(self, setup)
        free_free = self["free-free scattering"].evaluate(self, setup)
        bound_free = self["bound-free scattering"].evaluate(self, setup)
        free_bound = self["free-bound scattering"].evaluate(self, setup)

        return ionic + free_free + bound_free + free_bound

    # The following is required to jit a state
    def _tree_flatten(self):
        children = (
            self.Z_free,
            self.mass_density,
            self.T_e,
            self.T_i,
            self._overwritten["DH_screening_length"],
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
            DH_screening_length,
            ion_core_radius,
            obj.models,
        ) = children
        obj._overwritten = {
            "DH_screening_length": DH_screening_length,
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
            return self._eq_characteristic() == other._eq_characteristic()
        return NotImplemented


jax.tree_util.register_pytree_node(
    PlasmaState,
    PlasmaState._tree_flatten,
    PlasmaState._tree_unflatten,
)
