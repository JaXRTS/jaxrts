from abc import ABCMeta
from typing import List
import numpy as np
import logging
import jpu
from jax import numpy as jnp

from .elements import Element
from .units import ureg, Quantity, to_array
from .setup import Setup

logger = logging.getLogger(__name__)


class PlasmaState:

    def __init__(
        self,
        ions: List[Element],
        Z_free: List | Quantity,
        density_fractions: List | float,
        mass_density: List | Quantity,
        T_e: Quantity,
        T_i: List | Quantity | None = None,
    ):

        assert (
            (len(ions) == len(Z_free))
            and (len(ions) == len(density_fractions))
            and (len(ions) == len(mass_density))
        ), "WARNING: Input parameters should be the same shape as <ions>!"
        if T_i is not None:
            assert len(ions) == len(
                T_i
            ), "WARNING: Input parameters should be the same shape as <ions>!"

        self.ions = ions
        self.nions = len(ions)

        # Define charge configuration
        self.Z_free = to_array(Z_free)

        self.density_fractions = to_array(density_fractions)
        self.mass_density = to_array(mass_density)

        if isinstance(T_e.magnitude, jnp.ndarray):
            self.T_e = T_e[0]
        else:
            self.T_e = T_e
        T_i = T_i if T_i else T_e * jnp.ones(self.nions)
        self.T_i = to_array(T_i)

        self.models = {}

    def __len__(self) -> int:
        return len(self.ions)

    def __getitem__(self, key: str):
        return self.models[key]

    def __setitem__(self, key: str, model_class: ABCMeta) -> None:
        if key not in model_class.allowed_keys:
            raise KeyError(f"Model {model_class} not allowed for key {key}.")
        self.models[key] = model_class(self, key)

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

    def probe(self, setup: Setup) -> Quantity:
        ionic = self["ionic scattering"].evaluate(setup)
        free_free = self["free-free scattering"].evaluate(setup)
        bound_free = self["bound-free scattering"].evaluate(setup)
        free_bound = self["free-bound scattering"].evaluate(setup)

        return ionic + free_free + bound_free + free_bound
