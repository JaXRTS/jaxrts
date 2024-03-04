from .units import ureg, Quantity
from typing import List
import numpy as np
import logging
import jpu
from jax import numpy as jnp

from .elements import Element
from .setup import Setup

logger = logging.getLogger(__name__)


class PlasmaState:

    def __init__(
        self,
        ions: List[Element],
        Z_free: List | Quantity,
        density_fractions: List | float,
        mass_density: List | Quantity,
        T_e: List | Quantity,
        T_i: List | Quantity | None = None,
    ):

        assert (
            (len(ions) == len(Z_free))
            and (len(ions) == len(density_fractions))
            and (len(ions) == len(mass_density))
            and (len(ions) == len(T_e))
        ), "WARNING: Input parameters should be the same shape as <ions>!"
        if T_i is not None:
            assert len(ions) == len(
                T_i
            ), "WARNING: Input parameters should be the same shape as <ions>!"

        self.ions = ions
        self.nions = len(ions)

        # Define charge configuration
        self.Z_free = Z_free

        self.density_fractions = density_fractions
        self.mass_density = mass_density

        self.T_e = T_e
        self.T_i = T_i if T_i else T_e

        # Set some default models
        self.ionic_model = None
        self.free_free_model = None
        self.bound_free_model = None
        self.free_bound_model = None

    def Z_A(self) -> jnp.ndarray:
        """
        The atomic number of the atom-species.
        """
        return jnp.array([i.Z for i in self.ions])

    def Z_core(self) -> jnp.ndarray:
        """
        The number of electrons still bound to the ion.
        """
        return self.Z_A() - self.Z_free

    def atomic_masses(self) -> Quantity:
        """
        The atomic weight of the atoms.
        """
        return jnp.array(
            [i.atomic_mass.m_as(ureg.atomic_mass_constant) for i in self.ions]
        ) * (1 * ureg.atomic_mass_constant)

    def n_i(self):
        return (
            self.mass_density
            * self.density_fractions
            / jpu.numpy.sum(self.atomic_masses() * self.density_fractions)
        ).to_base_units()

    def n_e(self):
        return (jpu.numpy.sum(self.n_i() * self.Z_free)).to_base_units()

    def ee_coupling(self):
        d = (3 / (4 * np.pi * self.n_e())) ** (1.0 / 3.0)

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
        ionic = self.ionic_model.evaluate(setup)
        free_free = self.free_free_model.evaluate(setup)
        bound_free = self.bound_free_model.evaluate(setup)
        free_bound = self.free_bound_model.evaluate(setup)

        return ionic + free_free + bound_free + free_bound
