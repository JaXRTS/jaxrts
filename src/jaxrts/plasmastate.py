from .units import ureg, Quantity
from typing import List
import numpy as np
import logging
import jpu

logger = logging.getLogger(__name__)

class PlasmaState:

    def __init__(
        self,
        ions: List,
        Z_A: List | Quantity,
        Z_free: List | Quantity,
        atomic_masses: List | Quantity,
        density_fractions: List | float,
        mass_density: List | Quantity,
        T_e: List | Quantity,
        T_i: List | Quantity,
    ):

        assert (
            (len(ions) == len(Z_A))
            and (len(ions) == len(Z_free))
            and (len(ions) == len(atomic_masses))
            and (len(ions) == len(density_fractions))
            and (len(ions) == len(mass_density))
            and (len(ions) == len(T_e))
            and (len(ions) == len(T_i))
        ), "WARNING: Input parameters should be the same shape as <ions>!"

        self.ions = ions
        self.nions = len(ions)

        # Define charge configuration
        self.Z_A = Z_A
        self.Z_free = Z_free
        self.Z_core = Z_A - Z_free

        self.atomic_masses = atomic_masses
        self.density_fractions = density_fractions
        self.mass_density = mass_density

        self.T_e = T_e
        self.T_i = T_i if T_i else T_e

    def n_i(self):
        return (
            self.mass_density
            * self.density_fraction
            / jpu.numpy.sum(self.atomic_masses * self.density_fraction)
        ).to_base_units()

    def n_e(self):
        return (jpu.numpy.sum(self.ion_number_densities())).to_base_units()

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

        if type(kind) == str:
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
                            * ureg.atomic_masses[
                                np.argwhere(np.array(self.ions == par))
                            ]
                            * 1
                            * ureg.boltzmann_constant
                            * self.T_e
                        )
                    ).to_base_units()
                )

    def _jSii(self, k: Quantity, E: np.ndarray | List | Quantity):
        pass

    def Sii(self, k: Quantity, E: np.ndarray | List | Quantity):
        return np.array(self._jSii(k, E))

    def _jSee(self, k: Quantity, E: np.ndarray | List | Quantity):
        pass

    def See(self, k: Quantity, E: np.ndarray | List | Quantity):
        return np.array(self._jSee(k, E))

    def _jSee(self, k: Quantity, E: np.ndarray | List | Quantity):
        pass

    def Sce(self, k: Quantity, E: np.ndarray | List | Quantity):
        return np.array(self._jSce(k, E))

    def _jSce(self, k: Quantity, E: np.ndarray | List | Quantity):
        pass

    def Ss(self, k: Quantity, E: np.ndarray | List | Quantity):
        return np.array(self._jSs(k, E))

    def _jSs(self, k: Quantity, E: np.ndarray | List | Quantity):
        pass

    def probe(self, energy: Quantity, theta: float):

        # Calculate probe wavelength from probe energy

        lambda_0 = (
            1
            * ureg.speed_of_light
            / (energy.to(ureg.joule) / (1 * ureg.planck_constant))
        ).to_base_units()

        # Calculate wavenumber for choice of probe energy and scattering angle

        k = (4 * np.pi / lambda_0) * np.sin(np.deg2rad(theta) / 2.0)

        # Calculate S(k,w) and return it (maybe normalize it?)

        S_k_w = 1.0

        return S_k_w
