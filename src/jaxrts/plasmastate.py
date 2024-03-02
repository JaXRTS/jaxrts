from .units import ureg, Quantity
from typing import List
import numpy as np
import logging
import jpu
from jax import numpy as jnp

from .elements import Element

logger = logging.getLogger(__name__)


class PlasmaState:

    def __init__(
        self,
        ions: List[Element],
        Z_free: List | Quantity,
        density_fractions: List | float,
        mass_density: List | Quantity,
        T_e: List | Quantity,
        T_i: List | Quantity,
    ):

        assert (
            (len(ions) == len(Z_free))
            and (len(ions) == len(density_fractions))
            and (len(ions) == len(mass_density))
            and (len(ions) == len(T_e))
            and (len(ions) == len(T_i))
        ), "WARNING: Input parameters should be the same shape as <ions>!"

        self.ions = ions
        self.nions = len(ions)

        # Define charge configuration
        self.Z_free = Z_free

        self.density_fractions = density_fractions
        self.mass_density = mass_density

        self.T_e = T_e
        self.T_i = T_i if T_i else T_e

    @property
    def Z_A(self) -> jnp.ndarray:
        return jnp.array([i.Z for i in self.ions])

    @property
    def Z_core(self) -> jnp.ndarray:
        return self.Z_A - self.Z_free

    @property
    def atomic_masses(self) -> Quantity:
        return jnp.array(
            [i.atomic_mass.m_as(ureg.atomic_mass_constant) for i in self.ions]
        ) * (1 * ureg.atomic_mass_constant)

    @property
    def n_i(self):
        return (
            self.mass_density
            * self.density_fraction
            / jpu.numpy.sum(self.atomic_masses * self.density_fraction)
        ).to_base_units()

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
        return 1.0

    def Sii(self, k: Quantity, E: np.ndarray | List | Quantity):
        return np.array(self._jSii(k, E))

    def _jSee(self, k: Quantity, E: np.ndarray | List | Quantity):
        return 1.0

    def See(self, k: Quantity, E: np.ndarray | List | Quantity):
        return np.array(self._jSee(k, E))

    def Sce(self, k: Quantity, E: np.ndarray | List | Quantity):
        return np.array(self._jSce(k, E))

    def _jSce(self, k: Quantity, E: np.ndarray | List | Quantity):
        return 1.0

    def Ss(self, k: Quantity, E: np.ndarray | List | Quantity):
        return np.array(self._jSs(k, E))

    def _jSs(self, k: Quantity, E: np.ndarray | List | Quantity):
        return 1.0

    def _jf_I(self, k: Quantity):
        return 1.0

    def f_I(self, k: Quantity):
        return np.array(self._jf_I(k))

    def _jq(self, k: Quantity):
        return 1.0

    def q(self, k: Quantity):
        return np.array(self._jq(k))

    def probe(self, E: Quantity, theta: float, instrument: jnp.ndarray):

        # Calculate probe wavelength from probe energy

        lambda_0 = (
            1
            * ureg.speed_of_light
            / (E.to(ureg.joule) / (1 * ureg.planck_constant))
        ).to_base_units()

        # Calculate wavenumber for choice of probe energy and scattering angle

        k = (4 * np.pi / lambda_0) * np.sin(np.deg2rad(theta) / 2.0)

        # Calculate S(k,w) using the Chihara decomposition (normalization?)

        S_k_w = (
            (self._jf_I(k) + self._jq(k)) ** 2 * self._jSii(k, E)
            + self.Z_free * self._jSee(k, E)
            + self.Z_core * 1.0
        )

        return S_k_w
