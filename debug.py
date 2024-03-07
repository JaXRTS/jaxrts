import jaxrts

import jax
import jax.numpy as jnp
from jpu import numpy as jnpu

from jaxrts.models import Model, Neglect, Gregori2003IonFeat, ArkhipovIonFeat, PaulingFormFactors
from jaxrts.setup import Setup, convolve_stucture_factor_with_instrument
from jaxrts.elements import electron_distribution_ionized_state
from jaxrts.plasmastate import PlasmaState

from functools import partial


ureg = jaxrts.ureg
Quantity = jaxrts.units.Quantity


# The ion-feature
# -----------------


class GregoriChemPotential(Model):
    """
    A fitting formula for the chemical potential of a plasma between the
    classical and the quantum regime, given by :cite:`Gregori.2003`.
    """
    def evaluate(self, setup: Setup) -> Quantity:
        return jaxrts.plasma_physics.chem_pot_interpolation(
            self.plasma_state.T_e, self.plasma_state.n_e
        )


class BornMermin(Model):
    def __init__(self, state: PlasmaState) -> None:
        state.update_default_model("chemical potential", GregoriChemPotential)
        super().__init__(state)

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        mu = self.plasma_state["chemical potential"].evaluate(setup)
        epsilon = jaxrts.free_free.dielectric_function_BMA(
            setup.k,
            setup.measured_energy,
            mu,
            self.plasma_state.T_e,
            self.plasma_state.n_e,
            self.plasma_state.atomic_masses,
            self.plasma_state.Z_free,
        )
        See_0 = jaxrts.free_free.S0ee_from_dielectric_func_FDT(
            setup.k,
            self.plasma_state.T_e,
            self.plasma_state.n_e,
            setup.measured_energy,
            epsilon,
        )
        free_free = See_0 * self.plasma_state.Z_free
        return convolve_stucture_factor_with_instrument(free_free, setup)


class SchumacherImpulse(Model):
    def __init__(self, state: PlasmaState) -> None:
        state.update_default_model("form-factors", PaulingFormFactors)
        super().__init__(state)

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        k = setup.k
        omega_0 = setup.energy / ureg.hbar
        omega = omega_0 - setup.measured_energy / ureg.hbar
        Z_c = self.plasma_state.Z_core[0]
        E_b = self.plasma_state.ions[0].binding_energies

        Zeff = jaxrts.form_factors.pauling_effective_charge(
            self.plasma_state.ions[0].Z
        )
        population = electron_distribution_ionized_state(
            self.plasma_state.Z_core
        )
        # Gregori.2004, Eqn 20
        fi = self.plasma_state["form-factors"].evaluate(setup)
        r_k = 1 - jnp.sum(population / Z_c * fi**2)
        B = 1 + 1 / omega_0 * (ureg.hbar * k**2) / (2 * ureg.electron_mass)
        sbe = (
            r_k
            / (Z_c * B**3)
            * jaxrts.bound_free.J_impulse_approx(
                omega, k, population, Zeff, E_b
            )
        )

        return convolve_stucture_factor_with_instrument(sbe * Z_c, setup)


class QCSAFreeFree(Model):

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        See_0 = jaxrts.free_free.S0_ee_Salpeter(
            setup.k,
            self.plasma_state.T_e[0],
            self.plasma_state.n_e,
            setup.measured_energy - setup.energy,
        )

        free_free = See_0 * self.plasma_state.Z_free[0]
        return convolve_stucture_factor_with_instrument(free_free, setup)


class RPAFreeFree(Model):
    def __init__(self, state: PlasmaState) -> None:
        state.update_default_model("chemical potential", GregoriChemPotential)
        super().__init__(state)

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        mu = self.plasma_state["chemical potential"].evaluate(setup)
        See_0 = jaxrts.free_free.S0_ee_RPA_no_damping(
            setup.k,
            self.plasma_state.T_e[0],
            self.plasma_state.n_e,
            setup.measured_energy - setup.energy,
            mu,
        )

        free_free = See_0 * self.plasma_state.Z_free[0]
        return convolve_stucture_factor_with_instrument(free_free, setup)


element = jaxrts.elements.Element("Be")

state = jaxrts.PlasmaState(
    ions=[element],
    Z_free=jnp.array([2]),
    density_fractions=jnp.array([1]),
    mass_density=jnp.array([3e23])
    / (1 * ureg.centimeter**3)
    * element.atomic_mass
    / 2,
    T_e=jnp.array([40]) * ureg.electron_volt / ureg.k_B,
)

setup = Setup(
    ureg("160°"),
    ureg("4750 eV"),
    ureg("4750 eV") + jnp.linspace(-250, 100, 500) * ureg.electron_volt,
    partial(
        jaxrts.instrument_function.instrument_gaussian,
        sigma=ureg("50.0eV") / ureg.hbar / (2 * jnp.sqrt(2 * jnp.log(2))),
    ),
)

state["ionic scattering"] = Neglect
# state["free-free scattering"] = QCSAFreeFree
state["free-free scattering"] = BornMermin
# state["bound-free scattering"] = SchumacherImpulse
state["bound-free scattering"] = Neglect
state["free-bound scattering"] = Neglect


import matplotlib.pyplot as plt
import time

plt.plot(
    (setup.measured_energy - setup.energy).m_as(ureg.electron_volt),
    state.probe(setup).m_as(ureg.second),
)
state["free-free scattering"] = RPAFreeFree
plt.plot(
    (setup.measured_energy - setup.energy).m_as(ureg.electron_volt),
    state.probe(setup).m_as(ureg.second),
)

# setup.measured_energy = (
#     ureg("4768.6230 eV") + jnp.linspace(-250, 100, 100) * ureg.electron_volt
# )
# state.Z_free = jnp.array([2.5])
# state.mass_density = (
#     jnp.array([3e23])
#     / (1 * ureg.centimeter**3)
#     * element.atomic_mass
#     / state.Z_free[0]
# )
# plt.plot(
#     (setup.measured_energy - setup.energy).m_as(ureg.electron_volt),
#     state.probe(setup).m_as(ureg.second),
# )
# state.Z_free = jnp.array([3])
# state.mass_density = (
#     jnp.array([3e23])
#     / (1 * ureg.centimeter**3)
#     * element.atomic_mass
#     / state.Z_free[0]
# )
# plt.plot(
#     (setup.measured_energy - setup.energy).m_as(ureg.electron_volt),
#     state.probe(setup).m_as(ureg.second),
# )

plt.show()
