import jaxrts

import jax
import jax.numpy as jnp
from jpu import numpy as jnpu

from jaxrts.models import Model
from jaxrts.setup import Setup
from jaxrts.plasmastate import PlasmaState

from functools import partial


ureg = jaxrts.ureg
Quantity = jaxrts.units.Quantity


def electron_distribution_ionized_state(plasma_state):
    # Assume the population of electrons be behave like a neutral atom with
    # reduced number of electrons. I.e., a 1.5 times ionized carbon is like
    # Beryllium (and half a step to Boron).
    core_electron_floor = int(jnp.floor(plasma_state.Z_core[0]))
    pop_floor = jaxrts.elements.electron_distribution(core_electron_floor)
    pop_ceil = jaxrts.elements.electron_distribution(core_electron_floor + 1)

    population = pop_floor + (
        (plasma_state.Z_core[0] - core_electron_floor)
        * (pop_ceil - pop_floor)
    )
    return population


def conv_dync_stucture_with_instrument(
    Sfac: Quantity, setup: Setup
) -> Quantity:
    conv_grid = (
        setup.measured_energy - jnpu.mean(setup.measured_energy)
    ) / ureg.hbar
    return (
        jnp.convolve(
            Sfac.m_as(ureg.second),
            setup.instrument(conv_grid).m_as(ureg.second),
            mode="same",
        )
        * (1 * ureg.second**2)
        * (jnpu.diff(setup.measured_energy)[0] / ureg.hbar)
    )


class PaulingFormFactors(Model):
    def evaluate(self, setup: Setup) -> jnp.ndarray:
        Zstar = jaxrts.form_factors.pauling_effective_charge(
            self.plasma_state.Z_A[0]
        )
        form_factors = jaxrts.form_factors.pauling_all_ff(setup.k, Zstar)
        population = self.plasma_state.ions[0].electron_distribution
        return jnp.where(population > 0, form_factors, 0)


# The ion-feature
# -----------------


class ArkhipovIonFeat(Model):
    def __init__(self, state: PlasmaState) -> None:
        super().__init__(state)
        if "form-factors" not in state.models.keys():
            print("Setting Default form-factors model")
            state["form-factors"] = PaulingFormFactors

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        fi = self.plasma_state["form-factors"].evaluate(setup)
        population = electron_distribution_ionized_state(self.plasma_state)

        f = jnp.sum(fi * population)
        q = jaxrts.ion_feature.q(
            setup.k[jnp.newaxis],
            self.plasma_state.ions[0].atomic_mass,
            self.plasma_state.n_e,
            self.plasma_state.T_e[0],
            self.plasma_state.Z_free[0],
        )
        S_ii = jaxrts.ion_feature.S_ii_AD(
            setup.k,
            self.plasma_state.T_e[0],
            self.plasma_state.n_e,
            self.plasma_state.ions[0].atomic_mass,
            self.plasma_state.Z_free[0],
        )
        w_R = jnp.abs(f + q.m_as(ureg.dimensionless)) ** 2 * S_ii
        res = w_R * setup.instrument(
            (setup.measured_energy - setup.energy) / ureg.hbar
        )
        return res


class Gregori2003IonFeat(Model):
    def __init__(self, state: PlasmaState) -> None:
        super().__init__(state)
        if "form-factors" not in state.models.keys():
            print("Setting default 'form-factors' model")
            state["form-factors"] = PaulingFormFactors

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        fi = self.plasma_state["form-factors"].evaluate(setup)
        population = electron_distribution_ionized_state(self.plasma_state)

        T_eff = jaxrts.static_structure_factors.T_cf_Greg(
            self.plasma_state.T_e[0], self.plasma_state.n_e
        )
        f = jnp.sum(fi * population)
        q = jaxrts.ion_feature.q(
            setup.k[jnp.newaxis],
            self.plasma_state.ions[0].atomic_mass,
            self.plasma_state.n_e,
            T_eff,
            self.plasma_state.Z_free[0],
        )
        S_ii = jaxrts.ion_feature.S_ii_AD(
            setup.k,
            T_eff,
            self.plasma_state.n_e,
            self.plasma_state.ions[0].atomic_mass,
            self.plasma_state.Z_free[0],
        )
        w_R = jnp.abs(f + q.m_as(ureg.dimensionless)) ** 2 * S_ii
        res = w_R * setup.instrument(
            (setup.measured_energy - setup.energy) / ureg.hbar
        )
        return res


class GregoriChemPotential(Model):
    def evaluate(self, setup: Setup) -> Quantity:
        return jaxrts.plasma_physics.chem_pot_interpolation(
            self.plasma_state.T_e[0], self.plasma_state.n_e
        )


class BornMermin(Model):
    def __init__(self, state: PlasmaState) -> None:
        super().__init__(state)
        if "chemical potential" not in state.models.keys():
            print("Setting default 'chemical potential' model")
            state["chemical potential"] = GregoriChemPotential

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        mu = self.plasma_state["chemical potential"].evaluate(setup)
        epsilon = jaxrts.free_free.dielectric_function_BMA(
            setup.k,
            setup.measured_energy,
            mu,
            self.plasma_state.T_e[0],
            self.plasma_state.n_e,
            self.plasma_state.ions[0].atomic_mass,
            self.plasma_state.Z_free[0],
        )
        See_0 = jaxrts.free_free.S0ee_from_dielectric_func_FDT(
            setup.k,
            self.plasma_state.T_e[0],
            self.plasma_state.n_e,
            setup.measured_energy,
            epsilon,
        )
        free_free = See_0 * self.plasma_state.Z_free[0]
        return conv_dync_stucture_with_instrument(free_free, setup)


class SchumacherImpulse(Model):
    def __init__(self, state: PlasmaState) -> None:
        super().__init__(state)
        if "form-factors" not in state.models.keys():
            print("Setting default 'form-factors' model")
            state["form-factors"] = PaulingFormFactors

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        k = setup.k
        omega_0 = setup.energy / ureg.hbar
        omega = omega_0 - setup.measured_energy / ureg.hbar
        Z_c = self.plasma_state.Z_core[0]
        E_b = self.plasma_state.ions[0].binding_energies

        Zeff = jaxrts.form_factors.pauling_effective_charge(
            self.plasma_state.ions[0].Z
        )
        population = electron_distribution_ionized_state(self.plasma_state)
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

        return conv_dync_stucture_with_instrument(sbe * Z_c, setup)


class QCSAFreeFree(Model):

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        See_0 = jaxrts.free_free.S0_ee_Salpeter(
            setup.k,
            self.plasma_state.T_e[0],
            self.plasma_state.n_e,
            setup.measured_energy - setup.energy,
        )

        free_free = See_0 * self.plasma_state.Z_free[0]
        return conv_dync_stucture_with_instrument(free_free, setup)


class RPAFreeFree(Model):
    def __init__(self, state: PlasmaState) -> None:
        super().__init__(state)
        if "chemical potential" not in state.models.keys():
            print("Setting default 'chemical potential' model")
            state["chemical potential"] = GregoriChemPotential

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
        return conv_dync_stucture_with_instrument(free_free, setup)


class Neglect(Model):
    def evaluate(self, setup: Setup) -> jnp.ndarray:
        return jnp.zeros_like(setup.measured_energy) * (1 * ureg.second)


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

state["ionic scattering"] = Gregori2003IonFeat
# state["free-free scattering"] = QCSAFreeFree
state["free-free scattering"] = RPAFreeFree
# state["bound-free scattering"] = SchumacherImpulse
state["bound-free scattering"] = Neglect
state["free-bound scattering"] = Neglect

import matplotlib.pyplot as plt
import time

plt.plot(
    (setup.measured_energy - setup.energy).m_as(ureg.electron_volt),
    state.probe(setup),
)

# setup.measured_energy = (
#     ureg("4768.6230 eV") + jnp.linspace(-250, 100, 100) * ureg.electron_volt
# )
state.Z_free = jnp.array([2.5])
state.mass_density = (
    jnp.array([3e23]) / (1 * ureg.centimeter**3) * element.atomic_mass / state.Z_free[0]
)
plt.plot(
    (setup.measured_energy - setup.energy).m_as(ureg.electron_volt),
    state.probe(setup),
)
state.Z_free = jnp.array([3])
state.mass_density = (
    jnp.array([3e23]) / (1 * ureg.centimeter**3) * element.atomic_mass / state.Z_free[0]
)
plt.plot(
    (setup.measured_energy - setup.energy).m_as(ureg.electron_volt),
    state.probe(setup),
)

plt.show()
