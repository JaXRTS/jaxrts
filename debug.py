import jaxrts

import jax.numpy as jnp
from jpu import numpy as jnpu

from jaxrts.models import Model
from jaxrts.setup import Setup
from jaxrts.plasmastate import PlasmaState

from functools import partial


ureg = jaxrts.ureg
Quantity = jaxrts.units.Quantity

# Form Factor Models


class PaulingFormFactors(Model):
    def evaluate(self, setup: Setup) -> jnp.ndarray:
        Zstar = jaxrts.form_factors.pauling_effective_charge(
            self.plasma_state.Z_A()[0]
        )
        form_factors = jaxrts.form_factors.pauling_all_ff(setup.k(), Zstar)
        population = self.plasma_state.ions[0].electron_distribution
        return jnpu.sum(form_factors * population)


# The ion-feature
# -----------------


class ArphipovIonFeat(Model):
    def __init__(self, state: PlasmaState) -> None:
        super().__init__(state)
        if not hasattr(state, "form_factor_model"):
            print("Setting Default form_factor_model")
            state.form_factor_model = PaulingFormFactors(state)

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        f = self.plasma_state.form_factor_model.evaluate(setup)
        q = jaxrts.ion_feature.q(
            setup.k()[jnp.newaxis],
            self.plasma_state.ions[0].atomic_mass,
            self.plasma_state.n_e(),
            self.plasma_state.T_e[0],
            self.plasma_state.Z_free[0],
        )
        S_ii = jaxrts.ion_feature.S_ii_AD(
            setup.k(),
            self.plasma_state.T_e[0],
            self.plasma_state.n_e(),
            self.plasma_state.ions[0].atomic_mass,
            self.plasma_state.Z_free[0],
        )
        w_R = jnp.abs(f + q.m_as(ureg.dimensionless)) ** 2 * S_ii
        res = w_R * setup.instrument(setup.measured_energy - setup.energy)
        return res.magnitude


class GregoriChemPotential(Model):
    def evaluate(self, setup: Setup) -> Quantity:
        return jaxrts.plasma_physics.chem_pot_interpolation(
            self.plasma_state.T_e[0], self.plasma_state.n_e()
        )


class BornMermin(Model):
    def __init__(self, state: PlasmaState) -> None:
        super().__init__(state)
        if not hasattr(state, "chem_potential_model"):
            print("Setting Default chem_potential_model")
            state.chem_potential_model = GregoriChemPotential(state)

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        mu = self.plasma_state.chem_potential_model.evaluate(setup)
        epsilon = jaxrts.electron_feature.dielectric_function_BMA(
            setup.k(),
            setup.measured_energy,
            mu,
            self.plasma_state.T_e[0],
            self.plasma_state.n_e(),
            self.plasma_state.ions[0].atomic_mass,
            self.plasma_state.Z_free[0],
            )
        See_0 = jaxrts.electron_feature.S0ee_from_dielectric_func_FDT(
            setup.k(),
            self.plasma_state.T_e[0],
            self.plasma_state.n_e(),
            setup.measured_energy,
            epsilon,
        )
        free_free = See_0.m_as(ureg.second) * self.plasma_state.Z_free[0]
        return (
            jnp.convolve(
                free_free,
                setup.instrument(
                    setup.measured_energy - setup.energy
                ).magnitude,
                mode="same",
            )
            * ureg.second
            * (jnpu.diff(setup.measured_energy)[0] / ureg.hbar)
        )


class QCSAFreeFree(Model):

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        See_0 = jaxrts.electron_feature.S0_ee_Salpeter(
            setup.k(),
            self.plasma_state.T_e[0],
            self.plasma_state.n_e(),
            setup.measured_energy - setup.energy,
        )

        free_free = See_0.m_as(ureg.second) * self.plasma_state.Z_free[0]
        return (
            jnp.convolve(
                free_free,
                setup.instrument(
                    setup.measured_energy - setup.energy
                ).magnitude,
                mode="same",
            )
            * ureg.second
            * (jnpu.diff(setup.measured_energy)[0] / ureg.hbar)
        )


class RPAFreeFree(Model):
    def __init__(self, state: PlasmaState) -> None:
        super().__init__(state)
        if not hasattr(state, "chem_potential_model"):
            print("Setting Default chem_potential_model")
            state.chem_potential_model = GregoriChemPotential(state)

    def evaluate(self, setup: Setup) -> jnp.ndarray:
        mu = self.plasma_state.chem_potential_model.evaluate(setup)
        See_0 = jaxrts.electron_feature.S0_ee_RPA_no_damping(
            setup.k(),
            self.plasma_state.T_e[0],
            self.plasma_state.n_e(),
            setup.measured_energy - setup.energy,
            mu,
        )

        free_free = See_0.m_as(ureg.second) * self.plasma_state.Z_free[0]
        return (
            jnp.convolve(
                free_free,
                setup.instrument(
                    setup.measured_energy - setup.energy
                ).magnitude,
                mode="same",
            )
            * ureg.second
            * (jnpu.diff(setup.measured_energy)[0] / ureg.hbar)
        )


class Neglect(Model):
    def evaluate(self, setup: Setup) -> jnp.ndarray:
        return jnp.zeros_like(setup.measured_energy)


state = jaxrts.PlasmaState(
    [jaxrts.elements.Element("Be")],
    jnp.array([2]),
    jnp.array([1]),
    jnp.array([4.48943]) * ureg.gram / ureg.centimeter**3,
    jnp.array([10]) * ureg.electron_volt / ureg.k_B,
)

setup = Setup(
    ureg("160°"),
    ureg("4768.6230 eV"),
    ureg("4768.6230 eV") + jnp.linspace(-250, 100, 500) * ureg.electron_volt,
    partial(
        jaxrts.instrument_function.instrument_gaussian, sigma=ureg("10eV")
    ),
)

state.ionic_model = ArphipovIonFeat(state)
state.free_free_model = QCSAFreeFree(state)
state.bound_free_model = Neglect(state)
state.free_bound_model = Neglect(state)

import matplotlib.pyplot as plt
import time

for _ in range(5):
    t0 = time.time()
    state.probe(setup)
    print(time.time() - t0)

plt.plot(
    (setup.measured_energy - setup.energy).m_as(ureg.electron_volt),
    state.probe(setup),
)

setup.measured_energy = (
    ureg("4768.6230 eV") + jnp.linspace(-250, 100, 100) * ureg.electron_volt
)
plt.plot(
    (setup.measured_energy - setup.energy).m_as(ureg.electron_volt),
    state.probe(setup),
)

plt.show()
