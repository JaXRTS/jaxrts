import jaxrts

import jax
import jax.numpy as jnp
from jpu import numpy as jnpu

from jaxrts.models import (
    Model,
    Neglect,
    Gregori2003IonFeat,
    ArkhipovIonFeat,
    QCSalpeterApproximation,
    BornMermin,
    RPA_NoDamping,
    PaulingFormFactors,
    GregoriChemPotential,
    SchumacherImpulse
)
from jaxrts.setup import Setup, convolve_stucture_factor_with_instrument
from jaxrts.elements import electron_distribution_ionized_state
from jaxrts.plasmastate import PlasmaState

from functools import partial


ureg = jaxrts.ureg
Quantity = jaxrts.units.Quantity


element = jaxrts.elements.Element("Be")

state = jaxrts.PlasmaState(
    ions=[element],
    Z_free=jnp.array([2]),
    density_fractions=jnp.array([1]),
    mass_density=jnp.array([3e23])
    / (1 * ureg.centimeter**3)
    * element.atomic_mass
    / 2,
    T_e=jnp.array([10]) * ureg.electron_volt / ureg.k_B,
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

state["ionic scattering"] = ArkhipovIonFeat
state["free-free scattering"] = RPA_NoDamping
# state["free-free scattering"] = BornMermin
state["bound-free scattering"] = SchumacherImpulse
state["free-bound scattering"] = Neglect


import matplotlib.pyplot as plt
import time

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
