"""
Electron ion potentials
=======================

Plot electron-ion potentials which are used to calculate the screening cloud
from the results of linear response.
The examples here reproduce the findings by :cite:`Gericke.2010`.
"""

import matplotlib.pyplot as plt
import scienceplots
import jax.numpy as jnp

import jaxrts
from jaxrts import ureg

plt.style.use("science")

state = jaxrts.PlasmaState(
    [jaxrts.Element("Be")], [2], [1], [ureg("3g/cc")], ureg("1e5K")
)
state.ion_core_radius = jnp.array([0.75]) * ureg.angstrom

r = jnp.linspace(0.001, 4, 1000) * ureg.a_0
empty_core = jaxrts.hnc_potentials.EmptyCorePotential(state)
coulomb = jaxrts.hnc_potentials.CoulombPotential(state)

for pot in [empty_core, coulomb]:
    pot.include_electrons = True
    plt.plot(r.m_as(ureg.a_0), pot.full_r(r).m_as(ureg.rydberg)[0, 1, :])

plt.ylim(-4.0, 0.3)
plt.show()
