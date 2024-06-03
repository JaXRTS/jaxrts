"""
Electron ion potentials
=======================

Plot electron-ion potentials which are used to calculate the screening cloud
from the results of linear response.
The examples here reproduce the findings by :cite:`Gericke.2010`.
"""

import matplotlib.pyplot as plt
import scienceplots  # noqa F401
import jax.numpy as jnp

import jaxrts
from jaxrts import ureg

plt.style.use("science")

state = jaxrts.PlasmaState(
    [jaxrts.Element("Be")], [2], [1], [ureg("3g/cc")], ureg("1e5K")
)

r = jnp.linspace(0.001, 4, 1000) * ureg.a_0
# Note: Gericke.2010 seems to use r_cut = 1ang for the empty core potential and
# 0.75 for the soft-core potentials. We stick here with only one cutoff, for a
# fairer comparison.
state.ion_core_radius = jnp.array([0.75]) * ureg.angstrom

empty_core = jaxrts.hnc_potentials.EmptyCorePotential(state)
soft_core2 = jaxrts.hnc_potentials.SoftCorePotential(state, beta=2)
soft_core6 = jaxrts.hnc_potentials.SoftCorePotential(state, beta=6)
coulomb = jaxrts.hnc_potentials.CoulombPotential(state)

names = [
    "Empty core",
    "Soft core $\\beta=2$",
    "Soft core $\\beta=6$",
    "Coulomb",
]
for idx, pot in enumerate([empty_core, soft_core2, soft_core6, coulomb]):
    pot.include_electrons = True
    plt.plot(
        r.m_as(ureg.a_0),
        pot.full_r(r).m_as(ureg.rydberg)[0, 1, :],
        label=names[idx],
    )

plt.ylim(-4.0, 0.3)
plt.legend()
plt.show()
