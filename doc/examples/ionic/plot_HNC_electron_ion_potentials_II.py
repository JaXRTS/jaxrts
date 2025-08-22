"""
Electron ion potentials
=======================

Plot electron-ion potentials which are used to calculate the screening cloud
from the results of linear response.
The examples here reproduce the findings by :cite:`Gericke.2010`.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import scienceplots  # noqa F401

import jaxrts
from jaxrts import ureg

plt.style.use("science")

fig, ax = plt.subplots(3, figsize=(4, 9))

state = jaxrts.PlasmaState(
    [jaxrts.Element("Be")],
    [2],
    [ureg("1.848g/cc")],
    ureg("12eV") / ureg.k_B,
)
state["screening length"] = jaxrts.models.Gericke2010ScreeningLength()

r = jnp.linspace(0.001, 100, 5000) * ureg.a_0
k = jnp.pi / r[-1] + jnp.arange(len(r)) * (jnp.pi / (len(r) * (r[1] - r[0])))

# Note: Gericke.2010 seems to use r_cut = 1ang for the empty core potential and
# 0.75 for the soft-core potentials. We stick here with only one cutoff, for a
# fairer comparison.
state.ion_core_radius = jnp.array([1]) * ureg.angstrom

empty_core = jaxrts.hnc_potentials.EmptyCorePotential()
soft_core2 = jaxrts.hnc_potentials.SoftCorePotential(beta=2)
soft_core6 = jaxrts.hnc_potentials.SoftCorePotential(beta=6)
coulomb = jaxrts.hnc_potentials.CoulombPotential()

names = [
    "Empty core",
    "Soft core $\\beta=2$",
    "Soft core $\\beta=6$",
    "Coulomb",
]
for idx, pot in enumerate([empty_core, soft_core2, soft_core6, coulomb]):
    pot.include_electrons = "SpinAveraged"
    ax[0].plot(
        r.m_as(ureg.a_0),
        pot.full_r(state, r).m_as(ureg.rydberg)[0, 1, :],
        label=names[idx],
    )
    q = -jaxrts.ion_feature.free_electron_susceptilibily_RPA(
        k, 1 / state.screening_length
    ) * pot.full_k(state, k)
    ax[1].plot(
        k.m_as(1 / ureg.a_0),
        q.m_as(ureg.dimensionless)[0, 1, :],
        label=names[idx],
    )

for ls, r_cut in zip(["solid", "dashed", "dotted"], [0.5, 1, 2], strict=False):
    state.ion_core_radius = jnp.array([r_cut]) * ureg.angstrom
    q = -jaxrts.ion_feature.free_electron_susceptilibily_RPA(
        k, 1 / state.screening_length
    ) * soft_core6.full_k(state, k)
    ax[2].plot(
        k.m_as(1 / ureg.a_0),
        q.m_as(ureg.dimensionless)[0, 1, :],
        label=f"{r_cut:.1f} " + "$\\text{\\AA}$",
        color="C2",
        ls=ls,
    )

ax[0].set_ylim(-4.0, 0.3)
ax[0].set_xlim(0, 4.0)
ax[1].set_xlim(0, 5.0)
ax[2].set_xlim(0, 5.0)
ax[0].legend()
ax[2].legend()
ax[0].set_xlabel("$r$ [$\\text{\\AA}$]")
ax[1].set_xlabel("$k$ [$1/\\text{\\AA}$]")
ax[2].set_xlabel("$k$ [$1/\\text{\\AA}$]")
ax[0].set_ylabel("$V_{ei}(r)$ [Ryd]")
ax[1].set_ylabel("$q(k)$")
ax[2].set_ylabel("$q(k)$")

ax[0].set_title("Electron-ion Potentials")
ax[1].set_title("Screening function")
ax[2].set_title(
    "Influence of $r_{cut}$ in the screening function of a\n"
    + "Soft Core potential ($\\beta = 6$)"
)

plt.tight_layout()
plt.show()
