"""
HNC: Multicomponent
===================

This example reproduces Fig. 4.12 from :cite:`Wunsch.2011`, and shows how to
use the HNC approximation with different ion species.
"""

from pathlib import Path

import jax.numpy as jnp
import jpu
import matplotlib.pyplot as plt
import numpy as onp

import jaxrts
from jaxrts import hypernetted_chain as hnc
from jaxrts import ureg

plt.style.use("science")

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Set up the ionization, density and temperature for individual ion
# species.

state = jaxrts.PlasmaState(
    ions=[jaxrts.Element("H"), jaxrts.Element("C")],
    Z_free=[1, 4],
    mass_density=[
        2.5e23 / ureg.centimeter**3 * jaxrts.Element("H").atomic_mass,
        2.5e23 / ureg.centimeter**3 * jaxrts.Element("C").atomic_mass,
    ],
    T_e=2e4 * ureg.kelvin,
)

pot = 15
r = jpu.numpy.linspace(0.0001 * ureg.angstrom, 1000 * ureg.a0, 2**pot)

# We add densities, here. Maybe this is wrong.
d = jpu.numpy.cbrt(
    3 / (4 * jnp.pi * (state.n_i[:, jnp.newaxis] + state.n_i[jnp.newaxis, :]))
)

dr = r[1] - r[0]
dk = jnp.pi / (len(r) * dr)
k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk

# Set the Screening length for the Debye Screening. Verify where this might
# come form.
state["screening length"] = jaxrts.models.ConstantScreeningLength(
    2 / 3 * ureg.a_0
)

Potential = jaxrts.hnc_potentials.DebyeHueckelPotential()

V_s = Potential.short_r(state, r)
V_l_k = Potential.long_k(state, k)

g, niter = hnc.pair_distribution_function_HNC(
    V_s, V_l_k, r, Potential.T(state), state.n_i
)
S_ii = hnc.S_ii_HNC(k, g, state.n_i, r)

ax[0].plot(
    (r / d[0, 0]).m_as(ureg.dimensionless),
    g[0, 0, :].m_as(ureg.dimensionless),
    label="HH",
    color="C0",
)
ax[1].plot(
    (k * d[0, 0]).m_as(ureg.dimensionless),
    S_ii[0, 0, :].m_as(ureg.dimensionless),
    label="HH",
    color="C0",
)
ax[0].plot(
    (r / d[1, 0]).m_as(ureg.dimensionless),
    g[1, 0, :].m_as(ureg.dimensionless),
    label="CH",
    color="C1",
)
ax[1].plot(
    (k * d[0, 0]).m_as(ureg.dimensionless),
    S_ii[1, 0, :].m_as(ureg.dimensionless),
    label="CH",
    color="C1",
)
ax[0].plot(
    (r / d[1, 1]).m_as(ureg.dimensionless),
    g[1, 1, :].m_as(ureg.dimensionless),
    label="CC",
    color="C2",
)
ax[1].plot(
    (k * d[0, 0]).m_as(ureg.dimensionless),
    S_ii[1, 1, :].m_as(ureg.dimensionless),
    label="CC",
    color="C2",
)


# Compare to the literature

try:
    current_folder = Path(__file__).parent
except NameError:
    current_folder = Path.cwd()

for idx, gtype in enumerate(["HH", "CH", "CC"]):
    xlit, glit = onp.genfromtxt(
        current_folder
        / f"../../../tests/data/Wunsch2011/Fig4.12/g_{gtype}.csv",
        unpack=True,
        delimiter=",",
    )
    klit, Slit = onp.genfromtxt(
        current_folder
        / f"../../../tests/data/Wunsch2011/Fig4.12/S_{gtype}.csv",
        unpack=True,
        delimiter=",",
    )
    ax[0].plot(
        xlit,
        glit,
        ls="dashed",
        label="Literature" if idx == 0 else None,
        color="gray",
    )
    ax[1].plot(
        klit,
        Slit,
        ls="dashed",
        label="Literature" if idx == 0 else None,
        color="gray",
    )

ax[0].set_xlabel("$r [d_i]$")
ax[0].set_ylabel("$g(r)$")

ax[0].set_xlim(0, 3.5)
ax[0].set_ylim(0, 1.5)

ax[1].set_xlabel("$k [1/d_i]$")
ax[1].set_ylabel("$S_{ii}(k)$")
ax[1].set_xlim(0, 9)
ax[1].set_ylim(-0.4, 1.5)

ax[0].legend()

plt.tight_layout()
plt.show()
