"""
HNC-SVT: multi-component, multi-temperature (M-SVT)
==================================================
This example tests M-SVT model and reproduces Fig. 4.12 from :cite:`Wunsch.2011` 
for $T_e = T_i$, and reproduces Fig. 4 (b) from :cite:`Bredow.2013` 
for non-equilibrium case. For the equilibrium case, the M-SVT method gives identical
results as the Fig. 4.12 from :cite:`Wunsch.2011`. For the non-equilibrium case, 
the M-SVT result is slight different from cite:`Bredow.2013`.
"""

from pathlib import Path

import jax.numpy as jnp
import jpu.numpy as jnpu

# import scienceplots  # noqa: F401
import matplotlib.pyplot as plt
import numpy as onp

import jaxrts
from jaxrts import hypernetted_chain as hnc

# plt.style.use("science")
ureg = jaxrts.ureg

# ======================================================================
#                        reproduce wunsch 2011
# =====================================================================

# fig, ax = plt.subplots(1, 2, figsize=(8, 4))

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
    T_i=jnp.array([2e4, 2e4]) * ureg.kelvin,
)

pot = 12
r = jnpu.linspace(0.01 * ureg.angstrom, 200 * ureg.a0, 2**pot)

# We add densities, here. Maybe this is wrong.
d = jnpu.cbrt(
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

g, niter = hnc.pair_distribution_function_SVT_HNC(
    V_s,
    V_l_k,
    r,
    Potential.T(state),
    state.n_i,
    jaxrts.units.to_array([ion.atomic_mass for ion in state.ions]),
)
S_ii = hnc.S_ii_HNC(k, g, state.n_i, r)
fig, ax = plt.subplots(2, 1)
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
fig.suptitle("MSVT reproduce wunsch.2011 (Fig. 4.12)")
plt.tight_layout()
plt.show()

# =====================================================================
#              reproduce bredow2013hypernetted (Fig. 4b)
# =====================================================================
state = jaxrts.PlasmaState(
    ions=[jaxrts.Element("H")],
    Z_free=[1],
    mass_density=[1e23 * ureg.proton_mass / (1 * ureg.centimeter**3)],
    T_e=13.6 * 1.16e4 * ureg.kelvin,
    T_i=[2.72 * 1.16e4] * ureg.kelvin,
)

pot = 12
r = jnpu.linspace(1e-2 * ureg.a0, 200 * ureg.a0, 2**pot)
d = jnpu.cbrt(
    3
    / (
        4
        * jnp.pi
        * (state.n_i[:, jnp.newaxis] + state.n_i[jnp.newaxis, :])
        / 2
    )
)

dr = r[1] - r[0]
dk = jnp.pi / (len(r) * dr)
k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk


for ElectronIonPotential, ElectronElectronPotential, mix, tmult in [
    [
        jaxrts.hnc_potentials.DeutschPotential(),
        jaxrts.hnc_potentials.DeutschPotential(),
        0.8,
        [1.5],
    ],
]:
    fig, ax = plt.subplots(2, 2, sharex="col", figsize=(6, 8))

    IonIonPotential = jaxrts.hnc_potentials.DeutschPotential()
    CoulombPotential = jaxrts.hnc_potentials.CoulombPotential()

    for Potential in [
        IonIonPotential,
        ElectronIonPotential,
        ElectronElectronPotential,
        CoulombPotential,
    ]:
        Potential.include_electrons = "SpinAveraged"

    unit = ureg.electron_volt
    V_s = IonIonPotential.full_r(state, r).m_as(unit)

    V_s = V_s.at[-1, :, :].set(
        ElectronIonPotential.full_r(state, r)[-1, :, :].m_as(unit)
    )
    V_s = V_s.at[:, -1, :].set(
        ElectronIonPotential.full_r(state, r)[:, -1, :].m_as(unit)
    )
    V_s = V_s.at[-1, -1, :].set(
        ElectronElectronPotential.full_r(state, r)[-1, -1, :].m_as(unit)
    )
    V_s *= unit
    V_s -= CoulombPotential.long_r(state, r)

    ax[0, 0].plot(
        (r / d[0, 0]).m_as(ureg.dimensionless),
        V_s[0, 0, :].m_as(unit),
        label="$V_{ii}$",
    )
    ax[0, 0].plot(
        (r / d[1, 0]).m_as(ureg.dimensionless),
        V_s[1, 0, :].m_as(unit),
        label="$V_{ei}$",
    )
    ax[0, 0].plot(
        (r / d[1, 1]).m_as(ureg.dimensionless),
        V_s[1, 1, :].m_as(unit),
        label="$V_{ee}$",
    )

    unit = ureg.electron_volt * ureg.angstrom**3
    V_l_k = CoulombPotential.long_k(state, k).m_as(unit)

    ax[0, 1].plot(
        (k * d[0, 0]).m_as(ureg.dimensionless),
        V_l_k[0, 0, :],
        label="$V_{ii}$",
    )
    ax[0, 1].plot(
        (k * d[1, 0]).m_as(ureg.dimensionless),
        V_l_k[1, 0, :],
        label="$V_{ei}$",
    )
    ax[0, 1].plot(
        (k * d[1, 1]).m_as(ureg.dimensionless),
        V_l_k[1, 1, :],
        label="$V_{ee}$",
    )
    V_l_k *= unit
    n = jaxrts.units.to_array([*state.n_i, state.n_e])

    masses = jaxrts.units.to_array(
        [*[ion.atomic_mass for ion in state.ions], 1 * ureg.electron_mass]
    )
    g, niter = jaxrts.hypernetted_chain.pair_distribution_function_SVT_HNC(
        V_s,
        V_l_k,
        r,
        IonIonPotential.T(state),
        n,
        masses,
        mix=mix,
        tmult=tmult,
    )
    S_ii = jaxrts.hypernetted_chain.S_ii_HNC(k, g, n, r)

    # The Fist value should not be trusted
    ax[1, 1].plot(
        (k[1:]).m_as(1 / ureg.a0),
        S_ii[0, 0, 1:].m_as(ureg.dimensionless),
        label="$S_{ii}$",
    )
    ax[1, 1].plot(
        (k[1:]).m_as(1 / ureg.a0),
        S_ii[1, 0, 1:].m_as(ureg.dimensionless),
        label="$S_{ei}$",
    )
    ax[1, 1].plot(
        (k[1:]).m_as(1 / ureg.a0),
        S_ii[1, 1, 1:].m_as(ureg.dimensionless),
        label="$S_{ee}$",
    )
    ax[1, 0].plot(
        (r / d[0, 0]).m_as(ureg.dimensionless),
        g[0, 0, :].m_as(ureg.dimensionless),
        label="$g_{ii}$",
    )
    ax[1, 0].plot(
        (r / d[1, 0]).m_as(ureg.dimensionless),
        g[1, 0, :].m_as(ureg.dimensionless),
        label="$g_{ei}$",
    )
    ax[1, 0].plot(
        (r / d[1, 1]).m_as(ureg.dimensionless),
        g[1, 1, :].m_as(ureg.dimensionless),
        label="$g_{ee}$",
    )

    ax[1, 0].set_xlim(0, 3)
    ax[1, 1].set_xlim(0, 13)

    ax[1, 1].set_xlabel("$k$ [1/a$_i$]")
    ax[1, 0].set_xlabel("$r$ [a$_i$]")

    ax[0, 0].set_title("Short-ranged Potential")
    ax[0, 1].set_xlabel("long-ranged Potential (FT)")
    ax[1, 0].set_xlabel("Pair distribution function")
    ax[1, 1].set_xlabel("Static structure factor")

    for axis in ax.flatten():
        axis.legend()

    fig.suptitle(
        f"e-i Potential: {ElectronIonPotential.__name__}, e-e Potential: {ElectronElectronPotential.__name__}"  # noqa: E501
    )

plt.tight_layout()
plt.show()


data_ee = onp.genfromtxt(
    "../../../tests/data/bredow2013/right-ee.csv", delimiter=","
)
data_ei = onp.genfromtxt(
    "../../../tests/data/bredow2013/right-ei.csv", delimiter=","
)
data_ii = onp.genfromtxt(
    "../../../tests/data/bredow2013/right-ii.csv", delimiter=","
)
plt.figure()
plt.plot(
    (k[1:]).m_as(1 / ureg.a0),
    S_ii[0, 0, 1:].m_as(ureg.dimensionless),
    label="$S_{ii}$",
)
plt.plot(data_ii[:, 0], data_ii[:, 1], color="gray", ls="dashed")
plt.plot(
    (k[1:]).m_as(1 / ureg.a0),
    S_ii[1, 0, 1:].m_as(ureg.dimensionless),
    label="$S_{ei}$",
)
plt.plot(data_ei[:, 0], data_ei[:, 1], color="gray", ls="dashed")
plt.plot(
    (k[1:]).m_as(1 / ureg.a0),
    S_ii[1, 1, 1:].m_as(ureg.dimensionless),
    label="$S_{ee}$",
)
plt.plot(
    data_ee[:, 0],
    data_ee[:, 1],
    label="Bredow et al (2013).",
    color="gray",
    ls="dashed",
)
plt.xlabel("$k$ [1/a$_i$]")
plt.xlim(0, 2.5)
plt.ylabel("$S(k)$")
plt.legend()
plt.title("MSVT reproduce Bredow et al. 2013 (Fig. 4b)")
plt.show()
