"""
HNC-SVT: multi-component, multi-temperature (M-SVT)
===================================================

This example tests M-SVT model and reproduces Fig. 4.12 from
:cite:`Wunsch.2011`.

For equilibrium case, the M-SVT method gives identical results as the Fig. 4.12
from :cite:`Wunsch.2011`. And the results are also consistent with the results
from plot_HNC_multicomponent.py.

For non-equilibrium case, the M-SVT result is slightly different from
:cite:`Bredow.2013`.
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
#                   reproduce wunsch 2011 (Fig. 4.12)
# =====================================================================

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
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
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


ElectronIonPotential = jaxrts.hnc_potentials.DeutschPotential()
ElectronElectronPotential = jaxrts.hnc_potentials.DeutschPotential()
mix = 0.8
tmult = [1.5]

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


unit = ureg.electron_volt * ureg.angstrom**3
V_l_k = CoulombPotential.long_k(state, k).m_as(unit)

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
index_map = {"ii": (0, 0), "ei": (1, 0), "ee": (1, 1)}
x_calculated = (k[1:]).m_as(1 / ureg.a0)

plt.figure(figsize=(6, 4))
for gtype, indices in index_map.items():
    file_path = (
        current_folder / f"../../../tests/data/bredow2013/right-{gtype}.csv"
    )
    data_lit = onp.genfromtxt(file_path, delimiter=",")

    i, j = indices
    y_calculated = S_ii[i, j, 1:].m_as(ureg.dimensionless)
    plt.plot(x_calculated, y_calculated, label=f"$S_{{{gtype}}}$")
    plt.plot(data_lit[:, 0], data_lit[:, 1], color="gray", ls="dashed")

plt.plot([], [], label="literature", color="gray", ls="dashed")

info_text = (
    r"$T_e = 13.6\,$eV" + "\n"
    r"$T_i = 2.72\,$eV" + "\n"
    r"$n_{e} = n_{H^+}= 10^{23}\,cc$"
)

bbox_props = dict(
    boxstyle="round,pad=0.4", lw=0.5, fc="none", ec="none", alpha=1
)
plt.gca().text(
    0.97,
    0.75,
    info_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=bbox_props,
)

plt.xlabel("$k$ [1/a$_i$]")
plt.xlim(0, 2.5)
plt.ylabel("$S(k)$")
plt.legend(bbox_to_anchor=(1.01, 0.5), loc="upper right", frameon=False)
plt.title("MSVT reproduce Bredow et al. 2013 (Fig. 4b)")
plt.tight_layout()
plt.show()
