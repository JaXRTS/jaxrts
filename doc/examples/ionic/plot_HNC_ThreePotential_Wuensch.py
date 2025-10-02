"""
Static structure factor of the electron-ion system
==================================================

This shows the calculation of pair-distribution functions and static structure
factors of a relatively cold electron-ion system, as suggested by
:cite:`Wunsch.2008`. See fig. 3. of this publication. The work presented shows
to usage of quantum pseudopotentials for the electron-ion and electron-electron
interaction.
"""

import jax.numpy as jnp
import jpu.numpy as jnpu
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

import jaxrts

# plt.style.use("science")

ureg = jaxrts.ureg

state = jaxrts.PlasmaState(
    ions=[jaxrts.Element("H")],
    Z_free=[1],
    mass_density=[120] / (1 * ureg.centimeter**3) * ureg.gram,
    T_e=100 * 1.16e4 * ureg.kelvin,
    T_i=[100 * 1.16e4] * ureg.kelvin,
)

pot = 18
r = jnpu.linspace(1e-1 * ureg.a0, 5e3 * ureg.a0, 2**pot)

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


for ElectronIonPotential, ElectronElectronPotential, mix in [
    [
        jaxrts.hnc_potentials.DeutschPotential(),
        jaxrts.hnc_potentials.DeutschPotential(),
        0.8,
    ],
    [
        jaxrts.hnc_potentials.CoulombPotential(),
        jaxrts.hnc_potentials.KelbgPotential(),
        0.95,
    ],
    [
        jaxrts.hnc_potentials.CoulombPotential(),
        jaxrts.hnc_potentials.DeutschPotential(),
        0.8,
    ],
    [
        jaxrts.hnc_potentials.KlimontovichKraeftPotential(),
        jaxrts.hnc_potentials.KelbgPotential(),
        0.8,
    ],
]:
    fig, ax = plt.subplots(2, 2, sharex="col", figsize=(6, 8))

    IonIonPotential = jaxrts.hnc_potentials.CoulombPotential()
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
        V_s, V_l_k, r, IonIonPotential.T(state), n, masses, mix=mix
    )
    S_ii = jaxrts.hypernetted_chain.S_ii_HNC(k, g, n, r)

    # The Fist value should not be trusted
    ax[1, 1].plot(
        (k[1:] * d[0, 0]).m_as(ureg.dimensionless),
        S_ii[0, 0, 1:].m_as(ureg.dimensionless),
        label="$S_{ii}$",
    )
    ax[1, 1].plot(
        (k[1:] * d[1, 0]).m_as(ureg.dimensionless),
        S_ii[1, 0, 1:].m_as(ureg.dimensionless),
        label="$S_{ei}$",
    )
    ax[1, 1].plot(
        (k[1:] * d[1, 1]).m_as(ureg.dimensionless),
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
