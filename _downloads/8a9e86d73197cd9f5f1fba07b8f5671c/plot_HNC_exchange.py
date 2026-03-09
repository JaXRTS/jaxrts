"""
HNC calculations with Quantum exchange potentials
=================================================

This example shows the usage of the
:py:class:`jaxrts.hnc_potentials.SpinAveragedEEExchange` potential, reproducing
Fig. 6 from :cite:`Wunsch.2008`.

This example also shows how :py:class:`jaxrts.hnc_potentials.HNCPotential`
objects can be added.
"""

import jax.numpy as jnp
import jpu.numpy as jnpu
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

import jaxrts

plt.style.use("science")

ureg = jaxrts.ureg

element = jaxrts.Element("Be")

state = jaxrts.PlasmaState(
    ions=[element],
    Z_free=[2.2],
    mass_density=[1.23e23] / (1 * ureg.centimeter**3) * element.atomic_mass,
    T_e=1.39e5 * ureg.kelvin,
)

# Increasing max(r) results in better fits for the Sii@low k, but increases
# computation time
pot = 16
r = jnpu.linspace(5e-2 * ureg.a0, 1e3 * ureg.a0, 2**pot)
mix = 0.9

d = jnpu.cbrt(3 / (4 * jnp.pi * state.n_i))

dr = r[1] - r[0]
dk = jnp.pi / (len(r) * dr)
k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk

fig, ax = plt.subplots(2, 1, figsize=(6, 8))

for ElectronIonPotential, ElectronElectronPotential, ls in [
    [
        jaxrts.hnc_potentials.DeutschPotential(),
        jaxrts.hnc_potentials.DeutschPotential(),
        "solid",
    ],
    [
        jaxrts.hnc_potentials.DeutschPotential(),
        jaxrts.hnc_potentials.DeutschPotential()
        + jaxrts.hnc_potentials.SpinAveragedEEExchange(),
        "dashed",
    ],
]:

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
    print(V_s[:, :, 10])
    V_s -= CoulombPotential.long_r(state, r)

    unit = ureg.electron_volt * ureg.angstrom**3
    V_l_k = CoulombPotential.long_k(state, k)

    n = jaxrts.units.to_array([*state.n_i, state.n_e])

    g, niter = jaxrts.hypernetted_chain.pair_distribution_function_HNC(
        V_s, V_l_k, r, IonIonPotential.T(state), n, mix=mix
    )
    S_ab = jaxrts.hypernetted_chain.S_ii_HNC(k, g, n, r)

    # The Fist value should not be trusted
    ax[1].plot(
        (k[1:] * d).m_as(ureg.dimensionless),
        S_ab[0, 0, 1:].m_as(ureg.dimensionless),
        label="$ii$" if ls == "solid" else None,
        color="C0",
        ls=ls,
    )
    ax[1].plot(
        (k[1:] * d).m_as(ureg.dimensionless),
        S_ab[1, 0, 1:].m_as(ureg.dimensionless),
        label="$ei$" if ls == "solid" else None,
        color="C1",
        ls=ls,
    )
    ax[1].plot(
        (k[1:] * d).m_as(ureg.dimensionless),
        S_ab[1, 1, 1:].m_as(ureg.dimensionless),
        label="$ee$" if ls == "solid" else None,
        color="C2",
        ls=ls,
    )
    try:
        pot_label = ElectronElectronPotential.description
    except AttributeError:
        pot_label = ElectronElectronPotential.__name__

    ax[0].plot(
        (r / d).m_as(ureg.dimensionless),
        g[0, 0, :].m_as(ureg.dimensionless),
        label=pot_label,
        color="C0",
        ls=ls,
    )
    ax[0].plot(
        (r / d).m_as(ureg.dimensionless),
        g[1, 0, :].m_as(ureg.dimensionless),
        color="C1",
        ls=ls,
    )
    ax[0].plot(
        (r / d).m_as(ureg.dimensionless),
        g[1, 1, :].m_as(ureg.dimensionless),
        color="C2",
        ls=ls,
    )

# Perform the calulation for a an exchange which consideres different electron
# spins
IonIonPotential = jaxrts.hnc_potentials.CoulombPotential()
CoulombPotential = jaxrts.hnc_potentials.CoulombPotential()
ElectronIonPotential = jaxrts.hnc_potentials.DeutschPotential()
ElectronElectronPotential = jaxrts.hnc_potentials.DeutschPotential()
ExchangePotential = jaxrts.hnc_potentials.SpinSeparatedEEExchange()
ls = "dotted"

for Potential in [
    IonIonPotential,
    ElectronIonPotential,
    ElectronElectronPotential,
    CoulombPotential,
]:
    Potential.include_electrons = "SpinSeparated"

unit = ureg.electron_volt
V_s = IonIonPotential.full_r(state, r).m_as(unit)

V_s = V_s.at[-2, :, :].set(
    ElectronIonPotential.full_r(state, r)[-2, :, :].m_as(unit)
)
V_s = V_s.at[-1, :, :].set(
    ElectronIonPotential.full_r(state, r)[-1, :, :].m_as(unit)
)
V_s = V_s.at[:, -2, :].set(
    ElectronIonPotential.full_r(state, r)[:, -2, :].m_as(unit)
)
V_s = V_s.at[:, -1, :].set(
    ElectronIonPotential.full_r(state, r)[:, -1, :].m_as(unit)
)
V_s = V_s.at[-2:, -2:, :].set(
    ElectronElectronPotential.full_r(state, r)[-2:, -2:, :].m_as(unit)
)
V_s = V_s.at[-1, -1, :].set(
    (ElectronElectronPotential + ExchangePotential)
    .full_r(state, r)[-1, -1, :]
    .m_as(unit)
)
V_s = V_s.at[-2, -2, :].set(
    (ElectronElectronPotential + ExchangePotential)
    .full_r(state, r)[-2, -2, :]
    .m_as(unit)
)
V_s *= unit

print(V_s[:, :, 10])
V_s -= CoulombPotential.long_r(state, r)

unit = ureg.electron_volt * ureg.angstrom**3
V_l_k = CoulombPotential.long_k(state, k)

n = jaxrts.units.to_array([*state.n_i, state.n_e / 2, state.n_e / 2])

g, niter = jaxrts.hypernetted_chain.pair_distribution_function_HNC(
    V_s, V_l_k, r, IonIonPotential.T(state), n, mix=mix
)

# Sum up the contributions to the g_ee
g_sum = g[:-1, :-1, :].m_as(ureg.dimensionless)
g_sum = g_sum.at[-1, -1, :].set(
    (g[-1, -1, :] + g[-1, -2, :]).m_as(ureg.dimensionless) / 2
)
g_sum *= ureg.dimensionless
n = jaxrts.units.to_array([*state.n_i, state.n_e])
S_ab = jaxrts.hypernetted_chain.S_ii_HNC(k, g_sum, n, r)

# The Fist value should not be trusted
ax[1].plot(
    (k[1:] * d).m_as(ureg.dimensionless),
    S_ab[0, 0, 1:].m_as(ureg.dimensionless),
    label="$ii$" if ls == "solid" else None,
    color="C0",
    ls=ls,
)
ax[1].plot(
    (k[1:] * d).m_as(ureg.dimensionless),
    S_ab[1, 0, 1:].m_as(ureg.dimensionless),
    label="$ei$" if ls == "solid" else None,
    color="C1",
    ls=ls,
)
ax[1].plot(
    (k[1:] * d).m_as(ureg.dimensionless),
    S_ab[1, 1, 1:].m_as(ureg.dimensionless),
    label="$ee$" if ls == "solid" else None,
    color="C2",
    ls=ls,
)

pot_label = (ElectronElectronPotential + ExchangePotential).description

ax[0].plot(
    (r / d).m_as(ureg.dimensionless),
    g[0, 0, :].m_as(ureg.dimensionless),
    label=pot_label,
    color="C0",
    ls=ls,
)
ax[0].plot(
    (r / d).m_as(ureg.dimensionless),
    g[1, 0, :].m_as(ureg.dimensionless),
    color="C1",
    ls=ls,
)
ax[0].plot(
    (r / d).m_as(ureg.dimensionless),
    g[1, 2, :].m_as(ureg.dimensionless),
    color="C2",
    ls=ls,
    alpha=0.3,
)
ax[0].plot(
    (r / d).m_as(ureg.dimensionless),
    g[2, 2, :].m_as(ureg.dimensionless),
    color="C2",
    ls=ls,
    alpha=0.3,
)
ax[0].plot(
    (r / d).m_as(ureg.dimensionless),
    g_sum[-1, -1, :].m_as(ureg.dimensionless),
    color="C2",
    ls=ls,
)

ax[0].set_title("$g_{ab}(r)$")
ax[0].set_xlabel("$r$ [a$_i$]")
ax[0].set_xlim(0, 2.5)
ax[0].set_ylim(0, 2.5)
ax[1].set_title("$S_{ab}(k)$")
ax[1].set_xlabel("$k$ [1/a$_i$]")
ax[1].set_xlim(0, 7)

for axis in ax.flatten():
    axis.legend()

plt.tight_layout()
plt.show()
