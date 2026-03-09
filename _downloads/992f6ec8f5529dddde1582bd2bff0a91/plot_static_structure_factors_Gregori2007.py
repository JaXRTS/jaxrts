"""
Plot static structure factors in Charged Hard Sphere Approximation
==================================================================

This plot reproduces Fig. 3. of :cite:`Gregori.2007`, introducing the charged
hard sphere (CHS) approximation. Compare to
plot_static_structure_factors_Gregori2006 which shows results of Arkhipov-based
models at similar conditions.
"""

import matplotlib.pyplot as plt
from jax import numpy as jnp

import jaxrts

ureg = jaxrts.ureg

m_i = jaxrts.Element("Be").atomic_mass
n_e = ureg("2.5e23/cm³")
T_e = ureg("20eV") / (1 * ureg.k_B)
Z_f = 2

T_e_prime = jaxrts.static_structure_factors.T_cf_Greg(T_e, n_e)
T_D = jaxrts.static_structure_factors.T_Debye_Bohm_Staver(
    T_e_prime, n_e, m_i, Z_f
)
k_De = jaxrts.static_structure_factors._k_D_AD(T_e_prime, n_e)
k_f = jaxrts.plasma_physics.fermi_wavenumber(n_e)

k_over_kf = jnp.linspace(0, 6, 400)

k = k_over_kf * k_f


# plt.style.use("science")
fig, ax = plt.subplots(2, 2, sharex=True, figsize=(5, 4))

for i, T_i_over_T_e in enumerate([0.2, 0.5, 1.0]):
    T_i = T_e * T_i_over_T_e
    T_i_prime = jaxrts.static_structure_factors.T_i_eff_Greg(T_i, T_D)
    lfc = jaxrts.ee_localfieldcorrections.eelfc_farid(k, T_e, n_e)

    Sii = jaxrts.static_structure_factors.S_ii_CHS(
        k,
        T_e_prime,
        T_i_prime,
        n_e,
        m_i,
        Z_f,
        n_e / Z_f,
        lfc,
    )
    Sei = jaxrts.static_structure_factors.S_ei_CHS(
        k,
        T_e_prime,
        T_i_prime,
        n_e,
        m_i,
        Z_f,
        n_e / Z_f,
        lfc,
    )
    See = jaxrts.static_structure_factors.S_ee_CHS(
        k,
        T_e_prime,
        T_i_prime,
        n_e,
        m_i,
        Z_f,
        n_e / Z_f,
        lfc,
    )
    # This is the q calculated by Gregori.2006
    simple_q = jnp.sqrt(Z_f) * Sei / Sii
    # For comparison, also calculate the full q using e.g. Gregri 2003 and
    # assert that these results are not too different.
    q = jaxrts.ion_feature.q_Gregori2004(k, m_i, n_e, T_e, T_i, Z_f)
    q = jnp.real(q.m_as(ureg.dimensionless))

    ax[0, 0].plot(k_over_kf, Sii.m_as(ureg.dimensionless), label=T_i_over_T_e)
    ax[0, 0].hlines(
        [(1 + Z_f * T_e_prime / T_i_prime) ** (-1)],
        0,
        6,
        color="gray",
        ls="dashed",
    )
    ax[1, 0].plot(k_over_kf, Sei.m_as(ureg.dimensionless), label=T_i_over_T_e)
    ax[0, 1].plot(k_over_kf, See.m_as(ureg.dimensionless), label=T_i_over_T_e)
    ax[1, 1].plot(
        k_over_kf,
        simple_q.m_as(ureg.dimensionless),
        color=f"C{i}",
        label="Simple q" if T_i_over_T_e == 0.1 else None,
    )
    ax[1, 1].plot(
        k_over_kf[3:],
        q[3:],
        color=f"C{i}",
        label="Full q" if T_i_over_T_e == 0.1 else None,
        ls="dashed",
    )

for a in ax[1, :]:
    a.set_xlabel("$k/k_{f}$")

for a in [ax[0, 0], ax[1, 0], ax[0, 1]]:
    a.set_ylim(-0.05, 1.35)

ax[0, 0].set_ylabel("$S_{ii}(k)$")
ax[1, 0].set_ylabel("$S_{ei}(k)$")
ax[0, 1].set_ylabel("$S_{ee}(k)$")
ax[1, 1].set_ylabel("$q(k)$")

ax[0, 1].legend(loc="lower right")
ax[1, 1].legend(loc="upper right")

plt.tight_layout()

plt.show()
