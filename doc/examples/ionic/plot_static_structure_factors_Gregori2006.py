"""
Plot static structure Factors in the Approximation by Gregori for T_i != T_e
============================================================================

This scripts plots :math:`S_{ee}`, :math:`S_{ei}`, :math:`S_{ii}` and :math:`q`
for Beryllium with an ionization of 2 at :math:`T_e = 20\\text{eV}` and 
:math:`n_e = 2.5\\times 10^{23} 1 / \\text{cm}^3`.
This script reproduces Fig.1 in :cite:`Gregori.2006`, showing the impact of
differences between the ion- and electron temperature on the statig structure
factors as well as on the screening charge :math:`q`.
"""

import matplotlib.pyplot as plt
import scienceplots
import jaxrts
from jax import numpy as jnp

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

k_over_kDe = jnp.linspace(0, 10, 400)

k = k_over_kDe * k_De


plt.style.use("science")
fig, ax = plt.subplots(2, 2, sharex=True, figsize=(5, 4))

for i, T_i_over_T_e in enumerate([0.1, 0.5, 1.0]):
    T_i = T_e * T_i_over_T_e
    T_i_prime = jaxrts.static_structure_factors.T_i_eff_Greg(T_i, T_D)

    Sii = jaxrts.static_structure_factors.S_ii_AD(
        k, T_e_prime, T_i_prime, n_e, m_i, Z_f
    )
    Sei = jaxrts.static_structure_factors.S_ei_AD(
        k, T_e_prime, T_i_prime, n_e, m_i, Z_f
    )
    See = jaxrts.static_structure_factors.S_ee_AD(
        k, T_e_prime, T_i_prime, n_e, m_i, Z_f
    )
    # This is the q calculated by Gregori.2006
    simple_q = jnp.sqrt(Z_f) * Sei / Sii
    # For comparison, also calculate the full q using e.g. Gregri 2003 and
    # assert that these results are not too diffrent.
    q = jaxrts.ion_feature.q_Gregori2004(k, m_i, n_e, T_e, T_i, Z_f)
    q = jnp.real(q.m_as(ureg.dimensionless))

    ax[0, 0].plot(k_over_kDe, Sii.m_as(ureg.dimensionless), label=T_i_over_T_e)
    ax[1, 0].plot(k_over_kDe, Sei.m_as(ureg.dimensionless), label=T_i_over_T_e)
    ax[0, 1].plot(k_over_kDe, See.m_as(ureg.dimensionless), label=T_i_over_T_e)
    ax[1, 1].plot(
        k_over_kDe,
        simple_q.m_as(ureg.dimensionless),
        color=f"C{i}",
        label="Simple q" if T_i_over_T_e == 0.1 else None,
    )
    ax[1, 1].plot(
        k_over_kDe[3:],
        q[3:],
        color=f"C{i}",
        label="Full q" if T_i_over_T_e == 0.1 else None,
        ls="dashed",
    )

for a in ax[1, :]:
    a.set_xlabel("$k/k_{De}$")

for a in [ax[0, 0], ax[1, 0], ax[0, 1]]:
    a.set_ylim(-0.05, 1.05)

ax[0, 0].set_ylabel("$S_{ii}(k)$")
ax[1, 0].set_ylabel("$S_{ei}(k)$")
ax[0, 1].set_ylabel("$S_{ee}(k)$")
ax[1, 1].set_ylabel("$q(k)$")

ax[0, 1].legend(loc="lower right")
ax[1, 1].legend(loc="upper right")

plt.tight_layout()

plt.show()
