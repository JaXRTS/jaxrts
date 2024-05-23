"""
Compare HNC Calculations to Gregori.2006
========================================

This figure is reproducing Fig. 2 in :cite:`Schwarz.2007`, showing the notable
difference of the two approaches to the static structure factor :math:`S_{ii}`.
"""

from pathlib import Path
import jax.numpy as jnp
import jpu
import matplotlib.pyplot as plt
import scienceplots
import numpy as onp

import jaxrts
from jaxrts import hypernetted_chain as hnc
from jaxrts import ureg

plt.style.use("science")

fig, ax = plt.subplots()

Z = 2.5

q = hnc.construct_q_matrix(jnp.array([-1, Z]) * 1 * ureg.elementary_charge)
T_e = 12 * ureg.electron_volt / ureg.k_B
n = jnp.array([Z * 1.21e23, 1.21e23]) * (1 / ureg.centimeter**3)

alpha = hnc.construct_alpha_matrix(n)

r = jnp.linspace(0.05, 1000, 2**13) * ureg.angstrom

dr = r[1] - r[0]
dk = jnp.pi / (len(r) * dr)
k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk

m = (
    jnp.array(
        [
            (1 * ureg.electron_mass).m_as(ureg.gram),
            (9 * ureg.proton_mass).m_as(ureg.gram),
        ]
    )
    * ureg.gram
)

# (1/mu = 1/m1 + 1/m2)
mu = jpu.numpy.outer(m, m) / (m[:, jnp.newaxis] + m[jnp.newaxis, :])

for idx, frac in enumerate([1.0, 2.0, 4.0]):

    T_i = T_e / frac
    T = (m[0] * T_i + m[1] * T_e) / (m[0] + m[1])
    T = (
        jnp.array(
            [
                [T_e.m_as(ureg.kelvin), T.m_as(ureg.kelvin)],
                [T.m_as(ureg.kelvin), T_i.m_as(ureg.kelvin)],
            ]
        )[:, :, jnp.newaxis]
        * ureg.kelvin
    )

    T_e_prime = jaxrts.static_structure_factors.T_cf_Greg(T_e, n[0])
    T_D = jaxrts.static_structure_factors.T_Debye_Bohm_Staver(
        T_e_prime, n[0], m[1], Z
    )
    T_i_prime = jaxrts.static_structure_factors.T_i_eff_Greg(T_i, T_D)
    # Compared to Gregori.2003, there is a pi missing
    lambda_ab = (
        ureg.hbar
        * jpu.numpy.sqrt(1 / (2 * mu[:, :, jnp.newaxis] * ureg.k_B * T))[
            :, :, 0
        ]
    )

    KK = -hnc.V_Klimontovich_Kraeft_r(r, q, lambda_ab, T)
    Kelbg = hnc.V_Kelbg_r(r, q, lambda_ab) * (
        1
        - jpu.numpy.exp(
            -r[jnp.newaxis, jnp.newaxis, :] * alpha[:, :, jnp.newaxis]
        )
    )
    # Use the Coulomb - potential for the long-range part
    V_Coulomb_l_k = hnc.V_screened_C_l_k(k, q, alpha)

    V = Kelbg * jnp.eye(2)[:, :, jnp.newaxis]
    V_l_k = V_Coulomb_l_k * jnp.eye(2)[:, :, jnp.newaxis]
    V += KK * jnp.eye(2, k=1)[:, :, jnp.newaxis]
    V += KK * jnp.eye(2, k=-1)[:, :, jnp.newaxis]

    g, niter = hnc.pair_distribution_function_HNC(V, V_l_k, r, T, n)
    print(niter)
    S_ii = hnc.S_ii_HNC(k, g, n, r)

    ax.plot(
        (k * ureg.a_0).m_as(ureg.dimensionless),
        S_ii[1, 1, :].m_as(ureg.dimensionless),
        label=str(frac),
        color=f"C{idx}",
    )

    S_ii_Gregori = jaxrts.static_structure_factors.S_ii_AD(
        k, T_e_prime, T_i_prime, n[0], m[1], Z
    )
    ax.plot(
        (k * ureg.a_0).m_as(ureg.dimensionless),
        S_ii_Gregori.m_as(ureg.dimensionless),
        ls="dashed",
        color=f"C{idx}",
    )
ax.set_xlim(0, 5)
ax.set_xlabel("$k [1/a_0]$")
ax.set_ylabel("S_{ii}")
ax.text(2.4, 0.02, "Solid: HNC\nDashed: Gregori.2006")
ax.legend()
plt.tight_layout()
plt.show()
