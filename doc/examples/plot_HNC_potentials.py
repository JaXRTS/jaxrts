"""
HNC Potentials
==============
"""

from jaxrts import ureg
from jaxrts import hypernetted_chain as hnc
import jpu
import jax.numpy as jnp

import matplotlib.pyplot as plt

import scienceplots

plt.style.use("science")

r = jpu.numpy.linspace(0.001 * ureg.angstrom, 10 * ureg.a0, 2**12).to(
    ureg.angstrom
)

dr = r[1] - r[0]
dk = jnp.pi / (len(r) * dr)
k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk

q = hnc.construct_q_matrix(jnp.array([1]) * 1 * ureg.elementary_charge)
T = 10 * ureg.electron_volt / ureg.boltzmann_constant

Gamma = 30
di = 1 / (
    Gamma
    * (1 * ureg.boltzmann_constant)
    * T
    * 4
    * jnp.pi
    * ureg.epsilon_0
    / ureg.elementary_charge**2
)

n = (1 / (di**3 * (4 * jnp.pi / 3))).to(1 / ureg.angstrom**3)
n = jnp.array([n.m_as(1 / ureg.angstrom**3)]) * (1 / ureg.angstrom**3)
alpha = hnc.construct_alpha_matrix(n)

V_l = hnc.V_l(r, q, alpha)
V_s = hnc.V_s(r, q, alpha)

V_l_k_analytical = hnc.V_l_k(k, q, alpha)
V_l_k_transformed, _ = hnc.transformPotential(V_l, r)

fig, ax = plt.subplots(2)
ax[0].plot(
    r.m_as(ureg.angstrom),
    V_s[0, 0, :],
    label="$V_s^C$",
)
ax[0].plot(
    r.m_as(ureg.angstrom),
    V_l[0, 0, :],
    label="$V_l^C$",
)


ax[1].plot(
    k.m_as(1 / ureg.angstrom),
    V_l_k_analytical[0, 0, :],
    label="$V_l^C$ (analytic)",
)
ax[1].plot(
    k.m_as(1 / ureg.angstrom),
    V_l_k_transformed[0, 0, :],
    label="$V_l^C$ (transformed)",
)

ax[0].set_ylim(0.01, 25)
ax[0].set_xlim(-0.01, 0.151)
ax[1].set_xlim(-0.01, 25.1)
ax[0].set_xlabel("$r$ [$\\AA$]")
ax[1].set_xlabel("$k$ [1/$\\AA$]")
ax[0].legend()
ax[1].legend()

plt.tight_layout()
plt.show()
