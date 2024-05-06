"""
Multicomponent HNC
==================

This example reproduces Fig. 4.12 from :cite:`Wunsch.2011`, and shows how to
use the HNC approximation with different ion species.
"""

from pathlib import Path
from jaxrts import hypernetted_chain as hnc
import jax.numpy as jnp
import jpu
from jaxrts import ureg
import matplotlib.pyplot as plt
import scienceplots
import numpy as onp

plt.style.use("science")

fig, ax = plt.subplots(1, 2, sharex=True)

# Set up the ionization, density and temperature for individual ion
# species.
q = hnc.construct_q_matrix(jnp.array([1, 4]) * 1 * ureg.elementary_charge)
n = jnp.array([2.5e23, 2.5e23]) * (1 / ureg.centimeter**3)
T = 2e4 * ureg.kelvin

pot = 14
r = jpu.numpy.linspace(0.0001 * ureg.angstrom, 100 * ureg.a0, 2**pot)

d = jpu.numpy.cbrt(
    3 / (4 * jnp.pi * (n[:, jnp.newaxis] * n[jnp.newaxis, :]) ** (1 / 2))
)

dr = r[1] - r[0]
dk = jnp.pi / (len(r) * dr)
k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk

alpha = hnc.construct_alpha_matrix(n)
# V_l_k = hnc.V_l_k(k, q, alpha)
V_l_k, k = hnc.transformPotential(hnc.V_l(r, q, alpha), r)
V_s = hnc.V_s(r, q, alpha)

g, niter = hnc.pair_distribution_function_HNC(V_s, V_l_k, r, T, n)

ax[0].plot(
    (r / d[0, 0]).m_as(ureg.dimensionless),
    g[0, 0, :].m_as(ureg.dimensionless),
    label="HH",
)
ax[0].plot(
    (r / d[1, 0]).m_as(ureg.dimensionless),
    g[1, 0, :].m_as(ureg.dimensionless),
    label="CH",
)
ax[0].plot(
    (r / d[1, 1]).m_as(ureg.dimensionless),
    g[1, 1, :].m_as(ureg.dimensionless),
    label="CC",
)

for gtype in ["HH", "CH", "CC"]:
    xlit, glit = onp.genfromtxt(
        Path(__file__).parent
        / f"../../tests/data/Wunsch2011/Fig4.12/g_{gtype}.csv",
        unpack=True,
        delimiter=",",
    )
    ax[0].plot(xlit, glit, ls="dashed", label="literature {gtype}")

ax[0].set_xlabel("$r/d_i$")
ax[0].set_ylabel("$g(r)$")

ax[0].set_xlim(0, 3.5)
ax[0].set_ylim(0, 1.5)

plt.tight_layout()
plt.show()
