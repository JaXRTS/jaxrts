"""
HNC: Pair correlation function for different Γ
==============================================

This example reproduces Figure 4.4 in :cite:`Wunsch.2011`, showing the the pair
distribution function oscillating more with increasing :math:`Gamma` (e.g,
increasing density when temperature stays constant).

We assume fully ionized hydrogen at a temperature of 10 eV, and a statistically
screened Coulomb potential.

Due to limitations in the numerical discrete evaluation of the 3D Fourier
transform, we get reasonable results and are able to reproduce the literature
value well when starting from an analytical experission for the long range
potential in :math:`r` space, and performing a conversion to :math:`k` space
with :py:func:`jaxrts.hypernetted_chain.transformPotental`. 
"""

from jaxrts import hypernetted_chain as hnc
import jax.numpy as jnp
import jpu
from jaxrts import ureg
import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)

for idx, Gamma in enumerate([1, 10, 30, 100]):
    pot = [13, 13, 15, 16][idx]
    axis = [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]][idx]
    q = hnc.construct_q_matrix(jnp.array([1]) * 1 * ureg.elementary_charge)

    T = 10 * ureg.electron_volt / ureg.boltzmann_constant
    di = 1 / (
        Gamma
        * (1 * ureg.boltzmann_constant)
        * T
        * 4
        * jnp.pi
        * ureg.epsilon_0
        / ureg.elementary_charge**2
    )

    r = jpu.numpy.linspace(0.0001 * ureg.angstrom, 20 * ureg.a0, 2**pot)

    n = (1 / (di**3 * (4 * jnp.pi / 3))).to(1 / ureg.angstrom**3)

    n = jnp.array([n.m_as(1 / ureg.angstrom**3)]) * (1 / ureg.angstrom**3)

    d = jpu.numpy.cbrt(
        3 / (4 * jnp.pi * (n[:, jnp.newaxis] + n[jnp.newaxis, :]) / 2)
    )

    alpha = hnc.construct_alpha_matrix(n)

    V_s = hnc.V_screenedC_s_r(r, q, alpha)

    dr = r[1] - r[0]
    dk = jnp.pi / (len(r) * dr)
    k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk

    V_l_k = hnc.V_screened_C_l_k(k, q, alpha)
    V_l = hnc.V_screened_C_l_r(r, q, alpha)
    V_l_k, _ = hnc.transformPotential(V_l, r)

    g, niter = hnc.pair_distribution_function_HNC(V_s, V_l_k, r, T, n)
    print(f"Γ={Gamma}, {niter} iteration of the HNC scheme.")

    axis.plot(
        (r / d[0, 0]).m_as(ureg.dimensionless),
        g[0, 0, :].m_as(ureg.dimensionless),
    )
    axis.set_title("$\\Gamma =$ " + str(Gamma))
    axis.set_xlabel("$r/d_i$")
    axis.set_ylabel("$g(r)$")

plt.xlim(0, 5.0)
plt.ylim(0, 1.5)
plt.tight_layout()
plt.show()
