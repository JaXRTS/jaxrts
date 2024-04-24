import pytest

import sys
sys.path.append("C:/Users/Samuel/Desktop/PhD/Python_Projects/JAXRTS/jaxrts/src")
from jax import numpy as jnp
import jaxrts

import jpu

import jaxrts.hypernetted_chain as hnc

import matplotlib.pyplot as plt

from jaxrts.units import ureg

r = jpu.numpy.linspace(0.0001 * ureg.angstrom, 100 * ureg.a0, 10000)
q = hnc.construct_q_matrix(jnp.array([1]) * 1 * ureg.elementary_charge)
T = 10 * ureg.electron_volt / ureg.boltzmann_constant

Gamma = 30
d = 1 / (Gamma * (1 * ureg.boltzmann_constant) * T * 4 * jnp.pi * ureg.epsilon_0 / ureg.elementary_charge ** 2)

n = (1 / (d ** 3 * (4 * jnp.pi / 3))).to(1 / ureg.centimeter**3)

n = jnp.array([n.m_as(1 / ureg.centimeter**3)])* (1 / ureg.centimeter**3)

d = jpu.numpy.cbrt(
    3 / (4 * jnp.pi * (n[:, jnp.newaxis] + n[jnp.newaxis, :]) / 2)
)

alpha = hnc.construct_alpha_matrix(n)

V_s = hnc.V_s(r, q, alpha)
V_l = hnc.V_l(r, q, alpha)

g, niter = hnc.pair_distribution_function_HNC(V_s, V_l, T, n)

print(niter)

plt.plot((r/d[0,0]).m_as(ureg.dimensionless), g[0,0,:])
plt.xlim(0, 5.0)
plt.ylim(0, 1.2)
plt.show()
