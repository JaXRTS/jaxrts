from time import time

import sys

sys.path.append(
    "C:/Users/Samuel/Desktop/PhD/Python_Projects/JAXRTS/jaxrts/src"
)

import jaxrts.hypernetted_chain as hnc
import jpu.numpy as jnpu
from jaxrts import ureg
import jax.numpy as jnp
import matplotlib.pyplot as plt

r = jnp.linspace(0.0001, 1000, 5000) * ureg.nanometer
dr = r[1] - r[0]
dk = jnp.pi / (len(r) * dr)
k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk

# k_log = jnp.logspace(-3, 2, len(r)) / (1 * ureg.angstrom)

alpha = 2 / ureg.nanometer
q = ureg.elementary_charge**2

# V_r = q**2 / (4 * jnp.pi * ureg.epsilon_0 * r) * (1 - jnpu.exp(-alpha * r))
# V_k = q**2 / (k**2 * ureg.epsilon_0) * alpha**2 / (k**2 + alpha**2)

V_r = jnpu.exp(-r**2 * alpha**2)
V_k = jnpu.sqrt(jnp.pi / alpha**2)**3 * jnpu.exp(-k**2 / (4 * alpha**2))

t0 = time()
V_k_sine = hnc._3Dfour(k, r, V_r[jnp.newaxis, jnp.newaxis, :])
print("Time for sine:", time()-t0, "s.")
t0 = time()
V_k_ogata = hnc._3Dfour_ogata(k, r, V_r[jnp.newaxis, jnp.newaxis, :])
print("Time for ogata:", time()-t0, "s.")

# unit = ureg.angstrom ** 3
unit = V_k.units

plt.plot(
    k.m_as(1 / ureg.angstrom),
    V_k.m_as(unit),
    label="analytical",
)
plt.plot(
    k.m_as(1 / ureg.angstrom),
    V_k_sine[0, 0, :].m_as(unit),
    label="sine",
)
plt.plot(
    k.m_as(1 / ureg.angstrom),
    V_k_ogata[0, 0, :].m_as(unit),
    label="ogata",
)

plt.xlim(0,10)
plt.ylim(0,1.1)

plt.legend()

plt.show()

# plt.plot(k, (V_k_ogata[0,0,:] / V_k).m_as(ureg.dimensionless))
# plt.show()
