from time import time

import jaxrts.hypernetted_chain as hnc
import jpu.numpy as jnpu
from jaxrts import ureg
import jax.numpy as jnp
import matplotlib.pyplot as plt

r = jnp.linspace(0.1, 10, 1000) * ureg.angstrom
dr = r[1] - r[0]
dk = jnp.pi / (len(r) * dr)
k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk

alpha = 0.2 / ureg.angstrom
q = ureg.elementary_charge**2

V_r = q**2 / (4 * jnp.pi * ureg.epsilon_0 * r) * (1 - jnpu.exp(-alpha * r))
V_k = q**2 / (k**2 * ureg.epsilon_0) * alpha**2 / (k**2 + alpha**2)

V_k_sine = hnc._3Dfour(k, r, V_r[jnp.newaxis, jnp.newaxis, :])
V_k_ogata = hnc._3Dfour_ogata(k, r, V_r[jnp.newaxis, jnp.newaxis, :])

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

plt.legend()

plt.show()
