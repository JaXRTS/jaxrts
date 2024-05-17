"""
HNC: Electron-Ion Potentials
============================

This example reproduces Figure 4.6 in :cite:`Wunsch.2011`, showing
the different potentials implemented to treat the electron-ion interaction.

This plot shows quite plainly, notable differences between various models.

.. warning::

    We are not able to reproduce Fig. 4.6 from :cite:`Wunsch.2011`. Even if
    :math:`\\lambda_{ab}` is fixed to a forcing the Deutsch Potential to the
    currect value at :math:`r = 0`, would need to modify the temperature to
    match the Klimontvich-Kraeft potential.

"""

from jaxrts import hypernetted_chain as hnc
import jax.numpy as jnp
import jpu
from jaxrts import ureg
import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")

fig, ax = plt.subplots()

q = hnc.construct_q_matrix(jnp.array([-1, 2]) * 1 * ureg.elementary_charge)
T = 1e5 * ureg.kelvin
n = jnp.array([2 * 1.23e23, 1.23e23]) * ureg.centimeter ** (-3)

d = jpu.numpy.cbrt(
    3 / (4 * jnp.pi * (n[:, jnp.newaxis] + n[jnp.newaxis, :]) / 2)
)
r = jnp.linspace(0, 10, 1000) * ureg.angstrom

m = (
    jnp.array(
        [
            (1 * ureg.electron_mass).m_as(ureg.gram),
            (4 * ureg.proton_mass).m_as(ureg.gram),
        ]
    )
    * ureg.gram
)

# (1/mu = 1/m1 + 1/m2)
mu = jpu.numpy.outer(m, m) / (m[:, jnp.newaxis] + m[jnp.newaxis, :])
# See :cite:`Schwarz.2007`
lambda_ab = ureg.hbar * jpu.numpy.sqrt(1 / (2 * mu * ureg.k_B * T))
print(lambda_ab.to(ureg.angstrom))

# Get lambda_ab from the limit of Deutsch:
# for r-> 0: V_Deutsch -> q1q2/(4 pi eps_0 lam_ab)
# This will only be correct for the ei part!
lambda_ab = -q / (4 * jnp.pi * ureg.epsilon_0 * 5 * ureg.rydberg)
print(lambda_ab.to(ureg.angstrom))

# See :cite:`Gregori.2003`
lambda_ab = ureg.hbar * jpu.numpy.sqrt(1 / (2 * jnp.pi * mu * ureg.k_B * T))
print(lambda_ab.to(ureg.angstrom))

alpha = hnc.construct_alpha_matrix(n)

ax.plot(
    (r / d[0, 0]).m_as(ureg.dimensionless),
    -hnc.V_Klimontovich_Kraeft_r(r, q, lambda_ab, T)[1, 0, :].m_as(
        ureg.rydberg
    ),
    label="Klimontvich Kraeft",
)
ax.plot(
    (r / d[0, 0]).m_as(ureg.dimensionless),
    -hnc.V_Kelbg_r(r, q, lambda_ab)[1, 0, :].m_as(ureg.rydberg),
    label="Kelbg",
)

ax.plot(
    (r / d[0, 0]).m_as(ureg.dimensionless),
    -hnc.V_Deutsch_r(r, q, lambda_ab)[1, 0, :].m_as(ureg.rydberg),
    label="Deutsch",
)

ax.set_xlabel("$r\\; [d_i$]")
ax.set_ylabel("$V_{ei}\\; [\\mathrm{Ryd}]$")

ax.legend()
ax.set_xlim(0, 5.0)
# ax.set_ylim(0, 10)
plt.tight_layout()
plt.show()
