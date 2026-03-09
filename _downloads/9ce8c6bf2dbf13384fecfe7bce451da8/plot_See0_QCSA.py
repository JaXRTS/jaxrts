"""
Plot See0 using the Quantum corrected Salpetera approximation
=============================================================

The following plot is a reproduction of Fig 1b in :cite:`Gregori.2003`.
"""

# TO-DO: Add also Fig. 1 a+c

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp

import jaxrts
import jaxrts.free_free as free_free

ureg = jaxrts.units.ureg

plt.style.use("science")

lambda_0 = 4.13 * ureg.nanometer
theta = 160
k = (4 * jnp.pi / lambda_0) * jnp.sin(jnp.deg2rad(theta) / 2.0)

for count, T in enumerate(
    [
        0.5 * ureg.electron_volts,
        2.0 * ureg.electron_volts,
        8.0 * ureg.electron_volts,
    ]
):
    E = jnp.linspace(-10, 10, 500) * ureg.electron_volts
    vals = free_free.S0_ee_Salpeter(
        k,
        T_e=T / (1 * ureg.boltzmann_constant),
        n_e=1e21 / ureg.centimeter**3,
        E=E,
    )
    if count == 0:
        norm = onp.max(vals)
    plt.plot(
        E.m_as(ureg.electron_volt),
        (vals / norm).m_as(ureg.dimensionless),
        label="T = " + str(T.m_as(ureg.electron_volt)) + " eV",
    )

plt.xlabel(r"$\omega$ [eV]")
plt.ylabel(r"$S^0_{\text{ee}}$ [arb. units]")

plt.legend()
plt.tight_layout()
plt.show()
