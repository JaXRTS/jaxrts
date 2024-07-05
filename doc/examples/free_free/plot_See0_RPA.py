"""
Plot See0 using the RPA approximation
=====================================

This script reprocudes :cite:`Gregori.2003`, Fig. 1c. showing the calculation
of :math:`S_\\text{ee}^{0, \\text{RPA}}`.
"""

import matplotlib.pyplot as plt
import numpy as onp
import scienceplots

import jaxrts
import jaxrts.free_free as free_free

import jax.numpy as jnp

ureg = jaxrts.units.ureg

plt.style.use("science")

lambda_0 = 0.26 * ureg.nanometer
theta = 60
n_e = 1e23 / ureg.centimeter**3

E = jnp.linspace(-100, 100, 500) * ureg.electron_volts

k = (4 * jnp.pi / lambda_0) * jnp.sin(jnp.deg2rad(theta) / 2.0)

count = 0
norm = 1.0

for T in [
    0.8 * ureg.electron_volts,
    3.0 * ureg.electron_volts,
    13.0 * ureg.electron_volts,
]:
    mu = jaxrts.plasma_physics.chem_pot_interpolationIchimaru(
        T / (1 * ureg.boltzmann_constant), n_e
    )

    vals = free_free.S0_ee_RPA_no_damping(
        k,
        T_e=T / (1 * ureg.boltzmann_constant),
        n_e=n_e,
        E=E,
        chem_pot=mu,
    ).m_as(ureg.second)

    if count == 0:
        norm = onp.max(vals)

    plt.plot(
        E.m_as(ureg.electron_volt),
        (vals / norm),
        label="T = " + str(T.m_as(ureg.electron_volt)) + " eV",
        color=f"C{count}",
    )
    count += 1

plt.xlabel(r"$\omega$ [eV]")
plt.ylabel(r"$S^0_{\text{ee}}$ [arb. units]")

plt.legend()
plt.tight_layout()
plt.show()
