"""
Plot See0 using the RPA approximation
=====================================

This script reprocudes :cite:`Gregori.2003`, Fig. 1c. showing the calculation
of :math:`S_\\text{ee}^{0, \\text{RPA}}`.
"""

import matplotlib.pyplot as plt
import numpy as onp
import scienceplots

import sys
sys.path.append('C:/Users/Samuel/Desktop/PhD/Python_Projects/JAXRTS/jaxrts/src')

import jaxrts
import jaxrts.electron_feature as ef

import jax.numpy as jnp

ureg = jaxrts.units.ureg

plt.style.use("science")

lambda_0 = 4.13 * ureg.nanometer
theta = 160
n_e = 1e21 / ureg.centimeter**3

E = jnp.linspace(-15, 15, 500) * ureg.electron_volts

k = (4 * jnp.pi / lambda_0) * jnp.sin(jnp.deg2rad(theta) / 2.0)

count = 0
norm = 1.0

for T in [
    0.5 * ureg.electron_volts,
    2.0 * ureg.electron_volts,
    8.0 * ureg.electron_volts,
]:
    mu = jaxrts.plasma_physics.chem_pot_interpolation(
        T / (1 * ureg.boltzmann_constant), n_e
    )
    
    vals2 = ef.S0_ee_RPA_no_damping(
        k,
        T_e=T / (1 * ureg.boltzmann_constant),
        n_e=n_e,
        E=E,
        chem_pot=mu,
    ).m_as(ureg.second)

    vals = ef.S0_ee_BMA(
        k,
        T=T / (1 * ureg.boltzmann_constant),
        n_e=n_e,
        E=E,
        chem_pot=mu,
        m_ion = 1 * ureg.proton_mass,
        Zf = 1.0
    ).m_as(ureg.second)

    if count == 0:
        norm = onp.max(vals)

    plt.plot(
        E.m_as(ureg.electron_volt),
        (vals / norm),
        label="T = " + str(T.m_as(ureg.electron_volt)) + " eV, BM",
        color=f"C{count}",
    )
    
    if count == 0:
        norm = onp.max(vals)

    plt.plot(
        E.m_as(ureg.electron_volt),
        (vals2 / norm),
        label="T = " + str(T.m_as(ureg.electron_volt)) + " eV",
        linestyle = "dashed",
        color=f"C{count}",
    )
    count += 1
    
    

plt.xlabel(r"$\omega$ [eV]")
plt.ylabel(r"$S^0_{\text{ee}}$ [arb. units]")

plt.legend()
plt.tight_layout()
plt.show()
