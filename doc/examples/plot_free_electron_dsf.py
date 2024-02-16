"""
Comparison of free electron dynamic structures as calculated by Gregori et al. (Reproduction of Fig. 1b of Gregori.2003)
========================================================================================================================
"""

# TO-DO: Add also Fig. 1 a+c

import matplotlib.pyplot as plt
import numpy as onp
import scienceplots

import jaxrts
import jaxrts.electron_feature as ef

import jax.numpy as jnp

ureg = jaxrts.units.ureg

plt.style.use("science")

lambda_0 = 4.13 * ureg.nanometer
theta = 160
k = (4 * jnp.pi / lambda_0) * jnp.sin(jnp.deg2rad(theta) / 2.0)
    
count = 0
norm = 1.0
for T in [0.5 * ureg.electron_volts, 2.0 * ureg.electron_volts, 13.0 * ureg.electron_volts]:
    E = jnp.linspace(-10, 10, 500) * ureg.electron_volts
    vals = ef.S0_ee(k, T_e = T / (1 * ureg.boltzmann_constant), n_e = 1e21 / ureg.centimeter**3, E = E)
    count += 1
    if(count == 1):
        norm = onp.max(vals)
    plt.plot(E, vals / norm, label = 'T = ' + str(T.magnitude) + " eV")

plt.xlabel(r"$\omega$ [eV]")
plt.ylabel(r"$S^0_{\text{ee}}$ [arb. units]")

plt.legend()
plt.tight_layout()
plt.show()
