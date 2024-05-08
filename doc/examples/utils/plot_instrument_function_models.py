"""
All available instrument function models.
=========================================
"""

import matplotlib.pyplot as plt
import numpy as onp
import matplotlib.pyplot as plt
import scienceplots

import jaxrts
import jaxrts.instrument_function as ifs

import jax.numpy as jnp

ureg = jaxrts.units.ureg

plt.style.use("science")

fig, ax = plt.subplots(figsize=(7, 4))

x = jnp.linspace(-100, 100, 500) * ureg.electron_volts

plt.plot(
    x,
    ifs.instrument_gaussian(x, 20 * ureg.electron_volts),
    label=r"Gaussian Model",
)
plt.plot(
    x,
    ifs.instrument_supergaussian(x, 20 * ureg.electron_volts, 2),
    label=r"Super-Gaussian Model (p=1)",
)
plt.plot(
    x,
    ifs.instrument_lorentzian(x, 20 * ureg.electron_volts),
    label=r"Lorentzian Model",
)

plt.xlabel("E [eV]")
plt.ylabel("Intensity [arb. units]")

plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
