"""
All available instrument function models.
=========================================

To include an instrument function in a :py:class:`jaxrts.Setup`, we require it
to be Callable that takes only one argument, the shift in frequency space. The
function has to be normalized to unify from -infinity to +infinity.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

import jaxrts
import jaxrts.instrument_function as ifs

ureg = jaxrts.units.ureg

plt.style.use("science")

fig, ax = plt.subplots(figsize=(7, 4))

E = jnp.linspace(-100, 100, 500) * ureg.electron_volts
w = E / ureg.hbar


width = 20 * ureg.electron_volt / ureg.hbar

plt.plot(
    E.m_as(ureg.electron_volt),
    ifs.instrument_gaussian(w, width).m_as(ureg.hbar / ureg.electron_volt),
    label="Gaussian Model",
)
plt.plot(
    E.m_as(ureg.electron_volt),
    ifs.instrument_supergaussian(w, width, 2).m_as(
        ureg.hbar / ureg.electron_volt
    ),
    label="Super-Gaussian Model (p=2)",
)
plt.plot(
    E.m_as(ureg.electron_volt),
    ifs.instrument_lorentzian(w, width).m_as(ureg.hbar / ureg.electron_volt),
    label="Lorentzian Model",
)

plt.xlabel("E [eV]")
plt.ylabel("Intensity [hbar / eV]")

plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
