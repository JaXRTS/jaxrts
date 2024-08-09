"""
Multi-Species Plasmas
=====================

This example calculates a synthetic spectrum of a CHO plasma.
"""

from functools import partial

import matplotlib.pyplot as plt
from jax import numpy as jnp

import jaxrts

ureg = jaxrts.ureg

plt.style.use("science")

ions = [jaxrts.Element("C"), jaxrts.Element("H"), jaxrts.Element("O")]
rho = ureg("3g/cc")
# This helper function can be used to calculate density fractions from number
# fractions:
number_fraction = jnp.array([1 / 4, 1 / 2, 1 / 4])
mass_fraction = jaxrts.helpers.mass_from_number_fraction(number_fraction, ions)

# Initialize a plasma state with has different temperatures for the electrons
state = jaxrts.PlasmaState(
    ions=ions,
    Z_free=jnp.array([3.0, 1.0, 5.0]),  # This is the ionization per species
    mass_density=rho * mass_fraction,
    T_e=6 * ureg.electron_volt / ureg.k_B,
    T_i=jnp.array([5, 4, 4.5]) * ureg.electron_volt / ureg.k_B,
)

setup = jaxrts.Setup(
    scattering_angle=ureg("150°"),
    energy=ureg("8 keV"),
    measured_energy=jnp.linspace(7.4, 8.2, 500) * ureg.kiloelectron_volt,
    instrument=partial(
        jaxrts.instrument_function.instrument_gaussian,
        sigma=ureg("7.0eV") / ureg.hbar / (2 * jnp.sqrt(2 * jnp.log(2))),
    ),
)

# Add the required models

state["ionic scattering"] = jaxrts.models.OnePotentialHNCIonFeat()
state["free-free scattering"] = jaxrts.models.RPA_DandreaFit()
# Models can have arguments:
state["bound-free scattering"] = jaxrts.models.SchumacherImpulse(r_k=1)
state["free-bound scattering"] = jaxrts.models.DetailedBalance()

# Add additional models
state["ipd"] = jaxrts.models.StewartPyattIPD()

# Generate the spectrum
# Here, we evaluate the contributions, individually, with the evaluate method
See_el = state.evaluate("ionic scattering", setup)
See_ff = state.evaluate("free-free scattering", setup)
See_bf = state.evaluate("bound-free scattering", setup)
See_fb = state.evaluate("free-bound scattering", setup)

See_tot = See_el + See_bf + See_fb + See_ff

# Plot the result
plt.plot(
    setup.measured_energy.m_as(ureg.electron_volt),
    See_tot.m_as(ureg.second),
    lw=3,
    label="total",
)
plt.plot(
    setup.measured_energy.m_as(ureg.electron_volt),
    See_ff.m_as(ureg.second),
    ls="dotted",
    label="free-free",
)
plt.plot(
    setup.measured_energy.m_as(ureg.electron_volt),
    See_bf.m_as(ureg.second),
    ls="dashed",
    label="bound-free",
)
plt.plot(
    setup.measured_energy.m_as(ureg.electron_volt),
    See_bf.m_as(ureg.second),
    ls="dashdot",
    label="free-bound",
)
plt.legend()
plt.xlabel("Probed Energy [eV]")
plt.ylabel("$S_{ee}$ [s]")
plt.title("A Multicomponent Plasma")
plt.tight_layout()
plt.show()
