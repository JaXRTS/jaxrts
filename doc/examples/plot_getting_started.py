"""
Getting Started
===============

This example showcases a simple, one-component plasma, and might be a good
starting point.
"""

from functools import partial

import matplotlib.pyplot as plt

# jax provides a submodule which can be (mostly) used as a drop-in replacement
# for numpy. We use jnp to avoid confusion.
from jax import numpy as jnp

import jaxrts

# We use the jpu package (which is enabling the usage of pint with jax) to
# handle units. See below for how to use this
ureg = jaxrts.ureg

# Initialize a Plasma State (We use a Berillium Plasma, here)
state = jaxrts.PlasmaState(
    ions=[jaxrts.Element("Be")],  # Specify the elements as a list
    Z_free=jnp.array([2]),  # This is the ionization per species
    mass_density=jnp.array([1]) * ureg.gram / ureg.centimeter**3,
    T_e=5 * ureg.electron_volt / ureg.k_B,  # T_e is the electron temperature.
    # If no T_i (ion temperature) is given, we assume equilibrium of ion and
    # electron temperatures
)

# Now, we also have to define a Setup
# jpu and pint also allow to convert string to quantities with units, as you
# can see below.
setup = jaxrts.Setup(
    scattering_angle=ureg("60°"),
    energy=ureg("4700 eV"),
    measured_energy=ureg("4700 eV")
    + jnp.linspace(-100, 40, 500) * ureg.electron_volt,
    instrument=partial(
        jaxrts.instrument_function.instrument_gaussian,
        sigma=ureg("5.0eV") / ureg.hbar / (2 * jnp.sqrt(2 * jnp.log(2))),
    ),
)

# Add Models. In the typical way, we use the Chihara decomposition in the code,
# differentiating 4 contributions to the full XRTS signal, based on the
# different processes causing them.
#
# These processes are
#
# free-free scattering
# bound-free scattering
# free-bound scattering
# elastic scattering by electrons, which follow the motion of the ions, closely
#
# At a minimum, a user has to set models for these contributions. This is done
# by adding keys to the plasma state.


state["ionic scattering"] = jaxrts.models.ThreePotentialHNCIonFeat(SVT=True, mix=0.5)
state["free-free scattering"] = jaxrts.models.RPA_DandreaFit()
state["bound-free scattering"] = jaxrts.models.SchumacherImpulse()
state["free-bound scattering"] = jaxrts.models.DetailedBalance()

# Generate the spectrum
See_tot = state.probe(setup)

# Plot the result
plt.plot(
    setup.measured_energy.m_as(ureg.electron_volt),
    See_tot.m_as(ureg.second),
)
plt.xlabel("Probed Energy [eV]")
plt.ylabel("$S_{ee}^{tot}$ [s]")
plt.title("Be plasma at 2eV and 1g/cc with Z=2")
plt.show()
