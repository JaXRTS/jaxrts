"""
Moving edges vs cold ionoization edges
======================================

This script is used to test the bound-free edges using ionization energies from
FAC code calculated by the FAC code:
https://github.com/flexible-atomic-code/fac and compares them to a bound-free
model using only the cold edges.
"""

import matplotlib.pyplot as plt
from functools import partial
import jax
from jax import numpy as jnp
import time
import jaxrts
import os

import matplotlib as mpl

# ----------- Global settings ------------
mpl.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.family": "serif",
        "figure.dpi": 300,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2,
        "text.usetex": False,
    }
)

ureg = jaxrts.ureg


@partial(jax.jit)
def calc_BF(state, setup):
    See_bf = state.evaluate("bound-free scattering", setup)
    E = setup.measured_energy.m_as(ureg.electron_volt)
    return (
        E,
        See_bf.m_as(ureg.second),
    )


# ==========================================================
#                   Input parameters
# ----------------------------------------------------------
ions = (jaxrts.Element("C"), jaxrts.Element("H"))
rho = ureg("1g/cc")
number_fraction = jnp.array([1 / 2, 1 / 2])

temperature = 50.0
Z_free = jnp.array([3.5, 1.0])
scattering_angle = 166.0
photon_energy = 8.5
source_fwhm = 2.0
# ----------------------------------------------------------

measured_energy = jnp.linspace(5.5, 8.6, 4096) * ureg.kiloelectron_volt
T_e = temperature * ureg.electron_volt / ureg.k_B
T_i = jnp.array([temperature, temperature]) * ureg.electron_volt / ureg.k_B

# Calculate the spectrum
# ==========================================================

# Initialize plasma state with different temperatures for electrons
mass_fraction = jaxrts.helpers.mass_from_number_fraction(number_fraction, ions)
state = jaxrts.PlasmaState(
    ions=ions,
    Z_free=Z_free,  # Ionization per species
    mass_density=rho * mass_fraction,
    T_e=T_e,
    T_i=T_i,
)
setup = jaxrts.Setup(
    scattering_angle=ureg(f"{scattering_angle} deg"),  # Explicit units
    energy=ureg(f"{photon_energy} keV"),
    measured_energy=measured_energy,
    instrument=partial(
        jaxrts.instrument_function.instrument_gaussian,
        sigma=ureg(f"{source_fwhm}eV")
        / ureg.hbar
        / (2 * jnp.sqrt(2 * jnp.log(2))),
    ),
)
state["ipd"] = jaxrts.models.StewartPyattIPD()
state["bound-free scattering"] = jaxrts.models.SchumacherImpulse(r_k=1)

# =============================================================================

E, See_bf = calc_BF(state, setup)
state["bound-free scattering"] = jaxrts.models.SchumacherImpulseColdEdges(
    r_k=1
)
E, See_bf_cold = calc_BF(state, setup)
fig = plt.figure(figsize=(6, 4))
plt.plot(
    E - 8500,
    See_bf_cold,
    label="total bound-free (cold edge)",
    color="gray",
    lw=1.5,
)
plt.plot(E - 8500, See_bf, label="total bound-free", color="red", lw=1.5)
plt.xlabel("Energy [eV]")
plt.ylabel(r"$S_{ee}^{bf}$ [s]")
plt.title(r"Synthetic spectrum, [$C^{3.5+}+H^{1+}$] plasma")
plt.legend(frameon=False, loc="best")
plt.xlim([-800, 50])
plt.tight_layout()

plt.show()
