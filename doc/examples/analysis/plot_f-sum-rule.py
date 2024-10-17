"""
Imaginary time correlation function thermometry
===============================================

This example showcases how apply the analysis functions to get temperatures
from the Laplace transform of the structure, as proposed by
:cite:`Dornheim.2022`.
"""

import sys

sys.path.append(
    r"C:\Users\Samuel\Desktop\PhD\Python_Projects\JAXRTS\jaxrts\src"
)
from functools import partial

import jax
from jax import numpy as jnp
from jax import random
from jpu import numpy as jnpu

import matplotlib.pyplot as plt
import scienceplots  # noqa: F501

import jaxrts

ureg = jaxrts.ureg

plt.style.use("science")

state = jaxrts.PlasmaState(
    ions=[jaxrts.Element("N")],
    Z_free=jnp.array([7]),
    mass_density=jnp.array([1]) * ureg.gram / ureg.centimeter**3,
    T_e=40 * ureg.electron_volt / ureg.k_B,
)
setup = jaxrts.Setup(
    scattering_angle=ureg("60°"),
    energy=ureg("9 keV"),
    measured_energy=ureg("9 keV")
    + jnp.linspace(-2000, 2000, 5000) * ureg.electron_volt,
    instrument=partial(
        jaxrts.instrument_function.instrument_gaussian,
        sigma=ureg("5eV") / ureg.hbar / (2 * jnp.sqrt(2 * jnp.log(2))),
    ),
)

# Turn off the dispersion correction
setup.correct_k_dispersion = False

state["screening length"] = jaxrts.models.ArbitraryDegeneracyScreeningLength()
state["screening"] = jaxrts.models.LinearResponseScreening()
state["ionic scattering"] = jaxrts.models.OnePotentialHNCIonFeat()
# state["free-free scattering"] = jaxrts.models.RPA_DandreaFit()
state["free-free scattering"] = jaxrts.models.BornMermin_Fit()
state["BM S_ii"] = jaxrts.models.Sum_Sii()
state["bound-free scattering"] = jaxrts.models.Neglect()
state["free-bound scattering"] = jaxrts.models.DetailedBalance()


S_ee = state.probe(setup)
# fig, ax = plt.subplots()
# ax.plot(setup.measured_energy.m_as(ureg.electron_volt), S_ee.m_as(ureg.second))
# plt.show()
# We compare two approaches, the default minimization finding with the more
# forward grid finding
tau = jnp.linspace(1, 60, 500) / (1 * ureg.kiloelectron_volt)

# x is the cut-off Energy

x = jnp.linspace(1, 1800) * ureg.electron_volt

fsum = jax.vmap(
    jaxrts.analysis.ITCF_fsum,
    in_axes=(None, None, 0),
    out_axes=(0),
)(S_ee, setup, x)

fig, ax = plt.subplots()
ax.plot(
    x.m_as(ureg.electron_volt),
    (fsum).m_as(ureg.electron_volt),
    label="auto",
)

ax.hlines(
    [
        ((-1) * ureg.hbar**2 * setup.k**2 / (2 * ureg.electron_mass)).m_as(
            ureg.electron_volt
        )
    ],
    *ax.get_xlim(),
    color="gray",
    ls="dashed",
)


ax.set_ylabel(r"$\frac{\partial F}{\partial \tau}\vert_{\tau=0}$ [eV]")
ax.set_xlabel("$x$ [eV]")
ax.legend()
fig.tight_layout()

plt.show()
