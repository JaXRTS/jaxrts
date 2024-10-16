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
    ions=[jaxrts.Element("C")],
    Z_free=jnp.array([3]),
    mass_density=jnp.array([1]) * ureg.gram / ureg.centimeter**3,
    T_e=40 * ureg.electron_volt / ureg.k_B,
)
setup = jaxrts.Setup(
    scattering_angle=ureg("60°"),
    energy=ureg("9000 eV"),
    measured_energy=ureg("9000 eV")
    + jnp.linspace(-200, 200, 5000) * ureg.electron_volt,
    instrument=partial(
        jaxrts.instrument_function.instrument_gaussian,
        sigma=ureg("3eV") / ureg.hbar / (2 * jnp.sqrt(2 * jnp.log(2))),
    ),
)

# Turn off the dispersion correction
setup.correct_k_dispersion = False

state["screening length"] = jaxrts.models.ArbitraryDegeneracyScreeningLength()
state["screening"] = jaxrts.models.LinearResponseScreening()
state["ionic scattering"] = jaxrts.models.OnePotentialHNCIonFeat()
state["free-free scattering"] = jaxrts.models.RPA_DandreaFit()
state["bound-free scattering"] = jaxrts.models.SchumacherImpulse()
state["free-bound scattering"] = jaxrts.models.DetailedBalance()


S_ee = state.probe(setup)

# We compare two approaches, the default minimization finding with the more
# forward grid finding
tau = jnp.linspace(1, 60, 500) / (1 * ureg.kiloelectron_volt)

# x is the cut-off Energy

x = jnp.linspace(1, 180) * ureg.electron_volt
T_grid, L_grid = jax.vmap(
    jaxrts.analysis.ITCFT_grid, in_axes=(None, None, None, 0), out_axes=(0, 1)
)(S_ee, tau, setup, x)

T_auto = jax.vmap(
    jaxrts.analysis.ITCFT,
    in_axes=(None, None, None, 0),
    out_axes=(0),
)(S_ee, jnpu.max(tau), setup, x)

fig, ax = plt.subplots()
ax.plot(
    x.m_as(ureg.electron_volt),
    (T_grid * ureg.k_B).m_as(ureg.electron_volt),
    label="grid",
)
ax.plot(
    x.m_as(ureg.electron_volt),
    (T_auto * ureg.k_B).m_as(ureg.electron_volt),
    label="auto",
)
ax.set_ylim(
    jnp.array([-4, 1]) + (state.T_e * ureg.k_B).m_as(ureg.electron_volt)
)
ax.hlines(
    [(state.T_e * ureg.k_B).m_as(ureg.electron_volt)],
    *ax.get_xlim(),
    color="gray",
    ls="dashed",
)

ax.set_ylabel("$k_B T$ [eV]")
ax.set_xlabel("$x$ [eV]")
ax.legend()
fig.tight_layout()

# Plot different Laplace-Transforms for both methods.
fig, ax = plt.subplots()

E_shift = -(setup.measured_energy - setup.energy)
instrument = setup.instrument(E_shift / (1 * ureg.hbar))
for i, x in enumerate([10, 20, 30, 40]):
    minimizer = jaxrts.analysis.ITCF(
        S_ee, E_shift, instrument, E_shift, ureg(f"{x}eV")
    )

    T_auto = minimizer.get_T(60 / (1 * ureg.kiloelectron_volt))
    tau = jnp.linspace(1e-8, 60)
    L_auto = [minimizer.L(t * (1 / ureg.kiloelectron_volt)) for t in tau]
    T_grid, L_grid = jaxrts.analysis.ITCFT_grid(
        S_ee, tau / (1 * ureg.kiloelectron_volt), setup, ureg(f"{x}eV")
    )
    ax.plot(
        tau,
        L_auto,
        ls="dashed",
        color=f"C{i}",
        label="grid" if i == 0 else None,
    )
    ax.plot(
        tau,
        L_grid,
        ls="solid",
        color=f"C{i}",
        alpha=0.5,
        label="auto" if i == 0 else None,
    )
    ax.scatter(
        [0.5 / (T_grid * ureg.k_B).m_as(ureg.kiloelectron_volt)],
        [jnpu.min(L_grid)],
        color="black",
        label="minimum grid" if i == 0 else None,
    )
    ax.scatter(
        [0.5 / (T_auto * ureg.k_B).m_as(ureg.kiloelectron_volt)],
        [jnpu.min(L_grid)],
        color="black",
        marker="x",
        label="minumum auto" if i == 0 else None,
    )

ax.set_xlabel("$\\tau$ [1/keV]")
ax.set_ylabel("$\\mathcal{L}$")
ax.legend()

fig.tight_layout()
plt.show()
