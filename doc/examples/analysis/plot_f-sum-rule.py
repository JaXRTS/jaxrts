"""
f-sum rule for the dynamic stucture factor
==========================================

This example shows the computation of the f-sum rule for an example spectrum as
suggested by :cite:`Dornheim.2024`.
If fully ionized, we see excellent agreement for the RPA, and also the
Born-Mermin approximation for the free-free scattering yields in reasonable
results.
The implemented bound-free Model, however, does violate the f-sum rule. One
could leverage this to fix a value for
:py:attr:`jaxrts.models.SchumacherImpulse.r_k`, so that the f-sum rule would be
fulfilled.
"""

from functools import partial

import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp

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
