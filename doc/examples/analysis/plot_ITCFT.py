"""
Imaginary time correlation function thermometry
===============================================

This example showcases how apply the analysis functions to get temperatures
from the Laplace transform of the structure.
"""

from functools import partial

import jax
from jax import numpy as jnp
from jax import random
from jpu import numpy as jnpu

import matplotlib.pyplot as plt

import jaxrts

ureg = jaxrts.ureg

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

plt.plot(
    setup.measured_energy.m_as(ureg.electron_volt), S_ee.m_as(ureg.second)
)

# Add some noise
RNGKey = random.PRNGKey(42)
noise = (
    random.normal(RNGKey, (len(setup.measured_energy),))
    * 0.01
    * jnpu.max(S_ee)
)
S_ee += noise

plt.plot(
    setup.measured_energy.m_as(ureg.electron_volt),
    S_ee.m_as(ureg.second),
    alpha=0.5,
)
plt.show()


# We compare two approaches, the default minimization finding with the more
# forward grid finding
tau = jnp.linspace(1, 60, 500) / (1 * ureg.kiloelectron_volt)

x = jnp.linspace(1, 180) * ureg.electron_volt
T, L = jax.vmap(
    jaxrts.analysis.ITCFT_grid, in_axes=(None, None, None, 0), out_axes=(0, 1)
)(S_ee, tau, setup, x)


plt.plot(x, (T * ureg.k_B).m_as(ureg.electron_volt))
plt.ylim(ymax=1.5 * (state.T_e * ureg.k_B).m_as(ureg.electron_volt))
plt.hlines(
    [(state.T_e * ureg.k_B).m_as(ureg.electron_volt)],
    *plt.xlim(),
    color="gray",
    ls="dashed",
)

T = jax.vmap(
    jaxrts.analysis.ITCFT,
    in_axes=(None, None, None, 0),
    out_axes=(0),
)(S_ee, jnpu.max(tau), setup, x)
plt.plot(x, (T * ureg.k_B).m_as(ureg.electron_volt))

plt.show()

E_shift = -(setup.measured_energy - setup.energy)
instrument = setup.instrument(E_shift / (1 * ureg.hbar))
for x in [10, 20, 30, 40]:
    minimizer = jaxrts.analysis._ITCFT(
        S_ee, E_shift, instrument, E_shift, ureg(f"{x}eV")
    )

    T_scipy = minimizer.get_T(60 / (1 * ureg.kiloelectron_volt))
    tau = jnp.linspace(1e-8, 60)
    L = [minimizer.L(t * (1 / ureg.kiloelectron_volt)) for t in tau]
    plt.plot(tau, L, ls="solid")
    T, L = jaxrts.analysis.ITCFT_grid(
        S_ee, tau / (1 * ureg.kiloelectron_volt), setup, ureg(f"{x}eV")
    )
    plt.plot(tau, L, ls="dashed")
    plt.scatter(
        [0.5 / (T * ureg.k_B).m_as(ureg.kiloelectron_volt)],
        [jnpu.min(L)],
        color="black",
    )
    plt.scatter(
        [0.5 / (T_scipy * ureg.k_B).m_as(ureg.kiloelectron_volt)],
        [jnpu.min(L)],
        color="black",
        marker="x",
    )


plt.show()
