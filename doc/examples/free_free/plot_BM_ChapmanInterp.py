"""
Number of interpolation in the Born Mermin Chapman Interpolation
================================================================

The Chapman Interpolation of the Born Mermin approximation saves computation
time by evaluating the Structure factor not on all frequencies, but only on a
given number of points, and interpolates after. This example is investigating
the required number of points for the interpolation.

.. note::

    The time printed in this script is the time after the first compilation
    (which normally takes a notable time).

"""

import os

from functools import partial
import time

import matplotlib.pyplot as plt
import scienceplots
import jax
import jax.numpy as jnp


import jaxrts

# Allow jax to use 6 CPUs, see
# https://astralord.github.io/posts/exploring-parallel-strategies-with-jax/
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=6"

ureg = jaxrts.units.ureg

plt.style.use("science")

# Create a sharding for the probing energies
sharding = jax.sharding.PositionalSharding(jax.devices())
measured_energy = jnp.linspace(295, 305, 300) * ureg.electron_volt
input_energy = jax.device_put(measured_energy, sharding)

setup = jaxrts.setup.Setup(
    ureg("60°"),
    energy=ureg("300eV"),
    measured_energy=input_energy,
    instrument=partial(
        jaxrts.instrument_function.instrument_gaussian,
        sigma=(0.01 * ureg.electron_volt) / ureg.hbar,
    ),
)
state = jaxrts.PlasmaState(
    [jaxrts.Element("H")],
    jnp.array([1.0]),
    jnp.array([0.0017]) * ureg.gram / ureg.centimeter**3,
    jnp.array([2]) * ureg.electron_volt / ureg.k_B,
    jnp.array([2]) * ureg.electron_volt / ureg.k_B,
)

state["free-free scattering"] = jaxrts.models.BornMerminFull()
# This is required for the S_ii in the collision frequency
state["ionic scattering"] = jaxrts.models.ArkhipovIonFeat()

# This is required for the V_eiS in the collision frequency
state["BM V_eiS"] = jaxrts.models.DebyeHueckel_BM_V()
state.evaluate("free-free scattering", setup).m_as(ureg.second)
t0 = time.time()
BM_free_free_scatter = state.evaluate("free-free scattering", setup).m_as(
    ureg.second
)
print(f"Full BMA: {time.time()-t0}s")
state["free-free scattering"] = jaxrts.models.BornMermin()

for no_of_freq in [2, 4, 20, 100]:
    state["free-free scattering"].no_of_freq = no_of_freq
    state.evaluate("free-free scattering", setup).m_as(ureg.second)
    t0 = time.time()
    free_free_scatter = state.evaluate("free-free scattering", setup).m_as(
        ureg.second
    )
    print(
        f"{no_of_freq} interp points: {time.time()-t0}s      ",
        "Mean deviation from full RPA: ",
        jnp.mean(free_free_scatter - BM_free_free_scatter),
        "Max deviation from full RPA: ",
        jnp.max(free_free_scatter - BM_free_free_scatter),
    )
    plt.plot(
        setup.measured_energy.m_as(ureg.electron_volt),
        free_free_scatter,
        label=f"{no_of_freq} interpolation points",
        linestyle="solid",
        alpha=0.8,
    )

plt.plot(
    setup.measured_energy.m_as(ureg.electron_volt),
    BM_free_free_scatter,
    label="Full BMA",
    color="black",
)

state["free-free scattering"] = jaxrts.models.RPA_NoDamping()
state.evaluate("free-free scattering", setup).m_as(ureg.second)
t0 = time.time()
free_free_scatter = state.evaluate("free-free scattering", setup).m_as(
    ureg.second
)
print(f"RPA: {time.time()-t0}s")
plt.plot(
    setup.measured_energy.m_as(ureg.electron_volt),
    free_free_scatter,
    label="RPA",
    linestyle="dashed",
    color="gray",
)

plt.xlabel("Energy [eV]")
plt.ylabel("Scattering intensity")
plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1.00))
plt.show()
