"""
ITCFT Impact of noise
=====================

In this example, we apply the ITCFT to increasingly noisy data with different
scattering angles.
"""

from functools import partial

import jax
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from jax import numpy as jnp
from jax import random
from jpu import numpy as jnpu

import jaxrts

ureg = jaxrts.ureg

plt.style.use("science")

state = jaxrts.PlasmaState(
    ions=[jaxrts.Element("C")],
    Z_free=jnp.array([3]),
    mass_density=jnp.array([1]) * ureg.gram / ureg.centimeter**3,
    T_e=70 * ureg.electron_volt / ureg.k_B,
)

state["screening length"] = jaxrts.models.ArbitraryDegeneracyScreeningLength()
state["screening"] = jaxrts.models.LinearResponseScreening()
state["ionic scattering"] = jaxrts.models.OnePotentialHNCIonFeat()
state["free-free scattering"] = jaxrts.models.RPA_DandreaFit()
state["bound-free scattering"] = jaxrts.models.SchumacherImpulse()
state["free-bound scattering"] = jaxrts.models.DetailedBalance()

for angle, e_range in [[120, 600], [20, 200]]:
    setup = jaxrts.Setup(
        scattering_angle=ureg(f"{angle}°"),
        energy=ureg("9000 eV"),
        measured_energy=ureg("9000 eV")
        + jnp.linspace(-e_range, e_range, 1000) * ureg.electron_volt,
        instrument=partial(
            jaxrts.instrument_function.instrument_gaussian,
            sigma=ureg("3eV") / ureg.hbar / (2 * jnp.sqrt(2 * jnp.log(2))),
        ),
    )

    # Turn off the dispersion correction
    setup.correct_k_dispersion = False

    S_ee = state.probe(setup)

    fig, ax = plt.subplots(1, 2, figsize=(5, 3))

    # Add some noise
    sigma_rel = [0.0, 0.1, 0.2, 0.3]
    RNGKey = random.PRNGKey(123456789 * angle)
    noise_seed = random.normal(
        RNGKey,
        (
            len(sigma_rel),
            len(setup.measured_energy),
        ),
    )

    for i, sigma in enumerate(sigma_rel[::-1]):

        noise_norm = jnpu.max(
            state["free-free scattering"].evaluate_raw(state, setup)
        )
        noise = noise_seed[i, :] * sigma * noise_norm
        S_ee_noise = S_ee + noise

        ax[0].plot(
            setup.measured_energy.m_as(ureg.electron_volt),
            S_ee_noise.m_as(ureg.second),
            alpha=0.8,
            color=f"C{i}",
            label=sigma,
        )

        x = jnp.linspace(1, 0.9 * e_range) * ureg.electron_volt

        T_auto = jax.vmap(
            jaxrts.analysis.ITCFT,
            in_axes=(None, None, None, 0),
            out_axes=(0),
        )(S_ee_noise, ureg("60/keV"), setup, x)

        ax[1].plot(
            x.m_as(ureg.electron_volt),
            (T_auto * ureg.k_B).m_as(ureg.electron_volt),
            color=f"C{i}",
            ls="solid",
        )
    ax[0].set_ylim(jnp.array([0, 5 * noise_norm.m_as(ureg.second)]))
    ax[1].set_ylim(
        jnp.array([-40, 40]) + (state.T_e * ureg.k_B).m_as(ureg.electron_volt)
    )
    ax[1].hlines(
        [(state.T_e * ureg.k_B).m_as(ureg.electron_volt)],
        *ax[1].get_xlim(),
        color="gray",
        ls="dashed",
    )

    ax[1].set_ylabel("$k_B T$ [eV]")
    ax[0].set_ylabel(
        f"$S_{{ee}}(k={setup.k.m_as(1/ureg.angstrom):.2f}/$\\AA$, \\omega)$ [s]"
    )
    ax[1].set_xlabel("$x$ [eV]")
    ax[0].set_xlabel("$E$ [eV]")
    ax[0].legend()
    fig.tight_layout()
    fig.suptitle(f"Scattering angle = ${angle}^\\circ$")

plt.show()
