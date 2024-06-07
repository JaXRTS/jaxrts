"""
Calculate the full full structure factors for various plasma conditions
=======================================================================

This code reproduces Fig. 6 of :cite:`Gregori.2003`, showing the impact of
changing temperature, ionization and electron density on the scattering
spectra.

It also shows how to attach :py:class:`jaxrts.models.Model` objects to a
:py:class`jaxrts.plasma_state.PlasmaState`, to easily calculate the full S_ee
with :py:meth:`jaxrts.plasma_state.PlasmaState.probe`.
"""

import jaxrts

import jax.numpy as jnp

from jaxrts.models import (
    Neglect,
    Gregori2003IonFeat,
    RPA_NoDamping,
    SchumacherImpulse,
)

from functools import partial

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401


ureg = jaxrts.ureg

element = jaxrts.elements.Element("Be")


def set_state(state, Z_f, n_e, T):
    state.Z_free = jnp.array([Z_f])
    state.mass_density = (
        jnp.array([n_e])
        / (1 * ureg.centimeter**3)
        * element.atomic_mass
        / state.Z_free[0]
    )
    state.T_e = jnp.array([T]) * ureg.electron_volt / ureg.k_B


state = jaxrts.PlasmaState(
    ions=[element],
    Z_free=jnp.array([2]),
    mass_density=jnp.array([3e23])
    / (1 * ureg.centimeter**3)
    * element.atomic_mass
    / 2,
    T_e=jnp.array([10]) * ureg.electron_volt / ureg.k_B,
)

setup = jaxrts.setup.Setup(
    ureg("160°"),
    ureg("4750 eV"),
    ureg("4750 eV") + jnp.linspace(-250, 100, 500) * ureg.electron_volt,
    partial(
        jaxrts.instrument_function.instrument_gaussian,
        sigma=ureg("50.0eV") / ureg.hbar / (2 * jnp.sqrt(2 * jnp.log(2))),
    ),
)

state["ionic scattering"] = Gregori2003IonFeat
state["free-free scattering"] = RPA_NoDamping
state["bound-free scattering"] = SchumacherImpulse
state["free-bound scattering"] = Neglect

plt.style.use("science")
fig, ax = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(8, 8))

state.Z_free = jnp.array([2.5])
state.mass_density = (
    jnp.array([3e23])
    / (1 * ureg.centimeter**3)
    * element.atomic_mass
    / state.Z_free[0]
)
state.probe(setup).m_as(ureg.second),

for Z, ne, T in [(2, 0.5e23, 1), (2, 1e23, 1), (2, 5e23, 1)]:
    set_state(state, Z, ne, T)
    ax[0, 0].plot(
        (setup.measured_energy - setup.energy).m_as(ureg.electron_volt),
        state.probe(setup).m_as(ureg.second),
        label=f"$n_e = $ {ne}/cm$^3$",
    )
    ax[0, 0].set_title(f"$Z=$ {Z}, $T=$ {T}eV")

for Z, ne, T in [(2, 0.5e23, 10), (2, 1e23, 10), (2, 5e23, 10)]:
    set_state(state, Z, ne, T)
    ax[1, 0].plot(
        (setup.measured_energy - setup.energy).m_as(ureg.electron_volt),
        state.probe(setup).m_as(ureg.second),
        label=f"$n_e = $ {ne}/cm$^3$",
    )
    ax[1, 0].set_title(f"$Z=$ {Z}, $T=$ {T}eV")

for Z, ne, T in [(2, 0.5e23, 40), (2, 1e23, 40), (2, 5e23, 40)]:
    set_state(state, Z, ne, T)
    ax[2, 0].plot(
        (setup.measured_energy - setup.energy).m_as(ureg.electron_volt),
        state.probe(setup).m_as(ureg.second),
        label=f"$n_e = $ {ne}/cm$^3$",
    )
    ax[2, 0].set_title(f"$Z=$ {Z}, $T=$ {T}eV")

for Z, ne, T in [(2.0, 3e23, 1), (2.5, 3e23, 1), (3.0, 3e23, 1)]:
    set_state(state, Z, ne, T)
    ax[0, 1].plot(
        (setup.measured_energy - setup.energy).m_as(ureg.electron_volt),
        state.probe(setup).m_as(ureg.second),
        label=f"$Z=$ {Z}",
    )
    ax[0, 1].set_title(f"$n_e=$ {ne}/cm$^3$, $T=$ {T}eV")

for Z, ne, T in [(2.0, 3e23, 10), (2.5, 3e23, 10), (3.0, 3e23, 10)]:
    set_state(state, Z, ne, T)
    ax[1, 1].plot(
        (setup.measured_energy - setup.energy).m_as(ureg.electron_volt),
        state.probe(setup).m_as(ureg.second),
        label=f"$Z=$ {Z}",
    )
    ax[1, 1].set_title(f"$n_e=$ {ne}/cm$^3$, $T=$ {T}eV")

for Z, ne, T in [(2.0, 3e23, 40), (2.5, 3e23, 40), (3.0, 3e23, 40)]:
    set_state(state, Z, ne, T)
    ax[2, 1].plot(
        (setup.measured_energy - setup.energy).m_as(ureg.electron_volt),
        state.probe(setup).m_as(ureg.second),
        label=f"$Z=$ {Z}",
    )
    ax[2, 1].set_title(f"$n_e=$ {ne}/cm$^3$, $T=$ {T}eV")

for a in ax.flatten():
    a.legend()

for a in ax[2, :]:
    a.set_xlabel("Energy shift [eV]")
for a in ax[:, 0]:
    a.set_ylabel("$S_{ee}^{tot}$ [s]")

plt.show()
