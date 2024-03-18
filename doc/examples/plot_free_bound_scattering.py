"""
Showcase the :py:class:`jaxrts.models.DetailedBalace` free-bound scattering Model
=================================================================================

This plot is comparable to the target analzed by :cite:`Doppner.2023`, is this
example is one of the dataset revisited by :Bohme.2023`.
However, we used simpler for the relevant scattering processes, just to show
the usage of :py:class:`jaxrts.models.DetailedBalace`.
"""

from functools import partial
import matplotlib.pyplot as plt
import numpy as onp
import scienceplots

import jaxrts

import jax.numpy as jnp

ureg = jaxrts.units.ureg

plt.style.use("science")

setup = jaxrts.setup.Setup(
    ureg("120°"),
    energy=ureg("9keV"),
    measured_energy=jnp.linspace(8, 9.5, 300) * ureg.kiloelectron_volt,
    instrument=partial(
        jaxrts.instrument_function.instrument_gaussian,
        sigma=(30.0 * ureg.electron_volt) / ureg.hbar,
    ),
)
state = jaxrts.PlasmaState(
    [jaxrts.Element("Be")],
    jnp.array([3.0]),
    jnp.array([1.0]),
    jnp.array([9.3]) * ureg.gram / ureg.centimeter**3,
    jnp.array([119]) * ureg.electron_volt / ureg.k_B,
    jnp.array([119]) * ureg.electron_volt / ureg.k_B,
)


state["ionic scattering"] = jaxrts.models.Gregori2003IonFeat
state["free-free scattering"] = jaxrts.models.RPA_NoDamping
state["bound-free scattering"] = jaxrts.models.SchumacherImpulse
state["free-bound scattering"] = jaxrts.models.DetailedBalance

plt.plot(
    setup.measured_energy.m_as(ureg.electron_volt),
    state["free-free scattering"].evaluate(setup).m_as(ureg.second),
    label="free-free",
    linestyle="dashed",
)
plt.plot(
    setup.measured_energy.m_as(ureg.electron_volt),
    state["bound-free scattering"].evaluate(setup).m_as(ureg.second),
    label="bound-free",
)
plt.plot(
    setup.measured_energy.m_as(ureg.electron_volt),
    state["free-bound scattering"].evaluate(setup).m_as(ureg.second),
    label="free-bound",
)
plt.plot(
    setup.measured_energy.m_as(ureg.electron_volt),
    state.probe(setup).m_as(ureg.second),
    lw=2,
    color="black",
    label="full",
)
plt.xlabel("Energy [eV]")
plt.ylabel("Scattering intensity")
plt.legend()
plt.tight_layout()
plt.show()
