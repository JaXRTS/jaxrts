"""
Frequency redistribution correction
===================================

This example shows how the frequency redistribution correction
(:py:attr:`jaxrts.Setup.frc_exponent`) influences the outcome of
:py:meth:`jaxrts.PlasmaState.probe`.

For a discussion, see :cite:`Crowley.2013`.
"""

from functools import partial

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from jax import numpy as jnp

import jaxrts

ureg = jaxrts.ureg

state = jaxrts.PlasmaState(
    ions=[jaxrts.Element("Be")],  
    Z_free=jnp.array([2]),
    mass_density=jnp.array([1]) * ureg.gram / ureg.centimeter**3,
    T_e=2 * ureg.electron_volt / ureg.k_B, 
)

setup = jaxrts.Setup(
    scattering_angle=ureg("170°"),
    energy=ureg("8700 eV"),
    measured_energy=ureg("8700 eV")
    + jnp.linspace(-666, 100, 500) * ureg.electron_volt,
    instrument=partial(
        jaxrts.instrument_function.instrument_gaussian,
        sigma=ureg("5.0eV") / ureg.hbar / (2 * jnp.sqrt(2 * jnp.log(2))),
    ),
)

state["ionic scattering"] = jaxrts.models.OnePotentialHNCIonFeat()
state["free-free scattering"] = jaxrts.models.RPA_DandreaFit()
state["bound-free scattering"] = jaxrts.models.SchumacherImpulse()
state["free-bound scattering"] = jaxrts.models.DetailedBalance()


fig, ax = plt.subplots(2, sharex=True)
ax_inset1 = inset_axes(ax[0], width="50%", height="70%", loc="upper left")
# Generate the spectrum
for frc in [0.0, 1.0, 2.0]:
    setup.frc_exponent = frc
    See_tot = state.probe(setup)

    # Plot the result
    ax[0].plot(
        setup.measured_energy.m_as(ureg.electron_volt),
        See_tot.m_as(ureg.second),
        label = f"frc_exponent = {frc:.0f}"
    )
    ax_inset1.plot(
        setup.measured_energy.m_as(ureg.electron_volt),
        See_tot.m_as(ureg.second),
        label = f"frc_exponent = {frc:.0f}"
    )
    ax[1].plot(
        setup.measured_energy.m_as(ureg.electron_volt),
        setup.frequency_redistribution_correction,
        label = f"frc_exponent = {frc:.0f}"
    )



ax_inset1.set_xlim(8300, 8550)
ax_inset1.set_ylim(ymin=0.3e-18, ymax=3e-18)
ax[1].set_xlabel("Probed Energy [eV]")
ax[0].set_ylabel("$S_{ee}^{tot}$ or\n class. diff. crosssection or\n quantum diff. crosssection [s]")
ax[1].set_ylabel("FRC")
ax[1].legend()


ax_inset1.axes.get_xaxis().set_ticks([])
ax_inset1.axes.get_yaxis().set_ticks([])
mark_inset(ax[0], ax_inset1, loc1=3, loc2=1, fc="none", ec="gray", lw=1)

plt.show()
