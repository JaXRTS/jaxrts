"""
k dispersion and free-bound scattering
======================================

The free-bound and bound-free contributions only fulfill the detailed balance
relation in the (theoretical) case where the same :math:`k` is probed at every
energy-bin on an detector. We can model such a spectrum by setting
:py:attr:`jaxrts.setup.Setup.correct_k_dispersion` to ``False``.

In a real experiment, this is often not given and we therefore enable the
dispersion of :math:`k` as a default. When evaluating the free-bound part of
the spectrum, this has to be considered, as a mere "flipping" of the incoming
spectrum would result in an unphysical behaviour because the bound-free part
has to be evaluated at the :math:`k` values which corresponds to the
blue-shifted energies.
"""

from jpu import numpy as jnpu
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import scienceplots  # noqa: F401

import jaxrts

ureg = jaxrts.ureg

plt.style.use("science")

fig, ax = plt.subplots(figsize=(4.5, 4))
ax_inset1 = inset_axes(ax, width="30%", height="30%", loc="center right")
ax_inset2 = inset_axes(ax, width="30%", height="30%", loc="upper left")

test_setup = jaxrts.setup.Setup(
    ureg("145°"),
    ureg("5keV"),
    jnp.linspace(4.5, 5.5, 501) * ureg.kiloelectron_volts,
    lambda x: jaxrts.instrument_function.instrument_gaussian(
        x, sigma=ureg("1.0eV") / ureg.hbar / (2 * jnp.sqrt(2 * jnp.log(2)))
    ),
)

test_state = jaxrts.PlasmaState(
    ions=[jaxrts.Element("C")],
    Z_free=jnp.array([3]),
    mass_density=jnp.array([4]) * ureg.gram / ureg.centimeter**3,
    T_e=jnp.array([50]) * ureg.electron_volt / ureg.k_B,
)

test_state["electron-ion Potential"] = jaxrts.hnc_potentials.CoulombPotential()
test_state["screening length"] = (
    jaxrts.models.ArbitraryDegeneracyScreeningLength()
)
test_state["ionic scattering"] = jaxrts.models.OnePotentialHNCIonFeat()
test_state["free-free scattering"] = jaxrts.models.RPA_NoDamping()
test_state["bound-free scattering"] = jaxrts.models.SchumacherImpulse(r_k=1)
test_state["free-bound scattering"] = jaxrts.models.DetailedBalance()

test_setup.correct_k_dispersion = True

fb = test_state.evaluate("free-bound scattering", test_setup)
bf = test_state.evaluate("bound-free scattering", test_setup)


for axis in [ax, ax_inset1]:
    axis.plot(
        test_setup.measured_energy.m_as(ureg.electron_volt),
        fb.m_as(ureg.second),
        color="C0",
        ls="dashed",
        label="implemented",
        lw=2,
    )
for axis in [ax, ax_inset2]:
    axis.plot(
        test_setup.measured_energy.m_as(ureg.electron_volt),
        bf.m_as(ureg.second),
        color="C0",
        ls="solid",
        lw=2,
    )

test_setup.correct_k_dispersion = False

fb = test_state.evaluate("free-bound scattering", test_setup)
bf = test_state.evaluate("bound-free scattering", test_setup)

for axis in [ax, ax_inset1]:
    axis.plot(
        test_setup.measured_energy.m_as(ureg.electron_volt),
        fb.m_as(ureg.second),
        color="C1",
        ls="dashed",
        label="no $k$ dispersion",
    )
for axis in [ax, ax_inset2]:
    axis.plot(
        test_setup.measured_energy.m_as(ureg.electron_volt),
        bf.m_as(ureg.second),
        color="C1",
        ls="solid",
    )

test_setup.correct_k_dispersion = True

bf = test_state.evaluate("bound-free scattering", test_setup)
bf_raw = test_state["bound-free scattering"].evaluate_raw(
    test_state, test_setup
)
energy_shift = test_setup.measured_energy - test_setup.energy
fb_raw = bf_raw[::-1] * jnpu.exp(-energy_shift / (test_state.T_e * ureg.k_B))
fb = jaxrts.setup.convolve_stucture_factor_with_instrument(fb_raw, test_setup)

for axis in [ax, ax_inset1]:
    axis.plot(
        test_setup.measured_energy.m_as(ureg.electron_volt),
        fb.m_as(ureg.second),
        color="C2",
        ls="dashed",
        label="naive flipping",
    )

for axis in [ax, ax_inset2]:
    axis.plot(
        test_setup.measured_energy.m_as(ureg.electron_volt),
        bf.m_as(ureg.second),
        color="C2",
        ls="solid",
    )


ax.legend()

ax_inset1.set_xlim(5010, 5090)
ax_inset1.set_ylim(ymin=1.7e-18, ymax=2.7e-18)

ax_inset2.set_xlim(4865, 4955)
ax_inset2.set_ylim(ymin=8.9e-18, ymax=9.9e-18)

ax.set_ylabel("$I$ [s]")
ax.set_xlabel("$E$ [eV]")
ax.set_title("bound-free and free-bound scattering")

for axis in [ax_inset1, ax_inset2]:
    axis.axes.get_xaxis().set_ticks([])
    axis.axes.get_yaxis().set_ticks([])


mark_inset(ax, ax_inset1, loc1=2, loc2=4, fc="none", ec="gray", lw=1)
mark_inset(ax, ax_inset2, loc1=1, loc2=4, fc="none", ec="gray", lw=1)

plt.show()
