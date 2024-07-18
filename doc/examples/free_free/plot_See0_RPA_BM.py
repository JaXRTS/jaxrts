"""
Showing the relevance of the Born Mermin Approximation
======================================================

This script reprocudes :cite:`Glenzer.2009`, Fig. 9b. showing the calculation
of :math:`S_\\text{ee}^{0, \\text{RPA}}`.

This example shows a notable difference between RPA and BM for the relevant
conditions (temperatures between 0.5 and 8 eV, densities of 1e21 cubic
centimeres of fully ionized hydrogen).

Also, we show that the full RPA
(:py:func:`jaxrts.free_free.dielectric_function_RPA_no_damping`) and the RPA
without damping (:py:func:`jaxrts.free_free.dielectric_function_RPA`) give
identical results if the argument ``E`` is only real. (In the Born Mermin
Approximation, one introduces a damping term, resulting in an imaginary
contribution in ``E``). Therefore, the "full" RPA is required as part of the
module.
"""

import matplotlib.pyplot as plt
import numpy as onp
import scienceplots
import time

import time

import jaxrts
import jaxrts.free_free as free_free

import jax.numpy as jnp

import jax

ureg = jaxrts.units.ureg

plt.style.use("science")

lambda_0 = 4.13 * ureg.nanometer
theta = 60
n_e = 1e21 / ureg.centimeter**3


k = (4 * jnp.pi / lambda_0) * jnp.sin(jnp.deg2rad(theta) / 2.0)
w_pl = jaxrts.plasma_physics.plasma_frequency(n_e)
omega = jnp.linspace(-6, 6, 100) * w_pl
E = omega * ureg.hbar
count = 0

for T in [
    0.5 * ureg.electron_volts,
    2.0 * ureg.electron_volts,
    8.0 * ureg.electron_volts,
]:
    mu = jaxrts.plasma_physics.chem_pot_interpolationIchimaru(
        T / (1 * ureg.boltzmann_constant), n_e
    )

    t0 = time.time()
    vals1 = (
        free_free.S0_ee_RPA_no_damping(
            k,
            T_e=T / (1 * ureg.boltzmann_constant),
            n_e=n_e,
            E=E,
            chem_pot=mu,
        )
        / ureg.hbar
    ).m_as(1 / ureg.rydberg)
    print(f"RPA: {time.time() - t0}")

    @jax.tree_util.Partial
    def S_ii(q):
        return jaxrts.static_structure_factors.S_ii_AD(
            q,
            T / (1 * ureg.boltzmann_constant),
            T / (1 * ureg.boltzmann_constant),
            n_e,
            1 * ureg.proton_mass,
            Z_f=1.0,
        )

    kappa = 1 / jaxrts.plasma_physics.Debye_Hueckel_screening_length(
        n_e, T / (1 * ureg.boltzmann_constant)
    )

    @jax.tree_util.Partial
    def V_eiS(q):
        return jaxrts.free_free.statically_screened_ie_debye_potential(
            q, kappa, Zf=1.0
        )

    t0 = time.time()
    vals = (
        free_free.S0_ee_BMA(
            k,
            T=T / (1 * ureg.boltzmann_constant),
            n_e=n_e,
            E=E,
            chem_pot=mu,
            S_ii=S_ii,
            V_eiS=V_eiS,
            Zf=1.0,
        )
        / ureg.hbar
    ).m_as(1 / ureg.rydberg)
    print(f"BMA: {time.time() - t0}")

    t0 = time.time()
    vals3 = (
        free_free.S0_ee_BMA_chapman_interp(
            k,
            T=T / (1 * ureg.boltzmann_constant),
            n_e=n_e,
            E=E,
            chem_pot=mu,
            S_ii=S_ii,
            V_eiS=V_eiS,
            Zf=1.0,
            no_of_points=10,
        )
        / ureg.hbar
    ).m_as(1 / ureg.rydberg)
    print(f"BMA Chapman Interp: {time.time() - t0}")

    t0 = time.time()
    vals2 = (
        free_free.S0_ee_RPA(
            k,
            T_e=T / (1 * ureg.boltzmann_constant),
            n_e=n_e,
            E=E,
            chem_pot=mu,
        )
        / ureg.hbar
    ).m_as(1 / ureg.rydberg)
    print(f"RPA With damping: {time.time() - t0}")

    x = (E / ureg.hbar / w_pl).m_as(ureg.dimensionless)
    plt.plot(
        x,
        vals,
        label=(
            "T = " + str(T.m_as(ureg.electron_volt)) + " eV, BM"
            if count == 0
            else "T = " + str(T.m_as(ureg.electron_volt)) + " eV"
        ),
        color=f"C{count}",
    )
    plt.plot(
        x,
        vals3,
        label=(
            "T = "
            + str(T.m_as(ureg.electron_volt))
            + " eV, BMA, Chapman Interp"
            if count == 0
            else ""
        ),
        linestyle="dotted",
        color=f"C{count}",
    )
    plt.plot(
        x,
        vals1,
        label=(
            "T = " + str(T.m_as(ureg.electron_volt)) + " eV, RPA"
            if count == 0
            else ""
        ),
        linestyle="dashed",
        color=f"C{count}",
    )
    plt.plot(
        x,
        vals2,
        label=(
            "T = " + str(T.m_as(ureg.electron_volt)) + " eV, RPA, with damping"
            if count == 0
            else ""
        ),
        linestyle="dotted",
        color=f"black",
    )
    count += 1


plt.xlabel(r"$\omega/\omega_{pl}$")
plt.ylabel(r"$S^0_{\text{ee}}$ [Ryd$^{-1}$]")
plt.ylim(-0.01, 3)

plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1.00))
plt.show()
