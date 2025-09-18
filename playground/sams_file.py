from pathlib import Path
import sys

import pytest
import jpu
import numpy as onp
from jax import numpy as jnp

import jpu.numpy as jnpu

import jaxrts
import jaxrts.hypernetted_chain as hnc
from jaxrts import hnc_potentials
from jaxrts.units import Quantity, to_array, ureg
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)

from jaxrts.hypernetted_chain import pair_distribution_function_two_component_SVT_HNC_ei, pair_distribution_function_HNC, S_ii_HNC

if __name__ == "__main__":

    ions = [jaxrts.Element("H")]
    number_fraction = jnp.array([1.0])
    mass_fraction = jaxrts.helpers.mass_from_number_fraction(
        number_fraction, ions
    )

    exchange = False
    pot = 19
    mix = 0.8
    r = jnpu.linspace(1e-6 * ureg.a0, 100.0 * ureg.a0, 2**pot)

    dr = r[1] - r[0]
    dk = jnp.pi / (len(r) * dr)
    k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk

    plasma_state = jaxrts.PlasmaState(
        ions=ions,
        Z_free=jnp.array([1.0]),
        mass_density=ureg(f"{20.0}g/cc") * mass_fraction,
        T_e=1000.0 * ureg.electron_volt / ureg.boltzmann_constant,
        T_i=[1000.0 * ureg.electron_volt / ureg.boltzmann_constant],
    )

    IonIonPotential = jaxrts.hnc_potentials.CoulombPotential()
    ElectronIonPotential = jaxrts.hnc_potentials.KelbgPotential()
    ElectronElectronPotential = jaxrts.hnc_potentials.KelbgPotential()

    ExchangePotential = jaxrts.hnc_potentials.PauliClassicalMap()

    CoulombPotential = jaxrts.hnc_potentials.CoulombPotential()

    for Potential in [
        IonIonPotential,
        ElectronIonPotential,
        ElectronElectronPotential,
        ExchangePotential,
    ]:
        Potential.include_electrons = "SpinAveraged"

    unit = ureg.electron_volt

    # Ions
    V_s = IonIonPotential.full_r(plasma_state, r).m_as(unit)

    # Electrons
    V_s = V_s.at[0, -1, :].set(
        ElectronIonPotential.full_r(plasma_state, r)[0, -1, :].m_as(unit)
    )
    V_s = V_s.at[-1, 0, :].set(
        ElectronIonPotential.full_r(plasma_state, r)[-1, 0, :].m_as(unit)
    )

    V_s = V_s.at[-1, -1, :].set(
        (ElectronElectronPotential).full_r(plasma_state, r)[-1, -1, :].m_as(unit)
    )

    V_s *= unit

    V_s -= CoulombPotential.long_r(plasma_state, r)

    unit = ureg.electron_volt * ureg.angstrom**3

    V_l_k = CoulombPotential.long_k(plasma_state, k).m_as(unit)

    if exchange:

        V_l_k = V_l_k.at[-1, -1, :].add(
            ExchangePotential.full_k(plasma_state, k)[-1, -1, :].m_as(unit)
        )

    V_l_k *= unit

    n = jaxrts.units.to_array([*plasma_state.n_i, plasma_state.n_e])

    unit = ureg.electron_volt / ureg.boltzmann_constant
    T = IonIonPotential.T(plasma_state).m_as(unit)
    T *= unit

    ni = jaxrts.units.to_array(
        [*plasma_state.n_i, plasma_state.n_e]
    )
    ms = to_array([1.0 * ureg.proton_mass, 1.0 * ureg.electron_mass])

    def reduced_mass_matrix(m1, m2):

        reduced_mass = ((m1 * m2) / (m1 + m2)).m_as(ureg.electron_mass)

        mass_matrix = ([
            [m1.m_as(ureg.electron_mass), reduced_mass],
            [reduced_mass, m2.m_as(ureg.electron_mass)]
        ])

        return mass_matrix * 1 * ureg.electron_mass

    m_ab = reduced_mass_matrix(1.0 * ureg.proton_mass, 1.0 * ureg.electron_mass)
    m_ab = [1.0 * ureg.proton_mass, 1.0 * ureg.electron_mass]

    g, niter = pair_distribution_function_HNC(
        V_s, V_l_k, r, T, ni, mix=mix
    )

    print("Vlk")
    print(V_l_k)
    print("")
    print("V_s")
    print(V_s)
    print("")
    print("r")
    print(r)
    print("")
    print("T")
    print(T)
    print("")
    print("m_ab")
    print(m_ab)
    print("")
    print("ni")
    print(ni)
    g_svt, niter = pair_distribution_function_two_component_SVT_HNC_ei(
        V_s, V_l_k, r, T, ni, m_ab, mix=mix
    )

    fig, ax = plt.subplots(ncols = 2, figsize = (8, 6))

    ax[0].plot(r, g[1, 1, :], label = "ee", linestyle = "dashed")
    ax[0].plot(r, g_svt[1, 1, :], label = "ee SVT", alpha = 0.6)

    ax[0].set_xlabel("r")
    ax[0].set_xlim(0, 4)

    ax[0].legend()
    ax[0].set_ylabel(r"$g_\text{ee}(r)$")

    S_ab_HNC = S_ii_HNC(k, g, n, r).m_as(
        ureg.dimensionless
    )

    S_ab_HNC_SVT = S_ii_HNC(k, g_svt, n, r).m_as(
        ureg.dimensionless
    )

    ax[1].plot(k.m_as(1 / ureg.angstrom), S_ab_HNC[1, 1, :], label = "classic", alpha = 0.6)
    # ax[1].plot(k, S_ab_HNC_SVT[1, 1, :], label = "SVT", linestyle = "dashed")

    ax[1].set_xlabel("k")
    ax[1].set_xlim(0, 4)

    ax[1].legend()
    ax[1].set_ylabel(r"$S_\text{ee}(k)$")

    plt.tight_layout()

    plt.show()
