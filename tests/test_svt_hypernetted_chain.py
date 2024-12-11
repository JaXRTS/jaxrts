from pathlib import Path
import sys

sys.path.append(
    r"C:\Users\Samuel\Desktop\PhD\Python-Projects\JAXRTS\jaxrts\src"
)
import pytest
import jpu
import numpy as onp
from jax import numpy as jnp

import jpu.numpy as jnpu

import jaxrts
import jaxrts.hypernetted_chain as hnc
from jaxrts import hnc_potentials
from jaxrts.units import Quantity, to_array, ureg

from jaxrts.hypernetted_chain import pair_distribution_function_two_component_SVT_HNC

if __name__ == "__main__":

    ions = [jaxrts.Element("H")]
    number_fraction = jnp.array([1.0])
    mass_fraction = jaxrts.helpers.mass_from_number_fraction(
        number_fraction, ions
    )

    exchange = True

    pot = 19
    mix = 0.6
    r = jnpu.linspace(1e-3 * ureg.a0, 1000 * ureg.a0, 2**pot)

    dr = r[1] - r[0]
    dk = jnp.pi / (len(r) * dr)
    k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk

    plasma_state = jaxrts.PlasmaState(
        ions=ions,
        Z_free=jnp.array([1.0]),
        mass_density=ureg(f"{10}g/cc") * mass_fraction,
        T_e=300 * ureg.electron_volt / ureg.boltzmann_constant,
        T_i=[300 * ureg.electron_volt / ureg.boltzmann_constant],
    )

    IonIonPotential = jaxrts.hnc_potentials.CoulombPotential()
    ElectronIonPotential = jaxrts.hnc_potentials.KelbgPotential()
    ElectronElectronPotential = jaxrts.hnc_potentials.KelbgPotential()

    ExchangePotential = jaxrts.hnc_potentials.SpinAveragedEEExchange()

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

    print(m_ab)

    g, niter = pair_distribution_function_two_component_SVT_HNC(
        V_s, V_l_k, r, T, m_ab, ni, mix=0.0
    )

    print(g, niter)

    fig, ax = plt.subplots()

    ax.plot(r, g)
    ax.set_xlabel("r")
    ax.set_ylabel("g(r)")

    plt.show()
