import pytest
import sys

sys.path.append(
    "C:/Users/Samuel/Desktop/PhD/Python_Projects/JAXRTS/jaxrts/src"
)

from jax import numpy as jnp
import jaxrts
import copy

from pathlib import Path

import jpu.numpy as jnpu

import os

import numpy as np

ureg = jaxrts.ureg

from jaxrts.ipd import ipd_debye_hueckel, ipd_ion_sphere, ipd_stewart_pyatt, ipd_pauli_blocking

import matplotlib.pyplot as plt


def electron_number_density(mass_density, m_a, ionization):

    res = ionization * (ureg.avogadro_constant) * (mass_density / m_a)

    return res.to(1 / ureg.cc)


def test_IPD():

    test_setup = jaxrts.setup.Setup(
        ureg("145°"),
        ureg("5keV"),
        jnp.linspace(4.5, 5.5) * ureg.kiloelectron_volts,
        lambda x: jaxrts.instrument_function.instrument_gaussian(
            x, 1 / ureg.second
        ),
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    nes = []

    ipds = {"dh": [], "sp": [], "is": [], "pb": []}
    keys = ["dh", "sp", "is", "pb"]

    md = jnp.linspace(0.01, 600, 500)
    for m in md:

        test_state = jaxrts.PlasmaState(
            ions=[jaxrts.Element("C")],
            Z_free=jnp.array([5]),
            mass_density=jnp.array([m]) * ureg.gram / ureg.centimeter**3,
            T_e=jnp.array([100]) * ureg.electron_volt / ureg.k_B,
        )

        nes.append(test_state.n_e.m_as(1 / ureg.cc))

        for i, IPDModel in enumerate(
            [
                jaxrts.models.DebyeHueckelIPD(),
                jaxrts.models.StewartPyattIPD(),
                jaxrts.models.IonSphereIPD(),
                jaxrts.models.PauliBlockingIPD(),
            ]
        ):

            IPDModel.model_key = "ipd"
            shift = IPDModel.evaluate(
                plasma_state=test_state, setup=test_setup
            ).m_as(ureg.electron_volt)

            # print(test_state.n_e.m_as(1 / ureg.cc), shift)
            ipds[keys[i]].append(-shift[0])

    ax.plot(nes, ipds["dh"], label="Debye Hueckel Model")
    ax.plot(nes, ipds["sp"], label="Stewart Pyatt Model")
    ax.plot(nes, ipds["is"], label="Ion Sphere Model")
    ax.plot(nes, ipds["pb"], label="Pauli Blocking Model")

    plt.xscale("log")
    # plt.yscale("symlog")
    plt.xlabel("$n_i$ [cm$^{-3}$]")
    plt.ylabel("IPD [eV]")
    plt.legend()
    plt.xlim(1e23, 1e26)
    # plt.ylim(0.5, 15)
    plt.tight_layout()
    plt.show()
    
def ipd_db(Zi: float, ne, ni, Te, Ti):
    lambdaD = jnpu.sqrt(1 * ureg.epsilon_0 * ureg.boltzmann_constant * Te / ne / ureg.elementary_charge ** 2)
    return ((Zi+1) * 1 * ureg.elementary_charge ** 2 / (4 * ureg.epsilon_0 * jnp.pi * lambdaD)).to(ureg.electron_volt)

def test_ipd_zeng2022():

    T = 600.0 * ureg.electron_volt / ureg.boltzmann_constant

    data_path1 = Path(__file__).parent / f"data/Lin2017/Fig1/ipd_dh_lin.csv"
    data_path2 = Path(__file__).parent / f"data/Lin2017/Fig1/ipd_sp_lin.csv"
    rho1, ipd1 = np.genfromtxt(data_path1, unpack=True, delimiter=",")
    rho2, ipd2 = np.genfromtxt(data_path2, unpack=True, delimiter=",")

    sort1 = jnp.argsort(rho1)
    ipd1 = ipd1[sort1]
    rho1 = rho1[sort1]
    sort2 = jnp.argsort(rho2)
    ipd2 = ipd2[sort2]
    rho2 = rho2[sort2]

    Zi = 11.0
    ma = 26.98 * ureg.gram / ureg.mole
    ne1 = electron_number_density(rho1 * 1 * ureg.gram / ureg.cc, ma, Zi)
    ne2 = electron_number_density(rho2 * 1 * ureg.gram / ureg.cc, ma, Zi)

    ipd1_calc = ipd_debye_hueckel(Zi, ne1, ne1 / Zi, T, T).m_as(
        ureg.electron_volt
    )
    ipd2_calc = ipd_pauli_blocking(Zi, ne2, ne2 / Zi, T, T).m_as(ureg.electron_volt)
    ipd3_calc = ipd_stewart_pyatt(Zi, ne2, ne2 / Zi, T, T).m_as(
        ureg.electron_volt
    )

    plt.plot(rho1, ipd1, label="DH Lin2017", color="C0", linestyle="dashed", marker="+")
    plt.plot(rho2, ipd2, label="SP Lin2017", color="C3", linestyle="dashed", marker="+")
    plt.plot(rho1, ipd1_calc, label="DH calc", color="C0")
    plt.plot(rho2, ipd2_calc, label="PB calc", color="C1")
    plt.plot(rho2, ipd3_calc, label="SP calc", color="C2")

    plt.xscale("log")
    plt.xlabel(r"$\rho$ [g/cc]")
    plt.ylabel(f"IPD [eV]")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    test_ipd_zeng2022()
