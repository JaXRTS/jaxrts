from pathlib import Path

import jpu.numpy as jnpu
import numpy as np
from jax import numpy as jnp

import matplotlib.pyplot as plt

import jaxrts

from jaxrts.ipd import (
    ipd_debye_hueckel,
    ipd_ion_sphere,
    ipd_stewart_pyatt,
)

# from jaxrts.helpers import get_all_models

# all_ipd_models = get_all_models()["ipd"]

ureg = jaxrts.ureg


def electron_number_density(mass_density, m_a, ionization):

    res = ionization * (ureg.avogadro_constant) * (mass_density / m_a)

    return res.to(1 / ureg.cc)

def test_sp_ipd_zeng():

    data_path = Path(__file__).parent / "data/Zeng2022/Fig8/SP.csv"
    rhoSP, ipdSP = np.genfromtxt(data_path, unpack=True, delimiter=",")

    Zi = 20
    Te = 300 * ureg.electron_volt / ureg.boltzmann_constant
    element = jaxrts.Element("Fe")

    data_sp_calc = []
    data_is_calc = []
    data_db_calc = []

    rho_plot = jnp.array(range(1, 16, 1)) * 1 * ureg.gram / ureg.cc

    for rho in rho_plot:

        state = jaxrts.PlasmaState(
            ions=[element],
            Z_free=[Zi],
            mass_density=[rho],
            T_e=Te,
        )

        ipdSP_calc = -ipd_stewart_pyatt(Zi, state.n_e, state.n_e / Zi, Te, Te).m_as(
            ureg.electron_volt
        )

        ipdIS_calc = -ipd_ion_sphere(Zi, state.n_e, state.n_e / Zi).m_as(
        ureg.electron_volt
    )

        ipdDB_calc = -ipd_debye_hueckel(Zi, state.n_e, state.n_e / Zi, Te, Te).m_as(
            ureg.electron_volt
        )

        data_sp_calc.append(ipdSP_calc)
        data_is_calc.append(ipdIS_calc)
        data_db_calc.append(ipdDB_calc)

    fig, ax = plt.subplots()

    ax.scatter(rhoSP, ipdSP, label = "SP, Zeng 2022", ls = "dashed", color = "black")
    ax.plot(rho_plot.m_as(ureg.gram / ureg.cc), data_sp_calc, color = "blue", label = "SP, JaXRTS")
    ax.plot(rho_plot.m_as(ureg.gram / ureg.cc), data_is_calc, color = "green", label = "IS, JaXRTS")
    ax.plot(rho_plot.m_as(ureg.gram / ureg.cc), data_db_calc, color = "purple", label = "DB, JaXRTS")


    ax.set_xlabel(f"Mass density [g/cc]")
    ax.set_ylabel("IPD [eV]")
    ax.legend(fontsize = 12)
    plt.show()


def test_ipd_zeng2022():

    T = 600.0 * ureg.electron_volt / ureg.boltzmann_constant

    data_path1 = Path(__file__).parent / "data/Lin2017/Fig1/ipd_dh_lin.csv"
    data_path2 = Path(__file__).parent / "data/Lin2017/Fig1/ipd_sp_lin.csv"
    rhoDH, ipdDH = np.genfromtxt(data_path1, unpack=True, delimiter=",")
    rhoSP, ipdSP = np.genfromtxt(data_path2, unpack=True, delimiter=",")

    sort1 = jnp.argsort(rhoDH)
    ipdDH = ipdDH[sort1]
    rhoDH = rhoDH[sort1]
    sort2 = jnp.argsort(rhoSP)
    ipdSP = ipdSP[sort2]
    rhoSP = rhoSP[sort2]

    Zi = 11.0
    ma = 26.98 * ureg.gram / ureg.mole
    neDH = electron_number_density(rhoDH * 1 * ureg.gram / ureg.cc, ma, Zi)
    neSP = electron_number_density(rhoSP * 1 * ureg.gram / ureg.cc, ma, Zi)

    ipdDH_calc = ipd_debye_hueckel(Zi, neDH, neDH / Zi, T, T).m_as(
        ureg.electron_volt
    )
    ipdSP_calc = ipd_stewart_pyatt(Zi, neSP, neSP / Zi, T, T).m_as(
        ureg.electron_volt
    )

    # Relative errors are inflated for the first values, use absolute ones for
    # these.
    assert jnpu.nanmax(jnp.absolute((ipdDH - ipdDH_calc) / ipdDH)[10:]) < 0.08
    assert jnpu.nanmax(jnp.absolute((ipdSP - ipdSP_calc) / ipdSP)[10:]) < 0.02
    assert jnpu.max(jnp.absolute((ipdDH - ipdDH_calc))[:10]) < 10  # eV
    assert jnpu.max(jnp.absolute((ipdSP - ipdSP_calc))[:10]) < 4  # eV


def test_ipd_valid_for_Z_f_equals_0():
    Zf_0_state = jaxrts.PlasmaState(
        ions=[jaxrts.Element("C")],
        Z_free=jnp.array([0]),
        mass_density=jnp.array([3.5]) * ureg.gram / ureg.centimeter**3,
        T_e=jnp.array([80]) * ureg.electron_volt / ureg.k_B,
    )
    for model in all_ipd_models:
        if model == jaxrts.models.ConstantIPD:
            args = (23.42 * ureg.electron_volt,)
        else:
            args = ()
        Zf_0_state["ipd"] = model(*args)
        assert jnpu.isfinite(Zf_0_state.evaluate("ipd", None))

if __name__ == "__main__":

    test_sp_ipd_zeng()