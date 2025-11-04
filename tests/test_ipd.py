from pathlib import Path

import jpu.numpy as jnpu
import numpy as np
from jax import numpy as jnp

import jaxrts
from jaxrts.ipd import (
    ipd_debye_hueckel,
    ipd_ecker_kroell,
    ipd_ion_sphere,
    ipd_stewart_pyatt,
)
from jaxrts.units import to_array

from .helpers import get_all_models

all_ipd_models = get_all_models()["ipd"]

ureg = jaxrts.ureg


def electron_number_density(mass_density, m_a, ionization):

    res = ionization * (ureg.avogadro_constant) * (mass_density / m_a)

    return res.to(1 / ureg.cc)


def read_data(data_path, Zi, ma):
    rho, ipd = np.genfromtxt(data_path, unpack=True, delimiter=",")
    sort = jnp.argsort(rho)
    ipd = ipd[sort]
    rho = rho[sort]
    ipd = ipd[np.isfinite(rho)]
    rho = rho[np.isfinite(rho)]

    ne = electron_number_density(rho * 1 * ureg.gram / ureg.cc, ma, Zi)

    return ne, ipd


def test_ipd_zeng2022():

    T = 600.0 * ureg.electron_volt / ureg.boltzmann_constant

    Zi = 11.0
    ma = 26.98 * ureg.gram / ureg.mole
    neDH, ipdDH = read_data(
        Path(__file__).parent / "data/Lin2017/Fig1/ipd_dh_lin.csv", Zi, ma
    )
    neSP, ipdSP = read_data(
        Path(__file__).parent / "data/Lin2017/Fig1/ipd_sp_lin.csv", Zi, ma
    )
    neoEK, ipdoEK = read_data(
        Path(__file__).parent / "data/Lin2017/Fig1/ipd_oek_lin.csv", Zi, ma
    )
    nemEK, ipdmEK = read_data(
        Path(__file__).parent / "data/Lin2017/Fig1/ipd_mek_lin.csv", Zi, ma
    )
    neoIS, ipdoIS = read_data(
        Path(__file__).parent / "data/Lin2017/Fig1/ipd_ois_lin.csv", Zi, ma
    )
    nemIS, ipdmIS = read_data(
        Path(__file__).parent / "data/Lin2017/Fig1/ipd_mis_lin.csv", Zi, ma
    )

    ipdDH_calc = [
        ipd_debye_hueckel(Zi, n, n / Zi, T, T).m_as(ureg.electron_volt)
        for n in neDH
    ]
    ipdSP_calc = [
        ipd_stewart_pyatt(Zi, n, n / Zi, T, T).m_as(ureg.electron_volt)
        for n in neSP
    ]
    # the oIS model in Lin et al. uses the Zimmermann prefactor 9/5
    ipdoIS_calc = [
        ipd_ion_sphere(Zi, n, n / Zi, C=9 / 5).m_as(ureg.electron_volt)
        for n in neoIS
    ]
    # the mIS model in Lin et al. is our standard IS model in JaXRTS
    ipdmIS_calc = [
        ipd_ion_sphere(Zi, n, n / Zi).m_as(ureg.electron_volt) for n in nemIS
    ]
    ipdoEK_calc = [
        ipd_ecker_kroell(
            to_array([Zi]),
            n,
            to_array([n]) / Zi,
            T,
            to_array([T]),
            jnp.array([13]),
        )[0].m_as(ureg.electron_volt)
        for n in neoEK
    ]
    ipdmEK_calc = [
        ipd_ecker_kroell(
            to_array([Zi]),
            n,
            to_array([n]) / Zi,
            T,
            to_array([T]),
            jnp.array([13]),
            C=1,
        )[0].m_as(ureg.electron_volt)
        for n in nemEK
    ]

    # Relative errors are inflated for the first values, use absolute ones for
    # these.
    assert jnpu.nanmax(jnp.absolute((ipdDH - ipdDH_calc) / ipdDH)[10:]) < 0.08
    assert jnpu.nanmax(jnp.absolute((ipdSP - ipdSP_calc) / ipdSP)[10:]) < 0.02
    assert (
        jnpu.nanmax(jnp.absolute((ipdmIS - ipdmIS_calc) / ipdmIS)[5:]) < 0.03
    )
    assert (
        jnpu.nanmax(jnp.absolute((ipdoIS - ipdoIS_calc) / ipdoIS)[5:]) < 0.03
    )
    assert (
        jnpu.nanmax(jnp.absolute((ipdoEK - ipdoEK_calc) / ipdoEK)[20:]) < 0.02
    )
    # The cutoff frequency of the EK model seems slightly different in Lin et
    # al. Hence, the lowest energies are slightly off
    assert (
        jnpu.nanmax(jnp.absolute((ipdmEK - ipdmEK_calc) / ipdmEK)[3:]) < 0.05
    )
    assert jnpu.max(jnp.absolute(ipdDH - ipdDH_calc)[:10]) < 10  # eV
    assert jnpu.max(jnp.absolute(ipdSP - ipdSP_calc)[:10]) < 4  # eV
    assert jnpu.nanmax(jnp.absolute(ipdoEK - ipdoEK_calc)[:20]) < 2  # eV
    assert jnpu.nanmax(jnp.absolute(ipdmIS - ipdmIS_calc)[:5]) < 2  # eV
    assert jnpu.nanmax(jnp.absolute(ipdoIS - ipdoIS_calc)[:5]) < 2  # eV


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
        elif model == jaxrts.models.IPDSum:
            args = (
                [
                    jaxrts.models.StewartPyattIPD(),
                    jaxrts.models.EckerKroellIPD(),
                ],
            )
        else:
            args = ()
        Zf_0_state["ipd"] = model(*args)
        assert jnpu.isfinite(
            Zf_0_state.evaluate("ipd", None)
        ), f"{model.__name__} did not yield a finite result for Z=0"


def test_ipd_sum_model():
    energy = ureg("4eV")
    model1 = jaxrts.models.ConstantIPD(energy)

    model = jaxrts.models.IPDSum([model1, model1, model1])
    assert model.evaluate(None, None) == 3 * energy

    carbon_H_state = jaxrts.PlasmaState(
        ions=[jaxrts.Element("C"), jaxrts.Element("H")],
        Z_free=jnp.array([1.0, 1.0]),
        mass_density=jnp.array([3.5, 2.4]) * ureg.gram / ureg.centimeter**3,
        T_e=jnp.array([80]) * ureg.electron_volt / ureg.k_B,
    )
    assert len(model.all_element_states(carbon_H_state)) == 2
    assert jnp.all(
        model.all_element_states(carbon_H_state)[0].m_as(ureg.electron_volt)
        == jnp.ones(6) * 3 * energy.m_as(ureg.electron_volt)
    )
    assert jnp.all(
        model.all_element_states(carbon_H_state)[1].m_as(ureg.electron_volt)
        == jnp.ones(1) * 3 * energy.m_as(ureg.electron_volt)
    )
