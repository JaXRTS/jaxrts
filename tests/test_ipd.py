from pathlib import Path

import jpu.numpy as jnpu
import numpy as np
from jax import numpy as jnp

import jaxrts
from jaxrts.ipd import (
    ipd_debye_hueckel,
    ipd_stewart_pyatt,
)

from .helpers import get_all_models

all_ipd_models = get_all_models()["ipd"]

ureg = jaxrts.ureg


def electron_number_density(mass_density, m_a, ionization):

    res = ionization * (ureg.avogadro_constant) * (mass_density / m_a)

    return res.to(1 / ureg.cc)


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
    assert jnpu.max(jnp.absolute(ipdDH - ipdDH_calc)[:10]) < 10  # eV
    assert jnpu.max(jnp.absolute(ipdSP - ipdSP_calc)[:10]) < 4  # eV


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
