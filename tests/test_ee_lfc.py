import pathlib

import jax.numpy as jnp
import numpy as onp

import jaxrts

ureg = jaxrts.ureg


def test_staticInterp_Gregori2007_reproduction():
    for T in [4, 20]:
        data_path = (
            pathlib.Path(__file__).parent / f"data/Gregori2007/Fig1a_{T}eV.csv"
        )
        k_over_kf, G = onp.genfromtxt(data_path, unpack=True, delimiter=",")

        T = T * ureg.electron_volt / ureg.boltzmann_constant
        n_e = 2.5e23 / ureg.centimeter**3

        kf = (3 * jnp.pi**2 * n_e) ** (1 / 3)
        k = k_over_kf * kf

        G_calc = (
            jaxrts.ee_localfieldcorrections.eelfc_interpolationgregori2007(
                k, T, n_e
            )
        )
        assert jnp.max(jnp.abs(G - G_calc.m_as(ureg.dimensionless))) < 0.01


def test_Dornheim2021_interpolation_against_literature():
    rs = 2
    theta = 2
    data_path = (
        pathlib.Path(__file__).parent
        / f"data/Dornheim2021/Fig7G_rs2_theta2_FIT.csv"
    )
    k_over_kf_lit, G_lit = onp.genfromtxt(data_path, unpack=True, delimiter=",")

    assert (
        jnp.max(
            jnp.abs(
                [
                    jaxrts.ee_localfieldcorrections.dornheim_G(k, rs, theta)
                    for k in k_over_kf_lit
                ]
                - G_lit
            )
        )
    ) < 0.015
