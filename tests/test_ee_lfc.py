import pathlib

import numpy as onp
import jax.numpy as jnp
import jpu.numpy as jnpu
import jax
import pytest

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
        k = jnpu.linspace(0* kf, 30 * kf, 3000)

        G_calc = (
            jaxrts.ee_localfieldcorrections.eelfc_interpolationgregori2007(
                k, T, n_e
            )
        )
        # G_calc_farid = (
        #     jaxrts.ee_localfieldcorrections.eelfc_farid(
        #         k, T, n_e
        #     )
        # )

        import matplotlib.pyplot as plt

        plt.plot(k_over_kf, G)
        plt.plot(k/kf, G_calc.m_as(ureg.dimensionless), ls="dashed")
        # plt.plot(k/kf, G_calc_farid.m_as(ureg.dimensionless), ls="dotted")
    rs = (
        jaxrts.plasma_physics.interparticle_spacing(-1, -1, n_e) / ureg.a0
    ).m_as(ureg.dimensionless)
    z = 4 * (4 / (9 * jnp.pi)) ** (1 / 6) * (rs / jnp.pi) ** (1 / 2)

    g0ee = 1 / 8 * (z / jax.scipy.special.i1(z)) ** 2
    plt.plot([0, 30],[1-g0ee, 1-g0ee], color = "black")
    plt.title(n_e.m_as(1 / ureg.centimeter**3))
    plt.show()


test_staticInterp_Gregori2007_reproduction()
