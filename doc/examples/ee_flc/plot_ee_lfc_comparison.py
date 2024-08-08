import jax.numpy as jnp
import jpu.numpy as jnpu
import matplotlib.pyplot as plt

import jaxrts

ureg = jaxrts.ureg


def test_staticInterp_Gregori2007_reproduction():
    for T in [4, 20]:
        T = T * ureg.electron_volt / ureg.boltzmann_constant
        n_e = 2.5e23 / ureg.centimeter**3

        kf = (3 * jnp.pi**2 * n_e) ** (1 / 3)
        k = jnpu.linspace(0 * kf, 10 * kf, 5000)

        G_calc = (
            jaxrts.ee_localfieldcorrections.eelfc_interpolationgregori2007(
                k, T, n_e
            )
        )
        G_farid = (
            jaxrts.ee_localfieldcorrections.eelfc_interpolationgregori_farid(
                k, T, n_e
            )
        )
        plt.plot(k / kf, G_calc.m_as(ureg.dimensionless))
        plt.plot(k / kf, G_farid)

    plt.show()


test_staticInterp_Gregori2007_reproduction()
