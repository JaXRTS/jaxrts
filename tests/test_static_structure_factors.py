import pathlib

from unittest.mock import patch

import pytest
import numpy as onp
import jaxrts
from jax import numpy as jnp

ureg = jaxrts.ureg

def mock_Tcf(T_e,n_e):
    """
    Arkhipov does not use an effective temperature as Gregori does.
    Hence, replace jaxrts.static_structure_factors._T_cf_AD for this test to
    just return T_e.
    """
    return T_e


@patch("jaxrts.static_structure_factors._T_cf_AD", side_effect=mock_Tcf)
def test_arkhipov_literature(mock_Tcf_AD):
    """
    Test the calculations against the data displayed in Fig. 3 and Fig. 4
    of :cite:`Arkhipov`
    """
    r_s = 0.1
    Z_f = 1
    m_i = 1 * ureg.atomic_mass_constant
    prefactor = 4 * jnp.pi * ureg.epsilon_0
    a = r_s * ureg.a_0
    n_e = 3 / (4 * jnp.pi * a**3)

    for fig, gam in zip([3, 4], [0.1, 1]):
        data_dir = (
            pathlib.Path(__file__).parent / f"data/Arkhipov1998/Fig{fig}/"
        )
        ka_See, lit_S_ee = onp.genfromtxt(
            data_dir / "S_ee.csv", delimiter=",", unpack=True
        )
        ka_Sei, lit_S_ei = onp.genfromtxt(
            data_dir / "S_ei.csv", delimiter=",", unpack=True
        )
        k_See = ka_See / a
        k_Sei = ka_Sei / a

        T_e = ureg.elementary_charge**2 / (a * ureg.k_B * gam) / prefactor

        calc_See = jaxrts.static_structure_factors.S_ee_AD(
            k_See, T_e, n_e, m_i, Z_f
        ).m_as(ureg.dimensionless)
        calc_Sii = jaxrts.static_structure_factors.S_ii_AD(
            k_See, T_e, n_e, m_i, Z_f
        ).m_as(ureg.dimensionless)
        calc_Sei = jaxrts.static_structure_factors.S_ei_AD(
            k_Sei, T_e, n_e, m_i, Z_f
        ).m_as(ureg.dimensionless)

        # import matplotlib.pyplot as plt

        # plt.plot(ka_See, lit_S_ee, color="C0")
        # plt.plot(ka_See, calc_See, color="C0", ls="dashed")
        # plt.plot(ka_Sei, lit_S_ei, color="C1")
        # plt.plot(ka_Sei, calc_Sei, color="C1", ls="dashed")
        # plt.plot(ka_See, calc_Sii, color="C2", ls="dashed")

        # plt.show()


# test_arkhipov_literature()
