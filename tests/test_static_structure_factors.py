import pathlib

import pytest
import numpy as onp
import jaxrts
from jax import numpy as jnp

ureg = jaxrts.ureg


@pytest.mark.skip(
    reason="Re-producing figures from Arkhipov.1998 is not possible"
)
def test_arkhipov_static_structure_factors_literature():
    """
    Test the calculations against the data displayed in Fig. 3 and Fig. 4
    of :cite:`Arkhipov.1998`
    """
    r_s = 0.1
    Z_f = 1
    m_i = 1.0 * ureg.atomic_mass_constant
    a = r_s * ureg.a_0
    n_e = 3 / (4 * jnp.pi * a**3)

    for fig, gam in zip([3, 4], [0.1, 1]):
        data_dir = (
            pathlib.Path(__file__).parent / f"data/Arkhipov1998/Fig{fig}/"
        )
        ka_See, lit_See = onp.genfromtxt(
            data_dir / "S_ee.csv", delimiter=",", unpack=True
        )
        ka_Sei, lit_Sei = onp.genfromtxt(
            data_dir / "S_ei.csv", delimiter=",", unpack=True
        )
        k_See = ka_See / a
        k_Sei = ka_Sei / a

        T_e = ureg.elementary_charge**2 / (
            (4 * jnp.pi * ureg.vacuum_permittivity)
            * ureg.boltzmann_constant
            * gam
            * a
        )

        calc_See = jaxrts.static_structure_factors.S_ee_AD(
            k_See, T_e, T_e, n_e, m_i, Z_f
        ).m_as(ureg.dimensionless)
        calc_Sei = jaxrts.static_structure_factors.S_ei_AD(
            k_Sei, T_e, T_e, n_e, m_i, Z_f
        ).m_as(ureg.dimensionless)

        assert jnp.max(jnp.abs(lit_See - calc_See)) < 0.05
        assert jnp.max(jnp.abs(lit_Sei - calc_Sei)) < 0.05


def test_arkhipov_electron_electron_pair_correlation_function_literature():
    """
    Test the calculations against the data displayed in Fig. 2 of
    :cite:`Arkhipov.2000`
    """
    r_s = 0.1
    Z_f = 1
    m_i = 1.0 * ureg.atomic_mass_constant
    a = r_s * ureg.a_0
    n_e = 3 / (4 * jnp.pi * a**3)
    gam = 0.3
    T_e = ureg.elementary_charge**2 / (
        (4 * jnp.pi * ureg.vacuum_permittivity)
        * ureg.boltzmann_constant
        * gam
        * a
    )

    # Load the data
    data_dir = pathlib.Path(__file__).parent / "data/Arkhipov2000/"
    R_over_a_gee, lit_gee = onp.genfromtxt(
        data_dir / "Fig2.csv", delimiter=",", unpack=True
    )
    R_gee = R_over_a_gee * a
    calc_gee = jaxrts.static_structure_factors.g_ee_ABD(
        R_gee, T_e, T_e, n_e, m_i, Z_f
    ).m_as(ureg.dimensionless)

    assert jnp.max(jnp.abs(lit_gee - calc_gee)) < 0.015
    assert jnp.mean(jnp.abs(lit_gee - calc_gee)) < 0.003
