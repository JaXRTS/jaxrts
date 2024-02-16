import pytest

import numpy as onp
import jpu.numpy as jnpu
import jaxrts
import jaxrts.form_factors as ff

ureg = jaxrts.ureg


def test_screening_constants_tabulated_values():
    carbon_screening = ff.pauling_size_screening_constants(6)
    assert onp.abs(carbon_screening[1] - 2.04) < 0.1
    assert onp.abs(carbon_screening[2] - 2.91) < 0.1


def test_effective_charge_tabulated_values():
    carbon_z_eff = ff.pauling_effective_charge(6)
    assert onp.abs(carbon_z_eff[0] - 5.81) < 0.1
    assert onp.abs(carbon_z_eff[1] - 3.96) < 0.1
    assert onp.abs(carbon_z_eff[2] - 3.09) < 0.1


def test_tabulated_scattering_factors():
    """
    Test full scattering factors (i.e., the sum of individual f) against
    literature values. The scattering factors for full atoms are taken from
    table VII in :cite:`Pauling.1932`.
    Relative uncertainties are < ~3%.
    """
    k = onp.linspace(0, 1.3, 14) * (4 * onp.pi) / ureg.angstrom
    # fmt: off
    tabulated_values = {
        "C": onp.array([6, 5.21, 3.62, 2.42, 1.86, 1.66, 1.57, 1.51, 1.42, 1.32, 1.21, 1.10, 0.99, 0.89]),  # noqa: E501
        "Si": onp.array([14, 12.31, 9.72, 8.41, 7.58, 6.51, 5.45, 4.42, 3.59, 2.98, 2.53, 2.17, 1.91, 1.74]),  # noqa: E501
        "Ti": onp.array([22, 19.29, 15.68, 12.59, 10.03, 8.74, 8.03, 7.66, 7.27, 6.75, 6.21, 5.64, 5.04, 4.41]),  # noqa: E501
        "Kr": onp.array([36, 32.48, 26.87, 24.04, 21.86, 19.03, 16.09, 13.53, 11.54, 10.04, 9.00, 8.16, 7.54, 7.07]),  # noqa: E501
    }
    # fmt: on
    occupancy = {
        "C": jaxrts.helpers.orbital_array(2, 2, 2),
        "Si": jaxrts.helpers.orbital_array(2, 2, 6, 2, 2),
        "Ti": jaxrts.helpers.orbital_array(2, 2, 6, 2, 6, 2, 2),
        "Kr": jaxrts.helpers.orbital_array(2, 2, 6, 2, 6, 10, 2, 6),
    }
    for element, literature_F in tabulated_values.items():
        Zstar = ff.pauling_effective_charge(onp.sum(occupancy[element]))
        F = jnpu.sum(
            ff.pauling_all_ff(k, Zstar) * occupancy[element][:, onp.newaxis],
            axis=0,
        )
        assert (onp.max(onp.abs(literature_F - F) / F)) < 0.03
