import pytest

import numpy as onp
import jaxrts.form_factors as ff


def test_screening_constants_tabulated_values():
    carbon_screening = ff.pauling_size_screening_constants(6)
    assert onp.abs(carbon_screening[1] - 2.04) < 0.1
    assert onp.abs(carbon_screening[2] - 2.91) < 0.1


def test_effective_charge_tabulated_values():
    carbon_z_eff = ff.pauling_effective_charge(6)
    assert onp.abs(carbon_z_eff[0] - 5.81) < 0.1
    assert onp.abs(carbon_z_eff[1] - 3.96) < 0.1
    assert onp.abs(carbon_z_eff[2] - 3.09) < 0.1
