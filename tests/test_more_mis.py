import jax.numpy as jnp
import pathlib

import jaxrts
from jaxrts import ureg
from jaxrts.ionization import (
    calculate_mean_free_charge_more,
    _calculate_mean_free_charge_more_single_species,
)

import numpy as np

import pytest
import os

data_dir = pathlib.Path(__file__).parent / "data/Stanton2017/"


def test_compare_stanton():
    """
    Reproducing Fig. 14 (right) from cite:`Stanton.2017`.
    """

    TOLERANCE = 1e-1

    if not os.path.exists(data_dir):
        pytest.fail(f"Data directory '{data_dir}' not found. Test cannot run.")

    stanton_result = []

    T_vals = []

    # Load in Stanton data
    for k in range(1, 11):
        filename = data_dir / f"data_Z={k}.csv"

        data = np.loadtxt(filename, delimiter=",")
        x = data[:, 0]
        y = data[:, 1]
        T_vals.append(x)
        stanton_result.append(y)

    # Calculate using jaxrts
    n_i_base = jnp.ones(10) * 1e23 * (1 / ureg.cc)

    ions = [
        jaxrts.Element("H"),
        jaxrts.Element("He"),
        jaxrts.Element("Li"),
        jaxrts.Element("Be"),
        jaxrts.Element("B"),
        jaxrts.Element("C"),
        jaxrts.Element("N"),
        jaxrts.Element("O"),
        jaxrts.Element("F"),
        jaxrts.Element("Ne"),
    ]

    Z_free_init = jnp.ones(len(ions)) * 1.0
    jaxrts_result = []

    # This might be overkill, but it is cheap anyway
    for k, T_es in enumerate(T_vals):
        Zbars = []

        for T in T_es:
            state = jaxrts.PlasmaState(
                ions=ions,
                mass_density=n_i_base
                * jnp.array(
                    [
                        e.atomic_mass.m_as(ureg.atomic_mass_constant)
                        for e in ions
                    ]
                )
                * ureg.atomic_mass_constant,
                Z_free=Z_free_init,
                T_e=T * ureg.electron_volt / ureg.k_B,
            )

            Zbar_i = calculate_mean_free_charge_more(state)
            Zbars.append(Zbar_i[k])
        jaxrts_result.append(Zbars)

    errors = []
    for ent1, ent2 in zip(stanton_result, jaxrts_result):
        errors.append(np.max(np.array(ent1) - np.array(ent2)))

    assert np.max(errors) < TOLERANCE


def test_compatibility_with_one_species_plasma():
    state = jaxrts.PlasmaState(
        ions=[jaxrts.Element("C")],
        mass_density=[3 * ureg.gram / ureg.cc],
        Z_free=[3],
        T_e=16 * ureg.electron_volt / ureg.k_B,
    )

    Zbar_i = calculate_mean_free_charge_more(state)
    Zbar_i_one_species = _calculate_mean_free_charge_more_single_species(state)
    assert jnp.abs(Zbar_i - Zbar_i_one_species) < 1e-4
