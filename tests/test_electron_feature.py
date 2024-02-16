import pathlib

import numpy as onp
import pytest

import jaxrts

import matplotlib.pyplot as plt

ureg = jaxrts.ureg


def test_gregori2023_fig1_reprduction() -> None:
    # Set the scattering parameters
    lambda_0 = 4.13 * ureg.nanometer
    theta = 160
    k = (4 * onp.pi / lambda_0) * onp.sin(onp.deg2rad(theta) / 2.0)
    n_e = 1e21 / ureg.centimeter**3

    # Normalize

    count = 0
    norm = 1.0

    # Load the data
    data_dir = pathlib.Path(__file__).parent / "data/Gregori2003/Fig1/"
    # We have to sort, here, to assert the normalization works properly
    entries = list(data_dir.glob("*.csv"))
    for datafile in sorted(entries):
        energy_shift, literature_See = onp.genfromtxt(
            datafile, delimiter=",", unpack=True
        )
        # Read the temperature from the filename
        T = ureg(datafile.stem[2:])
        calc_See = jaxrts.electron_feature.S0_ee(
            k,
            T_e=T / (ureg.boltzmann_constant),
            n_e=n_e,
            E=energy_shift * ureg.electron_volt,
        )
        if count == 0:
            norm = onp.max(calc_See)
        calc_See /= norm
        # Calculate the deviation between our curves and the data ripped from
        # the literature
        error = onp.abs((calc_See - literature_See).m_as(ureg.dimensionless))
        # Be a bit more generous, for 0.5eV, where the peak is huge
        if count == 0:
            assert onp.max(error) < 0.06
            assert onp.mean(error) < 0.02
        else:
            assert onp.max(error) < 0.02
        count += 1