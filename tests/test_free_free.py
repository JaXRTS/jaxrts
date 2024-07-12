import pathlib

import numpy as onp
import pytest
import jax
import jax.numpy as jnp

import jaxrts

ureg = jaxrts.ureg


def test_BM_glenzer2009_fig9b_reprduction() -> None:
    # This should be the same data as gregori.2003, fig 1b. But the data
    # presented in 2009 does provide reasonable units.
    # This is in to test the Borm Mermin appoximation, only.

    # Set the scattering parameters
    lambda_0 = 4.13 * ureg.nanometer
    theta = 60
    k = (4 * onp.pi / lambda_0) * onp.sin(onp.deg2rad(theta) / 2.0)
    n_e = 1e21 / ureg.centimeter**3

    w_pl = jaxrts.plasma_physics.plasma_frequency(n_e)
    # Normalize

    count = 0

    # Load the data
    data_dir = pathlib.Path(__file__).parent / "data/Glenzer2009/Fig9/"
    # We have to sort, here, to assert the normalization works properly
    entries = list(data_dir.glob("c*.csv"))
    Tdict = {
        "c1": 0.5 * ureg.electron_volt,
        "c2": 2.0 * ureg.electron_volt,
        "c3": 8.0 * ureg.electron_volt,
    }

    for datafile in sorted(entries):
        omega_over_omega_pl, literature_See = onp.genfromtxt(
            datafile, delimiter=",", unpack=True
        )
        omega = omega_over_omega_pl * w_pl
        energy_shift = omega * ureg.hbar
        # Read the temperature from the filename
        T = Tdict[datafile.stem[:2]]

        @jax.tree_util.Partial
        def S_ii(q):
            return jaxrts.static_structure_factors.S_ii_AD(
                q,
                T / (1 * ureg.boltzmann_constant),
                T / (1 * ureg.boltzmann_constant),
                n_e,
                1 * ureg.proton_mass,
                Z_f=1.0,
            )
        mu = jaxrts.plasma_physics.chem_pot_interpolationIchimaru(
            T / (1 * ureg.boltzmann_constant), n_e
        )
        calc_See = (
            jaxrts.free_free.S0_ee_BMA(
                k,
                T=T / (ureg.boltzmann_constant),
                chem_pot=mu,
                S_ii=S_ii,
                n_e=n_e,
                Zf=1.0,
                E=energy_shift,
            )
            / ureg.hbar
        ).m_as(1 / ureg.rydberg)
        calc_See_Chapman = (
            jaxrts.free_free.S0_ee_BMA_chapman_interp(
                k,
                T=T / (ureg.boltzmann_constant),
                chem_pot=mu,
                S_ii=S_ii,
                n_e=n_e,
                Zf=1.0,
                E=energy_shift,
            )
            / ureg.hbar
        ).m_as(1 / ureg.rydberg)
        # Calculate the deviation between our curves and the data ripped from
        # the literature
        error = onp.abs(calc_See - literature_See)

        # The low-temperature curve has some notable difference in the hight of
        # the peak. However, we accept it here, for now.
        if count == 0:
            assert onp.max(error) < 0.35
            assert onp.quantile(error, 0.8) < 0.1
        else:
            assert onp.max(error) < 0.1
            assert onp.quantile(error, 0.8) < 0.05
        # import matplotlib.pyplot as plt
        # plt.plot(energy_shift, calc_See_Chapman, color=f"C{count}", alpha = 0.5)
        # plt.plot(energy_shift, literature_See, color=f"C{count}", ls=":")

        # Test the Chapman interpolation
        error_Chapman = onp.abs(calc_See - calc_See_Chapman)
        assert onp.max(error_Chapman) < 0.01
        count += 1


def test_glenzer2009_fig9a_reprduction() -> None:
    # This should be the same data as gregori.2003, fig 1a. But the data
    # presented in 2009 does provide reasonable units.
    # Set the scattering parameters
    lambda_0 = 532 * ureg.nanometer
    theta = 60
    k = (4 * onp.pi / lambda_0) * onp.sin(onp.deg2rad(theta) / 2.0)
    n_e = 1e19 / ureg.centimeter**3

    w_pl = jaxrts.plasma_physics.plasma_frequency(n_e)
    # Normalize

    count = 0

    # Load the data
    data_dir = pathlib.Path(__file__).parent / "data/Glenzer2009/Fig9/"
    # We have to sort, here, to assert the normalization works properly
    entries = list(data_dir.glob("a*.csv"))
    Tdict = {
        "a1": 200 * ureg.electron_volt,
        "a2": 600 * ureg.electron_volt,
        "a3": 3000 * ureg.electron_volt,
    }

    for datafile in sorted(entries):
        omega_over_omega_pl, literature_See = onp.genfromtxt(
            datafile, delimiter=",", unpack=True
        )
        omega = omega_over_omega_pl * w_pl
        energy_shift = omega * ureg.hbar
        # Read the temperature from the filename
        T = Tdict[datafile.stem]
        mu = jaxrts.plasma_physics.chem_pot_interpolationIchimaru(
            T / (1 * ureg.boltzmann_constant), n_e
        )
        calc_See = (
            jaxrts.free_free.S0_ee_RPA_no_damping(
                k,
                T_e=T / (ureg.boltzmann_constant),
                n_e=n_e,
                chem_pot=mu,
                E=energy_shift,
            )
            / ureg.hbar
        ).m_as(1 / ureg.rydberg)
        # Calculate the deviation between our curves and the data ripped from
        # the literature
        error = onp.abs(calc_See - literature_See)

        assert onp.max(error) < 5
        assert onp.mean(error) < 0.5
        count += 1


def test_gregori2003_fig1b_reprduction() -> None:
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
    entries = list(data_dir.glob("b_*.csv"))
    for datafile in sorted(entries):
        energy_shift, literature_See = onp.genfromtxt(
            datafile, delimiter=",", unpack=True
        )
        # Read the temperature from the filename
        T = ureg(datafile.stem[2:])
        calc_See = jaxrts.free_free.S0_ee_Salpeter(
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


def test_gregori2003_fig1c_reprduction() -> None:
    # Set the scattering parameters
    lambda_0 = 0.26 * ureg.nanometer
    theta = 60
    n_e = 1e23 / ureg.centimeter**3

    k = (4 * onp.pi / lambda_0) * onp.sin(onp.deg2rad(theta) / 2.0)

    # Normalize
    count = 0
    norm = 1.0

    # Load the data
    data_dir = pathlib.Path(__file__).parent / "data/Gregori2003/Fig1/"
    # We have to sort, here, to assert the normalization works properly
    entries = list(data_dir.glob("c_*.csv"))
    for datafile in sorted(entries):
        energy_shift, literature_See = onp.genfromtxt(
            datafile, delimiter=",", unpack=True
        )
        # Read the temperature from the filename
        T = ureg(datafile.stem[2:])

        mu = jaxrts.plasma_physics.chem_pot_interpolationIchimaru(
            T / (1 * ureg.boltzmann_constant), n_e
        )
        calc_See = jaxrts.free_free.S0_ee_RPA_no_damping(
            k,
            T_e=T / (1 * ureg.boltzmann_constant),
            n_e=n_e,
            E=energy_shift * ureg.electron_volt,
            chem_pot=mu,
        ).m_as(ureg.second)
        if count == 0:
            norm = onp.max(calc_See)
        calc_See /= norm
        # Calculate the deviation between our curves and the data ripped from
        # the literature
        error = onp.abs((calc_See - literature_See))

        assert onp.max(error) < 0.05
        assert onp.mean(error) < 0.02
        count += 1


def test_dandrea_fit_reproduces_calculated_RPA() -> None:
    lambda_0 = 0.26 * ureg.nanometer
    E = jnp.linspace(-200, 500, 1000) * ureg.electron_volt
    theta = 60
    n_e = 1e23 / ureg.centimeter**3
    T = 50000 * ureg.kelvin
    k = (4 * onp.pi / lambda_0) * onp.sin(onp.deg2rad(theta) / 2.0)

    mu = jaxrts.plasma_physics.chem_pot_interpolationIchimaru(T, n_e)
    calc_RPA = jaxrts.free_free.dielectric_function_RPA_no_damping(k, E, mu, T)
    dfit_RPA = jaxrts.free_free.dielectric_function_RPA_Dandrea1986(
        k, E, T, n_e
    )
    assert jnp.max(jnp.abs(jnp.real(calc_RPA) - jnp.real(dfit_RPA))) < 0.005
    assert jnp.max(jnp.abs(jnp.imag(calc_RPA) - jnp.imag(dfit_RPA))) < 0.005
