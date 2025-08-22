import pathlib

import jax
import jax.numpy as jnp
import jpu.numpy as jnpu
import numpy as onp
import pytest

import jaxrts

ureg = jaxrts.ureg


def test_BM_glenzer2009_fig9b_reprduction() -> None:
    # This should be the same data as gregori.2003, fig 1b. But the data
    # presented in 2009 does provide reasonable units.
    # This is in to test the Borm Mermin approximation, only.

    # Set the scattering parameters
    lambda_0 = 4.13 * ureg.nanometer
    theta = 60
    k = (4 * onp.pi / lambda_0) * onp.sin(onp.deg2rad(theta) / 2.0)
    n_e = 1e21 / ureg.centimeter**3

    w_pl = jaxrts.plasma_physics.plasma_frequency(n_e)
    # Normalize


    # Load the data
    data_dir = pathlib.Path(__file__).parent / "data/Glenzer2009/Fig9/"
    # We have to sort, here, to assert the normalization works properly
    entries = list(data_dir.glob("c*.csv"))
    Tdict = {
        "c1": 0.5 * ureg.electron_volt,
        "c2": 2.0 * ureg.electron_volt,
        "c3": 8.0 * ureg.electron_volt,
    }

    for count, datafile in enumerate(sorted(entries)):
        omega_over_omega_pl, literature_See = onp.genfromtxt(
            datafile, delimiter=",", unpack=True
        )
        omega = omega_over_omega_pl * w_pl
        energy_shift = omega * ureg.hbar
        # Read the temperature from the filename
        T = Tdict[datafile.stem[:2]]

        mu = jaxrts.plasma_physics.chem_pot_interpolationIchimaru(
            T / (1 * ureg.boltzmann_constant), n_e
        )
        kappa = 1 / jaxrts.plasma_physics.Debye_Hueckel_screening_length(
            n_e, T / (1 * ureg.boltzmann_constant)
        )
        # We cout reproduce the data well with this correction to kappa
        # kappa /= 0.9

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

        @jax.tree_util.Partial
        def V_eiS(q):
            return jaxrts.free_free.statically_screened_ie_debye_potential(
                q, kappa, 1.0
            )

        calc_See = (
            jaxrts.free_free.S0_ee_BMA(
                k,
                T=T / (ureg.boltzmann_constant),
                chem_pot=mu,
                S_ii=S_ii,
                V_eiS=V_eiS,
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
                V_eiS=V_eiS,
                n_e=n_e,
                Zf=1.0,
                E_cutoff_min=jnpu.min(jnpu.absolute(energy_shift)),
                E_cutoff_max=jnpu.max(jnpu.absolute(energy_shift)),
                E=energy_shift,
                no_of_points=20,
            )
            / ureg.hbar
        ).m_as(1 / ureg.rydberg)
        calc_See_Chapman_KKT = (
            jaxrts.free_free.S0_ee_BMA_chapman_interp(
                k,
                T=T / (ureg.boltzmann_constant),
                chem_pot=mu,
                S_ii=S_ii,
                V_eiS=V_eiS,
                n_e=n_e,
                Zf=1.0,
                E=energy_shift,
                E_cutoff_min=1 / 2 * jnpu.min(jnpu.absolute(energy_shift)),
                E_cutoff_max=10 * jnpu.max(jnpu.absolute(energy_shift)),
                no_of_points=100,
                KKT=True,
            )
            / ureg.hbar
        ).m_as(1 / ureg.rydberg)
        # Calculate the deviation between our curves and the data ripped from
        # the literature
        error = onp.abs(calc_See - literature_See)

        # The low-temperature curve has some notable difference in the height of
        # the peak. However, we accept it here, for now.
        if count == 0:
            assert onp.max(error) < 0.35
            assert onp.quantile(error, 0.8) < 0.1
        else:
            assert onp.max(error) < 0.1
            assert onp.quantile(error, 0.8) < 0.05
        # Test the Chapman interpolation
        error_Chapman = onp.abs(calc_See - calc_See_Chapman)
        assert onp.max(error_Chapman) < 0.05
        error_Chapman_KKT = onp.abs(calc_See - calc_See_Chapman_KKT)
        assert onp.max(error_Chapman_KKT) < 0.05


def test_glenzer2009_fig9a_reprduction() -> None:
    # This should be the same data as gregori.2003, fig 1a. But the data
    # presented in 2009 does provide reasonable units.
    # Set the scattering parameters
    lambda_0 = 532 * ureg.nanometer
    theta = 60
    k = (4 * onp.pi / lambda_0) * onp.sin(onp.deg2rad(theta) / 2.0)
    n_e = 1e19 / ureg.centimeter**3

    w_pl = jaxrts.plasma_physics.plasma_frequency(n_e)

    # Load the data
    data_dir = pathlib.Path(__file__).parent / "data/Glenzer2009/Fig9/"
    # We have to sort, here, to assert the normalization works properly
    entries = list(data_dir.glob("a*.csv"))
    Tdict = {
        "a1": 200 * ureg.electron_volt,
        "a2": 600 * ureg.electron_volt,
        "a3": 3000 * ureg.electron_volt,
    }

    for _count, datafile in enumerate(sorted(entries)):
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


def test_gregori2003_fig1b_reprduction() -> None:
    # Set the scattering parameters
    lambda_0 = 4.13 * ureg.nanometer
    theta = 160
    k = (4 * onp.pi / lambda_0) * onp.sin(onp.deg2rad(theta) / 2.0)
    n_e = 1e21 / ureg.centimeter**3

    # Load the data
    data_dir = pathlib.Path(__file__).parent / "data/Gregori2003/Fig1/"
    # We have to sort, here, to assert the normalization works properly
    entries = list(data_dir.glob("b_*.csv"))
    for count, datafile in enumerate(sorted(entries)):
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


def test_gregori2003_fig1c_reprduction() -> None:
    # Set the scattering parameters
    lambda_0 = 0.26 * ureg.nanometer
    theta = 60
    n_e = 1e23 / ureg.centimeter**3

    k = (4 * onp.pi / lambda_0) * onp.sin(onp.deg2rad(theta) / 2.0)

    # Load the data
    data_dir = pathlib.Path(__file__).parent / "data/Gregori2003/Fig1/"
    # We have to sort, here, to assert the normalization works properly
    entries = list(data_dir.glob("c_*.csv"))
    for count, datafile in enumerate(sorted(entries)):
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
        error = onp.abs(calc_See - literature_See)

        assert onp.max(error) < 0.05
        assert onp.mean(error) < 0.02


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


def calculate_fwhm(data, x):
    peak_value = jnp.max(data)

    # Find indices where the data crosses the half maximum
    idx = jnp.where(data >= peak_value / 2.0, jnp.arange(len(data)), jnp.nan)

    left_idx = jnp.nanmin(idx)
    right_idx = jnp.nanmax(idx)

    # Interpolate linearly between the points found
    left_x = jnp.interp(
        peak_value / 2,
        jnp.array(
            [data[left_idx.astype(int) - 1], data[left_idx.astype(int)]]
        ),
        jnp.array([x[left_idx.astype(int) - 1], x[left_idx.astype(int)]]),
    )
    right_x = jnp.interp(
        peak_value / 2,
        jnp.array(
            [data[right_idx.astype(int)], data[right_idx.astype(int) + 1]]
        ),
        jnp.array([x[right_idx.astype(int)], x[right_idx.astype(int) + 1]]),
    )
    fwhm = right_x - left_x
    return jnpu.absolute(fwhm)


@pytest.mark.skip(reason="Cannot Reproduce")
def test_BornCollisionFrequency_reproduces_literature_Fortmann2010() -> None:
    import matplotlib.pyplot as plt

    data_dir = pathlib.Path(__file__).parent / "data/Fortmann2010/Fig1"

    Zf = 1.0

    @jax.tree_util.Partial
    def S_ii(q):
        return jnpu.ones_like(q)

    for r_s in [1.0, 2.0, 5.0]:
        n_e = 3 / (4 * jnp.pi * (r_s * ureg.a0) ** 3)
        w_f = jaxrts.plasma_physics.fermi_energy(n_e) / (1 * ureg.hbar)

        @jax.tree_util.Partial
        def V_eiS(q):
            return jaxrts.plasma_physics.coulomb_potential_fourier(
                Zf, -1, q
            ) / jaxrts.free_free.dielectric_function_RPA_0K(
                q, 0 * ureg.electron_volt, n_e
            )

        E_f = jaxrts.plasma_physics.fermi_energy(n_e)
        E = jnp.linspace(-200, 200, 1500) * E_f
        E_over_Ef_real, nu_real = onp.genfromtxt(
            data_dir / f"Re_rs{r_s:.0f}.csv", unpack=True, delimiter=","
        )
        E_over_Ef_imag, nu_imag = onp.genfromtxt(
            data_dir / f"Im_rs{r_s:.0f}.csv", unpack=True, delimiter=","
        )
        nu = jaxrts.free_free.collision_frequency_BA_0K(
            E, S_ii, V_eiS, n_e, Zf
        )
        dimless_nu = (nu / w_f).m_as(ureg.dimensionless)

        # Interpolate the values
        interpnu_real = jnp.interp(
            E_over_Ef_real,
            (E / E_f).m_as(ureg.dimensionless),
            jnp.real(dimless_nu),
        )
        interpnu_imag = jnp.interp(
            E_over_Ef_imag,
            (E / E_f).m_as(ureg.dimensionless),
            jnp.imag(dimless_nu),
        )
        plt.plot(E_over_Ef_imag, nu_imag)
        plt.plot(
            (E / E_f).m_as(ureg.dimensionless),
            jnp.imag(dimless_nu),
            color="black",
        )
        plt.plot(E_over_Ef_real, nu_real, ls="dashed")
        plt.plot(
            (E / E_f).m_as(ureg.dimensionless),
            jnp.real(dimless_nu),
            ls="dashed",
            color="black",
        )
        plt.xscale("log")
        plt.xlim(0.1, 100)
        plt.show()
        assert jnp.max(jnp.abs(nu_real - interpnu_real)) < 0.05
        assert jnp.max(jnp.abs(nu_imag - interpnu_imag)) < 0.05


def test_Fortmann_with_LFC_reproduces_literature() -> None:
    data_dir = pathlib.Path(__file__).parent / "data/Fortmann2010/"

    # :cite:`Fortmann.2010` calculated these values at zero kelvin. This is
    # currently not implemented, in our code. We therefore use a finite
    # temperature, instead, accepting that we might deviate slightly from the
    # published result
    T = 0.3 * ureg.electron_volt / ureg.k_B
    r_s = 2

    n_e = 3 / (4 * jnp.pi * (r_s * ureg.a0) ** 3)

    k_f = jaxrts.plasma_physics.fermi_wavenumber(n_e)
    E_f = jaxrts.plasma_physics.fermi_energy(n_e)
    k = jnp.linspace(0, 2.5) * k_f
    E = jnp.linspace(-10, 5, 1000) * E_f
    mu = jaxrts.plasma_physics.chem_pot_interpolationIchimaru(T, n_e)

    Zf = 1

    sLFC = jaxrts.ee_localfieldcorrections.eelfc_farid(
        k[:, jnp.newaxis], T, n_e
    )

    @jax.tree_util.Partial
    def S_ii(q):
        return jnpu.ones_like(q)

    @jax.tree_util.Partial
    def V_eiS(q):
        return jaxrts.plasma_physics.coulomb_potential_fourier(
            Zf, -1, q
        ) / jaxrts.free_free.dielectric_function_RPA_Dandrea1986(
            q, 0 * ureg.electron_volt, T, n_e
        )

    S_ee_noLFC = jaxrts.free_free.S0_ee_BMA_Fortmann(
        k[:, jnp.newaxis],
        T,
        mu,
        S_ii,
        V_eiS,
        n_e,
        Zf,
        jnpu.min(jnpu.absolute(E)),
        jnpu.max(jnpu.absolute(E)),
        E[jnp.newaxis, :],
        0.0,
        no_of_points=150,
    )
    S_ee_sLFC = jaxrts.free_free.S0_ee_BMA_Fortmann(
        k[:, jnp.newaxis],
        T,
        mu,
        S_ii,
        V_eiS,
        n_e,
        Zf,
        jnpu.min(jnpu.absolute(E)),
        jnpu.max(jnpu.absolute(E)),
        E[jnp.newaxis, :],
        sLFC,
        no_of_points=150,
    )
    for S_ee, suffix in [(S_ee_noLFC, ""), (S_ee_sLFC, "sLFC")]:
        idx = jnpu.argmax(S_ee, axis=1)
        w_k_over_kf, w = onp.genfromtxt(
            data_dir / f"Fig7/BMA{suffix}.csv", delimiter=",", unpack=True
        )
        G_k_over_kf, G = onp.genfromtxt(
            data_dir / f"Fig11/BMA{suffix}.csv", delimiter=",", unpack=True
        )
        interpw = jnp.interp(
            w_k_over_kf,
            (k / k_f).m_as(ureg.dimensionless),
            (-E[idx] / E_f).m_as(ureg.dimensionless),
        )
        # Calculate the FWHM
        FWHM = jax.vmap(calculate_fwhm, in_axes=(0, None))(
            S_ee.m_as(ureg.second), (E / E_f).m_as(ureg.dimensionless)
        )
        interpG = jnp.interp(
            G_k_over_kf,
            (k / k_f).m_as(ureg.dimensionless),
            FWHM,
        )

        assert jnp.max(jnp.abs(w - interpw)) < 0.1
        assert jnp.max(jnp.abs(G - interpG)) < 0.15


def test_Fortman_reproduces_vanilla_BMA_without_LFC():
    Zf = 1.0

    lfc = 0.0

    lambda_0 = 4.13 * ureg.nanometer
    theta = 60
    n_e = 1e21 / ureg.centimeter**3

    k = (4 * jnp.pi / lambda_0) * jnp.sin(jnp.deg2rad(theta) / 2.0)
    w_pl = jaxrts.plasma_physics.plasma_frequency(n_e)
    omega = jnp.linspace(-6, 6, 200) * w_pl
    E = omega * ureg.hbar

    T = 50000 * ureg.kelvin

    mu = jaxrts.plasma_physics.chem_pot_interpolationIchimaru(T, n_e)

    @jax.tree_util.Partial
    def S_ii(q):
        return jnpu.ones_like(q)

    @jax.tree_util.Partial
    def V_eiS(q):
        return jaxrts.plasma_physics.coulomb_potential_fourier(
            Zf, -1, q
        ) / jaxrts.free_free.dielectric_function_RPA_Dandrea1986(
            q, 0 * ureg.electron_volt, T, n_e
        )

    classical_BMA = jaxrts.free_free.dielectric_function_BMA_chapman_interpFit(
        k,
        E,
        mu,
        T,
        n_e,
        S_ii,
        V_eiS,
        Zf,
        jnpu.min(jnpu.absolute(E)),
        jnpu.max(jnpu.absolute(E)),
    )
    fortmann_BMA = jaxrts.free_free.dielectric_function_BMA_Fortmann(
        k,
        E,
        mu,
        T,
        n_e,
        S_ii,
        V_eiS,
        Zf,
        jnpu.min(jnpu.absolute(E)),
        jnpu.max(jnpu.absolute(E)),
        lfc,
    )
    assert jnp.isclose(jnp.real(classical_BMA), jnp.real(fortmann_BMA)).all()
    assert jnp.isclose(jnp.imag(classical_BMA), jnp.imag(fortmann_BMA)).all()
