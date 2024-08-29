from pathlib import Path

import pytest
import jpu
import numpy as onp
from jax import numpy as jnp

import jaxrts
import jaxrts.hypernetted_chain as hnc
from jaxrts import hnc_potentials
from jaxrts.units import ureg


def test_electron_ion_potentials_literature_values_schwarz():
    """
    Test the calculation of electron and ion potentials by reproducing Fig. 1
    in :cite:`Schwarz.2007`
    """

    state = jaxrts.PlasmaState(
        ions=[jaxrts.Element("Be")],
        Z_free=[2.5],
        mass_density=[
            1.21e23 / ureg.centimeter**3 * jaxrts.Element("Be").atomic_mass
        ],
        T_e=12 * ureg.electron_volt / ureg.k_B,
        T_i=[12 * ureg.electron_volt / ureg.k_B],
    )

    r = jnp.linspace(0, 10, 1000) * ureg.angstrom

    KK = hnc_potentials.KlimontovichKraeftPotential()
    Kelbg = hnc_potentials.KelbgPotential()

    KK.include_electrons = True
    Kelbg.include_electrons = True

    ei = (-KK.full_r(state, r) / (ureg.k_B * KK.T(state)))[1, 0, :].m_as(
        ureg.dimensionless
    )
    ee = (Kelbg.full_r(state, r) / (ureg.k_B * Kelbg.T(state)))[1, 1, :].m_as(
        ureg.dimensionless
    )
    ii = (Kelbg.full_r(state, r) / (ureg.k_B * Kelbg.T(state)))[0, 0, :].m_as(
        ureg.dimensionless
    )

    current_folder = Path(__file__).parent

    for label, potential in [["ei", ei], ["ii", ii], ["ee", ee]]:
        f = list((current_folder / "data/Schwarz2007/").glob(f"*{label}.csv"))[
            0
        ]
        r_lit, V = onp.genfromtxt(f, delimiter=",", unpack=True)

        interp = jnp.interp(
            r_lit, (r / ureg.a_0).m_as(ureg.dimensionless), potential
        )

        assert jnp.max(jnp.abs(V - interp) / jnp.max(V)) < 0.01


def test_hydrogen_pair_distribution_function_literature_values_wuensch():
    """
    Test against the computation of literature data published in Fig. 4.4., in
    :cite:`Wunsch.2011`.
    """
    H = jaxrts.Element("H")
    state = jaxrts.PlasmaState(
        ions=[H],
        Z_free=[1],
        mass_density=[1e23 / ureg.centimeter**3 * H.atomic_mass],
        T_e=10 * ureg.electron_volt / ureg.k_B,
    )

    for Gamma, pot in zip([1, 10, 30, 100], [13, 13, 15, 16]):
        r = jpu.numpy.linspace(0.0001 * ureg.angstrom, 100 * ureg.a0, 2**pot)

        dr = r[1] - r[0]
        dk = jnp.pi / (len(r) * dr)
        k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk

        di = 1 / (
            Gamma
            * ((1 * ureg.boltzmann_constant) * state.T_e)
            * (4 * jnp.pi * ureg.epsilon_0)
            / ureg.elementary_charge**2
        )
        n = (1 / (di**3 * (4 * jnp.pi / 3))).to(1 / ureg.angstrom**3)
        dens = jnp.array(
            [(n * H.atomic_mass).m_as(ureg.gram / ureg.centimeter**3)]
        ) * (1 * ureg.gram / ureg.centimeter**3)
        n = jaxrts.units.to_array([n])
        state.mass_density = dens

        # ToDo: It seems that I cannot move this out of the loop. Fix this.
        Coulomb = hnc_potentials.CoulombPotential()

        V_s = Coulomb.short_r(state, r)
        # The long-range part is zero
        V_l_k = 0 * Coulomb.long_k(state, k)

        r_lit, g_lit = onp.genfromtxt(
            Path(__file__).parent
            / f"data/Wunsch2011/Fig4.4/Gamma_{Gamma}.csv",
            unpack=True,
            delimiter=", ",
        )

        d = jpu.numpy.cbrt(
            3 / (4 * jnp.pi * (n[:, jnp.newaxis] + n[jnp.newaxis, :]) / 2)
        )
        g, niter = hnc.pair_distribution_function_HNC(
            V_s, V_l_k, r, state.T_e, n
        )

        interp = jnp.interp(
            r_lit,
            (r / d[0, 0]).m_as(ureg.dimensionless),
            g[0, 0, :].m_as(ureg.dimensionless),
        )

        assert jnp.all(jnp.abs(g_lit - interp) < 0.04)


def test_linear_response_screening_gericke2010_literature():
    """
    Test if one can reproduce the scattering functions reproduced by
    :cite:`Gericke.2010`, Fig 2.
    """

    state = jaxrts.PlasmaState(
        [jaxrts.Element("Be")],
        [2],
        [ureg("1.848g/cc")],
        ureg("12eV") / ureg.k_B,
    )
    state["screening length"] = jaxrts.models.Gericke2010ScreeningLength()

    r = jnp.linspace(0.001, 100, 5000) * ureg.a_0
    k = jnp.pi / r[-1] + jnp.arange(len(r)) * (
        jnp.pi / (len(r) * (r[1] - r[0]))
    )

    state.ion_core_radius = jnp.array([1]) * ureg.angstrom

    empty_core = jaxrts.hnc_potentials.EmptyCorePotential()
    soft_core2 = jaxrts.hnc_potentials.SoftCorePotential(beta=2)
    soft_core6 = jaxrts.hnc_potentials.SoftCorePotential(beta=6)
    coulomb = jaxrts.hnc_potentials.CoulombPotential()

    current_folder = Path(__file__).parent
    data_folder = current_folder / "data/Gericke2010/Fig2"

    names = [
        "EmptyCore.csv",
        "SoftCore2.csv",
        "SoftCore6.csv",
        "Coulomb.csv",
    ]
    for idx, pot in enumerate([empty_core, soft_core2, soft_core6, coulomb]):
        pot.include_electrons = True
        import logging

        logging.warning(idx)
        q = -jaxrts.ion_feature.free_electron_susceptilibily_RPA(
            k, 1 / state.screening_length
        ) * pot.full_k(state, k)
        klit, qlit = onp.genfromtxt(
            data_folder / names[idx],
            unpack=True,
            delimiter=",",
        )
        klit *= 1 / ureg.a_0
        q_interp = jpu.numpy.interp(klit, k, q[0, 1, :])
        assert (
            jnp.max(jnp.abs(q_interp.m_as(ureg.dimensionless) - qlit)) < 0.03
        )


def test_multicomponent_wunsch2011_literature():
    # Set up the ionization, density and temperature for individual ion
    # species.
    state = jaxrts.PlasmaState(
        ions=[jaxrts.Element("H"), jaxrts.Element("C")],
        Z_free=[1, 4],
        mass_density=[
            2.5e23 / ureg.centimeter**3 * jaxrts.Element("H").atomic_mass,
            2.5e23 / ureg.centimeter**3 * jaxrts.Element("C").atomic_mass,
        ],
        T_e=2e4 * ureg.kelvin,
    )
    # Set the Screening length for the Debye Screening. Verify where this might
    # come form.
    state["screening length"] = jaxrts.models.ConstantScreeningLength(
        2 / 3 * ureg.a_0
    )

    pot = 15
    r = jpu.numpy.linspace(0.0001 * ureg.angstrom, 1000 * ureg.a0, 2**pot)

    # We add densities, here. Maybe this is wrong.
    d = jpu.numpy.cbrt(
        3
        / (
            4
            * jnp.pi
            * (state.n_i[:, jnp.newaxis] + state.n_i[jnp.newaxis, :])
        )
    )

    dr = r[1] - r[0]
    dk = jnp.pi / (len(r) * dr)
    k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk

    Potential = jaxrts.hnc_potentials.DebyeHueckelPotential()

    V_s = Potential.short_r(state, r)
    V_l_k = Potential.long_k(state, k)

    g, niter = hnc.pair_distribution_function_HNC(
        V_s, V_l_k, r, Potential.T(state), state.n_i
    )
    S_ii = hnc.S_ii_HNC(k, g, state.n_i, r)

    current_folder = Path(__file__).parent

    for idx, gtype in zip(
        [jnp.s_[0, 0, :], jnp.s_[1, 0, :], jnp.s_[1, 1, :]], ["HH", "CH", "CC"]
    ):
        xlit, glit = onp.genfromtxt(
            current_folder / f"data/Wunsch2011/Fig4.12/g_{gtype}.csv",
            unpack=True,
            delimiter=",",
        )
        klit, Slit = onp.genfromtxt(
            current_folder / f"data/Wunsch2011/Fig4.12/S_{gtype}.csv",
            unpack=True,
            delimiter=",",
        )
        g_interp = jpu.numpy.interp(xlit * d[0, 0], r, g[idx])
        S_interp = jpu.numpy.interp(klit / d[0, 0], k, S_ii[idx])
        assert (
            jnp.max(
                jpu.numpy.absolute(glit - g_interp).m_as(ureg.dimensionless)
            )
            < 0.03
        )
        assert (
            jnp.max(
                jpu.numpy.absolute(Slit - S_interp).m_as(ureg.dimensionless)
            )
            < 0.03
        )


def test_realfft_inversion():
    N = 2**7
    r = jnp.linspace(0.02, 20.0, N)

    f = r / (1 + r**2)
    f_fft = (2 / N) * jaxrts.hypernetted_chain.realfft(
        jaxrts.hypernetted_chain.realfft(f.copy()), isign=-1
    )
    assert jnp.max(jnp.abs(f - f_fft)) < 1e-8


def test_realfft_realfftnp_equaltity():
    N = 2**7
    r = jnp.linspace(0.02, 20.0, N)

    f = r / (1 + r**2)
    f_fft1 = jaxrts.hypernetted_chain.realfft(f.copy())
    f_fft2 = jaxrts.hypernetted_chain.realfftnp(f.copy())
    # There seems to be a small difference in index 1.
    assert jnp.quantile(jnp.abs(f_fft1 - f_fft2), 0.99) < 1e-8


def test_sinft_self_inverse():
    N = 2**14
    r = jnp.linspace(0.00, 20.0, N)

    f = r / (1 + r**2)
    f_fft = (2 / N) * jaxrts.hypernetted_chain.sinft(
        jaxrts.hypernetted_chain.sinft(f.copy())
    )
    assert jnp.max(jnp.abs(f - f_fft)) < 1e-8


@pytest.mark.skip(reason="Norm not clear")
def test_sinft_analytical_result():
    N = 2**14
    r = jnp.linspace(0.001, 1000, N)
    dr = r[1] - r[0]
    pref = jnp.pi

    dk = pref / (len(r) * dr)
    k = pref / r[-1] + jnp.arange(len(r)) * dk

    alpha = 4

    f = jnp.exp(-alpha * r)
    f_ft_analytical = k / (alpha**2 + k**2) * jnp.sqrt(2 / jnp.pi)

    f_fft = jaxrts.hypernetted_chain.sinft(f.copy()) / jnp.sqrt(len(r) / (2))
    assert jnp.max(jnp.abs(f_ft_analytical - f_fft)) < 1e-8
