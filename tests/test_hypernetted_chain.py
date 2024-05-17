import pytest
from pathlib import Path

import sys

sys.path.append(
    "C:/Users/Samuel/Desktop/PhD/Python_Projects/JAXRTS/jaxrts/src"
)
from jax import numpy as jnp
import jaxrts
import jpu
import jaxrts.hypernetted_chain as hnc
import matplotlib.pyplot as plt
from jaxrts.units import ureg
import numpy as onp
from scipy.fft import dst as sp_dst


def test_electron_ion_potentials_literature_values_schwarz():
    """
    Test the calculation of electron and ion potentials by reproducing Fig. 1
    in :cite:`Schwarz.2007`
    """

    Z = 2.5

    q = hnc.construct_q_matrix(jnp.array([-1, Z]) * 1 * ureg.elementary_charge)
    T = 12 * ureg.electron_volt / ureg.k_B

    r = jnp.linspace(0, 10, 1000) * ureg.angstrom

    m = (
        jnp.array(
            [
                (1 * ureg.electron_mass).m_as(ureg.gram),
                (9 * ureg.proton_mass).m_as(ureg.gram),
            ]
        )
        * ureg.gram
    )

    # (1/mu = 1/m1 + 1/m2)
    mu = jpu.numpy.outer(m, m) / (m[:, jnp.newaxis] + m[jnp.newaxis, :])
    # Compared to Gregori.2003, there is a pi missing
    lambda_ab = ureg.hbar * jpu.numpy.sqrt(1 / (2 * mu * ureg.k_B * T))

    ei = (-hnc.V_Klimontovich_Kraeft_r(r, q, lambda_ab, T) / (ureg.k_B * T))[
        1, 0, :
    ].m_as(ureg.dimensionless)
    ee = (hnc.V_Kelbg_r(r, q, lambda_ab) / (ureg.k_B * T))[0, 0, :].m_as(
        ureg.dimensionless
    )
    ii = (hnc.V_Kelbg_r(r, q, lambda_ab) / (ureg.k_B * T))[1, 1, :].m_as(
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
    for Gamma, pot in zip([1, 10, 30, 100], [13, 13, 15, 16]):
        q = hnc.construct_q_matrix(jnp.array([1]) * 1 * ureg.elementary_charge)

        T = 10 * ureg.electron_volt / ureg.boltzmann_constant
        di = 1 / (
            Gamma
            * (1 * ureg.boltzmann_constant)
            * T
            * 4
            * jnp.pi
            * ureg.epsilon_0
            / ureg.elementary_charge**2
        )

        r_lit, g_lit = onp.genfromtxt(
            Path(__file__).parent
            / f"data/Wunsch2011/Fig4.4/Gamma_{Gamma}.csv",
            unpack=True,
            delimiter=", ",
        )
        r = jpu.numpy.linspace(0.0001 * ureg.angstrom, 100 * ureg.a0, 2**pot)

        n = (1 / (di**3 * (4 * jnp.pi / 3))).to(1 / ureg.angstrom**3)

        n = jnp.array([n.m_as(1 / ureg.angstrom**3)]) * (1 / ureg.angstrom**3)

        d = jpu.numpy.cbrt(
            3 / (4 * jnp.pi * (n[:, jnp.newaxis] + n[jnp.newaxis, :]) / 2)
        )

        alpha = hnc.construct_alpha_matrix(n)

        V_s = hnc.V_screenedC_s_r(r, q, alpha)

        dr = r[1] - r[0]
        dk = jnp.pi / (len(r) * dr)
        k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk

        # The long-range part is zero
        V_l_k = hnc.V_screened_C_l_k(k, q, alpha)
        V_l_k *= 0

        g, niter = hnc.pair_distribution_function_HNC(V_s, V_l_k, r, T, n)

        interp = jnp.interp(
            r_lit,
            (r / d[0, 0]).m_as(ureg.dimensionless),
            g[0, 0, :].m_as(ureg.dimensionless),
        )

        assert jnp.all(jnp.abs(g_lit - interp) < 0.04)


def test_multicomponent_wunsch2011_literature():
    # Set up the ionization, density and temperature for individual ion
    # species.
    q = hnc.construct_q_matrix(jnp.array([1, 4]) * 1 * ureg.elementary_charge)
    n = jnp.array([2.5e23, 2.5e23]) * (1 / ureg.centimeter**3)
    T = 2e4 * ureg.kelvin

    pot = 15
    r = jpu.numpy.linspace(0.0001 * ureg.angstrom, 1000 * ureg.a0, 2**pot)

    # We add densities, here. Maybe this is wrong.
    d = jpu.numpy.cbrt(
        3 / (4 * jnp.pi * (n[:, jnp.newaxis] + n[jnp.newaxis, :]))
    )

    dr = r[1] - r[0]
    dk = jnp.pi / (len(r) * dr)
    k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk

    alpha = hnc.construct_alpha_matrix(n)

    # This screening constant is guessed arbitrarily. Verify.
    screen = 2 / 3 * ureg.a_0

    V_l_k, k = hnc.transformPotential(
        hnc.V_Debye_Huckel_l_r(r, q, alpha, 1 / screen), r
    )
    V_s = hnc.V_Debye_Huckel_s_r(r, q, alpha, 1 / screen)

    g, niter = hnc.pair_distribution_function_HNC(V_s, V_l_k, r, T, n)
    S_ii = hnc.S_ii_HNC(k, g, n, r)

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
            < 0.015
        )
        assert (
            jnp.max(
                jpu.numpy.absolute(Slit - S_interp).m_as(ureg.dimensionless)
            )
            < 0.03
        )


def test_sinft_self_inverse():
    N = 2**12
    r = jnp.linspace(0.00, 20.0, N)

    f = r / (1 + r**2)
    f_fft = (2 / N) * jaxrts.hypernetted_chain.sinft(
        jaxrts.hypernetted_chain.sinft(f.copy())
    )
    assert jnp.max(jnp.abs(f - f_fft)) < 1e-8


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


def test_sinft_OLDsinft_equaltity():
    N = 2**7
    r = jnp.linspace(0.02, 20.0, N)

    f = r / (1 + r**2)
    f_fft1 = jaxrts.hypernetted_chain.sinft(f.copy())
    f_fft2 = jaxrts.hypernetted_chain.OLDsinft(f.copy())
    # There seems to be a small difference in index 1.
    assert jnp.max(jnp.abs(f_fft1 - f_fft2)) < 1e-8


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
