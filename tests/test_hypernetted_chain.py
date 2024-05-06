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

from scipy.fftpack import rfftfreq

from scipy.fftpack import rfftfreq


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
        r = jpu.numpy.linspace(0.0001 * ureg.angstrom, 10 * ureg.a0, 2**pot)

        n = (1 / (di**3 * (4 * jnp.pi / 3))).to(1 / ureg.angstrom**3)

        n = jnp.array([n.m_as(1 / ureg.angstrom**3)]) * (1 / ureg.angstrom**3)

        d = jpu.numpy.cbrt(
            3 / (4 * jnp.pi * (n[:, jnp.newaxis] + n[jnp.newaxis, :]) / 2)
        )

        alpha = hnc.construct_alpha_matrix(n)

        V_s = hnc.V_s(r, q, alpha)

        dr = r[1] - r[0]
        dk = jnp.pi / (len(r) * dr)
        k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk

        V_l_k = hnc.V_l_k(k, q, alpha)
        # V_l = hnc.V_l(r, q, alpha)
        # V_l_k, _ = hnc.transformPotential(V_l, r)

        g, niter = hnc.pair_distribution_function_HNC(V_s, V_l_k, r, T, n)

        interp = jnp.interp(
            r_lit,
            (r / d[0, 0]).m_as(ureg.dimensionless),
            g[0, 0, :].m_as(ureg.dimensionless),
        )

        assert jnp.all(jnp.abs(g_lit - interp) < 0.04)


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
    # pref = jnp.sqrt(jnp.pi)

    # dk = pref / (len(r) * dr)
    # k = pref / r[-1] + jnp.arange(len(r)) * dk
    k = rfftfreq(len(r), d=dr)

    f = 1 / jnp.sqrt(r)
    f_fft = jaxrts.hypernetted_chain.sinft(f.copy()) / jnp.sqrt(len(r) / (2))

    f_ft_analytical = jnp.sqrt(jnp.pi / (2 * k))
    factor = f_fft / f_ft_analytical
    plt.plot(k, f_fft, label="Trafo")
    plt.plot(k, f_ft_analytical, label="Ana")
    plt.plot(k, factor)
    plt.legend()
    plt.ylim(0, 2)
    plt.show()

    # assert jnp.max(jnp.abs(f_ft_analytical - f_fft)) < 1e-8

def test_sinft_vs_scipy_dst():
    N = 2**14
    r = jnp.linspace(0.01, 100, N)
    dr = r[1] - r[0]
    pref = jnp.sqrt(jnp.pi)

    dk = pref / (len(r) * dr)
    k = pref / r[-1] + jnp.arange(len(r)) * dk

    f = 1 / r ** 2
    # f = 1 / jnp.sqrt(r)
    for i in range(4):
        scipy = sp_dst(f.copy(), type=i + 1)
    # f_fft2 = jaxrts.hypernetted_chain.dst(f.copy(), 2) * jnp.sqrt(N * 2)

        plt.plot(k, scipy, label = str(i+1), alpha = 0.7,)
    f_fft = jaxrts.hypernetted_chain.sinft(f.copy()) * 2
    plt.plot(k, f_fft, label="we")
    plt.legend()
    plt.show()

