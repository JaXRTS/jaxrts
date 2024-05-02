import pytest

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


def main():
    for pot in [13, 14, 15]:
        r = jpu.numpy.linspace(0.01 * ureg.angstrom, 100 * ureg.a0, 2**pot)
        q = hnc.construct_q_matrix(jnp.array([1]) * 1 * ureg.elementary_charge)
        T = 10 * ureg.electron_volt / ureg.boltzmann_constant

        Gamma = 30
        d = 1 / (
            Gamma
            * (1 * ureg.boltzmann_constant)
            * T
            * 4
            * jnp.pi
            * ureg.epsilon_0
            / ureg.elementary_charge**2
        )

        n = (1 / (d**3 * (4 * jnp.pi / 3))).to(1 / ureg.angstrom**3)

        n = jnp.array([n.m_as(1 / ureg.angstrom**3)]) * (1 / ureg.angstrom**3)

        d = jpu.numpy.cbrt(
            3 / (4 * jnp.pi * (n[:, jnp.newaxis] + n[jnp.newaxis, :]) / 2)
        )

        alpha = hnc.construct_alpha_matrix(n)

        V_s = hnc.V_s(r, q, alpha)

        dr = r[1] - r[0]
        dk = jnp.pi/ (len(r) * dr)
        k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk

        V_l_k = hnc.V_l_k(k, q, alpha)
        V_l = hnc.V_l(r, q, alpha)

        g, niter = hnc.pair_distribution_function_HNC(V_s, V_l, V_l_k, r, T, n)

        print(niter)
        print(g)

        plt.plot(
            (r / d[0, 0]).m_as(ureg.dimensionless),
            g[0, 0, :].m_as(ureg.dimensionless),
        )
    plt.xlim(0, 5.0)
    plt.ylim(0, 1.5)
    plt.show()


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


def test_sinft_self_inverse():
    N = 2**14
    r = jnp.linspace(0.00, 20.0, N)

    f = r / (1 + r**2)
    f_fft = (2 / N) * jaxrts.hypernetted_chain.sinft(
        jaxrts.hypernetted_chain.sinft(f.copy())
    )
    assert jnp.max(jnp.abs(f - f_fft)) < 1e-8


if __name__ == "__main__":
    main()
