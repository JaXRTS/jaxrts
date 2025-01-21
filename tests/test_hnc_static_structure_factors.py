import jpu
import matplotlib.pyplot as plt
from jax import numpy as jnp

import jaxrts
import jaxrts.hypernetted_chain as hnc
from jaxrts.units import ureg


def main():

    fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, squeeze=True)

    axis_count = 0

    axis = ax.flatten()

    for T in [
        20 * ureg.electron_volt / ureg.boltzmann_constant,
        10 * ureg.electron_volt / ureg.boltzmann_constant,
        4 * ureg.electron_volt / ureg.boltzmann_constant,
        1 * ureg.electron_volt / ureg.boltzmann_constant,
    ]:
        Z = 2

        n_i = (1.5e23 / ureg.cc).to(1 / ureg.angstrom**3)
        n_i = jnp.array([n_i.m_as(1 / ureg.angstrom**3)]) * (
            1 / ureg.angstrom**3
        )

        d = jpu.numpy.cbrt(
            3 / (4 * jnp.pi * (n_i[:, jnp.newaxis] + n_i[jnp.newaxis, :]) / 2)
        )

        q = jaxrts.hnc_potentials.construct_q_matrix(jnp.array([1]) * Z * ureg.elementary_charge)

        Gamma = (
            Z**2
            * 1
            * ureg.elementary_charge**2
            / (4 * jnp.pi * 1 * ureg.epsilon_0 * d)
        ) / (1 * ureg.boltzmann_constant * T)

        r = jpu.numpy.linspace(0.001 * ureg.angstrom, 1000 * ureg.a0, 2**14)

        alpha = jaxrts.hnc_potentials.construct_alpha_matrix(n_i)

        V_s = hnc.V_screenedC_s_r(r, q, alpha)

        dr = r[1] - r[0]
        dk = jnp.pi / (len(r) * dr)
        k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk

        V_l_k = hnc.V_screened_C_l_k(k, q, alpha)
        # V_l = hnc.V_screened_C_l_r(r, q, alpha)
        # V_l_k, _ = hnc.transformPotential(V_l, r)

        g, niter = jaxrts.hypernetted_chain.pair_distribution_function_HNC(V_s, V_l_k, r, T, n_i)

        S_k = jaxrts.hypernetted_chain.S_ii_HNC(k, g, n_i, r)

        axis[axis_count].plot(
            (k * d[0, 0]).m_as(ureg.dimensionless),
            S_k[0, 0].m_as(ureg.dimensionless),
        )

        axis[axis_count].annotate(
            r"T = "
            + str(T.magnitude)
            + " eV\n"
            + r"$\Gamma_{\text{ii}} = $"
            + "{:.1f}".format(Gamma.m_as(ureg.dimensionless)[0, 0]),
            (0.4, 1.2),
        )

        axis[axis_count].set_xlim(0, 10.0)
        if axis_count <= 1:
            axis[axis_count].set_ylim(0, 1.6)
        else:
            axis[axis_count].set_ylim(0, 1.6)
        if axis_count > 1:
            axis[axis_count].set_xlabel("k [d$_i^{-1}$]")
        if axis_count in [0, 2]:
            axis[axis_count].set_ylabel(r"S$_{ii}$(k)")
        axis_count += 1
    plt.show()

    # assert jnp.max(jnp.abs(f_ft_analytical - f_fft)) < 1e-8


if __name__ == "__main__":
    main()
    # test_sinft_analytical_result()
