import jax.numpy as jnp
import jpu.numpy as jnpu
import numpy as onp

import jaxrts
import jaxrts.form_factors as ff

ureg = jaxrts.ureg


def test_screening_constants_tabulated_values():
    carbon_screening = ff.pauling_size_screening_constants(6)
    assert onp.abs(carbon_screening[1] - 2.04) < 0.1
    assert onp.abs(carbon_screening[2] - 2.91) < 0.1


def test_effective_charge_tabulated_values():
    carbon_z_eff = ff.pauling_effective_charge(6)
    assert onp.abs(carbon_z_eff[0] - 5.81) < 0.1
    assert onp.abs(carbon_z_eff[1] - 3.96) < 0.1
    assert onp.abs(carbon_z_eff[2] - 3.09) < 0.1


def test_tabulated_scattering_factors():
    """
    Test full scattering factors (i.e., the sum of individual f) against
    literature values. The scattering factors for full atoms are taken from
    table VII in :cite:`Pauling.1932`.
    Relative uncertainties are < ~3%.
    """
    k = onp.linspace(0, 1.3, 14) * (4 * onp.pi) / ureg.angstrom
    # fmt: off
    tabulated_values = {
        "C": onp.array([6, 5.21, 3.62, 2.42, 1.86, 1.66, 1.57, 1.51, 1.42, 1.32, 1.21, 1.10, 0.99, 0.89]),  # noqa: E501
        "Si": onp.array([14, 12.31, 9.72, 8.41, 7.58, 6.51, 5.45, 4.42, 3.59, 2.98, 2.53, 2.17, 1.91, 1.74]),  # noqa: E501
        "Ti": onp.array([22, 19.29, 15.68, 12.59, 10.03, 8.74, 8.03, 7.66, 7.27, 6.75, 6.21, 5.64, 5.04, 4.41]),  # noqa: E501
        "Kr": onp.array([36, 32.48, 26.87, 24.04, 21.86, 19.03, 16.09, 13.53, 11.54, 10.04, 9.00, 8.16, 7.54, 7.07]),  # noqa: E501
    }
    # fmt: on
    occupancy = {
        "C": jaxrts.helpers.orbital_array(2, 2, 2),
        "Si": jaxrts.helpers.orbital_array(2, 2, 6, 2, 2),
        "Ti": jaxrts.helpers.orbital_array(2, 2, 6, 2, 6, 2, 2),
        "Kr": jaxrts.helpers.orbital_array(2, 2, 6, 2, 6, 10, 2, 6),
    }
    for element, literature_F in tabulated_values.items():
        Zstar = ff.pauling_effective_charge(onp.sum(occupancy[element]))
        F = jnpu.sum(
            ff.pauling_all_ff(k, Zstar) * occupancy[element][:, onp.newaxis],
            axis=0,
        )
        assert (onp.max(onp.abs(literature_F - F) / F)) < 0.03


def test_effective_charge_ffl_approaches_pauling_0_ipd():
    """
    Test that the effective charge (and with it the form factors) in the form
    factor lowering formalism approach the values of Pauling for 0 IPD.
    """
    Z_A = jnp.arange(2, 36)

    Zeff_pauling_H_like = Z_A
    Zeff_pauling_He_like = (
        Z_A - jaxrts.form_factors.pauling_size_screening_constants(Z_A)[0]
    )

    Zeff_ffl_H_like = []
    Zeff_ffl_He_like = []

    Zeff_ffl_H_like_corr = []
    Zeff_ffl_He_like_corr = []
    for Zs in Z_A:
        binding_E = jnpu.sort(jaxrts.Element(Zs).ionization.energies)[-2:][
            ::-1
        ]
        H_like, He_like = jaxrts.form_factors.form_factor_lowering_Zeff_10(
            binding_E, Zs, Z_squared_correction=False
        )
        Zeff_ffl_H_like.append(H_like)
        Zeff_ffl_He_like.append(He_like)

        H_like, He_like = jaxrts.form_factors.form_factor_lowering_Zeff_10(
            binding_E, Zs, Z_squared_correction=True
        )
        Zeff_ffl_H_like_corr.append(H_like)
        Zeff_ffl_He_like_corr.append(He_like)

    Zeff_ffl_H_like = jnp.array(Zeff_ffl_H_like)
    Zeff_ffl_He_like = jnp.array(Zeff_ffl_He_like)

    Zeff_ffl_H_like_corr = jnp.array(Zeff_ffl_H_like_corr)
    Zeff_ffl_He_like_corr = jnp.array(Zeff_ffl_He_like_corr)

    assert jnp.all(jnp.abs(Zeff_ffl_H_like - Zeff_pauling_H_like)[:15] < 0.05)
    assert jnp.all(jnp.abs(Zeff_ffl_H_like_corr - Zeff_pauling_H_like) < 0.03)

    assert jnp.all(
        jnp.abs(Zeff_ffl_He_like - Zeff_pauling_He_like)[:15] < 0.05
    )
    assert jnp.all(
        jnp.abs(Zeff_ffl_He_like_corr - Zeff_pauling_He_like) < 0.03
    )


def test_0_idp_ffl_approaches_pauling():
    """
    Compare the 1s form factors of ffl and pauling for z IPD.
    """
    k = jnp.linspace(0, 170, 400) / (1 * ureg.angstrom)
    for Z_A in [4, 25]:
        for Z_C in [0.5, 1.0, 1.5, 2.0, 2.5]:
            Zeff_pauling = (
                Z_A
                - jaxrts.form_factors.pauling_size_screening_constants(Z_C)[0]
            )
            f_pauling = jaxrts.form_factors.pauling_f10(k, Zeff_pauling)

            binding_E = jaxrts.Element(Z_A).ionization.energies[::-1][:2]
            f_ffl = jaxrts.form_factors.form_factor_lowering_10(
                k, binding_E, Z_C, Z_A, Z_squared_correction=True
            )
            assert (
                jnpu.max(jnpu.absolute(f_ffl - f_pauling)).m_as(
                    ureg.dimensionless
                )
                < 0.001
            )
