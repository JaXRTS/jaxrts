import pytest

from pathlib import Path

import numpy as onp

import jaxrts
import jax.numpy as jnp
import jpu.numpy as jnpu

ureg = jaxrts.ureg


@pytest.mark.skip(
    reason="Re-producing figure with D. Chapmans data, directly, fails"
)
def test_literature_chapman2015():
    """
    This reprocudes the data in D. Chapman's thesis, Fig 3.7, but it shows some
    notable discrepancy.
    """
    current_folder = Path(__file__).parent
    k = 8.4 / ureg.angstrom
    for Z_b in [1, 2, 3, 4, 5, 6]:
        E, S_bf_lit = onp.genfromtxt(
            current_folder / f"data/Chapman2015/Fig3_7/Z_C{Z_b}.csv",
            delimiter=",",
            unpack=True,
        )
        sort = onp.argsort(E)
        E = E[sort] * ureg.electron_volt
        omega = E / ureg.hbar
        S_bf_lit *= ureg.hbar

        S_bf_lit = S_bf_lit[sort] * 1e-2 / ureg.electron_volt

        E_b = jaxrts.Element("C").binding_energies

        Zeff = jaxrts.form_factors.pauling_effective_charge(6)
        population = jaxrts.elements.electron_distribution_ionized_state(Z_b)

        S_bf = jaxrts.bound_free.J_impulse_approx(
            omega, k, population, Zeff, E_b
        )

        assert (
            jnpu.quantile(
                jnpu.absolute((S_bf_lit - S_bf)) / jnpu.max(S_bf), 0.90
            ).to(ureg.dimensionless)
            < 0.1
        )


def test_modified_literature_chapman2015():
    """
    This does reprocude the data in D. Chapman's thesis, Fig 3.7, but does not
    show the given input variables.
    """
    current_folder = Path(__file__).parent
    k = 7.9 / ureg.angstrom
    for Z_b in [1, 2, 3, 4, 5, 6]:
        E, S_bf_lit = onp.genfromtxt(
            current_folder / f"data/Chapman2015/Fig3_7/Z_C{Z_b}.csv",
            delimiter=",",
            unpack=True,
        )
        sort = onp.argsort(E)
        E = E[sort] * ureg.electron_volt
        omega = E / ureg.hbar
        S_bf_lit *= ureg.hbar

        S_bf_lit = S_bf_lit[sort] * 1e-2 / ureg.electron_volt

        E_b = jaxrts.Element("C").binding_energies

        Zeff = jaxrts.form_factors.pauling_effective_charge(6)
        if Z_b == 1:
            population = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif Z_b == 2:
            population = [2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            population = [2, (Z_b - 2) / 2, (Z_b - 2) / 2, 0, 0, 0, 0, 0, 0, 0]
        population = jnp.array(population)

        S_bf = jaxrts.bound_free.J_impulse_approx(
            omega, k, population, Zeff, E_b
        )

        assert (
            jnpu.quantile(
                jnpu.absolute((S_bf_lit - S_bf)) / jnpu.max(S_bf), 0.90
            ).to(ureg.dimensionless)
            < 0.1
        )


def test_holm_first_order_corrections():
    p = jnp.linspace(-0.2, 0.2, 1500)
    E = ureg("56keV")
    k = 2 * E / (ureg.hbar * ureg.c) * jnpu.sin(ureg("173°") / 2)

    Zeff = jaxrts.form_factors.pauling_effective_charge(11)

    omega_c = (ureg.hbar * k**2) / (2 * ureg.m_e)
    omega_plus = (p * ureg.c * k) + omega_c
    omega_min = (-p * ureg.c * k) + omega_c

    J10BM = jaxrts.bound_free._J10_BM(omega_c[jnp.newaxis], k, Zeff[0])
    J10HRp = jaxrts.bound_free._J10_HR(omega_plus, k, Zeff[0])
    J10HRm = jaxrts.bound_free._J10_HR(omega_min, k, Zeff[0])

    J20BM = jaxrts.bound_free._J20_BM(omega_c[jnp.newaxis], k, Zeff[1])
    J20HRp = jaxrts.bound_free._J20_HR(omega_plus, k, Zeff[1])
    J20HRm = jaxrts.bound_free._J20_HR(omega_min, k, Zeff[1])

    J21BM = jaxrts.bound_free._J21_BM(omega_c[jnp.newaxis], k, Zeff[2])
    J21HRp = jaxrts.bound_free._J21_HR(omega_plus, k, Zeff[2])
    J21HRm = jaxrts.bound_free._J21_HR(omega_min, k, Zeff[2])

    import matplotlib.pyplot as plt

    plt.plot(p, 100 * -(J10HRp - J10HRm) / J10BM, label = "1s")
    plt.plot(p, 100 * -(J20HRp - J20HRm) / J20BM, label = "2s")
    plt.plot(p, 100 * -(J21HRp - J21HRm) / J21BM, label = "2p")
    plt.show()


def test_Schum_vs_BM():
    """
    Test that the Schumacher and Bloch Mendelson J's give compatible values for
    1s, 2s and 2p.
    """
    omega = jnp.linspace(0, 200) * ureg.electron_volt / ureg.hbar
    k = 4 / ureg.angstrom
    Zeff = 2

    J10BM = jaxrts.bound_free._J10_BM(omega, k, Zeff)
    J10Schum = jaxrts.bound_free._J10_Schum75(omega, k, Zeff)

    J20BM = jaxrts.bound_free._J20_BM(omega, k, Zeff)
    J20Schum = jaxrts.bound_free._J20_Schum75(omega, k, Zeff)

    J21BM = jaxrts.bound_free._J21_BM(omega, k, Zeff)
    J21Schum = jaxrts.bound_free._J21_Schum75(omega, k, Zeff)

    assert jnpu.max(jnpu.absolute(J21BM - J21Schum)/jnpu.max(J21BM)) < 1e-6
    assert jnpu.max(jnpu.absolute(J20BM - J20Schum)/jnpu.max(J20BM)) < 1e-6
    assert jnpu.max(jnpu.absolute(J10BM - J10Schum)/jnpu.max(J10BM)) < 1e-6


test_holm_first_order_corrections()
