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
