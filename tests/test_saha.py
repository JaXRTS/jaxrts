import pathlib

import jax.numpy as jnp
import numpy as onp

import jaxrts
from jaxrts.saha import solve_saha
from jaxrts.units import ureg


def test_against_MALGS_calculation():
    number_fraction = jnp.array(
        [
            1,
            10 ** (10.925 - 12),
            10 ** (6.29 - 12),
            10 ** (5.11 - 12),
            10 ** (3.28 - 12),
            10 ** (7.46 - 12),
            10 ** (6.31 - 12),
        ]
    )
    number_fraction /= jnp.sum(number_fraction)
    ions = [
        jaxrts.Element("H"),
        jaxrts.Element("He"),
        jaxrts.Element("Na"),
        jaxrts.Element("K"),
        jaxrts.Element("Li"),
        jaxrts.Element("Fe"),
        jaxrts.Element("Ca"),
    ]
    mass_fraction = jaxrts.helpers.mass_from_number_fraction(
        number_fraction, ions
    )

    plasma_state = jaxrts.PlasmaState(
        ions=ions,
        Z_free=jnp.ones_like(mass_fraction),
        mass_density=ureg("1e-5g/cc") * mass_fraction,
        T_e=(
            jnp.array([40]) * ureg.electron_volt / ureg.k_B
        ),  # This is just a placeholder
    )

    # Prepare the output dictionary
    ion_population = {element.Z: [] for element in ions}
    for element in ions:
        for charge in onp.arange(element.Z + 1):  # noqa: B007
            ion_population[element.Z].append([])

    Tes = jnp.logspace(2, 6, 150) * ureg.kelvin
    for Te in Tes:
        sol, _ = solve_saha(
            tuple(plasma_state.ions),
            Te,
            (plasma_state.mass_density / plasma_state.atomic_masses),
        )
        sol = sol.m_as(1 / ureg.meter**3)

        idx = 0

        for element in ions:
            for charge in onp.arange(element.Z + 1):
                ion_population[element.Z][charge].append(sol[idx])
                idx += 1

    # numberdensity of electrons per species (reached as n_free when species is
    # fully ionized)
    full_electron_density = (plasma_state.Z_A * plasma_state.n_i).m_as(
        1 / ureg.meter**3
    )

    data_dir = pathlib.Path(__file__).parent / "data/Schoettler/"
    for idx, element in enumerate(ions):
        for charge in onp.arange(min(element.Z + 1, 3)):
            csv = data_dir / f"n_{element.symbol}_{charge}.csv"
            x, y = onp.genfromtxt(csv, unpack=True)
            # Interpolate to the literature data
            interp = jnp.interp(Tes.m_as(ureg.kelvin), x, y)

            # the literature is only ionizing up to the second degree. Collect
            # all other ionizations
            if charge < 2:
                calc = jnp.array(ion_population[element.Z][charge])
            else:
                calc = jnp.sum(
                    jnp.array(ion_population[element.Z][charge:]), axis=0
                )

            # calculate maximal relavtive difference
            diff = jnp.max(
                jnp.abs((calc - interp) / full_electron_density[idx])
            )
            assert diff < 5e-3
