"""
Saha equation for hot jupiter materials
=======================================

This example showcases the use of the saha equation solver. This example
consideres abundances as reported by :cite:`Lodders.2010`, and reproduces a
plot generated with M. Schöttler's MALGS programm -- which is discussed, e.g.
in :cite:`Kumar.2021`.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxrts
from jaxrts.saha import solve_saha
from jaxrts.units import ureg

if __name__ == "__main__":
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
        T_e=jnp.array([40]) * ureg.electron_volt / ureg.k_B,
    )
    ion_population = {element.Z: [] for element in ions}
    ion_population["n_free"] = []
    for element in ions:
        for charge in np.arange(element.Z + 1):  # noqa: B007
            ion_population[element.Z].append([])

    Tes = jnp.logspace(2, 6, 250) * ureg.kelvin
    for Te in Tes:
        ion_population["n_free"].append(jnp.array([0.0]))
        sol, ne, _ = solve_saha(
            tuple(plasma_state.ions),
            Te,
            (plasma_state.mass_density / plasma_state.atomic_masses),
        )
        sol = sol.m_as(1 / ureg.meter**3)
        ne = ne.m_as(1 / ureg.meter**3)

        idx = 0

        for element in ions:
            for charge in np.arange(element.Z + 1):
                ion_population[element.Z][charge].append(sol[idx])
                ion_population["n_free"][-1] += charge * sol[idx]
                idx += 1

    fig, ax = plt.subplots()

    ls = ["solid", "dashed", "dotted"]

    for idx, element in enumerate(ions):
        for charge in np.arange(min(element.Z + 1, 3)):
            plt.plot(
                Tes.m_as(ureg.kelvin),
                ion_population[element.Z][charge],
                label=f"{element.symbol}" + "$^{" + f" {charge}+" + "}$",
                color=f"C{idx}",
                ls=ls[charge],
            )

    plt.plot(
        Tes.m_as(ureg.kelvin),
        ion_population["n_free"],
        label="$n_e$",
        color="black",
        lw=2,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e9, 1e25)
    ax.set_xlabel("T [K]")
    ax.set_ylabel("ion density [1/m³]")
    plt.legend(ncols=5, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    plt.tight_layout()
    plt.show()
