from .elements import Element

from typing import List
from functools import partial
from .units import Quantity, ureg
import numpy as onp
import jax

import jax.numpy as jnp
import jpu.numpy as jnpu


h = 1 * ureg.planck_constant
k_B = 1 * ureg.boltzmann_constant
m_e = 1 * ureg.electron_mass


@jax.jit
def bisection(func, a, b, tolerance=1e-4, max_iter=1e4, min_iter=40):

    def condition(state):
        prev_x, next_x, count = state

        return (
            (count < max_iter)
            & (jnp.abs(func(next_x)) > tolerance)
            & (jnp.abs(prev_x - next_x) > tolerance)
        ) | (count < min_iter)

    def body(state):
        a, b, i = state
        c = (a + b) / 2  # middlepoint
        bound = jnp.where(jnp.sign(func(c)) == jnp.sign(func(b)), a, b)
        return bound, c, i + 1

    initial_state = (a, b, 0)

    _, final_state, iterations = jax.lax.while_loop(
        condition, body, initial_state
    )

    return final_state, iterations


@jax.jit
def saha_equation(gi: float, gj: float, T_e: Quantity, energy_diff: Quantity):

    return (
        2
        * (gj / gi)
        * ((2 * jnp.pi * 1 * m_e * k_B * T_e) ** 1.5 / (1 * h**3))
        * jnpu.exp(((-energy_diff) / (1 * k_B * T_e)))
    )


@partial(jax.jit, static_argnames=["element_list"])
def solve_saha(element_list, T_e: Quantity, ion_number_densities: Quantity):

    Z = [i.Z for i in element_list]
    M = jnp.zeros(
        (
            len(element_list) + onp.sum(Z) + 1,
            len(element_list) + onp.sum(Z) + 1,
        )
    )

    max_ne = jnpu.sum(
        jnp.array([elem.Z for elem in element_list]) * ion_number_densities
    )

    # ne_scale = max_ne
    ne_scale = 1e0 / (1 * ureg.m**3)

    skip = 0
    ionization_states = []

    for ion_dens, element in zip(ion_number_densities, element_list):

        stat_weight = element.ionization.statistical_weights
        Eb = element.ionization.energies

        coeff = (
            saha_equation(
                stat_weight[:-1],
                stat_weight[1:],
                T_e,
                Eb,
            )
            / ne_scale
        ).m_as(ureg.dimensionless)

        diag = jnp.diag((-1) * coeff)
        dens_row = jnp.ones((element.Z + 1))

        M = M.at[skip : skip + element.Z, skip : skip + element.Z].set(diag)
        M = M.at[skip : skip + element.Z + 1, skip + element.Z].set(dens_row)

        M = M.at[-1, skip + element.Z].set(
            (ion_dens / ne_scale).m_as(ureg.dimensionless)
        )

        skip += element.Z + 1
        ionization_states += list(jnp.arange(element.Z + 1))

    M = M.at[:-1, -1].set(jnp.array(ionization_states))

    def insert_ne(_M, ne):

        ne_line = jnp.ones(len(element_list) + onp.sum(Z)) * ne

        skip = -1
        for element in element_list:

            ne_line = ne_line.at[skip + element.Z + 1].set(0.0)
            skip += element.Z + 1

        _diag = jnp.diag(ne_line, -1)
        out = M + _diag
        out = out.at[-1, -1].set(ne)
        return out.T

    def det_M(ne):
        res = jnp.linalg.det(insert_ne(M, ne))
        return res

    sol_ne, iterations = bisection(
        jax.tree_util.Partial(det_M),
        0,
        (max_ne / ne_scale).m_as(ureg.dimensionless),
        tolerance=1e-2,
        max_iter=1e2,
        min_iter=0,
    )

    # jax.debug.print("Needed iterations for convergence: {x}", x=iterations)

    M = insert_ne(M, sol_ne)

    MM = jnp.array(M)
    M1 = MM[: (len(MM[0]) - 1), 0 : (len(MM[0]) - 1)]
    M2 = MM[: (len(MM[0]) - 1), (len(MM[0]) - 1)]

    # The solution in form of (nh0,nh1,nhe0,nhe1,nhe2,...)
    ionised_number_densities = jnp.linalg.solve(M1, M2)

    return ionised_number_densities * ne_scale, sol_ne * ne_scale


def calculate_mean_free_charge_saha(plasma_state):
    """
    Calculates the mean charge of each ion in a plasma using the Saha-Boltzmann
    equation.

    Parameters:
    -----------
    plasma_state (PlasmaState): The plasma state object.

    Returns:
    --------
    jnp.ndarray: An array containing the mean charge of each ion in the plasma.
    """

    sol, _ = solve_saha(
        tuple(plasma_state.ions),
        plasma_state.T_e,
        (plasma_state.mass_density / plasma_state.atomic_masses),
    )
    sol = sol.m_as(1 / ureg.cc)

    indices = jnp.cumsum(
        jnp.array([0] + list([ion.Z + 1 for ion in plasma_state.ions]))
    )
    Z_total = []
    Z_free = []
    for i in range(len(indices) - 1):
        idx = jnp.arange(len(sol))
        relevant_part = jnp.where(
            (idx >= indices[i]) & (idx < indices[i + 1]), sol, 0
        )
        ionizations = jnp.where(
            (idx >= indices[i]) & (idx < indices[i + 1]),
            jnp.arange(len(sol)) - indices[i],
            0,
        )
        Z_total.append(jnp.sum(relevant_part))
        Z_free.append(jnp.sum(relevant_part / Z_total[i] * ionizations))
    Z_free = jnp.array(Z_free)

    return Z_free
