"""
Module containing functions to solve the
`Saha-equation <https://en.wikipedia.org/wiki/Saha_ionization_equation>`_,
linking the temperature of a plasma to it's ionization.
"""

from functools import partial
from typing import List

import jax
import jax.numpy as jnp
import jpu.numpy as jnpu
import numpy as onp

from .elements import Element
from .units import Quantity, ureg

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
def saha_equation(
    gi: float, gj: float, T_e: Quantity, energy_diff: Quantity
) -> Quantity:
    """
    Calculate the Saha equation for a given set of statistical weights,
    temperatures and energy-differences.

    Returns the value n_j / n_i * n_e

    .. math::

       \\frac{n_{i+1}n_e}{n_i} =
       2\\frac{g_j}{g_i} \\frac{\\left(2\\pi m_e k_B T_e\\right)^{1.5}}{h^3}
       \\exp\\left({\\frac{-\\Delta E}{k_B T_e}}\\right)

    Parameters
    ----------
    gi: float
        The statistical weight of the lower-energy state.
    gj: float
        The statistical weight of the upper-energy state, often g_{i+1}.
    T_e: Quantity
        The electron temperature.
    energy_diff
        The difference in energy levels.

    Returns
    -------
    n_j / n_i * n_e : Quantity
        The ratio of the population of the two states, multiplied with the
        free-electron density.
    """

    return (
        2
        * (gj / gi)
        * ((2 * jnp.pi * 1 * m_e * k_B * T_e) ** 1.5 / (1 * h**3))
        * jnpu.exp(((-energy_diff) / (1 * k_B * T_e)))
    )


@partial(jax.jit, static_argnames=["element_list"])
def solve_saha(
    element_list: List[Element],
    T_e: Quantity,
    ion_number_densities: Quantity,
    continuum_lowering: Quantity = 0 * ureg.electron_volt,
) -> (Quantity, Quantity):
    """
    Solve the Saha equation for a list of elements at a given temperature.

    This function uses a similar approach to solve the set of equations as
    Jamal El Kuweiss' `many-ion-saha-equation tool
    <https://github.com/jelkuweiss/many-ion-saha-equation>`_
    by

    #. Creating an abstract matrix which for each element consists of

        #. `Z + 1` rows of which have the solution of the Saha equation as
           diagonal entries, multiplied by -i and the free electron density
           `n_e` on the first off-diagonal element. These rows reflect that
           every neighboring number densities `n_{i+1}` and `n_i` fulfill
           the Saha equation

        #. A row with `Z + 1` entries of 1, ending with the diagonal entry,
           and the ion number density as last entry of the row. This guarantees
           that the individual ionization number densities add up to the full
           ion number density.

       And additionally a final row, which contains the possible charge
       states for all ions, respectively, and, finally, the free electron
       number density n_e.
       For all not-specified entries, this matrix is sparse:

       .. image:: ../images/SahaMatrix.svg

       Throughout the functions, densities are converted to non-dimensional
       quantities. The scale has impact in numerical stability.

    #. Finding the correct free electron density, which is given by the root
       of the determinant of the matrix.

    #. Building a concrete matrix by inserting the free electron density

    #. Finding the number densities for all ionization levels by solving the
       remaining set of equations by stripping the last row and column of the
       matrix, where the latter is the in-homogeneity of the set of
       equations.

    The ionization energies are taken from the provided
    :py:class:`jaxrts.element.Element`, but we allow for a modification in form
    of continuum_lowering, which is an optional parameter.

    Parameters
    ----------
    element_list
        A list of :py:class:`jaxrts.element.Element`.
    T_e
        The electron temperature of the plasma.
    ion_number_densities
        A list of number densities for the individual ions. Has to have the
        same size as `element_list`.
    continuum_lowering: Quantity, default: 0 eV
        A fixed value that is subtracted from all binding energies. Defaults to
        0 eV.

    Returns
    -------
    ionised_number_densities: Quantity
        A jax.numpy.ndarray of number densities per ionization state. It is
        ordered per-ion-species, and then with increasing ionization degree.
        I.e., `[nIon0 0+, nIon0 1+, ..., nIon1 0+, nion1 1+, ...]`.
    n_e: Quantity
        The free electron number density.

    See Also
    --------
    jaxrts.saha.saha_equation
         Function used to calculate the Saha equation for two ionization
         degrees.
    """

    Z = [i.Z for i in element_list]

    # Set up the empty matrix
    M = jnp.zeros(
        (
            len(element_list) + int(onp.sum(Z)) + 1,
            len(element_list) + int(onp.sum(Z)) + 1,
        )
    )

    # Maximal electron number density per element, if fully ionized -> Sum up
    # for maximal full electron number density
    max_ne = jnpu.sum(
        jnp.array([elem.Z for elem in element_list]) * ion_number_densities
    )

    # ne_scale = max_ne
    ne_scale = 1e0 / (1 * ureg.m**3)

    # Offset (each element will have a block of size Z+1) This value specifies
    # the block.
    skip = 0
    ionization_states = []

    # Fill the matrix, without any entries containing n_e, the free electron
    # number density (the Saha part does contain n_e, but implicitly).
    # Note: The matrix is transposed, here. It will be transposed when n_e is
    # inserted.
    for ion_dens, element in zip(ion_number_densities, element_list):

        stat_weight = element.ionization.statistical_weights
        Eb = element.ionization.energies + continuum_lowering

        # Don't allow negative binding energies.
        Eb = jnpu.where(Eb < 0, 0, Eb)

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

        # Set the diagonal for the Saha-rows
        M = M.at[skip : skip + element.Z, skip : skip + element.Z].set(diag)
        # Create ones for the density row.
        M = M.at[skip : skip + element.Z + 1, skip + element.Z].set(dens_row)
        # Set the last element of the density row to the ion number density
        M = M.at[-1, skip + element.Z].set(
            (ion_dens / ne_scale).m_as(ureg.dimensionless)
        )

        # Each element requires a block of Z+1 size.
        skip += element.Z + 1

        # This is for the final row, which lists all ionization states for the
        # elements
        ionization_states += list(jnp.arange(element.Z + 1))

    # Set the row for all ionization states.
    M = M.at[:-1, -1].set(jnp.array(ionization_states))

    def insert_ne(M, ne):
        """
        Add n_e to the off-diagonal end the last entry (last row and column.
        """

        ne_line = jnp.ones(len(element_list) + int(onp.sum(Z))) * ne

        skip = -1
        for element in element_list:

            # Not the full off-diagonal equals n_e: The density rows don't
            # contain it!
            ne_line = ne_line.at[skip + element.Z + 1].set(0.0)
            skip += element.Z + 1

        _diag = jnp.diag(ne_line, -1)
        out = M + _diag
        out = out.at[-1, -1].set(ne)
        # Transpose
        return out.T

    def det_M(M, ne):
        res = jnp.linalg.det(insert_ne(M, ne))
        return res

    # Find n_e by finding the root where the determinant of M is 0
    # Use the bisection method, boundaries are fixed by max_ne and 0.
    sol_ne, iterations = bisection(
        jax.tree_util.Partial(lambda ne: det_M(M=M, ne=ne)),
        0,
        (max_ne / ne_scale).m_as(ureg.dimensionless),
        tolerance=1e-2,
        max_iter=1e2,
        min_iter=0,
    )

    # jax.debug.print("Needed iterations for convergence: {x}", x=iterations)

    # Create the matrix that describes the linear system of equations we solve.
    # Insert n_e.
    concrete_M = insert_ne(M, sol_ne)

    # Strip the last row and column. Use the latter as inhomogeneity.
    M1 = concrete_M[: (len(concrete_M[0]) - 1), 0 : (len(concrete_M[0]) - 1)]
    M2 = concrete_M[: (len(concrete_M[0]) - 1), (len(concrete_M[0]) - 1)]

    # Get the solution of the set of linear equations of the form
    # (nIon0_0+,nIon0_1+, ..., nIon1_0+,nIon1_1+, ...)
    ionised_number_densities = jnp.linalg.solve(M1, M2)

    # Rescale the dimensionless number densities so that units are
    # 1/[length]**3.
    return ionised_number_densities * ne_scale, sol_ne * ne_scale


def calculate_mean_free_charge_saha(plasma_state, ipd=True):
    """
    Calculates the mean charge of each ion in a plasma using the Saha-Boltzmann
    equation.

    Parameters
    ----------
    plasma_state : PlasmaState
        The plasma state object.
    ipd : bool
        If true, the ipd correction of the plasma state is used to reduce the
        continuum. Note: the IPD can very much depend on the ionization state.
        this could result in some circular dependency.

    Returns
    -------
    jnp.ndarray
        An array containing the mean charge of each ion in the plasma.

    See Also
    --------
    jaxrts.saha.solve_saha
        Function used to solve the saha equation
    """

    if ipd:
        cl = jnpu.mean(plasma_state["ipd"].evaluate(plasma_state, None))
        cl = jnpu.where(jnp.isnan(cl.magnitude), 0 * ureg.electron_volt, cl)
    else:
        cl = 0 * ureg.electron_volt
    sol, _ = solve_saha(
        tuple(plasma_state.ions),
        plasma_state.T_e,
        (plasma_state.mass_density / plasma_state.atomic_masses),
        continuum_lowering=cl,
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
