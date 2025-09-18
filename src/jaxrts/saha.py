"""
Module containing functions to solve the
`Saha-equation <https://en.wikipedia.org/wiki/Saha_ionization_equation>`_,
linking the temperature of a plasma to it's ionization.
"""

from functools import partial

import jax
import jax.numpy as jnp
import jpu.numpy as jnpu

import matplotlib.pyplot as plt

import numpy as onp

from .elements import Element
from .units import Quantity, ureg, to_array
from .plasma_physics import (
    chem_pot_interpolationIchimaru,
    chem_pot_sommerfeld_fermi_interpolation,
)

h = 1 * ureg.planck_constant
k_B = 1 * ureg.boltzmann_constant
m_e = 1 * ureg.electron_mass

# fig, ax = plt.subplots()
debug_detM = False


@jax.jit
def bisection(func, a, b, tolerance=1e-4, max_iter=1e4, min_iter=1e2):

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
def gen_saha_equation(
    gi: float,
    gj: float,
    T_e: Quantity,
    n_e: Quantity,
    energy_diff: Quantity,
) -> Quantity:

    chem_pot = chem_pot_sommerfeld_fermi_interpolation(T_e, n_e).to(
        ureg.electron_volt
    )
    return (
        (gj / gi)
        * n_e
        * jnpu.exp(((-energy_diff - chem_pot) / (1 * k_B * T_e)))
    )


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
        * jnpu.exp((-energy_diff) / (1 * k_B * T_e))
    )


@partial(jax.jit, static_argnames=["element_list"])
def solve_saha(
    element_list: list[Element],
    T_e: Quantity,
    ion_number_densities: Quantity,
    continuum_lowering: Quantity | None = None,
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

    Z = [ion.Z for ion in element_list]
    if continuum_lowering is None:
        continuum_lowering = [
            jnp.zeros(ion.Z) * ureg.electron_volt for ion in element_list
        ]

    # Set up the empty matrix
    M = jnp.zeros(
        (
            len(element_list) + int(onp.sum(Z)) + 1,
            len(element_list) + int(onp.sum(Z)) + 1,
        )
    )

    # Maximal electron number density per element, if fully ionized -> Sum up
    # for maximal full electron number density
    max_ne = jnpu.sum(jnp.array(Z) * ion_number_densities)

    # ne_scale = max_ne
    # ne_scale = 1e0 / (1 * ureg.cm**3)

    # Interpolate between a small ne and the maximal ne, depending on the
    # temperature to avoid numerical instabilities
    # jax.debug.print("{x}",x = max_ne.m_as(1 / ureg.m**3) // 1)

    # Calculate how many states have a energy below 0, as it can happen due to
    # ipd.abs

    Ebs = []
    for element, ipd in zip(element_list, continuum_lowering, strict=True):

        stat_weight = element.ionization.statistical_weights

        Ebs.append(
            (jnpu.sort(element.ionization.energies) + ipd).m_as(ureg.electron_volt)
        )

    all_binding_energies = jnp.concatenate(Ebs)
    ratio = jnp.sum(
        jnp.heaviside(all_binding_energies, 0) / len(all_binding_energies)
    )

    _ne_range = jnp.array(
        [
            max_ne.m_as(1 / ureg.m**3) * (1 - ratio) + 1,
            max_ne.m_as(1 / ureg.m**3),
        ]
    )
    ne_scale = jnp.interp(
        (T_e * k_B).m_as(ureg.electron_volt),
        jnp.array([1, 1000]),
        _ne_range,
        left=_ne_range[0],
        right=_ne_range[-1],
    ) * (1 / ureg.m**3)

    # jax.debug.print("{x}", x= max_ne)
    # ne_scale = max_ne

    # Offset (each element will have a block of size Z+1) This value specifies
    # the block.
    skip = 0
    ionization_states = []

    # Fill the matrix, without any entries containing n_e, the free electron
    # number density (the Saha part does contain n_e, but implicitly).
    # Note: The matrix is transposed, here. It will be transposed when n_e is
    # inserted.
    for ion_dens, element, Eb in zip(
        ion_number_densities, element_list, Ebs, strict=True
    ):

        stat_weight = element.ionization.statistical_weights

        # jax.debug.print("Before: {x}", x = jnp.sum(jnp.heaviside(Eb.m_as(ureg.electron_volt), 0)))

        def scan_fn(pref, inputs):
            E, g = inputs
            output = jnp.where(E > 0, pref * g, 1.0)
            new_pref = jnp.where(E > 0, 1.0, pref * g)
            return new_pref, output

        initial_pref = 1.0
        final_pref, out_values = jax.lax.scan(
            scan_fn,
            initial_pref,
            (jnp.array([*Eb, 1.0]), stat_weight),
        )

        # stat_weight = out_values

        # jax.debug.print("After: {x}", x = stat_weight)

        # jax.debug.print("{x}, {z}", x=element.ionization.energies + ipd, z = continuum_lowering)

        # Eb = jnpu.where(Eb < 0, 0, Eb)
        # jax.debug.print("{x}{y}", x =stat_weight[:-1], y = stat_weight[1:])

        coeff = (
            saha_equation(
                stat_weight[:-1],
                stat_weight[1:],
                T_e,
                Eb * ureg.electron_volt,
            )
            / ne_scale
        ).m_as(ureg.dimensionless)

        diag = jnp.diag(
            jnp.where(Eb > 0, (-1) * coeff, 1)
        )
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
        for element, Eb in zip(element_list, Ebs, strict=True):

            # jax.debug.print("{x}", x = jnp.heaviside(Eb, 0))

            # Not the full off-diagonal equals n_e: The density rows don't
            # contain it!
            ne_line = ne_line.at[skip + 1 : skip + element.Z + 1].multiply(
                jnp.heaviside(Eb, 0)
            )
            ne_line = ne_line.at[skip + element.Z + 1].set(0.0)
            skip += element.Z + 1

        _diag = jnp.diag(ne_line, -1)
        out = M + _diag
        out = out.at[-1, -1].set(ne)
        # Transpose
        return out.T

    def det_M(M, ne):
        res = jnp.linalg.det(insert_ne(M, ne))
        # jax.debug.print("{x}", x=res)
        return res

    det_M_func = (
        lambda ne: det_M(M=M, ne=ne)
        / (ne + 1) ** (M.shape[0] + 1)
        * T_e.m_as(ureg.electron_volt / ureg.boltzmann_constant)
        ** (-M.shape[0])
        * 1e16
    )

    # if debug_detM:
    #     ne_plot = onp.linspace(0, 1, 100)
    #     ax.hlines(0.0, ne_plot[0], ne_plot[-1], color="black", ls="dashed")
    #     ax.plot(ne_plot, [det_M_func(ne) for ne in ne_plot])
    #     ax.vlines(sol_ne, 0, 0.0001, color="black", ls="dashed")
    #     plt.show(block=False)
    #     plt.pause(2.0)
    #     jax.debug.print("sol_ne: {x}, {y}", x=sol_ne, y=ne_scale)

    # ax.set_yscale("log")

    # Find n_e by finding the root where the determinant of M is 0
    # Use the bisection method, boundaries are fixed by max_ne and 0.
    sol_ne, iterations = bisection(
        # jax.tree_util.Partial(lambda ne: det_M(M=M, ne=ne)),
        jax.tree_util.Partial(det_M_func),
        0,
        (max_ne / ne_scale).m_as(ureg.dimensionless),
        tolerance=1e-16,
        max_iter=1e4,
        min_iter=40,
    )

    # sol_ne = Broyden(fun=det_M_func).run((max_ne / ne_scale).m_as(ureg.dimensionless)).params
    # sol_ne = Bisection(optimality_fun = det_M_func, lower = 1E-7, upper = (max_ne / ne_scale).m_as(ureg.dimensionless)).run().params

    # Create the matrix that describes the linear system of equations we solve.
    # Insert n_e.
    concrete_M = insert_ne(M, sol_ne)

    # jax.debug.print("{x}", x = concrete_M)

    # Strip the last row and column. Use the latter as inhomogeneity.
    M1 = concrete_M[:-1, :-1]
    M2 = concrete_M[:-1, -1]

    # Get the solution of the set of linear equations of the form
    # (nIon0_0+,nIon0_1+, ..., nIon1_0+,nIon1_1+, ...)
    ionised_number_densities = jnp.linalg.solve(M1, M2)

    # Rescale the dimensionless number densities so that units are
    # 1/[length]**3.
    res = jnpu.multiply(ionised_number_densities, ne_scale)
    # jax.debug.print("{x}", x = ionised_number_densities)

    skip = 0
    Z_mean = []
    for element in element_list:
        Z_mean.append(
            jnpu.sum(
                res[skip : skip + element.Z + 1]
                * jnp.arange(element.Z + 1)
                / jnpu.sum(res[skip : skip + element.Z + 1])
            )
        )
        skip += element.Z + 1
    # Z_mean = to_array(
    #     (sol_ne * ne_scale) / jnpu.sum(ion_number_densities)
    # ).m_as(ureg.dimensionless)
    # print(len(res))
    # print(jnp.abs(jnpu.sum((jnp.arange(len(res)) * res) / jnpu.sum(res)).m_as(ureg.dimensionless) - Z_mean) < 1E-3)

    return (
        to_array(res),
        sol_ne * ne_scale,
        to_array(Z_mean).m_as(ureg.dimensionless),
    )


@partial(jax.jit, static_argnames=["element_list"])
def solve_gen_saha(
    element_list: list[Element],
    T_e: Quantity,
    n_e: Quantity,
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
    if continuum_lowering is None:
        continuum_lowering = [
            jnp.zeros(ion.Z) * ureg.electron_volt for ion in element_list
        ]

    # Set up the empty matrix
    M = jnp.zeros(
        (
            len(element_list) + int(onp.sum(Z)) + 1,
            len(element_list) + int(onp.sum(Z)) + 1,
        )
    )

    # Maximal electron number density per element, if fully ionized -> Sum up
    # for maximal full electron number density
    max_ne = jnpu.sum(jnp.array(Z) * ion_number_densities)
    # ne_scale = max_ne
    # ne_scale = 1e0 / (1 * ureg.m**3)

    # Interpolate between a small ne and the maximal ne, depending on the
    # temperature to avoid numerical instabilities
    Ebs = []
    for element, ipd in zip(element_list, continuum_lowering, strict=True):

        stat_weight = element.ionization.statistical_weights

        Ebs.append(
            (jnpu.sort(element.ionization.energies) + ipd).m_as(ureg.electron_volt)
        )

    all_binding_energies = jnp.concatenate(Ebs)
    ratio = jnp.sum(
        jnp.heaviside(all_binding_energies, 0) / len(all_binding_energies)
    )

    _ne_range = jnp.array(
        [
            max_ne.m_as(1 / ureg.m**3) * (1 - ratio) + 1,
            max_ne.m_as(1 / ureg.m**3),
        ]
    )
    ne_scale = jnp.interp(
        (T_e * k_B).m_as(ureg.electron_volt),
        jnp.array([1, 1000]),
        _ne_range,
        left=_ne_range[0],
        right=_ne_range[-1],
    ) * (1 / ureg.m**3)

    # jax.debug.print("{x}", x= max_ne)
    # ne_scale = max_ne

    # Offset (each element will have a block of size Z+1) This value specifies
    # the block.
    skip = 0
    ionization_states = []

    # Fill the matrix, without any entries containing n_e, the free electron
    # number density (the Saha part does contain n_e, but implicitly).
    # Note: The matrix is transposed, here. It will be transposed when n_e is
    # inserted.
    for ion_dens, element, Eb in zip(
        ion_number_densities, element_list, Ebs, strict=True
    ):

        stat_weight = element.ionization.statistical_weights

        # jax.debug.print("Before: {x}", x = jnp.sum(jnp.heaviside(Eb.m_as(ureg.electron_volt), 0)))

        def scan_fn(pref, inputs):
            E, g = inputs
            output = jnp.where(E > 0, pref * g, 1.0)
            new_pref = jnp.where(E > 0, 1.0, pref * g)
            return new_pref, output

        initial_pref = 1.0
        final_pref, out_values = jax.lax.scan(
            scan_fn,
            initial_pref,
            (jnp.array([*Eb, 1.0]), stat_weight),
        )

        # stat_weight = out_values

        # jax.debug.print("After: {x}", x = stat_weight)

        # jax.debug.print("{x}, {z}", x=element.ionization.energies + ipd, z = continuum_lowering)

        # Eb = jnpu.where(Eb < 0, 0, Eb)
        # jax.debug.print("{x}{y}", x =stat_weight[:-1], y = stat_weight[1:])

        coeff = (
            gen_saha_equation(
                stat_weight[:-1],
                stat_weight[1:],
                T_e,
                n_e,
                Eb * ureg.electron_volt,
            )
            / ne_scale
        ).m_as(ureg.dimensionless)

        diag = jnp.diag(
            jnp.where(Eb > 0, (-1) * coeff, 1)
        )
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

    det_M_func = (
        lambda ne: det_M(M=M, ne=ne)
        / (ne + 1) ** (M.shape[0] + 1)
        * T_e.m_as(ureg.electron_volt / ureg.boltzmann_constant)
        ** (-M.shape[0])
        * 1e16
    )

    sol_ne, iterations = bisection(
        jax.tree_util.Partial(det_M_func),
        0,
        (max_ne / ne_scale).m_as(ureg.dimensionless),
        tolerance=1e-16,
        max_iter=1e4,
        min_iter=40,
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

    # jax.debug.print("{x}", x=M1)

    ionised_number_densities = jnp.linalg.solve(M1, M2)

    # Rescale the dimensionless number densities so that units are
    # 1/[length]**3.

    res = jnpu.multiply(ionised_number_densities, ne_scale)

    skip = 0
    Z_mean = []
    for element in element_list:
        Z_mean.append(
            jnpu.sum(
                res[skip : skip + element.Z + 1]
                * jnp.arange(element.Z + 1)
                / jnpu.sum(res[skip : skip + element.Z + 1])
            )
        )
        skip += element.Z + 1

    return (
        to_array(res),
        sol_ne * ne_scale,
        to_array(Z_mean).m_as(ureg.dimensionless),
    )


def calculate_charge_state_distribution(plasma_state):

    cl = plasma_state["ipd"].all_element_states(plasma_state)

    sol, ne, Z_mean = solve_saha(
        tuple(plasma_state.ions),
        plasma_state.T_e,
        (plasma_state.mass_density / plasma_state.atomic_masses),
        continuum_lowering=cl,
    )

    return (sol / jnpu.sum(sol)).m_as(ureg.dimensionless)


def calculate_mean_free_charge_saha(plasma_state, ipd: bool = False, degenerate = False):
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

    plasma_state.Z_free = jnp.array([1.0])
    if ipd:
        cl = plasma_state["ipd"].all_element_states(plasma_state)
    else:
        cl = [
            jnp.zeros(ion.Z) * ureg.electron_volt for ion in plasma_state.ions
        ]

    if not degenerate:
        charge_distribution, ne, Z_mean = solve_saha(
        tuple(plasma_state.ions),
        plasma_state.T_e,
        (plasma_state.mass_density / plasma_state.atomic_masses),
        continuum_lowering=cl,
    )
    else:

        for k in range(10):
            charge_distribution, ne, Z_mean = solve_gen_saha(
            tuple(plasma_state.ions),
            plasma_state.T_e,
            plasma_state.n_e,
            (plasma_state.mass_density / plasma_state.atomic_masses),
            continuum_lowering=cl,
        )
            plasma_state.Z_free = jnp.array([Z_mean])
            # print(Z_mean)
        # print(plasma_state.n_e)
    

    return Z_mean
