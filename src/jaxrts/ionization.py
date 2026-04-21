"""
Module to model the ionization state of a plasma.
It contains functions to solve the
`Saha-equation <https://en.wikipedia.org/wiki/Saha_ionization_equation>`_,
linking the temperature of a plasma to its ionization.
"""

from functools import partial

import jax
from jax.tree_util import Partial
import jax.numpy as jnp
import jpu.numpy as jnpu
import numpy as onp

import logging

from .elements import Element
from .plasma_physics import therm_de_broglie_wl
from .units import Quantity, to_array, ureg
from .helpers import bisection

h = 1 * ureg.planck_constant
k_B = 1 * ureg.boltzmann_constant
m_e = 1 * ureg.electron_mass

logger = logging.getLogger(__name__)


# Balance relations


@jax.jit
def gen_saha_equation(
    gi: float,
    gj: float,
    T_e: Quantity,
    n_e: Quantity,
    energy_diff: Quantity,
    chem_pot: Quantity,
) -> Quantity:
    """
    Generic Saha equation, using
    :py:func:`jaxrts.plasma_physics.chem_pot_sommerfeld_fermi_interpolation`
    calculate the chemical potential which is plugged into the equation

    Returns the value n_j / n_i * n_e

    .. math::

       \\frac{n_{i+1}n_e}{n_i} = 2\\frac{g_j}{g_i}
       \\exp\\left({\\frac{-\\Delta E - \\mu}{k_B T_e}}\\right)

    .. note::

       Since the free electron number :math:`n_e` is an input to this function,
       solving the gen_saha_equation has to be done iteratively, until the
       result is self-consistent.

    Parameters
    ----------
    gi: float
        The statistical weight of the lower-energy state.
    gj: float
        The statistical weight of the upper-energy state, often g_{i+1}.
    T_e: Quantity
        The electron temperature.
    n_e: Quantity
        The free electron density.
    energy_diff
        The difference in energy levels.

    Returns
    -------
    n_j / n_i * n_e : Quantity
        The ratio of the population of the two states, multiplied with the
        free-electron density.
    """

    return (
        (gj / gi)
        * n_e
        * (jnpu.exp((-energy_diff) / (1 * k_B * T_e)))
        * jnpu.exp(-chem_pot / (1 * k_B * T_e))
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
        (gj / gi)
        * 2
        / therm_de_broglie_wl(T_e) ** 3
        * (jnpu.exp((-energy_diff) / (1 * k_B * T_e)))
    )


# Solver


@partial(
    jax.jit, static_argnames=["element_list", "exclude_non_negative_energies"]
)
def solve_ionization(
    element_list: list[Element],
    T_e: Quantity,
    ion_number_densities: Quantity,
    balance_fn: Partial,
    continuum_lowering: Quantity | None = None,
    exclude_non_negative_energies: bool = True,
) -> tuple[Quantity, Quantity, Quantity]:
    """
    Solve for the ionization state of a plasma given an arbitrary balance
    relation.

    The matrix structure and bisection root-find are the same regardless of
    which balance relation is used.  The only changes between the classic Saha
    case and, e.g., the generalised (degenerate) Saha case is how the ratio
    :math:`n_{i+1} n_e / n_i` is computed from the level energies and
    statistical weights.  This function therefore accepts a ``balance_fn`` that
    encapsulates that computation.

    ``balance_fn`` must have the signature::

        balance_fn(gi, gj, Eb_quantity) -> Quantity   [units: 1/volume]

    where ``Eb_quantity`` is the ionization energy of the transition as a
    :py:class:`~jaxrts.units.Quantity`.  Any additional physical parameters
    (temperature, chemical potential, ...) should be bound in advance via
    :py:func:`jax.tree_util.Partial`

    Built-in balance relations
    --------------------------
    * :py:func:`saha_balance` — classic non-degenerate Saha equation.
    * :py:func:`gen_saha_balance` — generalised (degenerate) Saha equation.

    Algorithm
    ---------
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
    element_list : list[Element]
        Ion species to include.
    T_e : Quantity
        Electron temperature.
    ion_number_densities : Quantity
        Total number density of each species (same order as ``element_list``).
    balance_fn : jax.tree_util.Partial
        ``(gi, gj, Eb_quantity) -> Quantity`` — the ionization balance
        relation.  Bind extra parameters (T_e, n_e, μ, ...) before passing.
    continuum_lowering : Quantity or None, default None
        Per-level energy shift (IPD) subtracted from binding energies.
        Defaults to zero for all levels.
    exclude_non_negative_energies : bool, default True
        When ``True``, levels whose shifted binding energy ≤ 0 are excluded
        from the balance equations (pressure ionization).

    Returns
    -------
    ionised_number_densities : Quantity
        Per-level number densities ordered as
        ``[n(Ion0, 0+), n(Ion0, 1+), ..., n(Ion1, 0+), ...]``.
    n_e : Quantity
        Self-consistent free-electron number density.
    Z_mean : Quantity
        Mean ionization charge of each species.
    """

    Z = [ion.Z for ion in element_list]

    if continuum_lowering is None:
        continuum_lowering = [
            jnp.zeros(ion.Z) * ureg.electron_volt for ion in element_list
        ]

    # Binding energies (shifted by continuum lowering / IPD)
    Ebs = [
        (jnpu.sort(element.ionization.energies) + ipd).m_as(ureg.electron_volt)
        for element, ipd in zip(element_list, continuum_lowering, strict=True)
    ]

    max_ne = jnpu.sum(jnp.array(Z) * ion_number_densities)

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

    # Build the abstract matrix (without n_e on the off-diagonal)
    size = len(element_list) + int(onp.sum(Z)) + 1
    M = jnp.zeros((size, size))

    skip = 0
    ionization_states = []

    for ion_dens, element, Eb in zip(
        ion_number_densities, element_list, Ebs, strict=True
    ):
        stat_weight = element.ionization.statistical_weights

        # Call the pluggable balance relation
        coeff = (
            balance_fn(
                stat_weight[:-1],
                stat_weight[1:],
                Eb * ureg.electron_volt,
            )
            / ne_scale
        ).m_as(ureg.dimensionless)

        if exclude_non_negative_energies:
            diag = jnp.diag(jnp.where(Eb > 0, (-1) * coeff, 1))
        else:
            diag = jnp.diag(-coeff)

        dens_row = jnp.ones(element.Z + 1)

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
        Add n_e to the off-diagonal end the last entry (last row and column).
        """

        ne_line = jnp.ones(len(element_list) + int(onp.sum(Z))) * ne

        skip = -1
        for element, Eb in zip(element_list, Ebs, strict=True):
            # Not the full off-diagonal equals n_e: The density rows don't
            # contain it!
            ne_line = ne_line.at[skip + 1 : skip + element.Z + 1].multiply(
                jnp.heaviside(Eb, 0) if exclude_non_negative_energies else 1
            )
            ne_line = ne_line.at[skip + element.Z + 1].set(0.0)
            skip += element.Z + 1

        _diag = jnp.diag(ne_line, -1)
        out = M + _diag
        out = out.at[-1, -1].set(ne)
        # Transpose
        return out.T

    # Bisection: find n_e such that det(M) = 0
    def det_M_func(ne):
        return jnp.linalg.det(insert_ne(M, ne))

    # Find n_e by finding the root where the determinant of M is 0
    # Use the bisection method, boundaries are fixed by max_ne and 0.
    sol_ne, iterations = bisection(
        Partial(det_M_func),
        0,
        (max_ne / ne_scale).m_as(ureg.dimensionless),
        tolerance=1e-5,
        max_iter=1e4,
        min_iter=40,
    )

    # Create the matrix that describes the linear system of equations we solve.
    # Insert n_e.
    concrete_M = insert_ne(M, sol_ne)

    # Strip the last row and column. Use the latter as inhomogeneity.
    M1 = concrete_M[:-1, :-1]
    M2 = concrete_M[:-1, -1]

    # Get the solution of the set of linear equations of the form
    # (nIon0_0+,nIon0_1+, ..., nIon1_0+,nIon1_1+, ...)
    ionised_number_densities = jnp.linalg.solve(M1, M2)

    # Rescale the dimensionless number densities so that units are
    # 1/[length]**3.
    res = jnpu.multiply(ionised_number_densities, ne_scale)

    # Mean ionization per species
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


# Wrapper for certain balance equations


@partial(
    jax.jit,
    static_argnames=["element_list", "exclude_non_negative_energies"],
)
def solve_saha(
    element_list: list[Element],
    T_e: Quantity,
    ion_number_densities: Quantity,
    continuum_lowering: Quantity | None = None,
    exclude_non_negative_energies: bool = True,
) -> tuple[Quantity, Quantity, Quantity]:
    """
    Solve the classic (non-degenerate) Saha equation.

    Thin wrapper around :py:func:`~.solve_ionization` using
    :py:func:`~.saha_equation` as the balance relation.

    Parameters
    ----------
    element_list, T_e, ion_number_densities, continuum_lowering,
    exclude_non_negative_energies
        See :py:func:`solve_ionization`.

    Returns
    -------
    ionised_number_densities, n_e, Z_mean
        See :py:func:`solve_ionization`.
    """
    balance_fn = Partial(lambda gi, gj, E: saha_equation(gi, gj, T_e, E))
    return solve_ionization(
        element_list,
        T_e,
        ion_number_densities,
        balance_fn=balance_fn,
        continuum_lowering=continuum_lowering,
        exclude_non_negative_energies=exclude_non_negative_energies,
    )


@partial(
    jax.jit,
    static_argnames=["element_list", "exclude_non_negative_energies"],
)
def solve_gen_saha(
    element_list: list[Element],
    T_e: Quantity,
    n_e: Quantity,
    ion_number_densities: Quantity,
    continuum_lowering: Quantity = 0 * ureg.electron_volt,
    chem_pot_ideal: Quantity = 0 * ureg.electron_volt,
    exclude_non_negative_energies: bool = True,
) -> tuple[Quantity, Quantity, Quantity]:
    """
    Solve the generalised (degenerate) Saha equation.

    Thin wrapper around :py:func:`~.solve_ionization` using
    :py:func:`~.gen_saha_equation` as the balance relation.

    Parameters
    ----------
    element_list, T_e, n_e, ion_number_densities, continuum_lowering,
    chem_pot_ideal, exclude_non_negative_energies
        See :py:func:`solve_ionization` and :py:func:`gen_saha_balance`.

    Returns
    -------
    ionised_number_densities, n_e, Z_mean
        See :py:func:`solve_ionization`.
    """
    balance_fn = Partial(
        lambda gi, gj, E: gen_saha_equation(
            gi, gj, T_e, n_e, E, chem_pot_ideal
        )
    )
    return solve_ionization(
        element_list,
        T_e,
        ion_number_densities,
        balance_fn=balance_fn,
        continuum_lowering=continuum_lowering,
        exclude_non_negative_energies=exclude_non_negative_energies,
    )


# Higher-level helpers


def calculate_charge_state_distribution(plasma_state):
    """
    Calculates the charge state distribution in fractions using the
    Saha-Boltzmann equation assuming thermal equilibrium.

    Parameters
    ----------
    plasma_state : PlasmaState
        The plasma state object.

    Returns
    -------
    jnp.ndarray
        The charge state distribution in fractions.
    """

    cl = plasma_state["ipd"].all_element_states(plasma_state)

    sol, ne, Z_mean = solve_saha(
        tuple(plasma_state.ions),
        plasma_state.T_e,
        (plasma_state.mass_density / plasma_state.atomic_masses),
        continuum_lowering=cl,
    )

    return (sol / jnpu.sum(sol)).m_as(ureg.dimensionless)


def calculate_mean_free_charge_saha(
    plasma_state,
    use_ipd: bool = False,
    use_chem_pot: bool = False,
    use_distribution: bool = False,
    exclude_non_negative_energies: bool = True,
):
    """
    Calculates the mean charge of each ion in a plasma using the Saha-Boltzmann
    equation.

    Parameters
    ----------
    plasma_state : PlasmaState
        The plasma state object.
    use_ipd : bool
        If true, the ipd correction of the plasma state is used to reduce the
        continuum. Note: the IPD can very much depend on the ionization state.
        this could result in some circular dependency.
    use_chem_pot : bool
        If true, the chemical potential of the plasma state is used, instead of
        the non-degenerate limiting case.
    use_distribution:
        If true, the ipd is evaluated for each ion species separately.
    exclude_non_negative_energies : bool, default = True
        If true, bound states for which the ionization energy is pushed into
        the continuum are removed from the calculation and do not appear with
        their Boltzmann factors in the Saha equations.

    Returns
    -------
    jnp.ndarray
        The charge state distribution in fractions.
    jnp.ndarray
        An array containing the mean charge of each ion in the plasma.

    See Also
    --------
    jaxrts.saha.solve_saha
        Function used to solve the saha equation
    """

    plasma_state.Z_free = jnp.array(
        jnp.max(jnp.array([i.Z for i in plasma_state.ions]))
    )

    if not use_chem_pot:
        charge_distribution, ne, Z_mean = solve_saha(
            tuple(plasma_state.ions),
            plasma_state.T_e,
            (plasma_state.mass_density / plasma_state.atomic_masses),
        )

        for k in range(6):
            if use_ipd:
                cl = plasma_state["ipd"].all_element_states(
                    plasma_state,
                    charge_distribution if use_distribution else None,
                )

            else:
                cl = [
                    jnp.zeros(ion.Z) * ureg.electron_volt
                    for ion in plasma_state.ions
                ]

            charge_distribution, ne, Z_mean = solve_saha(
                tuple(plasma_state.ions),
                plasma_state.T_e,
                (plasma_state.mass_density / plasma_state.atomic_masses),
                continuum_lowering=cl,
                exclude_non_negative_energies=exclude_non_negative_energies,
            )
            plasma_state.Z_free = jnp.array(Z_mean)

    else:
        charge_distribution, ne, Z_mean = solve_saha(
            tuple(plasma_state.ions),
            plasma_state.T_e,
            (plasma_state.mass_density / plasma_state.atomic_masses),
        )

        for k in range(6):
            chem_pot_ideal = plasma_state["chemical potential"].evaluate(
                plasma_state, None
            )
            if use_ipd:
                cl = plasma_state["ipd"].all_element_states(
                    plasma_state,
                    charge_distribution if use_distribution else None,
                )
            else:
                cl = [
                    jnp.zeros(ion.Z) * ureg.electron_volt
                    for ion in plasma_state.ions
                ]

            charge_distribution, ne, Z_mean = solve_gen_saha(
                tuple(plasma_state.ions),
                plasma_state.T_e,
                plasma_state.n_e,
                (plasma_state.mass_density / plasma_state.atomic_masses),
                continuum_lowering=cl,
                chem_pot_ideal=chem_pot_ideal,
                exclude_non_negative_energies=exclude_non_negative_energies,
            )

            plasma_state.Z_free = jnp.array(Z_mean)

    return charge_distribution, Z_mean


# Thomas-Fermi / More fits

@jax.jit
def _calculate_single_species_ionization_more(rho, T_e, m_A, Z_A) -> Quantity:
    alpha = 14.3139
    beta = 0.6624
    a1 = 0.003323
    a2 = 0.9718
    a3 = 9.26148e-5
    a4 = 3.10165
    b0 = -1.7630
    b1 = 1.43175
    b2 = 0.31546
    c1 = -0.366667
    c2 = 0.983333

    R = (rho.m_as(ureg.gram / ureg.cc) / m_A.m_as(ureg.u)) / Z_A
    T0 = T_e.m_as(ureg.eV / ureg.k_B) / Z_A ** (4.0 / 3.0)
    Tf = T0 / (1 + T0)
    A = a1 * T0**a2 + a3 * T0**a4
    B = -jnp.exp(b0 + b1 * Tf + b2 * Tf**7)
    C = c1 * Tf + c2
    Q1 = A * R**B
    Q = (R**C + Q1**C) ** (1 / C)
    x = alpha * Q**beta

    return Z_A * x / (1 + x + jnp.sqrt(1 + 2.0 * x))


@jax.jit
def _calculate_mean_free_charge_more_single_species(plasma_state) -> Quantity:
    """

    Deprecated: Use `calculate_mean_free_charge_more` instead.

    Finite Temperature Thomas Fermi Charge State using an analytical fit
    provided by :cite:`More.1985` p. 332 (Table IV).

    .. warning::

        This fit is currently only applicable to a OCP.

    Parameters
    ----------
    plasma_state : PlasmaState
        The plasma state object.

    Returns
    -------
    Z_f : Quantity
        The mean free charge of the ions in the OCP

    """

    logger.warning(
        "'_calculate_mean_free_charge_more_single_species' is deprecated, please use 'calculate_mean_free_charge_more' instead."  # noqa: E501
    )

    rho = plasma_state.mass_density
    m_A = plasma_state.atomic_masses

    if len(m_A) > 1:
        logger.warning(
            "Multiple ion species detected. '_calculate_mean_free_charge_more_single_species' supports only one species and evaluates ionization per ion independently."  # noqa: E501
        )

    Z_A = plasma_state.Z_A
    T_e = plasma_state.T_e

    return _calculate_single_species_ionization_more(rho, T_e, m_A, Z_A)


@jax.jit
def calculate_mean_free_charge_more(plasma_state) -> jnp.ndarray:
    """

    Uses the finite Temperature Thomas Fermi Charge State fits provided by
    :cite:`More.1985` p. 332 (Table IV) using an Average-Atom model to
    calculate the mean ionization state of each species at an effective
    density.

    Based on the Multispecies TF-AA Ionization solver of M. Murillo et al.:
    https://github.com/MurilloGroupMSU/Thomas-Fermi-Multispecies-Ionization/
    which is published in :cite:`Stanton.2017`.


    Parameters
    ----------
    plasma_state : PlasmaState
        The plasma state object.

    Returns
    -------
    Z_f : Quantity
        The mean free charge of each ion in the plasma

    """

    # Convert to number densities
    rho_i = plasma_state.mass_density
    n_i = rho_i / plasma_state.atomic_masses

    # Initial guess
    Zbar = 0.5 * plasma_state.Z_A

    max_iter = 100
    tol = 1e-5

    def body_fun(state):
        Zbar_old = state

        # Electron density from all species
        n_e = jnpu.sum(n_i * Zbar_old)

        # Effective density per species calculated from current n_e
        rho_eff = (n_e / Zbar_old) * plasma_state.atomic_masses

        Zbar_new = _calculate_single_species_ionization_more(
            rho_eff,
            plasma_state.T_e,
            plasma_state.atomic_masses,
            plasma_state.Z_A,
        )

        return Zbar_new

    def cond_fun(state):
        Zbar_old, Zbar_new, i = state
        err = jnp.max(jnp.abs(Zbar_new - Zbar_old))

        return jnp.logical_and(err > tol, i < max_iter)

    def loop_body(state):
        Zbar_old, Zbar_new, i = state
        Zbar_next = body_fun(Zbar_new)
        return (Zbar_new, Zbar_next, i + 1)

    Zbar_new = body_fun(Zbar)
    state = (Zbar, Zbar_new, 0)

    Zbar_old, Z_mean, _ = jax.lax.while_loop(cond_fun, loop_body, state)

    return Z_mean
