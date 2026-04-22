"""
Module to model the ionization state of a plasma.
It contains functions to solve the
`Saha-equation <https://en.wikipedia.org/wiki/Saha_ionization_equation>`_,
linking the temperature of a plasma to its ionization.

The central solver is :py:func:`solve_ionization`, which accepts a
*balance term* -- an object that supplies per-transition coefficients and the
corresponding "still-bound" mask.  Two concrete balance terms are provided:

* :py:class:`SahaBalanceTerm` -- classic / generalised Saha (the generalised
  form reduces to the classic one when ``n_e = 0`` and ``chem_pot = 0``).
* :py:class:`BUBalanceTerm` -- Bethe-Uhlenbeck, using Planck-Larkin
  partition sums instead of bare statistical weights.
"""

from typing import NamedTuple
import logging
from functools import partial

import jax
from jax.tree_util import Partial
import jax.numpy as jnp
import jpu.numpy as jnpu
import numpy as onp

from .elements import Element
from .plasma_physics import therm_de_broglie_wl
from .units import Quantity, to_array, ureg
from .helpers import bisection, read_nist_file

h = 1 * ureg.planck_constant
k_B = 1 * ureg.boltzmann_constant
m_e = 1 * ureg.electron_mass

logger = logging.getLogger(__name__)


# Partition sums.


@jax.jit
def planck_larkin_partition_sum(
    g: jnp.ndarray,
    E: Quantity,
    T: Quantity,
) -> jnp.ndarray:
    """
    Planck-Larkin partition sum with a cutoff for negative energies.

    .. math::

        u_i = \\sum_s g_s
              \\left(e^{-E_s / k_B T} - 1 + \\frac{E_s}{k_B T}\\right)
              \\Theta(E_s)

    See :cite:`Kremp.2005` eqn 6.182.

    Parameters
    ----------
    g : jnp.ndarray
        Degeneracy of each level.
    E : Quantity
        Energy of each level relative to the ionization threshold.
    T : Quantity
        Temperature.

    Returns
    -------
    jnp.ndarray
        Scalar partition-sum value.
    """
    ratio = (E / (T * ureg.boltzmann_constant)).m_as(ureg.dimensionless)
    return jnp.sum(
        g * (jnp.exp(-ratio) - 1 + ratio) * jnp.heaviside(E.m_as(ureg.eV), 0)
    )


# Low-level balance equations


@jax.jit
def saha_equation(
    gi: float,
    gj: float,
    T_e: Quantity,
    energy_diff: Quantity,
) -> Quantity:
    """
    Classic (non-degenerate) Saha balance relation.

    Returns :math:`n_{i+1} n_e / n_i`:

    .. math::

       \\frac{n_{i+1}n_e}{n_i} =
       2\\frac{g_j}{g_i}
       \\frac{(2\\pi m_e k_B T_e)^{3/2}}{h^3}
       \\exp\\!\\left(\\frac{-\\Delta E}{k_B T_e}\\right)

    Parameters
    ----------
    gi, gj : float
        Statistical weights of the lower and upper states.
    T_e : Quantity
        Electron temperature.
    energy_diff : Quantity
        Ionization energy :math:`\\Delta E`.
    """
    return (
        (gj / gi)
        * 2
        / therm_de_broglie_wl(T_e) ** 3
        * jnpu.exp((-energy_diff) / (k_B * T_e))
    )


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
def gen_balance_equation(
    part_i: float | jnp.ndarray,
    part_j: float | jnp.ndarray,
    T_e: Quantity,
    n_e: Quantity,
    chem_pot: Quantity,
) -> Quantity:
    """
    Generic balance relation expressed in terms of partition functions.

    The energy dependence is already folded into ``part_i`` and ``part_j``
    (e.g. Planck-Larkin sums), so no explicit Boltzmann factor for
    :math:`\\Delta E` appears here:

    .. math::

       \\frac{n_{i+1}n_e}{n_i}
       = \\frac{u_{i+1}}{u_i} n_e \\exp(-\\mu / k_B T_e})

    Parameters
    ----------
    part_i, part_j : array-like
        Partition functions of adjacent charge states.
    T_e : Quantity
        Electron temperature.
    n_e : Quantity
        Free-electron density.
    chem_pot : Quantity
        Electron chemical potential :math:`\\mu`.

    Returns
    -------
    n_j / n_i * n_e : Quantity
        The ratio of the population of the two states, multiplied with the
        free-electron density.
    """
    return (part_j / part_i) * n_e * jnpu.exp(-chem_pot / (1 * k_B * T_e))


class BalanceTerm(NamedTuple):
    """
    Per-element data produced by a balance-term factory.

    Attributes
    ----------
    coeff : Quantity
        Dimensioned ratio :math:`n_{i+1} n_e / n_i` (units: 1/volume) for each
        transition (length ``element.Z``).  The solver divides by ``ne_scale``.
    mask : jnp.ndarray
        1.0 where state i is still bound, 0.0 where pressure-ionised (length
        ``element.Z``).  Used both to set the diagonal sign and to gate the
        :math:`n_e` sub-diagonal.
    """

    coeff: jnp.ndarray  # shape (Z,)  units: 1/volume
    mask: jnp.ndarray  # shape (Z,)  values in {0.0, 1.0}


# balance-term factories


@partial(
    jax.jit,
    static_argnames=["element"],
)
def saha_balance_term(
    element: Element,
    ipd: Quantity,
    T_e: Quantity,
) -> BalanceTerm:
    """
    Classic (non-degenerate) Saha balance term.

    Parameters
    ----------
    element : Element
    ipd : Quantity
        Continuum-lowering / IPD correction (subtracted from binding energies).
    T_e : Quantity
        Electron temperature.
    """
    g = element.ionization.statistical_weights
    Eb = (jnpu.sort(element.ionization.energies) + ipd).m_as(
        ureg.electron_volt
    )
    coeff = saha_equation(g[:-1], g[1:], T_e, Eb * ureg.electron_volt)
    mask = jnp.heaviside(Eb, 0)
    return BalanceTerm(coeff=coeff, mask=mask)


def gen_saha_balance_term(
    element: Element,
    ipd: Quantity,
    T_e: Quantity,
    n_e: Quantity,
    chem_pot: Quantity,
) -> BalanceTerm:
    """
    Generalised (degenerate) Saha balance term.

    Parameters
    ----------
    element : Element
    ipd : Quantity
        Continuum-lowering correction.
    T_e : Quantity
        Electron temperature.
    n_e : Quantity
        Free-electron density (supplied externally; iterated to
        self-consistency by the caller).
    chem_pot : Quantity
        Electron chemical potential :math:`\\mu`.
    """
    g = element.ionization.statistical_weights
    Eb = (jnpu.sort(element.ionization.energies) + ipd).m_as(
        ureg.electron_volt
    )
    coeff = gen_saha_equation(
        g[:-1], g[1:], T_e, n_e, Eb * ureg.electron_volt, chem_pot
    )
    mask = jnp.heaviside(Eb, 0)
    return BalanceTerm(coeff=coeff, mask=mask)


def bu_balance_term(
    element: Element,
    ipd: Quantity,
    T_e: Quantity,
    n_e: Quantity,
    chem_pot: Quantity,
) -> BalanceTerm:
    """
    Bethe-Uhlenbeck balance term using Planck-Larkin partition sums.

    Instead of bare statistical weights the partition sums

    .. math::

        u_i = \\sum_s g_s
              \\left(e^{-E_s/k_BT} - 1 + \\frac{E_s}{k_BT}\\right)\\Theta(E_s)

    are used.  The energy dependence is absorbed into the partition functions,
    so the balance equation reduces to :py:func:`gen_balance_equation`.

    Parameters
    ----------
    element : Element
    ipd : Quantity
        Continuum-lowering correction (subtracted from binding energies
        before computing partition sums).
    T_e : Quantity
        Electron temperature.
    n_e : Quantity
        Free-electron density.
    chem_pot : Quantity
        Electron chemical potential :math:`\\mu`.

    Returns
    -------
    BalanceTerm
    """
    Ebs = jnpu.sort(element.ionization.energies) + ipd

    pls = []
    for charge, Eb in zip(range(element.Z), Ebs):
        g, E = read_nist_file(f"{element.symbol}{charge}")
        pls.append(planck_larkin_partition_sum(g, Eb - E, T_e))
    pls.append(1.0)  # fully-stripped ion has partition function 1
    part_func = jnp.array(pls)

    coeff = gen_balance_equation(
        part_func[:-1], part_func[1:], T_e, n_e, chem_pot
    )
    mask = jnp.heaviside(part_func[:-1], 0)
    return BalanceTerm(coeff=coeff, mask=mask)


@partial(
    jax.jit,
    static_argnames=["element_list"],
)
def _ne_scale(
    element_list: list[Element],
    ion_number_densities: Quantity,
    T_e: Quantity,
    masks: list[jnp.ndarray],
) -> Quantity:
    """
    Numerical density scale used to non-dimensionalise the matrix.

    The scale is interpolated between a small value (for low-T, mostly
    pressure-ionised plasmas) and the fully-ionised maximum, improving
    matrix conditioning across a wide temperature range.

    Parameters
    ----------
    element_list, ion_number_densities, T_e
        As in :py:func:`solve_ionization`.
    masks : list[jnp.ndarray]
        Per-element bound-state masks (used to estimate the pressure-ionised
        fraction and shift the lower bound of the scale accordingly).

    Returns
    -------
    Quantity:
        n_e scale in units of 1/volume.
    """
    Z = [ion.Z for ion in element_list]
    max_ne = jnpu.sum(jnp.array(Z) * ion_number_densities)
    all_masks = jnp.concatenate(masks)
    ratio = jnp.sum(all_masks) / len(all_masks)
    _ne_range = jnp.array(
        [
            jnp.max(
                jnp.array([max_ne.m_as(1 / ureg.m**3) * (1 - ratio), 1e-6])
            ),
            max_ne.m_as(1 / ureg.m**3),
        ]
    )
    return jnp.interp(
        (T_e * k_B).m_as(ureg.electron_volt),
        jnp.array([1.0, 1000.0]),
        _ne_range,
        left=_ne_range[0],
        right=_ne_range[-1],
    ) * (1 / ureg.m**3)


# Core solver


@partial(
    jax.jit,
    static_argnames=["element_list"],
)
def solve_ionization(
    element_list: list[Element],
    T_e: Quantity,
    ion_number_densities: Quantity,
    balance_terms: list[BalanceTerm],
    ne_scale: Quantity,
) -> tuple[Quantity, Quantity, Quantity]:
    """
    Solve for the ionization state of a plasma given pre-computed balance
    terms.

    This is the inner workhorse.  Callers should normally use one of the
    convenience wrappers (:py:func:`solve_saha`, :py:func:`solve_gen_saha`,
    :py:func:`solve_BU`) which construct ``balance_terms`` and ``ne_scale``
    automatically.

    The algorithm mirrors the `many-ion-saha-equation
    <https://github.com/jelkuweiss/many-ion-saha-equation>`_ approach:

    1. Build a sparse matrix whose diagonal encodes the balance relation and
       whose last row/column encodes number conservation.
    2. Find :math:`n_e` as the root of :math:`\\det(M)=0` via bisection.
    3. Solve the resulting linear system for per-level number densities.

    Parameters
    ----------
    element_list : list[Element]
        Ion species.
    T_e : Quantity
        Electron temperature.
    ion_number_densities : Quantity
        Total number density per species.
    balance_terms : list[BalanceTerm]
        One :py:class:`BalanceTerm` per element (same order as
        ``element_list``).  Each term supplies a ``coeff`` array (the ratio
        :math:`n_{i+1} n_e / n_i`) and a boolean ``mask`` array indicating
        which transitions are still bound.
    ne_scale : Quantity
        Numerical scale used to non-dimensionalise :math:`n_e`.

    Returns
    -------
    ionised_number_densities : Quantity
        Per-level densities ordered as
        ``[n(Ion0,0+), n(Ion0,1+), ..., n(Ion1,0+), ...]``.
    n_e : Quantity
        Self-consistent free-electron density.
    Z_mean : Quantity
        Mean ionization of each species.
    """
    Z = [ion.Z for ion in element_list]

    size = len(element_list) + int(onp.sum(Z)) + 1
    M = jnp.zeros((size, size))
    max_ne = jnpu.sum(jnp.array(Z) * ion_number_densities)

    skip = 0
    ionization_states = []

    for ion_dens, element, bt in zip(
        ion_number_densities, element_list, balance_terms, strict=True
    ):
        coeff = (bt.coeff / ne_scale).m_as(ureg.dimensionless)

        # Diagonal: -coeff where state is still bound, 1 where state is
        # pressure-ionized, i.e., does not exist, anymore.
        diag = jnp.diag(jnp.where(bt.mask > 0, -coeff, 1.0))
        dens_row = jnp.ones(element.Z + 1)

        # Set the diagonal for the balance-rows
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

    ne_mask = jnp.zeros(len(element_list) + int(onp.sum(Z)))
    skip = -1
    for element, bt in zip(element_list, balance_terms, strict=True):
        ne_mask = ne_mask.at[skip + 1 : skip + element.Z + 1].set(bt.mask)
        # density rows stay zero (already zero from initialization)
        skip += element.Z + 1

    def insert_ne(M, ne):
        # ne_mask is captured from outer scope
        ne_line = ne_mask * ne
        _diag = jnp.diag(ne_line, -1)
        out = (M + _diag).at[-1, -1].set(ne)
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
    continuum_lowering: list[Quantity] | None = None,
    exclude_non_negative_energies: bool = True,
) -> tuple[Quantity, Quantity, Quantity]:
    """
    Solve the classic (non-degenerate) Saha equation.

    Thin wrapper around :py:func:`solve_ionization` using
    :py:func:`saha_balance_term` to build the balance terms.

    Parameters
    ----------
    element_list : list[Element]
    T_e : Quantity
        Electron temperature.
    ion_number_densities : Quantity
        Total number density per species.
    continuum_lowering : list[Quantity] or None
        Per-species IPD shift (default: zero for all levels).
    exclude_non_negative_energies : bool
        If ``True`` (default), pressure-ionised levels (shifted
        :math:`E_b \\leq 0`) are excluded from the balance equations.

    Returns
    -------
    ionised_number_densities, n_e, Z_mean
        See :py:func:`solve_ionization`.
    """
    if continuum_lowering is None:
        continuum_lowering = [
            jnp.zeros(ion.Z) * ureg.electron_volt for ion in element_list
        ]

    bts = [
        saha_balance_term(el, ipd, T_e)
        for el, ipd in zip(element_list, continuum_lowering, strict=True)
    ]
    if not exclude_non_negative_energies:
        bts = [
            BalanceTerm(coeff=bt.coeff, mask=jnp.ones_like(bt.mask))
            for bt in bts
        ]

    scale = _ne_scale(
        element_list, ion_number_densities, T_e, [bt.mask for bt in bts]
    )
    return solve_ionization(
        element_list, T_e, ion_number_densities, bts, scale
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
    continuum_lowering: list[Quantity] | None = None,
    chem_pot_ideal: Quantity = 0 * ureg.electron_volt,
    exclude_non_negative_energies: bool = True,
) -> tuple[Quantity, Quantity, Quantity]:
    """
    Solve the generalised (degenerate) Saha equation.

    Thin wrapper around :py:func:`solve_ionization` using
    :py:func:`gen_saha_balance_term`.

    Parameters
    ----------
    element_list : list[Element]
    T_e : Quantity
        Electron temperature.
    n_e : Quantity
        Free-electron density (initial estimate; caller should iterate for
        self-consistency, e.g. via :py:func:`calculate_mean_free_charge_saha`).
    ion_number_densities : Quantity
    continuum_lowering : list[Quantity] or None
    chem_pot_ideal : Quantity
        Electron chemical potential :math:`\\mu` (default 0 eV).
    exclude_non_negative_energies : bool

    Returns
    -------
    ionised_number_densities, n_e, Z_mean
        See :py:func:`solve_ionization`.
    """
    if continuum_lowering is None:
        continuum_lowering = [
            jnp.zeros(ion.Z) * ureg.electron_volt for ion in element_list
        ]

    bts = [
        gen_saha_balance_term(el, ipd, T_e, n_e, chem_pot_ideal)
        for el, ipd in zip(element_list, continuum_lowering, strict=True)
    ]
    if not exclude_non_negative_energies:
        bts = [
            BalanceTerm(coeff=bt.coeff, mask=jnp.ones_like(bt.mask))
            for bt in bts
        ]

    scale = _ne_scale(
        element_list,
        ion_number_densities,
        T_e,
        [bt.mask for bt in bts],
    )
    return solve_ionization(
        element_list, T_e, ion_number_densities, bts, scale
    )


def solve_BU(
    element_list: list[Element],
    T_e: Quantity,
    n_e: Quantity,
    ion_number_densities: Quantity,
    continuum_lowering: list[Quantity] | None = None,
    chem_pot_ideal: Quantity = 0 * ureg.electron_volt,
) -> tuple[Quantity, Quantity, Quantity]:
    """
    Solve the Bethe-Uhlenbeck equation using Planck-Larkin partition sums.

    Thin wrapper around :py:func:`solve_ionization` using
    :py:func:`bu_balance_term`.

    Parameters
    ----------
    element_list : list[Element]
    T_e : Quantity
    n_e : Quantity
    ion_number_densities : Quantity
    continuum_lowering : list[Quantity] or None
    chem_pot_ideal : Quantity

    Returns
    -------
    ionised_number_densities, n_e, Z_mean
        See :py:func:`solve_ionization`.

    See Also
    --------
    jaxrts.ionization.planck_larkin_partition_sum
    jaxrts.ionization.gen_balance_equation
    """
    if continuum_lowering is None:
        continuum_lowering = [
            jnp.zeros(ion.Z) * ureg.electron_volt for ion in element_list
        ]

    bts = [
        bu_balance_term(el, ipd, T_e, n_e, chem_pot_ideal)
        for el, ipd in zip(element_list, continuum_lowering, strict=True)
    ]

    scale = _ne_scale(
        element_list,
        ion_number_densities,
        T_e,
        [bt.mask for bt in bts],
    )
    return solve_ionization(
        element_list, T_e, ion_number_densities, bts, scale
    )


# Higher-level helpers


def calculate_charge_state_distribution_saha(plasma_state):
    """
    Charge-state distribution as fractions, assuming thermal equilibrium
    (classic Saha + IPD).

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
    Mean ionization charge per species via Saha-Boltzmann.

    Iterates up to 6 times for self-consistency between the ionization state
    and the IPD / chemical potential.

    Parameters
    ----------
    plasma_state : PlasmaState
    use_ipd : bool
        Apply IPD continuum-lowering.  Note: the IPD often depends on the
        ionization state, creating a circular dependency that the iteration
        resolves.
    use_chem_pot : bool
        Use the generalised Saha equation with chemical potential.
    use_distribution : bool
        Evaluate IPD per-species rather than from the mean state.
    exclude_non_negative_energies : bool
        Exclude pressure-ionised bound states.

    Returns
    -------
    charge_distribution : jnp.ndarray
        The charge state distribution in fractions.
    Z_mean : jnp.ndarray
        An array containing the mean charge of each ion in the plasma.

    See Also
    --------
    jaxrts.ionization.solve_saha
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


def calculate_mean_free_charge_BU(
    plasma_state,
    use_ipd: bool = False,
    use_distribution: bool = False,
):
    """
    Mean ionization charge per species via the Bethe-Uhlenbeck equation.

    Parameters
    ----------
    plasma_state : PlasmaState
    use_ipd : bool
        Apply IPD continuum-lowering.
    use_distribution : bool
        Evaluate IPD per-species rather than from the mean state.

    Returns
    -------
    charge_distribution : jnp.ndarray
    Z_mean : jnp.ndarray

    See Also
    --------
    jaxrts.saha.solve_BU
    """
    _ions = tuple(plasma_state.ions)
    _n_ions = plasma_state.mass_density / plasma_state.atomic_masses

    plasma_state.Z_free = jnp.array(
        jnp.max(jnp.array([i.Z for i in plasma_state.ions]))
    )

    def _get_cl(charge_distribution):
        if use_ipd:
            return plasma_state["ipd"].all_element_states(
                plasma_state,
                charge_distribution if use_distribution else None,
            )
        return [
            jnp.zeros(ion.Z) * ureg.electron_volt for ion in plasma_state.ions
        ]

    chem_pot = plasma_state["chemical potential"].evaluate(plasma_state, None)
    cl = [jnp.zeros(ion.Z) * ureg.electron_volt for ion in plasma_state.ions]

    charge_distribution, _ne, Z_mean = solve_BU(
        _ions,
        plasma_state.T_e,
        plasma_state.n_e,
        _n_ions,
        continuum_lowering=cl,
        chem_pot_ideal=chem_pot,
    )

    for _ in range(6):
        chem_pot = plasma_state["chemical potential"].evaluate(
            plasma_state, None
        )
        cl = _get_cl(charge_distribution)
        charge_distribution, _ne, Z_mean = solve_BU(
            _ions,
            plasma_state.T_e,
            plasma_state.n_e,
            _n_ions,
            continuum_lowering=cl,
            chem_pot_ideal=chem_pot,
        )
        plasma_state.Z_free = jnp.array(Z_mean)

    return charge_distribution, Z_mean


# Thomas-Fermi / More fits


@jax.jit
def _calculate_single_species_ionization_more(rho, T_e, m_A, Z_A) -> Quantity:
    alpha = 14.3139
    beta = 0.6624
    a1, a2, a3, a4 = 0.003323, 0.9718, 9.26148e-5, 3.10165
    b0, b1, b2 = -1.7630, 1.43175, 0.31546
    c1, c2 = -0.366667, 0.983333

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
