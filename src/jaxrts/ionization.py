"""
Module to model the ionization state of a plasma.
It contains functions to solve the
`Saha-equation <https://en.wikipedia.org/wiki/Saha_ionization_equation>`_,
linking the temperature of a plasma to its ionization.

The central solver is :py:func:`solve_ionization`, which accepts a
:py:class:`~.BalanceTerm` -- an object that supplies per-transition
coefficients and the corresponding "still-bound" mask.

:py:class:`~.BalanceTerm` s can be created with the factories provided, e.g.
:py:class:`~.saha_balance_term`, :py:class:`~.gen_balance_term`, or
:py:class:`bu_balance_term`.
"""

from typing import NamedTuple
import logging
from functools import partial

import jax
from jax.tree_util import Partial
import jax.numpy as jnp
import jpu.numpy as jnpu

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
        transition (length ``element.Z``).
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


# Core solver


def _log_partition_products(
    log_coeffs: jnp.ndarray,
    mask: jnp.ndarray,
    log_ne: float,
) -> jnp.ndarray:
    """
    mask[k] = 1 means charge state k exists as a bound state.
    mask[k] = 0 means charge state k is pressure-ionized (does not exist).
    State Z (fully stripped) always exists.
    """
    # Zero out pressure-ionized states in log_ratios so cumsum is unaffected by
    # them. They will be masked to -inf afterwards anyway (in hte log!).
    log_ratios = log_coeffs - log_ne  # shape (Z,)
    log_ratios_clean = jnp.where(mask > 0, log_ratios, 0.0)

    log_P_raw = jnp.concatenate([
        jnp.zeros(1),
        jnp.cumsum(log_ratios_clean),
    ])

    # Extend mask to cover the fully-stripped state (always exists)
    mask_ext = jnp.concatenate([mask, jnp.ones(1)])

    # vanished states get -inf, others states keep the cumsum value
    return jnp.where(mask_ext > 0, log_P_raw, -jnp.inf)


def _zbar_and_weights(log_P: jnp.ndarray) -> tuple[float, jnp.ndarray]:
    """
    Stable mean charge and fractional populations from log partition products.
    Handles -inf entries (states pushed to the continuum) and large positive
    entries.
    """
    charges = jnp.arange(len(log_P), dtype=float)

    # Clip before softmax to prevent exp(inf-inf)=NaN in fully-ionized limit.
    log_P_clipped = jnp.clip(log_P, min=-500.0, max=500.0)
    weights = jax.nn.softmax(log_P_clipped)
    return jnp.sum(charges * weights), weights


@partial(
    jax.jit,
    static_argnames=["element_list"],
)
def solve_ionization(
    element_list,
    T_e,
    ion_number_densities,
    balance_terms,
):
    """
    Solve for the ionization state of a plasma given pre-computed balance
    terms.

    Callers should normally use one of the convenience wrappers
    (:py:func:`solve_saha`, :py:func:`solve_gen_saha`, :py:func:`solve_BU`)
    which construct ``balance_terms`` automatically.

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
    Z = [el.Z for el in element_list]
    N_i_arr = ion_number_densities.m_as(
        1 / ureg.m**3
    )  # strip units for arithmetic
    max_ne = sum(z * n for z, n in zip(Z, N_i_arr))

    # Precompute log coefficients
    log_coeffs_list = []
    for bt in balance_terms:
        coeff_vals = bt.coeff.m_as(1 / ureg.m**3)
        # Maybe clip here, if required
        # coeff_vals = jnp.clip(coeff_vals, min=1e-300, max=1e300)
        log_coeffs_list.append(jnp.log(coeff_vals))

    def residual(log_ne):
        total_charge = sum(
            N_i
            * _zbar_and_weights(_log_partition_products(lc, bt.mask, log_ne))[
                0
            ]
            for lc, bt, N_i in zip(log_coeffs_list, balance_terms, N_i_arr)
        )
        return jnp.exp(log_ne) - total_charge

    log_ne_lo = jnp.log(jnp.array(1e-8))
    log_ne_hi = jnp.log(jnp.array(max_ne))

    sol_log_ne, _ = bisection(
        Partial(residual),
        log_ne_lo,
        log_ne_hi,
        tolerance=1e-8,
        max_iter=1000,
        min_iter=1,
    )

    sol_ne = jnp.exp(sol_log_ne)

    # Recover populations
    all_densities = []
    Z_mean = []
    for bt, N, lc in zip(balance_terms, N_i_arr, log_coeffs_list):
        log_P = _log_partition_products(lc, bt.mask, sol_log_ne)
        zbar, weights = _zbar_and_weights(log_P)
        all_densities.append(N * weights / (1 * ureg.m**3))
        Z_mean.append(zbar)

    return (
        to_array(
            jnp.concatenate([d.m_as(1 / ureg.m**3) for d in all_densities])
            / (1 * ureg.m**3)
        ),
        sol_ne / (1 * ureg.m**3),
        jnp.array(Z_mean),
    )


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

    return solve_ionization(
        element_list, T_e, ion_number_densities, bts
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

    return solve_ionization(
        element_list, T_e, ion_number_densities, bts
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

    return solve_ionization(
        element_list, T_e, ion_number_densities, bts
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
