"""
HNC Potentials
==============

This module contains a set of Parameters to be used when performing HNC
calculations. This includes that the potential is split into a long-range and a
short-range part and is also Fourier-transformed to k-space.
"""

import abc
import logging
from typing import Literal

import jax
import jax.interpreters
import jpu.numpy as jnpu
from jax import numpy as jnp

from jaxrts import literature
from jaxrts.hypernetted_chain import (
    _3Dfour,
    fourier_transform_sine,
    hnc_interp,
    mass_weighted_T,
)
from jaxrts.plasma_physics import (
    chem_pot_sommerfeld_fermi_interpolation,
    fermi_dirac,
    fermi_wavenumber,
)
from jaxrts.units import Quantity, to_array, ureg

logger = logging.getLogger(__name__)


@jax.jit
def pauli_potential_from_classical_map_SpinAveraged(
    r1: jnp.ndarray, r2: jnp.ndarray, n_e: Quantity, T: Quantity
):
    """
    This method utilizes the exact non-interacting solution for the pair
    distribution function (PDF) as input (see :cite:`Bredow.2017`). The
    effective potential is then derived iteratively to reproduce the input PDF
    using the HNC algorithm. This approach effectively inverts the HNC
    framework to determine the interaction potential from a known PDF.
    """

    SMALL = 1e-14

    # Step 1: Calculate g^0_T(r)

    k_F = fermi_wavenumber(n_e)

    dr1 = r1[1] - r1[0]
    dk1 = jnp.pi / (len(r1) * dr1)
    k1 = jnp.pi / r1[-1] + jnp.arange(len(r1)) * dk1
    dr2 = r2[1] - r2[0]
    dk2 = jnp.pi / (len(r2) * dr2)
    k2 = jnp.pi / r2[-1] + jnp.arange(len(r2)) * dk2

    # Use this instead ...
    chem_pot = chem_pot_sommerfeld_fermi_interpolation(T, n_e)

    # Calculate Fermi distribution
    n_k = fermi_dirac(k1, chem_pot, T)
    F_T_r = (
        (6 * jnp.pi**2 / k_F**3)
        * fourier_transform_sine(r1, k1, n_k)
        / (2 * jnp.pi) ** 3
    )

    g0_T_r = 1 - (F_T_r**2).m_as(ureg.dimensionless)

    # Average between the interacting and the non-interacting g (the latter is
    # just unity! for all k)

    g0_T_r = g0_T_r / 2 + 1 / 2

    # Clip g0, as negative values are problematic
    g0_T_r = jnp.where(g0_T_r <= 0.0, SMALL, g0_T_r)

    # Interpolate g0 to the r-spacing on which the HNC alogrithm is performed
    # on
    g0_T_r = jnp.interp(
        r2.m_as(ureg.a0),
        r1.m_as(ureg.a0),
        g0_T_r,
        left=0.5,
        right=1.0,
    )

    # Step 2: Invert HNC algorithm to calculate the potential which produces
    # g0_T_r (this is the Pauli exclusion potential by construction)

    h0_T_r = g0_T_r - 1
    Ns_r = jnpu.log(g0_T_r)

    N_k = fourier_transform_sine(k2, r2, Ns_r * ureg.dimensionless)
    h0_T_k = fourier_transform_sine(k2, r2, h0_T_r * ureg.dimensionless)

    cs_k = h0_T_k - N_k
    c_k = h0_T_k / (1 + n_e * h0_T_k)

    potential = (cs_k - c_k) * (1 * ureg.boltzmann_constant * T)

    # Interpolate values close to origin to fix divergences

    index1 = jnp.argmax(
        jnp.where(k2.m_as(1 / ureg.angstrom) < 0.8, jnp.arange(k2.size), -1.0)
    )
    index2 = jnp.argmax(
        jnp.where(k2.m_as(1 / ureg.angstrom) < 1.0, jnp.arange(k2.size), -1.0)
    )

    f1 = potential.at[index1].get() * 1 * ureg.electron_volt * ureg.a0**3
    f2 = potential.at[index2].get() * 1 * ureg.electron_volt * ureg.a0**3
    x1 = k2.at[index1].get() / (1 * ureg.a0)
    x2 = k2.at[index2].get() / (1 * ureg.a0)

    def interp_func(x):

        a = (f2 - f1) / (x2**2 - x1**2)
        c = f1 - a * x1**2

        return a * x**2 + c

    potential = jnpu.where((k2 < x1), interp_func(k2), potential)

    return k2, potential, g0_T_r


@jax.jit
def pauli_potential_from_classical_map_SpinSeparated(
    r1: jnp.ndarray, r2: jnp.ndarray, n_e: Quantity, T: Quantity
):
    """
    This method utilizes the exact non-interacting solution for the pair
    distribution function (PDF) as input. The effective potential is then
    derived iteratively to reproduce the input PDF using the HNC algorithm.
    This approach effectively inverts the HNC framework to determine the
    interaction potential from a known PDF.
    """

    n = to_array([n_e / 2, n_e / 2])

    d = jnp.eye(n.shape[0]) * n

    # Step 1: Calculate g^0_T(r)

    k_F = fermi_wavenumber(d)

    dr1 = r1[1] - r1[0]
    dk1 = jnp.pi / (len(r1) * dr1)
    k1 = jnp.pi / r1[-1] + jnp.arange(len(r1)) * dk1
    dr2 = r2[1] - r2[0]
    dk2 = jnp.pi / (len(r2) * dr2)
    k2 = jnp.pi / r2[-1] + jnp.arange(len(r2)) * dk2

    # Don't use Maxwellian distribution (see Bredow 2017), use this instead
    chem_pot = chem_pot_sommerfeld_fermi_interpolation(T, d)

    # Calculate Fermi distribution
    n_k = fermi_dirac(
        k1[jnp.newaxis, jnp.newaxis, :], chem_pot[:, :, jnp.newaxis], T
    )
    F_T_r = (
        (6 * jnp.pi**2 / k_F**3)[:, :, jnp.newaxis]
        * _3Dfour(r1, k1, n_k)
        / (2 * jnp.pi) ** 3
    )

    g0_T_r = 1 - (F_T_r**2).m_as(ureg.dimensionless)

    # This is only valid for T=0, keeping it for documentation reasons

    # g_same = (
    #     1
    #     - (
    #         3
    #         * (jnpu.sin(k_F * r) - (k_F * r) * jnpu.cos(k_F * r))
    #         / (k_F * r) ** 3
    #     ).m_as(ureg.dimensionless)
    #     ** 2
    # )
    # g_diff = jnp.ones(r.shape)
    # g0_T_r_analytic = jnp.array([[g_same, g_diff], [g_diff, g_same]])

    # Clip g0, as negative values are problematic
    g0_T_r = jnp.where(g0_T_r <= 0.0, 1e-14, g0_T_r)
    g0_T_r = jnp.interp(
        r2.m_as(ureg.a0),
        r1.m_as(ureg.a0),
        g0_T_r[0, 0, :],
        left=0.5,
        right=1.0,
    )
    g0_T_r = jnp.array(
        [[g0_T_r, jnp.ones_like(r2)], [jnp.ones_like(r2), g0_T_r]]
    )
    # Step 2: Invert HNC algorithm to calculate the potential which produces
    # g0_T_r (this is the Pauli exclusion potential by construction)

    h0_T_r = g0_T_r - 1
    Ns_r = jnpu.log(g0_T_r)

    N_k = _3Dfour(k2, r2, Ns_r * ureg.dimensionless)
    h0_T_k = _3Dfour(k2, r2, h0_T_r * ureg.dimensionless)

    cs_k = h0_T_k - N_k

    def ozr_inverted(input_vec):
        """
        Inverted Ornstein-Zernicke Relation
        """
        return jnpu.matmul(
            input_vec,
            jnp.linalg.inv(
                (jnp.eye(n.shape[0]) + jnpu.matmul(d, input_vec)).m_as(
                    ureg.dimensionless
                )
            ),
        )

    c_k = jax.vmap(ozr_inverted, in_axes=2, out_axes=2)(h0_T_k)

    potential = (cs_k - c_k) * (1 * ureg.boltzmann_constant * T)

    index1 = jnp.argmax(
        jnp.where(k2.m_as(1 / ureg.angstrom) < 1.0, jnp.arange(k2.size), -1.0)
    )
    index2 = jnp.argmax(
        jnp.where(k2.m_as(1 / ureg.angstrom) < 2.0, jnp.arange(k2.size), -1.0)
    )

    f1 = potential.at[1, 1, index1].get() * 1 * ureg.electron_volt * ureg.a0**3

    f2 = potential.at[1, 1, index2].get() * 1 * ureg.electron_volt * ureg.a0**3

    x1 = k2.at[index1].get() / (1 * ureg.a0)
    x2 = k2.at[index2].get() / (1 * ureg.a0)

    def interp_func(x):

        a = (f2 - f1) / (x2**2 - x1**2)
        c = f1 - a * x1**2

        return a * x**2 + c

    potential = jnpu.where((k2 < x1), interp_func(k2), potential)

    # Step 3: Calculate V_(k)
    return k2, potential, g0_T_r


@jax.jit
def construct_alpha_matrix(n: jnp.ndarray | Quantity):
    d = jnpu.cbrt(
        3 / (4 * jnp.pi * (n[:, jnp.newaxis] * n[jnp.newaxis, :]) ** (1 / 2))
    )

    return 2 / d


@jax.jit
def construct_q_matrix(q: jnp.ndarray) -> jnp.ndarray:
    return jnpu.outer(q, q)


class HNCPotential(metaclass=abc.ABCMeta):
    """
    Potentials, intended to be used in the HNC scheme. By default, the results
    of methods evaluating this Potential (in k or r space), return a
    :math:`(n\\times n\\times m)` matrix, where ``n`` is the number of ion
    species and ``m`` is the number of r or k points.
    However, if :py:attr:`~.include_electrons` is ``"SpinAveraged"`` or
    ``"SpinSeparated"``, electrons are added as another ion species, and so one
    the first two dimensions get one or two additional entries, respectively.
    """

    #: A list of keys where this Potential is adequate for. Valid keys are
    #: "ion-ion Potential", "electron-ion Potential", and
    #: "electron-electron Potential".
    allowed_keys = [
        "ion-ion Potential",
        "electron-ion Potential",
        "electron-electron Potential",
    ]
    #: A list of bibtex keys. Can be in the format ``[key1, key2]``, for
    #: general keys, ``[(key1, comment1), (key2, comment2)]``, if comments are
    #: desired, of ``[([key1, key2], comment1), (key3, comment2)]`` if the
    #: comment should apply to multiple keys.
    cite_keys: (
        list[str] | list[tuple[str, str]] | list[tuple[list[str], str]]
    ) = []

    def __init__(
        self,
        include_electrons: Literal[
            "off", "SpinAveraged", "SpinSeparated"
        ] = "off",
    ):
        self._transform_r = jnpu.linspace(1e-4, 1e4, 2**21) * ureg.a_0

        self.model_key = ""

        #: If `"SpinAveraged"`, the electrons are added as the n+1th ion
        #: species to the potential. The relevant entries are the last row and
        #: column, respectively (i.e., the colored lines in the image above).
        #: If `"SpinSeparated"`, two species of electrons (with half the
        #: electron number density, each) are added as the n+1th and n+2th ion.
        self.include_electrons: Literal[
            "off", "SpinAveraged", "SpinSeparated"
        ] = include_electrons

    def citation(
        self,
        style: Literal["plain", "bibtex", "cite"] = "plain",
        comment: str | None = None,
    ) -> str:
        """
        Return bibliographic information for the Model used.

        Parameters
        ----------
        style: "plain", "cite", or "bibtex"
            When ``"plain"``, the literature references are formatted in a
            human-readable format. If ``"bibtex"``, the citations are given as
            bibtex entries, which can then be copied into a literature
            collection. If ``"cite"``, the citation keys are not evaluated, but
            just retuned, wrapped in a tex ``\\cite{...}`` command.
        comment: str or None, default None
            (Additional) comment to give to the citation entry.

        Returns
        -------
        str
            The information about the model used
        """
        if style == "plain":
            citation_function = literature.get_formatted_ref_string
        elif style == "cite":
            citation_function = literature.get_cite_ref_string
        else:
            citation_function = literature.get_bibtex_ref_string
        citations = []
        for entry in self.cite_keys:
            if isinstance(entry, str):
                citations.append(citation_function(entry, comment))
            else:
                key, key_comment = entry
                if comment is not None:
                    key_comment = comment + ". " + key_comment
                citations.append(citation_function(key, key_comment))
        return "\n".join(citations)

    def check(self, plasma_state) -> None:  # noqa: B027
        """
        Test if the HNCPotential is applicable to the PlasmaState. Might raise
        logged messages and errors. Is automatically called after
        :py:meth:`~.__init__`.
        """
        pass

    def prepare(self, plasma_state, key: str) -> None:  # noqa: B027
        pass

    @abc.abstractmethod
    def full_r(self, plasma_state, r: Quantity) -> Quantity: ...

    @jax.jit
    def short_r(self, plasma_state, r: Quantity) -> Quantity:
        """
        This is the short-range part of :py:meth:`full_r`:

        .. math::

            V(r) \\cdot \\exp(-\\alpha r)

        """
        _r = r[jnp.newaxis, jnp.newaxis, :]
        return self.full_r(plasma_state, r) * jnpu.exp(
            -self.alpha(plasma_state) * _r
        )

    @jax.jit
    def long_r(self, plasma_state, r: Quantity) -> Quantity:
        """
        This is the long-range part of :py:meth:`full_r`:

        .. math::

            V(r) * (1 - \\exp(-\\alpha r))

        """
        _r = r[jnp.newaxis, jnp.newaxis, :]
        return self.full_r(plasma_state, r) * (
            1 - jnpu.exp(-self.alpha(plasma_state) * _r)
        )

    @jax.jit
    def short_k(self, plasma_state, k: Quantity) -> Quantity:
        """
        The Fourier transform of :py:meth:`~short_r`.
        """
        V_k, _k = transformPotential(
            self.short_r(plasma_state, self._transform_r), self._transform_r
        )
        return hnc_interp(k, _k, V_k)

    @jax.jit
    def long_k(self, plasma_state, k: Quantity) -> Quantity:
        """
        The Foutier transform of :py:meth:`~short_k`.
        """
        V_k, _k = transformPotential(
            self.long_r(plasma_state, self._transform_r), self._transform_r
        )
        return hnc_interp(k, _k, V_k)

    @jax.jit
    def full_k(self, plasma_state, k):
        return self.short_k(plasma_state, k) + self.long_k(plasma_state, k)

    def q2(self, plasma_state):
        """
        This is :math:`q^2`!
        """
        if self.include_electrons == "SpinAveraged":
            Z = to_array([*plasma_state.Z_free, -1])
        elif self.include_electrons == "SpinSeparated":
            Z = to_array([*plasma_state.Z_free, -1, -1])
        else:
            Z = plasma_state.Z_free
        charge = construct_q_matrix(Z * ureg.elementary_charge)
        return charge[:, :, jnp.newaxis]

    def alpha(self, plasma_state):
        if self.include_electrons == "SpinAveraged":
            n = to_array([*plasma_state.n_i, plasma_state.n_e])
        elif self.include_electrons == "SpinSeparated":
            n = to_array(
                [*plasma_state.n_i, plasma_state.n_e / 2, plasma_state.n_e / 2]
            )
        else:
            n = plasma_state.n_i
        a = construct_alpha_matrix(n)
        return a[:, :, jnp.newaxis]

    def mu(self, plasma_state):
        """
        The geometric mean of two masses (or reciprocal sum)
        """
        if self.include_electrons == "SpinAveraged":
            m = to_array([*plasma_state.atomic_masses, 1 * ureg.electron_mass])
        elif self.include_electrons == "SpinSeparated":
            m = to_array(
                [
                    *plasma_state.atomic_masses,
                    1 * ureg.electron_mass,
                    1 * ureg.electron_mass,
                ]
            )
        else:
            m = plasma_state.atomic_masses
        mu = jnpu.outer(m, m) / (m[:, jnp.newaxis] + m[jnp.newaxis, :])
        return mu[:, :, jnp.newaxis]

    def T(self, plasma_state):
        """
        The mass_weighted temperature average of a pair, according to
        :cite:`Schwarz.2007`.

        .. math::

           \\bar{T}_{ab} = \\frac{T_a m_b + T_b m_a}{m_a + m_b}

        """
        if self.include_electrons == "SpinAveraged":
            m = to_array([*plasma_state.atomic_masses, 1 * ureg.electron_mass])
            T = to_array([*plasma_state.T_i, plasma_state.T_e])
        elif self.include_electrons == "SpinSeparated":
            m = to_array(
                [
                    *plasma_state.atomic_masses,
                    1 * ureg.electron_mass,
                    1 * ureg.electron_mass,
                ]
            )
            T = to_array(
                [*plasma_state.T_i, plasma_state.T_e, plasma_state.T_e]
            )
        else:
            m = plasma_state.atomic_masses
            T = plasma_state.T_i
        return mass_weighted_T(m, T)[:, :, jnp.newaxis]

    def r_cut(self, plasma_state) -> Quantity:
        """
        This casts :attr:`jaxrts.PlasmaState.ion_core_radius` in the form
        required to be used with an :py:class:~.HNCPotential`. However, this
        quantity is only relevant if we consider electron-ion interactions.
        Hence, the returned array will be 0 for all ion-ion pairs and the
        electron-electron pair(s).
        """
        r = jnp.zeros_like(self.q2(plasma_state).magnitude, dtype=float)
        if self.include_electrons == "SpinAveraged":
            r = r.at[:-1, -1, :].set(
                plasma_state.ion_core_radius.m_as(ureg.angstrom)[
                    :, jnp.newaxis
                ]
            )
            r = r.at[-1, :-1, :].set(
                plasma_state.ion_core_radius.m_as(ureg.angstrom)[
                    :, jnp.newaxis
                ]
            )
        elif self.include_electrons == "SpinSeparated":
            r = r.at[:-2, -2, :].set(
                plasma_state.ion_core_radius.m_as(ureg.angstrom)[
                    :, jnp.newaxis
                ]
            )
            r = r.at[:-2, -1, :].set(
                plasma_state.ion_core_radius.m_as(ureg.angstrom)[
                    :, jnp.newaxis
                ]
            )
            r = r.at[-1, :-2, :].set(
                plasma_state.ion_core_radius.m_as(ureg.angstrom)[
                    :, jnp.newaxis
                ]
            )
            r = r.at[-2, :-2, :].set(
                plasma_state.ion_core_radius.m_as(ureg.angstrom)[
                    :, jnp.newaxis
                ]
            )
        return r * ureg.angstrom

    def lambda_ab(self, plasma_state):
        # Compared to Gregori.2003, there is a pi missing
        l_ab = ureg.hbar * jnpu.sqrt(
            1 / (2 * self.mu(plasma_state) * ureg.k_B * self.T(plasma_state))
        )
        return l_ab

    def __add__(self, other) -> "PotentialSum":
        if not isinstance(other, HNCPotential):
            raise NotImplementedError(
                "You can only add other HNCPotentials to an HNCPotential."
                + f"Other is type {type(other)}."
            )
        if isinstance(self, PotentialSum):
            list_of_potentials = self.potentials
        else:
            list_of_potentials = [self]
        if isinstance(other, PotentialSum):
            list_of_potentials = [*list_of_potentials, *other.potentials]
        else:
            list_of_potentials.append(other)
        return PotentialSum(list_of_potentials)

    def __mul__(self, other) -> "ScaledPotential":
        if not isinstance(other, float | int):
            raise NotImplementedError(
                "You can only add scale an HNCPotentials with a float."
                + f"Other is type {type(other)}."
            )
        return ScaledPotential(self, factor=other)

    # The following is required to jit a state
    def _tree_flatten(self):
        children = (self._transform_r,)
        aux_data = {
            "include_electrons": self.include_electrons,
            "model_key": self.model_key,
        }  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        (obj._transform_r,) = children
        obj.model_key = aux_data["model_key"]
        obj.include_electrons = aux_data["include_electrons"]
        return obj


class PotentialSum(HNCPotential):
    """
    A sum of several :py:class:`HNCPotential` s. Can be used e.g., if
    higher-order corrections should be added to a model.
    """

    __name__ = "PotentialSum"

    def __init__(self, list_of_potentials: list[HNCPotential]) -> None:
        self.potentials = list_of_potentials
        if any(
            [
                pot.include_electrons == "SpinSeparated"
                for pot in self.potentials
            ]
        ):
            self.include_electrons = "SpinSeparated"
        elif any(
            [
                pot.include_electrons == "SpinAveraged"
                for pot in self.potentials
            ]
        ):
            self.include_electrons = "SpinAveraged"
        else:
            self.include_electrons = "off"
        self.model_key = ""

    @property
    def include_electrons(self):
        return self._include_electrons

    @include_electrons.setter
    def include_electrons(
        self, value: Literal["off", "SpinSeparated", "SpinAveraged"]
    ):
        self._include_electrons = value
        # Pass the setting if electrons should be included down to all the
        # potentials considered
        for pot in self.potentials:
            pot.include_electrons = value

    @property
    def description(self) -> str:
        return f"Sum of {[pot.__name__ for pot in self.potentials]}"

    @jax.jit
    def full_r(self, plasma_state, r: Quantity) -> Quantity:
        unit = ureg.electron_volt
        out = jnp.array(
            [pot.full_r(plasma_state, r).m_as(unit) for pot in self.potentials]
        )
        return jnp.sum(out, axis=0) * unit

    @jax.jit
    def long_r(self, plasma_state, r: Quantity) -> Quantity:
        unit = ureg.electron_volt
        out = jnp.array(
            [pot.long_r(plasma_state, r).m_as(unit) for pot in self.potentials]
        )
        return jnp.sum(out, axis=0) * unit

    @jax.jit
    def short_r(self, plasma_state, r: Quantity) -> Quantity:
        unit = ureg.electron_volt
        out = jnp.array(
            [
                pot.short_r(plasma_state, r).m_as(unit)
                for pot in self.potentials
            ]
        )
        return jnp.sum(out, axis=0) * unit

    @jax.jit
    def full_k(self, plasma_state, k: Quantity) -> Quantity:
        unit = ureg.electron_volt * ureg.angstrom**3
        out = jnp.array(
            [pot.full_k(plasma_state, k).m_as(unit) for pot in self.potentials]
        )
        return jnp.sum(out, axis=0) * unit

    @jax.jit
    def long_k(self, plasma_state, k: Quantity) -> Quantity:
        unit = ureg.electron_volt * ureg.angstrom**3
        out = jnp.array(
            [pot.long_k(plasma_state, k).m_as(unit) for pot in self.potentials]
        )
        return jnp.sum(out, axis=0) * unit

    @jax.jit
    def short_k(self, plasma_state, k: Quantity) -> Quantity:
        unit = ureg.electron_volt * ureg.angstrom**3
        out = jnp.array(
            [
                pot.short_k(plasma_state, k).m_as(unit)
                for pot in self.potentials
            ]
        )
        return jnp.sum(out, axis=0) * unit

    # The following is required to jit a state
    def _tree_flatten(self):
        children = (self.potentials,)
        aux_data = {
            "include_electrons": self._include_electrons,
            "model_key": self.model_key,
        }  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        (obj.potentials,) = children
        obj.model_key = aux_data["model_key"]
        obj._include_electrons = aux_data["include_electrons"]
        return obj


class ScaledPotential(HNCPotential):
    """
    A :py:class:`HNCPotential`, scaled with a constant factor.
    """

    __name__ = "ScaledPotential"

    def __init__(self, potential: HNCPotential, factor: float = 1.0) -> None:
        self.potential = potential
        self.factor = factor
        self.include_electrons = self.potential.include_electrons
        self.model_key = ""

    @property
    def include_electrons(self):
        return self._include_electrons

    @include_electrons.setter
    def include_electrons(
        self, value: Literal["off", "SpinAveraged", "SpinSeparated"]
    ):
        self._include_electrons = value
        # Pass the setting if electrons should be included down to all the
        # potentials considered
        self.potential.include_electrons = value

    @property
    def description(self) -> str:
        return f"{self.potential.__name__}, scaled with {self.factor}"

    @jax.jit
    def full_r(self, plasma_state, r: Quantity) -> Quantity:
        return self.potential.full_r(plasma_state, r) * self.factor

    @jax.jit
    def long_r(self, plasma_state, r: Quantity) -> Quantity:
        return self.potential.long_r(plasma_state, r) * self.factor

    @jax.jit
    def short_r(self, plasma_state, r: Quantity) -> Quantity:
        return self.potential.short_r(plasma_state, r) * self.factor

    @jax.jit
    def full_k(self, plasma_state, k: Quantity) -> Quantity:
        return self.potential.full_k(plasma_state, k) * self.factor

    @jax.jit
    def long_k(self, plasma_state, k: Quantity) -> Quantity:
        return self.potential.long_k(plasma_state, k) * self.factor

    @jax.jit
    def short_k(self, plasma_state, k: Quantity) -> Quantity:
        return self.potential.short_k(plasma_state, k) * self.factor

    def __mul__(self, other) -> "ScaledPotential":
        if not isinstance(other, float | int):
            raise NotImplementedError(
                "You can only add scale an HNCPotentials with a float. "
                + f"Other is type {type(other)}."
            )
        return ScaledPotential(self.potential, factor=other * self.factor)

    # The following is required to jit a state
    def _tree_flatten(self):
        children = (self.potential, self.factor)
        aux_data = {
            "include_electrons": self._include_electrons,
            "model_key": self.model_key,
        }  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        (obj.potential, obj.factor) = children
        obj.model_key = aux_data["model_key"]
        obj._include_electrons = aux_data["include_electrons"]
        return obj


class CoulombPotential(HNCPotential):
    """
    A full Coulomb Potential.
    """

    __name__ = "CoulombPotential"

    @jax.jit
    def full_r(self, plasma_state, r: Quantity) -> Quantity:
        """
        .. math::

           q^2 / (4 jnp.pi \\varepsilon_0 * r)

        """
        _r = r[jnp.newaxis, jnp.newaxis, :]
        return self.q2(plasma_state) / (4 * jnp.pi * ureg.epsilon_0 * _r)

    @jax.jit
    def full_k(self, plasma_state, k: Quantity):
        _k = k[jnp.newaxis, jnp.newaxis, :]
        return self.q2(plasma_state) / ureg.vacuum_permittivity / _k**2

    @jax.jit
    def long_k(self, plasma_state, k: Quantity):
        """
        .. math::

            q^2 / (k^2 \\varepsilon_0) \\cdot (\\alpha^2 / (k^2 + \\alpha^2))

        """
        _k = k[jnp.newaxis, jnp.newaxis, :]

        return (
            self.q2(plasma_state)
            / (_k**2 * ureg.epsilon_0)
            * self.alpha(plasma_state) ** 2
            / (_k**2 + self.alpha(plasma_state) ** 2)
        )


class DebyeHueckelPotential(HNCPotential):
    """
    The Debye-Hückel screening potential as defined, e.g., in
    :cite:`Wunsch.2008`.
    """

    __name__ = "DebyeHueckelPotential"

    cite_keys = ["Wunsch.2008"]

    def check(self, plasma_state) -> None:
        if not hasattr(plasma_state, "screening_length"):
            logger.error(
                f"The PlasmaState {plasma_state} has no attribute 'screening_length, which is required for DebyeHueckelPotential."  # noqa: E501
            )

    def kappa(self, plasma_state):
        # This is called if kappa is defined per ion species
        if (
            isinstance(plasma_state.screening_length.magnitude, jnp.ndarray)
            and len(plasma_state.screening_length.shape) == 2
        ):
            return 1 / plasma_state.screening_length[:, :, jnp.newaxis]
        else:
            return 1 / plasma_state.screening_length

    @jax.jit
    def full_r(self, plasma_state, r):
        """
        .. math::

            \\frac{q^2}{(4 \\pi \\epsilon_0 r)} \\exp(-\\kappa r)

        """

        _r = r[jnp.newaxis, jnp.newaxis, :]

        return (
            self.q2(plasma_state)
            / (4 * jnp.pi * ureg.epsilon_0 * _r)
            * jnpu.exp(-self.kappa(plasma_state) * r)
        )

    @jax.jit
    def long_k(self, plasma_state, k):
        """
        See :cite:`Chapman.2015`, (Eqn 3.58)
        """

        _k = k[jnp.newaxis, jnp.newaxis, :]

        pref = self.q2(plasma_state) / (_k**2 * ureg.epsilon_0) * _k**2
        numerator = self.alpha(plasma_state) ** 2 + 2 * self.alpha(
            plasma_state
        ) * self.kappa(plasma_state)
        denumerator = (_k**2 + self.kappa(plasma_state) ** 2) * (
            _k**2 + (self.kappa(plasma_state) + self.alpha(plasma_state)) ** 2
        )

        return pref * numerator / denumerator

    @jax.jit
    def full_k(self, plasma_state, k):
        """
        See :cite:`Chapman.2015`, (Eqn 3.49)
        """
        _k = k[jnp.newaxis, jnp.newaxis, :]
        V_C = self.q2(plasma_state) / ureg.vacuum_permittivity / _k**2
        return V_C / (1 + (self.kappa(plasma_state) / _k) ** 2)


class KelbgPotential(HNCPotential):
    """
    Quantum diffraction potential suggested by Kelbg.
    See :cite:`Wunsch.2011` Eqn. 4.43, who cites :cite:`Kelbg.1963`, and
    :cite:`Schwarz.2007`, Eqn 14. We use the definition of
    :math:`\\lambda_{ab}` from :cite:`Schwarz.2007`.

    .. note::

        Only applicable for weakly coupled systems with :math:`\\Gamma < 1`.

    """

    __name__ = "KelbgPotential"
    cite_keys = [
        "Kelbg.1963",
        "Wunsch.2011",
        ("Schwarz.2007", "Definition of lambda_ab"),
    ]

    @jax.jit
    def full_r(self, plasma_state, r: Quantity) -> Quantity:
        """

        .. math::

            V_{a b}^{\\mathrm{Kelbg}}(r) =
            \\frac{q_{a}q_{b}}{4 \\pi \\varepsilon_0 r}
            \\left[1-\\exp\\left(-\\frac{r^2}{\\lambda_{a b}^2}\\right) +
            \\frac{\\sqrt\\pi r}{\\lambda_{a b}}
            \\left(1-\\mathrm{erf}
            \\left(\\frac{r}{\\lambda_{a b}}\\right)
            \\right)
            \\right]

        In the above equation, :math:`\\mathrm{erf}` is the Gaussian error
        function.

        For :math:`r\\rightarrow 0: V_{a b} \\rightarrow
        \\frac{q_{a}q_{b}\\sqrt{\\pi}}{4 \\pi \\varepsilon_0 \\lambda_{a b}}`.
        """

        _r = r[jnp.newaxis, jnp.newaxis, :]

        return (
            self.q2(plasma_state)
            / (4 * jnp.pi * ureg.epsilon_0 * _r)
            * (
                1
                - jnpu.exp(-(_r**2) / self.lambda_ab(plasma_state) ** 2)
                + (jnp.sqrt(jnp.pi) * _r / self.lambda_ab(plasma_state))
                * (
                    1
                    - jax.scipy.special.erf(
                        (_r / self.lambda_ab(plasma_state)).m_as(
                            ureg.dimensionless
                        )
                    )
                )
            )
        )

    @jax.jit
    def short_r(self, plasma_state, r: Quantity) -> Quantity:
        return self.full_r(plasma_state, r) - self.long_r(plasma_state, r)

    @jax.jit
    def long_r(self, plasma_state, r: Quantity) -> Quantity:
        """
        The long-range behavior of the Deutsch-Potentials should be the
        identical to the Coulomb-Potential. Therefore, define a long-range term
        with is the Coulomb-Potential, which also has the known Fourier
        transform.

        .. math::

           q^2 / (4 jnp.pi \\varepsilon_0 r) \\cdot (1 - \\exp(-\\alpha r))

        """
        _r = r[jnp.newaxis, jnp.newaxis, :]
        c_full_r = self.q2(plasma_state) / (4 * jnp.pi * ureg.epsilon_0 * _r)
        return c_full_r * (1 - jnpu.exp(-self.alpha(plasma_state) * _r))

    @jax.jit
    def long_k(self, plasma_state, k: Quantity):
        """
        The known Fourier transform of the Coulomb's potential long-range part.

        .. math::

            q^2 / (k^2 \\varepsilon_0) \\cdot (\\alpha^2 / (k^2 + \\alpha^2))

        """
        _k = k[jnp.newaxis, jnp.newaxis, :]

        return (
            self.q2(plasma_state)
            / (_k**2 * ureg.epsilon_0)
            * self.alpha(plasma_state) ** 2
            / (_k**2 + self.alpha(plasma_state) ** 2)
        )

    # ===
    # These short-and longrange terms were calculate by splitting the Kelbg
    # potential, directly. However, a Fourier transform of the long-range part
    # might not be easy. Instead, we leverage that for big k, the Kelbg and
    # Coulomb-Potential should be the same (see above).
    # ===
    # def short_r(self, plasma_state, r: Quantity) -> Quantity:
    #     """

    #     .. math::

    #         V_{a b}^{\\mathrm{Kelbg}}(r) =
    #         \\frac{q_{a}q_{b}}{4 \\pi \\varepsilon_0 r}
    #         \\left[
    #         \\frac{\\sqrt\\pi r}{\\lambda_{a b}}
    #         \\left(1-\\mathrm{erf}
    #         \\left(\\frac{r}{\\lambda_{a b}}\\right)
    #         \\right)
    #         \\right]

    #     In the above equation, :math:`\\mathrm{erf}` is the Gaussian error
    #     function.

    #     For :math:`r\\rightarrow 0: V_{a b} \\rightarrow
    #     \\frac{q_{a}q_{b}\\sqrt{\\pi}}{4 \\pi \\varepsilon_0 \\lambda_{a b}}`
    #     .
    #     """

    #     _r = r[jnp.newaxis, jnp.newaxis, :]

    #     return (
    #         self.q2(plasma_state)
    #         / (4 * jnp.pi * ureg.epsilon_0 * _r)
    #         * (
    #             (jnp.sqrt(jnp.pi) * _r / self.lambda_ab(plasma_state))
    #             * (
    #                 1
    #                 - jax.scipy.special.erf(
    #                     (_r / self.lambda_ab(plasma_state)).m_as(
    #                         ureg.dimensionless
    #                     )
    #                 )
    #             )
    #         )
    #     )

    # def long_r(self, plasma_state, r: Quantity) -> Quantity:
    #     """

    #     .. math::

    #         V_{a b}^{\\mathrm{Kelbg}}(r) =
    #         \\frac{q_{a}q_{b}}{4 \\pi \\varepsilon_0 r}
    #         \\left[1-\\exp\\left(-\\frac{r^2}{\\lambda_{a b}^2}\\right)
    #         \\right]

    #     In the above equation, :math:`\\mathrm{erf}` is the Gaussian error
    #     function.

    #     For :math:`r\\rightarrow 0: V_{a b} \\rightarrow
    #     \\frac{q_{a}q_{b}\\sqrt{\\pi}}{4 \\pi \\varepsilon_0 \\lambda_{a b}}`
    #     .
    #     """

    #     _r = r[jnp.newaxis, jnp.newaxis, :]

    #     return (
    #         self.q2(plasma_state)
    #         / (4 * jnp.pi * ureg.epsilon_0 * _r)
    #         * (1 - jnpu.exp(-(_r**2) / self.lambda_ab(plasma_state) ** 2))
    #     )


class KlimontovichKraeftPotential(HNCPotential):
    """
    Quantum diffraction potential suggested by Klimontovich and Kraeft.
    See :cite:`Schwarz.2007` Eqn (15) and :cite:`Wunsch.2011` Eqn. (4.43).

    .. note::

        This potential is only defined for electron-ion interactions. However,
        for the output to have the same shape as other potentials, we calculate
        it for all inputs. The most sensible treatment is to only use the
        off-diagonal entries for the `ei` Potential.

    """

    allowed_keys = [
        "electron-ion Potential",
    ]
    __name__ = "KlimontovichKraeftPotential"
    cite_keys = [
        "Klimontovich.1974",
        "Wunsch.2011",
        "Schwarz.2007",
    ]

    @jax.jit
    def full_r(self, plasma_state, r):
        """
        .. math::

            V_{e i}^{\\mathrm{KK}}(r)=-\\frac{k_{B}T\\xi_{e i}^{2}}{16}
            \\left[1+
            \\frac{4\\pi\\varepsilon_0 k_{B}T\\xi_{e i}^{2}}{16Z e^{2}}
            r\\right]^{-1}


        In the above equation, :math:`\\xi{e i} = (Z e^2 \\beta) / (\\lambda_{e
        i} 4 \\pi \\varepsilon_0)`.

        :math:`Z e^{2} = q^2`, and :math:`\\beta = 1/(k_B T)` (see note above)
        """
        _r = r[jnp.newaxis, jnp.newaxis, :]

        beta = 1 / (ureg.k_B * self.T(plasma_state))
        xi = (
            self.q2(plasma_state)
            * beta
            / (4 * jnp.pi * ureg.epsilon_0 * self.lambda_ab(plasma_state))
        )
        pref = -(ureg.k_B * self.T(plasma_state) * xi**2 / 16)
        factor = ureg.k_B * self.T(plasma_state) * xi**2
        denominator = (
            16
            * jnpu.absolute(self.q2(plasma_state))
            / (4 * jnp.pi * ureg.epsilon_0)
        )
        return pref * (1 + (factor / denominator) * _r) ** (-1)


class DeutschPotential(HNCPotential):
    """
    Quantum diffraction potential suggested by C. Deutsch. See
    :cite:`Wunsch.2011` Eqn. 4.43, who cites :cite:`Deutsch.1977`.

    .. math::

        V_{a b}^{\\mathrm{Deutsch}}(r) =
        \\frac{q_{a}q_{b}}{4 \\pi \\varepsilon_0 r}
        \\left[1-\\exp\\left(-\\frac{r}{\\lambda_{a b}}\\right)\\right]

    """

    __name__ = "DeutschPotential"
    cite_keys = [
        "Kelbg.1963",
        "Wunsch.2011",
        ("Schwarz.2007", "Definition of lambda_ab"),
    ]

    @jax.jit
    def full_r(self, plasma_state, r):
        _r = r[jnp.newaxis, jnp.newaxis, :]
        return (
            self.q2(plasma_state)
            / (4 * jnp.pi * ureg.epsilon_0 * _r)
            * (
                1
                - jnpu.exp(
                    -_r / (self.lambda_ab(plasma_state) / jnp.sqrt(jnp.pi))
                )
            )
        )

    @jax.jit
    def full_k(self, plasma_state, k: Quantity):
        """
        .. math::

            q^2 / (k^2 * \\varepsilon_0) *
            ((1/\\lambda{a b})**2 / (k^2 + (1/\\lambda{a b})^2))

        """
        _k = k[jnp.newaxis, jnp.newaxis, :]

        return (
            self.q2(plasma_state)
            / (_k**2 * ureg.epsilon_0)
            * (1 / self.lambda_ab(plasma_state)) ** 2
            / (
                _k**2
                + (1 / (self.lambda_ab(plasma_state) / jnp.sqrt(jnp.pi))) ** 2
            )
        )

    @jax.jit
    def short_r(self, plasma_state, r: Quantity) -> Quantity:
        return self.full_r(plasma_state, r) - self.long_r(plasma_state, r)

    @jax.jit
    def long_r(self, plasma_state, r: Quantity) -> Quantity:
        """
        The long-range behavior of the Deutsch-Potentials should be the
        identical to the Coulomb-Potential. Therefore, define a long-range term
        with is the Coulomb-Potential, which also has the known Fourier
        transform.

        .. math::

           q^2 / (4 jnp.pi \\varepsilon_0 r) \\cdot (1 - \\exp(-\\alpha r))

        """
        _r = r[jnp.newaxis, jnp.newaxis, :]
        c_full_r = self.q2(plasma_state) / (4 * jnp.pi * ureg.epsilon_0 * _r)
        return c_full_r * (1 - jnpu.exp(-self.alpha(plasma_state) * _r))

    @jax.jit
    def long_k(self, plasma_state, k: Quantity):
        """
        The known Fourier transform of the Coulomb's potential long-range part.

        .. math::

            q^2 / (k^2 \\varepsilon_0) \\cdot (\\alpha^2 / (k^2 + \\alpha^2))

        """
        _k = k[jnp.newaxis, jnp.newaxis, :]

        return (
            self.q2(plasma_state)
            / (_k**2 * ureg.epsilon_0)
            * self.alpha(plasma_state) ** 2
            / (_k**2 + self.alpha(plasma_state) ** 2)
        )


class EmptyCorePotential(HNCPotential):
    """
    The Empty core potential, which is essentially a
    :py:class:`~.CoulombPotential` for all radii bigger
    than :math:`r_\\text{cut}`. For all radii smaller
    than :math:`r_\\text{cut}`, the potential is forced to zero.

    We define :math:`r_\\text{cut}` in the :py:class:`jaxrts.PlasmaState`.

    .. warning::

       This potential is only defined for the electron-ion interaction. Hence,
       it will always and automatically set ~:py:meth:`include_electrons` to
       ``"SpinAveraged"``. For the rest, we return a Coulomb-potential -- but
       this is really just to be compatible with the other potentials defined
       here.

    """

    allowed_keys = [
        "electron-ion Potential",
    ]
    __name__ = "EmptyCorePotential"
    cite_keys = ["Gericke.2010"]

    def __init__(
        self,
    ):
        super().__init__()
        self.include_electrons = "SpinAveraged"

    @jax.jit
    def full_r(self, plasma_state, r: Quantity) -> Quantity:
        """
        .. math::

           \\begin{cases}
           q^2 / (4 jnp.pi \\varepsilon_0 * r) \\text{if} r \\geq r_{cut}\\\\
           0 \\text{else}
           \\end{cases}

        """
        _r = r[jnp.newaxis, jnp.newaxis, :]
        return jnp.heaviside(
            (_r - self.r_cut(plasma_state)).m_as(ureg.angstrom),
            0.0,
        ) * (self.q2(plasma_state) / (4 * jnp.pi * ureg.epsilon_0 * _r))

    @jax.jit
    def full_k(self, plasma_state, k):
        _k = k[jnp.newaxis, jnp.newaxis, :]
        return (
            self.q2(plasma_state)
            / ureg.vacuum_permittivity
            / _k**2
            * jnpu.cos(_k * self.r_cut(plasma_state))
        )

    @jax.jit
    def long_r(self, plasma_state, r: Quantity) -> Quantity:
        return self.full_r(plasma_state, r)

    @jax.jit
    def short_r(self, plasma_state, r: Quantity) -> Quantity:
        return (
            jnp.zeros([*self.q2(plasma_state).shape[:2], len(r)])
            * ureg.electron_volt
        )

    @jax.jit
    def long_k(self, plasma_state, k: Quantity) -> Quantity:
        return self.full_k(plasma_state, k)

    @jax.jit
    def short_k(self, plasma_state, k: Quantity) -> Quantity:
        return (
            jnp.zeros([*self.q2(plasma_state).shape[:2], len(k)])
            * ureg.electron_volt
            * ureg.angstrom**3
        )


class YukawaShortRangeRepulsion(HNCPotential):
    """
    Yukawa Potential with a short-range repulsion as used in
    :cite:`Fletcher.2015`.
    """

    __name__ = "YukawaShortRangeRepulsion"
    cite_keys = ["Wunsch.2009", "Fletcher.2015"]

    def __init__(
        self,
        sigma: Quantity = 0.0 * ureg.angstrom,
    ):
        #: This gives the strength of the repulsion, and is a fitting parameter
        self.sigma = sigma
        super().__init__()

    def kappa(self, plasma_state):
        # This is called if kappa is defined per ion species
        if (
            isinstance(plasma_state.screening_length.magnitude, jnp.ndarray)
            and len(plasma_state.screening_length.shape) == 2
        ):
            return 1 / plasma_state.screening_length[:, :, jnp.newaxis]
        else:
            return 1 / plasma_state.screening_length

    @jax.jit
    def full_r(self, plasma_state, r: Quantity) -> Quantity:
        """
        .. math::

           q^2 / (4 jnp.pi \\varepsilon_0 * r) \\exp(-\\kappa r) +
           \\frac{\\sigma^4}{r^4}

        """
        _r = r[jnp.newaxis, jnp.newaxis, :]
        return (
            self.q2(plasma_state)
            / (4 * jnp.pi * ureg.epsilon_0 * _r)
            * jnpu.exp(-self.kappa(plasma_state) * r)
        ) + 1 * ureg.k_B * self.T(plasma_state) * self.sigma**4 / _r

    @jax.jit
    def long_r(self, plasma_state, r):
        """
        .. math::

            \\frac{q^2}{(4 \\pi \\epsilon_0 r)} \\exp(-\\kappa r)

        """

        _r = r[jnp.newaxis, jnp.newaxis, :]

        return (
            self.q2(plasma_state)
            / (4 * jnp.pi * ureg.epsilon_0 * _r)
            * jnpu.exp(-self.kappa(plasma_state) * r)
        )

    @jax.jit
    def short_r(self, plasma_state, r: Quantity) -> Quantity:
        _r = r[jnp.newaxis, jnp.newaxis, :]
        return (self.sigma**4 / _r**4) * 1 * ureg.k_B * self.T(plasma_state)

    @jax.jit
    def long_k(self, plasma_state, k):
        _k = k[jnp.newaxis, jnp.newaxis, :]
        V_C = self.q2(plasma_state) / ureg.vacuum_permittivity / _k**2
        return V_C / (1 + (self.kappa(plasma_state) / _k) ** 2)

    # The following is required to jit a state
    def _tree_flatten(self):
        children = (self._transform_r, self.sigma)
        aux_data = {
            "include_electrons": self.include_electrons,
            "model_key": self.model_key,
        }  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj._transform_r, obj.sigma = children
        obj.model_key = aux_data["model_key"]
        obj.include_electrons = aux_data["include_electrons"]
        return obj


class SoftCorePotential(HNCPotential):
    """
    Comparable to :py:class:`EmptyCorePotential`, but to circumvent the hard
    cutoff, rather introduce a soft, exponential cutoff. It's strength is given
    by the attribute :py:attr:`~.beta`. See :cite:`Gericke.2010`.

    We define `r_cut` in the :py:class:`jaxrts.PlasmaState`.

    .. warning::

       This potential is only defined for the electron-ion interaction. Hence,
       it will always and automatically set ~:py:meth:`include_electrons` to
       ``"SpinAveraged"``. For the rest, we return a Coulomb-potential -- but
       this is really just to be compatible with the other potentials defined
       here.

    """

    allowed_keys = [
        "electron-ion Potential",
    ]
    __name__ = "SoftCorePotential"
    cite_keys = ["Gericke.2010"]

    def __init__(
        self,
        beta: float = 4.0,
    ):
        #: This is the exponent which gives the steepness of the soft-core's
        #: edge
        self.beta = beta
        super().__init__()
        self.include_electrons = "SpinAveraged"

    @jax.jit
    def full_r(self, plasma_state, r: Quantity) -> Quantity:
        """
        .. math::

           q^2 / (4 jnp.pi \\varepsilon_0 * r)
           \\left[1 -
           \\exp\\left(-\\frac{r^\\beta}{r_{cut}^\\beta}\\right)\\right]

        """
        _r = r[jnp.newaxis, jnp.newaxis, :]
        exp_part = 1 - jnpu.exp(
            -(
                (_r / self.r_cut(plasma_state)).m_as(ureg.dimensionless)
                ** self.beta
            )
        )
        return (
            self.q2(plasma_state)
            / (4 * jnp.pi * ureg.epsilon_0 * _r)
            * exp_part
        )

    @jax.jit
    def long_r(self, plasma_state, r: Quantity) -> Quantity:
        return self.full_r(plasma_state, r)

    @jax.jit
    def short_r(self, plasma_state, r: Quantity) -> Quantity:
        return jnp.zeros([*self.q.shape[:2], len(r)]) * ureg.electron_volt

    @jax.jit
    def full_k(self, plasma_state, k):
        _k = k[jnp.newaxis, jnp.newaxis, :]
        # Define a auxiliary short-range version of this potential; While it is
        # not used, actually, it is easier to Fourier transform.
        exp_part = jnpu.exp(
            -(
                (self._transform_r / self.r_cut(plasma_state)).m_as(
                    ureg.dimensionless
                )
                ** self.beta
            )
        )
        V_s_transform_r = (
            self.q2(plasma_state)
            / (4 * jnp.pi * ureg.epsilon_0 * self._transform_r)
            * exp_part
        )
        _V_s_transform_k, _k = transformPotential(
            V_s_transform_r, self._transform_r
        )
        V_s_k = hnc_interp(k, _k, _V_s_transform_k)

        # Subtract this from the full, known solution of the Coulomb Potential
        # in k space.

        _k = k[jnp.newaxis, jnp.newaxis, :]
        V_full_Coulomb_k = (
            self.q2(plasma_state) / ureg.vacuum_permittivity / _k**2
        )

        return V_full_Coulomb_k - V_s_k

    @jax.jit
    def long_k(self, plasma_state, k: Quantity) -> Quantity:
        return self.full_k(k)

    @jax.jit
    def short_k(self, plasma_state, k: Quantity) -> Quantity:
        return (
            jnp.zeros([*self.q.shape[:2], len(k)])
            * ureg.electron_volt
            * ureg.angstrom**3
        )

    # The following is required to jit a state
    def _tree_flatten(self):
        children = (self._transform_r, self.beta)
        aux_data = {
            "include_electrons": self.include_electrons,
            "model_key": self.model_key,
        }  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj._transform_r, obj.beta = children
        obj.model_key = aux_data["model_key"]
        obj.include_electrons = aux_data["include_electrons"]
        return obj


class PauliClassicalMap(HNCPotential):
    """
    Calculates the Pauli potential from the exact non-interacting
    UEG pair distribution function by inverting the
    Hyper-Netted-Chain calculations:cite:`DharmaWardana.2012`.

    """

    __name__ = "PauliClassicalMap"

    cite_keys = ["Bredow.2015", "DharmaWardana.2000", "DharmaWardana.2012"]

    def __init__(self):
        """
        Sets :py:attr:`~PauliClassicalMap.include_electrons` to
        ``"SpinSeparated"``, automatically.
        """

        super().__init__()
        self.include_electrons = "SpinSeparated"

    @jax.jit
    def full_k(self, plasma_state, k):

        dk = k[1] - k[0]
        dr = jnp.pi / (len(k) * dk)
        _r = jnp.pi / k[-1] + jnp.arange(len(k)) * dr

        pot = 21
        r_calc = jnpu.linspace(1e-3 * ureg.a0, 1e4 * ureg.a0, 2**pot)

        if self.include_electrons == "SpinSeparated":

            k_, exchange, _ = pauli_potential_from_classical_map_SpinSeparated(
                r_calc, _r, plasma_state.n_e, plasma_state.T_e
            )

            # Set the parts that are not electron_electron exchange to zero
            potential = jnp.zeros(
                (
                    len(plasma_state.ions) + 2,
                    len(plasma_state.ions) + 2,
                    len(k),
                )
            )
            potential = (
                (
                    potential.at[-2:, -2:, :].set(
                        exchange.m_as(ureg.electron_volt * ureg.angstrom**3)
                    )
                )
                * ureg.electron_volt
                * ureg.angstrom**3
            )
        elif self.include_electrons == "SpinAveraged":
            k_, exchange, g0_T_r = (
                pauli_potential_from_classical_map_SpinAveraged(
                    r_calc, _r, plasma_state.n_e, plasma_state.T_e
                )
            )

            # Set the parts that are not electron_electron exchange to zero
            potential = jnp.zeros(
                (
                    len(plasma_state.ions) + 1,
                    len(plasma_state.ions) + 1,
                    len(k),
                )
            )
            potential = (
                (
                    potential.at[-1:, -1:, :].set(
                        exchange.m_as(ureg.electron_volt * ureg.angstrom**3)
                    )
                )
                * ureg.electron_volt
                * ureg.angstrom**3
            )

        return potential

    @jax.jit
    def full_r(self, plasma_state, r):

        dr = r[1] - r[0]
        dk = jnp.pi / (len(r) * dr)
        k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk

        V_P_r = _3Dfour(r, k, self.full_k(plasma_state, k)) / (2 * jnp.pi) ** 3

        return V_P_r

    def _tree_flatten(self):
        children = (self._transform_r,)
        aux_data = {
            "include_electrons": self.include_electrons,
            "model_key": self.model_key,
        }  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        (obj._transform_r,) = children
        obj.model_key = aux_data["model_key"]
        obj.include_electrons = aux_data["include_electrons"]
        return obj


class SpinSeparatedEEExchange(HNCPotential):
    """
    See :cite:`Wunsch.2008`, Eqn (17), Citing :cite:`Huang.1987`. Eqn. (9.57),
    however, there is a factor of :math:`\\sqrt{\\pi}` difference in the
    definition of lambda. (one is from :cite:`Deutsch.1982` presenting a
    reduced quantity :math:`\\lambda_{ee}`, and then them having another
    factor of :math:`\\sqrt{2}` in the mass term. We here present the
    corrected version of the equation, which reproduces
    Figure 6. in :cite:`Wunsch.2008` and results in a higher repulsion
    than the :py:class:`~.SpinAveragedEEExchange`,
    which should be expected.
    """

    __name__ = "SpinSeparatedEEExchange"
    cite_keys = ["Wunsch.2008", "Huang.1987", "Deutsch.1982"]

    def __init__(
        self,
    ):
        """
        Sets :py:attr:`~SpinSeparatedEEExchange.include_electrons` to
        ``"SpinSeparated"``, automatically.
        """
        super().__init__()
        self.include_electrons = "SpinSeparated"

    @jax.jit
    def full_r(self, plasma_state, r):
        _r = r[jnp.newaxis, jnp.newaxis, :]
        exchange = (
            (-1 * ureg.k_B * self.T(plasma_state))
            * jnpu.log(
                1 - jnpu.exp(-(_r**2 / self.lambda_ab(plasma_state) ** 2))
            )
            * jnp.eye(len(self.mu(plasma_state)))[:, :, jnp.newaxis]
        )
        # Set the parts that are not electron_electron exchange to zero
        exchange = (
            exchange.m_as(ureg.electron_volt)
            .at[: len(plasma_state.ions), : len(plasma_state.ions), :]
            .set(
                jnp.zeros(
                    (len(plasma_state.ions), len(plasma_state.ions), len(_r))
                )
            )
            * ureg.electron_volt
        )
        return exchange


class SpinAveragedEEExchange(HNCPotential):
    """
    See :cite:`Wunsch.2008`, Eqn (18).
    """

    __name__ = "SpinAveragedEEExchange"
    cite_keys = ["Wunsch.2008"]

    @jax.jit
    def full_r(self, plasma_state, r):
        _r = r[jnp.newaxis, jnp.newaxis, :]

        exchange = (
            (ureg.k_B * self.T(plasma_state))
            * jnp.log(2)
            * jnpu.exp(
                -1
                / (jnp.pi * jnp.log(2))
                * (
                    (
                        _r / (self.lambda_ab(plasma_state) / jnp.sqrt(jnp.pi))
                    ).m_as(ureg.dimensionless)
                )
                ** 2
            )
            * jnp.eye(plasma_state.nions + 1)[:, :, jnp.newaxis]
        )
        # Set the parts that are not electron_electron exchange to zero
        exchange = (
            exchange.m_as(ureg.electron_volt)
            .at[: plasma_state.nions, : plasma_state.nions, :]
            .set(jnp.zeros((plasma_state.nions, plasma_state.nions, len(_r))))
            * ureg.electron_volt
        )
        return exchange

    @jax.jit
    def full_k(self, plasma_state, k):

        dk = k[1] - k[0]
        dr = jnp.pi / (len(k) * dk)
        r = jnp.pi / k[-1] + jnp.arange(len(k)) * dr

        V_P_k = _3Dfour(k, r, self.full_r(plasma_state, r))

        return V_P_k


@jax.jit
def transformPotential(V, r) -> Quantity:
    """
    ToDo: Test this, potentially, there is something wrong, here.
    """
    dr = r[1] - r[0]
    dk = jnp.pi / (len(r) * dr)
    k = jnp.pi / r[-1] + jnp.arange(len(r)) * dk
    V_k = _3Dfour(
        k,
        r,
        V,
    )
    return V_k, k


_all_hnc_potentals = [
    CoulombPotential,
    DebyeHueckelPotential,
    DeutschPotential,
    EmptyCorePotential,
    KelbgPotential,
    KlimontovichKraeftPotential,
    PotentialSum,
    ScaledPotential,
    SoftCorePotential,
    SpinAveragedEEExchange,
    PauliClassicalMap,
    SpinSeparatedEEExchange,
    YukawaShortRangeRepulsion,
]

for _pot in _all_hnc_potentals:
    jax.tree_util.register_pytree_node(
        _pot,
        _pot._tree_flatten,
        _pot._tree_unflatten,
    )
