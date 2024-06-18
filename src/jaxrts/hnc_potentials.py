"""
HNC Potentials
==============

This module contains a set of Parameters to be used when performing HNC
calculations. This includes that the potential is split into a long-range and a
shortrange part and is also Fourier-transformed to k-space.
"""

import abc
import logging

from jax import numpy as jnp
import jax.interpreters
import jax
import jpu

from jaxrts.units import ureg, Quantity, to_array
from jaxrts.hypernetted_chain import _3Dfour, mass_weighted_T, hnc_interp

logger = logging.getLogger(__name__)


@jax.jit
def construct_alpha_matrix(n: jnp.ndarray | Quantity):
    d = jpu.numpy.cbrt(
        3 / (4 * jnp.pi * (n[:, jnp.newaxis] * n[jnp.newaxis, :]) ** (1 / 2))
    )

    return 2 / d


@jax.jit
def construct_q_matrix(q: jnp.ndarray) -> jnp.ndarray:
    return jpu.numpy.outer(q, q)


class HNCPotential(metaclass=abc.ABCMeta):
    """
    Potentials, intended to be used in the HNC scheme. Per default, the results
    of methonds evaluating this Potential (in k or r space), return a
    :math:`(n\\times n\\times m)` matrix, where ``n`` is the number of ion
    species and ``m`` is the number of r or k points.
    However, if :py:attr:`~.include_electrons` is ``True``, electrons are added
    as another ion species, and so one the first two dimensions get an
    additional entry.
    """

    allowed_keys = [
        "ion-ion Potential",
        "electron-ion Potential",
        "electron-electron Potential",
    ]

    def __init__(
        self,
    ):
        self._transform_r = jpu.numpy.linspace(1e-3, 1e3, 2**14) * ureg.a_0

        self.model_key = ""

        #: If `True`, the electrons are added as the n+1th ion species to the
        #: potential. The relevant entries are the last row and column,
        #: respectively (i.e., the colored lines in the image above).
        self.include_electrons: bool = False

    def check(self, plasma_state) -> None:
        """
        Test if the HNCPotential is applicable to the PlasmaState. Might raise
        logged messages and errors. Is automatically called after
        :py:meth:`~.__init__`.
        """
        pass

    def prepare(self, plasma_state) -> None:
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
        return self.full_r(plasma_state, r) * jpu.numpy.exp(
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
            1 - jpu.numpy.exp(-self.alpha(plasma_state) * _r)
        )

    @jax.jit
    def short_k(self, plasma_state, k: Quantity) -> Quantity:
        """
        The Foutier transform of :py:meth:`~short_r`.
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
        if self.include_electrons:
            Z = to_array([*plasma_state.Z_free, -1])
        else:
            Z = plasma_state.Z_free
        charge = construct_q_matrix(Z * ureg.elementary_charge)
        return charge[:, :, jnp.newaxis]

    def alpha(self, plasma_state):
        if self.include_electrons:
            n = to_array([*plasma_state.n_i, plasma_state.n_e])
        else:
            n = plasma_state.n_i
        a = construct_alpha_matrix(n)
        return a[:, :, jnp.newaxis]

    def mu(self, plasma_state):
        """
        The geometric mean of two masses (or reciprocal sum)
        """
        if self.include_electrons:
            m = to_array([*plasma_state.atomic_masses, 1 * ureg.electron_mass])
        else:
            m = plasma_state.atomic_masses
        mu = jpu.numpy.outer(m, m) / (m[:, jnp.newaxis] + m[jnp.newaxis, :])
        return mu[:, :, jnp.newaxis]

    def T(self, plasma_state):
        """
        The mass_weighted temperature average of a pair, according to
        :cite:`Schwarz.2007`.

        .. math::

           \\bar{T}_{ab} = \\frac{T_a m_b + T_b m_a}{m_a m_b}

        """
        if self.include_electrons:
            m = to_array([*plasma_state.atomic_masses, 1 * ureg.electron_mass])
            T = to_array([*plasma_state.T_i, plasma_state.T_e])
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
        electron-electron pair.
        """
        r = jnp.zeros_like(self.q2(plasma_state).magnitude, dtype=float)
        if self.include_electrons:
            r = r.at[:-1, -1, :].set(
                plasma_state.ion_core_radius.m_as(ureg.angstrom)
            )
            r = r.at[-1, :-1, :].set(
                plasma_state.ion_core_radius.m_as(ureg.angstrom)
            )
        return r * ureg.angstrom

    def lambda_ab(self, plasma_state):
        # Compared to Gregori.2003, there is a pi missing
        l_ab = ureg.hbar * jpu.numpy.sqrt(
            1 / (2 * self.mu(plasma_state) * ureg.k_B * self.T(plasma_state))
        )
        return l_ab

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

            q^2 / (k^2 * \\varepsilon_0) * (\\alpha**2 / (k^2 + \\alpha^2))

        """
        _k = k[jnp.newaxis, jnp.newaxis, :]

        return (
            self.q2(plasma_state)
            / (_k**2 * ureg.epsilon_0)
            * self.alpha(plasma_state) ** 2
            / (_k**2 + self.alpha(plasma_state) ** 2)
        )


class DebyeHuckelPotential(HNCPotential):
    __name__ = "DebyeHuckelPotential"

    def check(self, plasma_state) -> None:
        if not hasattr(plasma_state, "DH_screening_length"):
            logger.error(
                f"The PlasmaState {plasma_state} has no attribute 'DH_screening_length, which is required for DebyeHuckelPotential."  # noqa: 501
            )

    def kappa(self, plasma_state):
        # This is called if kappa is defined per ion species
        if (
            isinstance(plasma_state.DH_screening_length.magnitude, jnp.ndarray)
            and len(plasma_state.DH_screening_length.shape) == 2
        ):
            return 1 / plasma_state.DH_screening_length[:, :, jnp.newaxis]
        else:
            return 1 / plasma_state.DH_screening_length

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
            * jpu.numpy.exp(-self.kappa(plasma_state) * r)
        )

    @jax.jit
    def long_k(self, plasma_state, k):

        _k = k[jnp.newaxis, jnp.newaxis, :]

        pref = self.q2(plasma_state) / (_k**2 * ureg.epsilon_0) * _k**2
        numerator = self.alpha(plasma_state) ** 2 + 2 * self.alpha(
            plasma_state
        ) * self.kappa(plasma_state)
        denumerator = (_k**2 + self.kappa(plasma_state) ** 2) * (
            _k**2 + (self.kappa(plasma_state) + self.alpha(plasma_state)) ** 2
        )

        return pref * numerator / denumerator


class KelbgPotential(HNCPotential):
    """
    See :cite:`Wunsch.2011` Eqn. 4.43, who cites :cite:`Kelbg.1963`, and
    :cite:`Schwarz.2007`, Eqn 14. We use the definition of lambda_ab from
    :cite:`Schwarz.2007`.

    .. note::

        Only applicable for weakly coupled systems with :math:`\\Gamma < 1`.

    """

    __name__ = "KelbgPotential"

    @jax.jit
    def full_r(self, plasma_state, r: Quantity) -> Quantity:
        """

        .. math::

            V_{a b}^{\\mathrm{Deutsch}}(r) =
            \\frac{q_{a}q_{b}}{4 \\pi \\varepsilon_0 r}
            \\left[1-\\exp\\left(-\\frac{r^2}{\\lambda_{a b}^2}\\right) +
            \\frac{\\sqrt\\pi r}{\\lambda_{a b}}
            \\left(1-\\mathrm{erf}
            \\left(\\frac{r}{\\lambda_{a b}}\\right)
            \\right)
            \\right]

        In the aboce equation, :math:`\\mathrm{erf}` is the Gaussian error
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
                - jpu.numpy.exp(-(_r**2) / self.lambda_ab(plasma_state) ** 2)
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


class KlimontovichKraeftPotential(HNCPotential):
    """
    See ::cite:`Schwarz.2007` Eqn 15 and cite:`Wunsch.2011` Eqn. 4.43.

    .. note::

        This potential is only defined for electron-ion interactions. However,
        for the output to have the same shape as other potentials, we calculate
        it for all inputs. The most sensible treatment is to only use the
        off-diagnonal entries for the `ei` Potential.

    """

    allowed_keys = [
        "electron-ion Potential",
    ]
    __name__ = "KlimontovichKraeftPotential"

    @jax.jit
    def full_r(self, plasma_state, r):
        """
        .. math::

            V_{e i}^{\\mathrm{KK}}(r)=-\\frac{k_{B}T\\xi_{e i}^{2}}{16}
            \\left[1+
            \\frac{4\\pi\\varepsilon_0 k_{B}T\\xi_{e i}^{2}}{16Z e^{2}}
            r\\right]^{-1}


        In the aboce equation, :math:`\\xi{e i} = (Z e^2 \\beta) / (\\lambda_{e
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
            * jpu.numpy.absolute(self.q2(plasma_state))
            / (4 * jnp.pi * ureg.epsilon_0)
        )
        return pref * (1 + (factor / denominator) * _r) ** (-1)


class DeutschPotential(HNCPotential):
    """
    See :cite:`Wunsch.2011` Eqn. 4.43, who cites :cite:`Deutsch.1977`.

    .. math::

        V_{a b}^{\\mathrm{Deutsch}}(r) =
        \\frac{q_{a}q_{b}}{4 \\pi \\varepsilon_0 r}
        \\left[1-\\exp\\left(-\\frac{r}{\\lambda_{a b}}\\right)\\right]

    """

    __name__ = "DeutschPotential"

    @jax.jit
    def full_r(self, plasma_state, r):
        _r = r[jnp.newaxis, jnp.newaxis, :]
        return (
            self.q2(plasma_state)
            / (4 * jnp.pi * ureg.epsilon_0 * _r)
            * (1 - jpu.numpy.exp(-_r / self.lambda_ab(plasma_state)))
        )


class EmptyCorePotential(HNCPotential):
    """
    The Empty core potential, which is essentially a
    :py:class:`~.CoulombPotential` for all radii bitter than `r_cut`.
    For all radii smaller than `r_cut` (this is the short-range part of the
    potential, for now), the potential is forced to zero.

    We definie `r_cut` in the :py:class:`jaxrts.PlasmaState`.

    .. warning::

       This potential is only defined for the electron-ion interaction. Hence,
       it will always and automatically set ~:py:meth:`include_electrons` to
       ``True``. For the rest, we reurn a Coulomb-potential -- but this is
       really just to be compatible with the other potentials defined here.

    """

    allowed_keys = [
        "electron-ion Potential",
    ]
    __name__ = "EmptyCorePotential"

    def __init__(
        self,
    ):
        super().__init__()
        self.include_electrons = True

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
    def long_r(self, plasma_state, r: Quantity) -> Quantity:
        return self.full_r(plasma_state, r)

    @jax.jit
    def short_r(self, plasma_state, r: Quantity) -> Quantity:
        return (
            jnp.zero(*self.q2(plasma_state).shape[:2], len(r))
            * ureg.electron_volt
        )

    @jax.jit
    def full_k(self, plasma_state, k):
        _k = k[jnp.newaxis, jnp.newaxis, :]
        return (
            self.q2(plasma_state)
            / ureg.vacuum_permittivity
            / _k**2
            * jpu.numpy.cos(_k * self.r_cut(plasma_state))
        )

    @jax.jit
    def long_k(self, plasma_state, k: Quantity) -> Quantity:
        return self.full_k(plasma_state, k)

    @jax.jit
    def short_k(self, plasma_state, k: Quantity) -> Quantity:
        return (
            jnp.zero(*self.q2(plasma_state).shape[:2], len(k))
            * ureg.electron_volt
            * ureg.angstrom**3
        )


class SoftCorePotential(HNCPotential):
    """
    This potential is very comparable to :py:class:`EmptyCorePotential`, but to
    circumvent the hard cutoff, we rather introduce a soft cutoff, exponential
    cutoff. It's strength is given by the attribute :py:attr:`~.beta`.
    See :cite:`Gericke.2010`.

    We definie `r_cut` in the :py:class:`jaxrts.PlasmaState`.

    .. warning::

       This potential is only defined for the electron-ion interaction. Hence,
       it will always and automatically set ~:py:meth:`include_electrons` to
       ``True``. For the rest, we reurn a Coulomb-potential -- but this is
       really just to be compatible with the other potentials defined here.

    """

    allowed_keys = [
        "electron-ion Potential",
    ]
    __name__ = "SoftCorePotential"

    def __init__(
        self,
        beta: float = 4.0,
    ):
        #: This is the exponent which gives the steepness of the soft-core's
        #: edge
        self.beta = beta
        super().__init__()
        self.include_electrons = True

    @jax.jit
    def full_r(self, plasma_state, r: Quantity) -> Quantity:
        """
        .. math::

           q^2 / (4 jnp.pi \\varepsilon_0 * r)
           \\left[1 -
           \\exp\\left(-\\frac{r^\\beta}{r_{cut}^\\beta}\\right)\\right]

        """
        _r = r[jnp.newaxis, jnp.newaxis, :]
        exp_part = 1 - jpu.numpy.exp(
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
        return jnp.zero(*self.q.shape[:2], len(r)) * ureg.electron_volt

    @jax.jit
    def full_k(self, plasma_state, k):
        _k = k[jnp.newaxis, jnp.newaxis, :]
        # Define a auxiliary short-range version of this potential; While it is
        # not used, acutally, it is easier to Fourier transform.
        exp_part = jpu.numpy.exp(
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
            jnp.zero(*self.q.shape[:2], len(k))
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


for pot in [
    CoulombPotential,
    DebyeHuckelPotential,
    DeutschPotential,
    EmptyCorePotential,
    KelbgPotential,
    KlimontovichKraeftPotential,
    SoftCorePotential,
]:
    jax.tree_util.register_pytree_node(
        pot,
        pot._tree_flatten,
        pot._tree_unflatten,
    )