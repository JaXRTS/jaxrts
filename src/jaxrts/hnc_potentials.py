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
        state,
        model_key="",
    ):
        self.state = state
        self._transform_r = jpu.numpy.linspace(1e-3, 1e3, 2**14) * ureg.a_0

        self.model_key = model_key

        #: If `True`, the electrons are added as the n+1th ion species to the
        #: potential. The relevant entries are the last row and column,
        #: respectively (i.e., the colored lines in the image above).
        self.include_electrons: bool = False
        self.check()

    def check(self) -> None:
        """
        Test if the HNCPotential is applicable to the PlasmaState. Might raise
        logged messages and errors. Is automatically called after
        :py:meth:`~.__init__`.
        """
        pass

    @abc.abstractmethod
    def full_r(self, r: Quantity) -> Quantity: ...

    @jax.jit
    def short_r(self, r: Quantity) -> Quantity:
        """
        This is the short-range part of :py:meth:`full_r`:

        .. math::

            V(r) \\cdot \\exp(-\\alpha r)

        """
        _r = r[jnp.newaxis, jnp.newaxis, :]
        return self.full_r(r) * jpu.numpy.exp(-self.alpha * _r)

    @jax.jit
    def long_r(self, r: Quantity) -> Quantity:
        """
        This is the long-range part of :py:meth:`full_r`:

        .. math::

            V(r) * (1 - \\exp(-\\alpha r))

        """
        _r = r[jnp.newaxis, jnp.newaxis, :]
        print(self.alpha)
        return self.full_r(r) * (1 - jpu.numpy.exp(-self.alpha * _r))

    @jax.jit
    def short_k(self, k: Quantity) -> Quantity:
        """
        The Foutier transform of :py:meth:`~short_r`.
        """
        V_k, _k = transformPotential(
            self.short_r(self._transform_r), self._transform_r
        )
        return hnc_interp(k, _k, V_k)

    @jax.jit
    def long_k(self, k: Quantity) -> Quantity:
        """
        The Foutier transform of :py:meth:`~short_k`.
        """
        V_k, _k = transformPotential(
            self.long_r(self._transform_r), self._transform_r
        )
        return hnc_interp(k, _k, V_k)

    @jax.jit
    def full_k(self, k):
        return self.short_k(k) + self.long_k(k)

    @property
    def q2(self):
        """
        This is :math:`q^2`!
        """
        if self.include_electrons:
            Z = to_array([*self.state.Z_free, -1])
        else:
            Z = self.state.Z_free
        charge = construct_q_matrix(Z * ureg.elementary_charge)
        return charge[:, :, jnp.newaxis]

    @property
    def alpha(self):
        if self.include_electrons:
            n = to_array([*self.state.n_i, self.state.n_e])
        else:
            n = self.state.n_i
        a = construct_alpha_matrix(n)
        return a[:, :, jnp.newaxis]

    @property
    def mu(self):
        """
        The geometric mean of two masses (or reciprocal sum)
        """
        if self.include_electrons:
            m = to_array([*self.state.atomic_masses, 1 * ureg.electron_mass])
        else:
            m = self.state.atomic_masses
        mu = jpu.numpy.outer(m, m) / (m[:, jnp.newaxis] + m[jnp.newaxis, :])
        return mu[:, :, jnp.newaxis]

    @property
    def T(self):
        """
        The mass_weighted temperature average of a pair, according to
        :cite:`Schwarz.2007`.

        .. math::

           \\bar{T}_{ab} = \\frac{T_a m_b + T_b m_a}{m_a m_b}

        """
        if self.include_electrons:
            m = to_array([*self.state.atomic_masses, 1 * ureg.electron_mass])
            T = to_array([*self.state.T_i, self.state.T_e])
        else:
            m = self.state.atomic_masses
            T = self.state.T_i
        return mass_weighted_T(m, T)[:, :, jnp.newaxis]

    @property
    def lambda_ab(self):
        # Compared to Gregori.2003, there is a pi missing
        l_ab = ureg.hbar * jpu.numpy.sqrt(
            1 / (2 * self.mu * ureg.k_B * self.T)
        )
        return l_ab

    # The following is required to jit a state
    def _tree_flatten(self):
        children = (self.state,)
        aux_data = {
            "include_electrons": self.include_electrons,
            "transform_r": self._transform_r,
            "model_key": self.model_key,
        }  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        new_obj = cls(*children)
        new_obj._transform_r = aux_data["transform_r"]
        new_obj.model_key = aux_data["model_key"]
        new_obj.include_electrons = aux_data["include_electrons"]
        return new_obj


class CoulombPotential(HNCPotential):
    """
    A full Coulomb Potential.
    """

    @jax.jit
    def full_r(self, r: Quantity) -> Quantity:
        """
        .. math::

           q^2 / (4 jnp.pi \\varepsilon_0 * r)

        """
        _r = r[jnp.newaxis, jnp.newaxis, :]
        return self.q2 / (4 * jnp.pi * ureg.epsilon_0 * _r)

    @jax.jit
    def long_k(self, k: Quantity):
        """
        .. math::

            q^2 / (k^2 * \\varepsilon_0) * (\\alpha**2 / (k^2 + \\alpha^2))

        """
        _k = k[jnp.newaxis, jnp.newaxis, :]

        return (
            self.q2
            / (_k**2 * ureg.epsilon_0)
            * self.alpha**2
            / (_k**2 + self.alpha**2)
        )


class DebyeHuckelPotential(HNCPotential):
    def check(self) -> None:
        if not hasattr(self.state, "DH_screening_length"):
            logger.error(
                f"The PlasmaState {self.state} has no attribute 'DH_screening_length, which is required for DebyeHuckelPotential."  # noqa: 501
            )

    @property
    def kappa(self):
        # This is called if kappa is defined per ion species
        if (
            isinstance(self.state.DH_screening_length.magnitude, jnp.ndarray)
            and len(self.state.DH_screening_length.shape) == 2
        ):
            return 1 / self.state.DH_screening_length[:, :, jnp.newaxis]
        else:
            return 1 / self.state.DH_screening_length

    @jax.jit
    def full_r(self, r):
        """
        .. math::

            \\frac{q^2}{(4 \\pi \\epsilon_0 r)} \\exp(-\\kappa r)

        """

        _r = r[jnp.newaxis, jnp.newaxis, :]

        return (
            self.q2
            / (4 * jnp.pi * ureg.epsilon_0 * _r)
            * jpu.numpy.exp(-self.kappa * r)
        )

    @jax.jit
    def long_k(self, k):

        _k = k[jnp.newaxis, jnp.newaxis, :]

        pref = self.q2 / (_k**2 * ureg.epsilon_0) * _k**2
        numerator = self.alpha**2 + 2 * self.alpha * self.kappa
        denumerator = (_k**2 + self.kappa**2) * (
            _k**2 + (self.kappa + self.alpha) ** 2
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

    @jax.jit
    def full_r(self, r: Quantity) -> Quantity:
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
            self.q2
            / (4 * jnp.pi * ureg.epsilon_0 * _r)
            * (
                1
                - jpu.numpy.exp(-(_r**2) / self.lambda_ab**2)
                + (jnp.sqrt(jnp.pi) * _r / self.lambda_ab)
                * (
                    1
                    - jax.scipy.special.erf(
                        (_r / self.lambda_ab).m_as(ureg.dimensionless)
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

    @jax.jit
    def full_r(self, r):
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

        beta = 1 / (ureg.k_B * self.T)
        xi = self.q2 * beta / (4 * jnp.pi * ureg.epsilon_0 * self.lambda_ab)
        pref = -(ureg.k_B * self.T * xi**2 / 16)
        factor = ureg.k_B * self.T * xi**2
        denominator = (
            16 * jpu.numpy.absolute(self.q2) / (4 * jnp.pi * ureg.epsilon_0)
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

    @jax.jit
    def full_r(self, r):
        _r = r[jnp.newaxis, jnp.newaxis, :]
        return (
            self.q2
            / (4 * jnp.pi * ureg.epsilon_0 * _r)
            * (1 - jpu.numpy.exp(-_r / self.lambda_ab))
        )


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


jax.tree_util.register_pytree_node(
    CoulombPotential,
    CoulombPotential._tree_flatten,
    CoulombPotential._tree_unflatten,
)
jax.tree_util.register_pytree_node(
    DebyeHuckelPotential,
    DebyeHuckelPotential._tree_flatten,
    DebyeHuckelPotential._tree_unflatten,
)
jax.tree_util.register_pytree_node(
    DeutschPotential,
    DeutschPotential._tree_flatten,
    DeutschPotential._tree_unflatten,
)
jax.tree_util.register_pytree_node(
    KelbgPotential,
    KelbgPotential._tree_flatten,
    KelbgPotential._tree_unflatten,
)
jax.tree_util.register_pytree_node(
    KlimontovichKraeftPotential,
    KlimontovichKraeftPotential._tree_flatten,
    KlimontovichKraeftPotential._tree_unflatten,
)
