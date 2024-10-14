"""
Some analysis functions, mainly used for benchmarking and testing.
"""

import jax
from jax import numpy as jnp
from jpu import numpy as jnpu

from .setup import Setup
from .units import ureg, Quantity
from .helpers import secant_extrema_finding


@jax.jit
def twoSidedLaplace(
    intensity: Quantity,
    tau: Quantity,
    E_shift: Quantity,
    E_min: Quantity | None = None,
    E_max: Quantity | None = None,
):
    # Set default values for E_min and E_max if they are not given
    E_min = jnpu.min(E_shift) if E_min is None else E_min
    E_max = jnpu.max(E_shift) if E_max is None else E_max

    # Integrate in energy space (this way, the integral is numerically more
    # stable compared to omega_space. This results in a dimensionless result,
    # when considering dynamic structure factors
    kernel = intensity * jnpu.exp(-tau * E_shift) * ureg.hbar
    kernel_unit = kernel.units
    # Set the kernel to 0 where the energy is outside of the defined range,
    # i.e., don't include it for the integral
    cut_kernel = jnpu.where(
        (E_shift >= E_min) & (E_shift <= E_max), kernel.m_as(kernel_unit), 0
    )
    return (
        jnp.trapezoid(cut_kernel, (E_shift).m_as(ureg.electron_volt))
        * kernel_unit
        * ureg.electron_volt
    )


@jax.jit
def _ITCFT_grid(
    S_ee_conv: Quantity,
    tau: Quantity,
    E_shift: Quantity,
    instrument: Quantity,
    instrument_E: Quantity,
    E_cut: Quantity,
) -> (Quantity, Quantity):
    """
    1. Calculate :math:`\\mathscr{L}[S] where S is the real dynamic structure
       factor (i.e., not convolved with the instrument function) See
       :cite:`Dornheim.2022`, Eqn (2).
    2. Find the :math:`\\tau` for which the Laplace transform is minimal
    3. Get the corresponding temperature
    """
    L_inst = jax.vmap(
        twoSidedLaplace, in_axes=(None, 0, None, None, None), out_axes=0
    )(instrument, tau, instrument_E, -E_cut, E_cut)
    L_S_ee = jax.vmap(
        twoSidedLaplace, in_axes=(None, 0, None, None, None), out_axes=0
    )(S_ee_conv, tau, E_shift, -E_cut, E_cut)
    L = L_S_ee / L_inst
    tau_min = tau[jnpu.argmin(L)]
    T = 1 / (2 * tau_min * ureg.k_B)
    return T.to(ureg.kelvin), L


class _ITCFT:
    def __init__(
        self,
        S_ee_conv: Quantity,
        E_shift: Quantity,
        instrument: Quantity,
        instrument_E: Quantity,
        E_cut: Quantity,
    ) -> (Quantity, Quantity):
        """
        1. Calculate :math:`\\mathscr{L}[S] where S is the real dynamic
           structure factor (i.e., not convolved with the instrument function)
           See :cite:`Dornheim.2022`, Eqn (2).
        2. Find the :math:`\\tau` for which the Laplace transform is minimal
        3. Get the corresponding temperature
        """
        self.S_ee_conv = S_ee_conv
        self.E_shift = E_shift
        self.instrument = instrument
        self.instrument_E = instrument_E
        self.E_cut = E_cut

    @jax.jit
    def _L(self, tau_dimensionless):
        tau = tau_dimensionless/(1 * ureg.kiloelectron_volt)
        return self.L(tau)

    @jax.jit
    def L(self, tau):
        L_inst = twoSidedLaplace(
            self.instrument, tau, self.instrument_E, -self.E_cut, self.E_cut
        )
        L_S_ee = twoSidedLaplace(
            self.S_ee_conv, tau, self.E_shift, -self.E_cut, self.E_cut
        )
        return (L_S_ee / L_inst).m_as(ureg.dimensionless)

    @jax.jit
    def get_T(self, tau_max):
        t_min = 1e-8  # / ureg.kiloelectron_volt
        t_max = tau_max.m_as(1 / ureg.kiloelectron_volt)

        sol, iterations = secant_extrema_finding(
            jax.tree_util.Partial(self._L), t_min, t_max
        )

        minimizing_tau = sol / (1 * ureg.kiloelectron_volt)

        T = 1 / (2 * minimizing_tau * ureg.k_B)
        return T.to(ureg.kelvin)

    def _tree_flatten(self):
        children = (
            self.S_ee_conv,
            self.E_shift,
            self.instrument,
            self.instrument_E,
            self.E_cut,
        )
        aux_data = ()
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = _ITCFT.__new__(cls)
        (
            obj.S_ee_conv,
            obj.E_shift,
            obj.instrument,
            obj.instrument_E,
            obj.E_cut,
        ) = children
        return obj


@jax.jit
def ITCFT(
    S_ee_conv: Quantity,
    tau_max: Quantity,
    setup: Setup,
    E_cut: Quantity,
) -> (Quantity, Quantity):
    """
    1. Calculate :math:`\\mathscr{L}[S] where S is the real dynamic structure
       factor (i.e., not convolved with the instrument function) See
       :cite:`Dornheim.2022`, Eqn (2).
    2. Find the :math:`\\tau` for which the Laplace transform is minimal
    3. Get the corresponding temperature
    """
    # Note the sign, here! Lower energy in a spectrum is a positive energy
    # shift.
    E_shift = -(setup.measured_energy - setup.energy)
    instrument = setup.instrument(E_shift / (1 * ureg.hbar))
    ITCF_minimizer = _ITCFT(S_ee_conv, E_shift, instrument, E_shift, E_cut)
    return ITCF_minimizer.get_T(tau_max)


@jax.jit
def ITCFT_grid(
    S_ee_conv: Quantity,
    tau: Quantity,
    setup: Setup,
    E_cut: Quantity,
) -> (Quantity, Quantity):
    """
    1. Calculate :math:`\\mathscr{L}[S] where S is the real dynamic structure
       factor (i.e., not convolved with the instrument function) See
       :cite:`Dornheim.2022`, Eqn (2).
    2. Find the :math:`\\tau` for which the Laplace transform is minimal
    3. Get the corresponding temperature
    """
    # Note the sign, here! Lower energy in a spectrum is a positive energy
    # shift.
    E_shift = -(setup.measured_energy - setup.energy)
    instrument = setup.instrument(E_shift / (1 * ureg.hbar))
    return _ITCFT_grid(S_ee_conv, tau, E_shift, instrument, E_shift, E_cut)


jax.tree_util.register_pytree_node(
    _ITCFT,
    _ITCFT._tree_flatten,
    _ITCFT._tree_unflatten,
)
