"""
Some analysis functions, mainly used for benchmarking and testing.
"""

import jax
from jax import numpy as jnp
from jpu import numpy as jnpu

from .setup import Setup
from .units import ureg, Quantity


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
def _ITCF(
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


@jax.jit
def ITCF(
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
    return _ITCF(S_ee_conv, tau, E_shift, instrument, E_shift, E_cut)
