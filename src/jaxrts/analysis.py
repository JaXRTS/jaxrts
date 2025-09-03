"""
Some analysis functions, mainly used for benchmarking and testing.
"""

from functools import partial

import jax
from jax import numpy as jnp
from jpu import numpy as jnpu

from .helpers import secant_extrema_finding
from .setup import Setup
from .units import Quantity, ureg


@jax.jit
def twoSidedLaplace(
    intensity: Quantity,
    tau: Quantity,
    E_shift: Quantity,
    E_min: Quantity | None = None,
    E_max: Quantity | None = None,
):
    """
    Perform the two-sided Lapace transform of intensity numerically.

    .. math::

       \\mathscr{L}[I] = \\int_{E_{min}}^{E_{max}}
       I \\exp(-\\tau E) \\mathrm{d}E

    """
    # Set default values for E_min and E_max if they are not given
    E_min = jnpu.min(E_shift) if E_min is None else E_min
    E_max = jnpu.max(E_shift) if E_max is None else E_max

    # Sort energy and intensity for the integration
    inten_sorted = intensity[jnpu.argsort(E_shift)]
    E_sorted = E_shift[jnpu.argsort(E_shift)]

    # Integrate in energy space (this way, the integral is numerically more
    # stable compared to omega_space. This results in a dimensionless result,
    # when considering dynamic structure factors
    kernel = inten_sorted * jnpu.exp(-tau * E_sorted) / ureg.hbar
    kernel_unit = kernel.units
    # Set the kernel to 0 where the energy is outside of the defined range,
    # i.e., don't include it for the integral
    cut_kernel = jnpu.where(
        (E_sorted >= E_min) & (E_sorted <= E_max), kernel.m_as(kernel_unit), 0
    )
    return (
        jnp.trapezoid(cut_kernel, (E_sorted).m_as(ureg.electron_volt))
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
    1. Calculate :math:`\\mathscr{L}[S]` where S is the real dynamic structure
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


class ITCF:
    def __init__(
        self,
        S_ee_conv: Quantity,
        E_shift: Quantity,
        instrument: Quantity,
        instrument_E: Quantity,
        E_cut: Quantity,
        raw: bool = False,
    ) -> (Quantity, Quantity):
        """
        1. Calculate :math:`F=\\mathscr{L}[S]` where S is the real dynamic
           structure factor (i.e., not convolved with the instrument function)
           See :cite:`Dornheim.2022`, Eqn (2).
        """
        self.S_ee_conv = S_ee_conv
        self.E_shift = E_shift
        self.instrument = instrument
        self.instrument_E = instrument_E
        self.E_cut = E_cut
        #: If ``raw==True``, don't deconvolve with the instrument function.
        #: This can be used for testing, benchmarking and toubleshooting.
        self.raw = raw

    @jax.jit
    def _F(self, tau_dimensionless):
        tau = tau_dimensionless / (1 * ureg.kiloelectron_volt)
        return self.F(tau)

    @jax.jit
    def __call__(self, tau):
        """
        Alias to :py:meth:`~.F`.
        """
        return self.F(tau)

    @jax.jit
    def F(self, tau):
        """
        Calculate the Laplace Transform for the given structure factor.
        """
        L_S_ee = twoSidedLaplace(
            self.S_ee_conv, tau, self.E_shift, -self.E_cut, self.E_cut
        )
        L_inst = twoSidedLaplace(
            self.instrument,
            tau,
            self.instrument_E,
            -self.E_cut,
            self.E_cut,
        )
        if self.raw:
            return L_S_ee.m_as(ureg.dimensionless)
        else:
            return (L_S_ee / L_inst).m_as(ureg.dimensionless)

    @jax.jit
    def get_T(self, tau_max):
        """
        1. Find the :math:`\\tau` for which the Laplace transform is minimal
        2. Get the corresponding temperature

        See :cite:`Dornheim.2022`.
        """
        t_min = 1e-8  # / ureg.kiloelectron_volt
        t_max = tau_max.m_as(1 / ureg.kiloelectron_volt)

        sol, iterations = secant_extrema_finding(
            jax.tree_util.Partial(self._F), t_min, t_max
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
        aux_data = (self.raw,)
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = ITCF.__new__(cls)
        (
            obj.S_ee_conv,
            obj.E_shift,
            obj.instrument,
            obj.instrument_E,
            obj.E_cut,
        ) = children
        (obj.raw,) = aux_data
        return obj


@partial(jax.jit, static_argnames="raw")
def ITCFT(
    S_ee_conv: Quantity,
    tau_max: Quantity,
    setup: Setup,
    E_cut: Quantity,
    raw: bool = False,
) -> (Quantity, Quantity):
    """
    1. Calculate :math:`F=\\mathscr{L}[S]` where S is the real dynamic
       structure factor (i.e., not convolved with the instrument function) See
       :cite:`Dornheim.2022`, Eqn (2).
    2. Find the :math:`\\tau` for which the Laplace transform is minimal
    3. Get the corresponding temperature
    """
    # Note the sign, here! Lower energy in a spectrum is a positive energy
    # shift.
    ITCF_func = ITCF_from_setup(S_ee_conv, setup, E_cut, raw)
    return ITCF_func.get_T(tau_max)


@jax.jit
def ITCFT_grid(
    S_ee_conv: Quantity,
    tau: Quantity,
    setup: Setup,
    E_cut: Quantity,
) -> (Quantity, Quantity):
    """
    This function is akin to :py:func:`~.ITCFT`, but rather than an adoptive
    minimization, it runs on a naive :math:`\\tau` grid.

    1. Calculate :math:`F=\\mathscr{L}[S]` where S is the real dynamic
       structure factor (i.e., not convolved with the instrument function) See
       :cite:`Dornheim.2022`, Eqn (2).
    2. Find the :math:`\\tau` for which the Laplace transform is minimal
    3. Get the corresponding temperature
    """
    # Note the sign, here! Lower energy in a spectrum is a positive energy
    # shift.
    E_shift = -(setup.measured_energy - setup.energy)
    instrument = setup.instrument(E_shift / (1 * ureg.hbar))
    return _ITCFT_grid(S_ee_conv, tau, E_shift, instrument, E_shift, E_cut)


@partial(jax.jit, static_argnames="raw")
def ITCF_from_setup(
    S_ee_conv: Quantity,
    setup: Setup,
    E_cut: Quantity,
    raw: bool = False,
) -> "ITCF":
    """
    Returns a :py:class:`~.ITCF`object from the given given
    :py:class:`jaxrts.setup.Setup` instance and a dynamic structure factor.
    If ``raw`` is True, the returned object will omit the deconvolution
    with the instrument function.
    """
    # Note the sign, here! Lower energy in a spectrum is a positive energy
    # shift.
    E_shift = -(setup.measured_energy - setup.energy)
    instrument = setup.instrument(E_shift / (1 * ureg.hbar))
    return ITCF(S_ee_conv, E_shift, instrument, E_shift, E_cut, raw)


@partial(jax.jit, static_argnames="raw")
def ITCF_fsum(
    S_ee_conv: Quantity,
    setup: Setup,
    E_cut: Quantity,
    raw: bool = False,
) -> (Quantity, Quantity):
    """
    Calculate the f-sum rule by evaluating the first derivative of the Laplace
    transform at 0/eV.
    See :cite:`Dornheim.2024`.

    1. Calculate :math:`F=\\mathscr{L}[S]` where S is the real dynamic
       structure factor (i.e., not convolved with the instrument function) See
       :cite:`Dornheim.2022`, Eqn (2).
    2. Calculate the derivative of the Laplace transform at :math:`\\tau = 0`.
    3. Get the f-sum.
    """
    # Note the sign, here! Lower energy in a spectrum is a positive energy
    # shift.
    ITCF_func = ITCF_from_setup(S_ee_conv, setup, E_cut, raw)
    return (jax.grad(ITCF_func._F)(0.0)) * 1 * ureg.kiloelectron_volt


@partial(jax.jit, static_argnames="raw")
def ITCF_ssf(
    S_ee_conv: Quantity,
    setup: Setup,
    E_cut: Quantity,
    raw: bool = False,
) -> (Quantity, Quantity):
    """
    Calculate the Static structure Factor by evaluating the Laplace transform
    at 0/eV.

    1. Calculate :math:`F=\\mathscr{L}[S]` where S is the real dynamic
       structure factor (i.e., not convolved with the instrument function) See
       :cite:`Dornheim.2022`, Eqn (2).
    2. Calculate :math:`F(0/eV) = S_ee(k)`, the static structure factor
    """
    # Note the sign, here! Lower energy in a spectrum is a positive energy
    # shift.
    ITCF_func = ITCF_from_setup(S_ee_conv, setup, E_cut, raw)
    return ITCF_func._F(0.0)


jax.tree_util.register_pytree_node(
    ITCF,
    ITCF._tree_flatten,
    ITCF._tree_unflatten,
)
