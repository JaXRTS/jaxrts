"""
Module to account for Ionization Potential Depression (IPD)

All these implementations and formulas are taken from 'Modifications of Bound
States in Dense Plasma Environments' by Rory A. Baggott. :cite:`Baggott.2017`
"""

import logging
from functools import partial

import jax
from jax import numpy as jnp
from jpu import numpy as jnpu
from quadax import quadts as quad

from .math import fermi_neg12_rational_approximation_antia
from .plasma_physics import (
    chem_pot_interpolationIchimaru as chem_pot_interpolation,
)
from .units import Quantity, ureg

logger = logging.getLogger(__name__)


@jax.jit
def inverse_screening_length_e(ne: Quantity, Te: Quantity) -> Quantity:
    """
    Inverse screening length for arbitrary degeneracy for the electron
    as needed for WDM applications. (Taken from :cite:`Baggott.2017`)

    .. note::

       This result reproduces eqn. 19 in :cite:`Ropke.2019`, up to a factor of
       :math:`4 \\pi`.

    """
    q = -1 * ureg.elementary_charge

    chem_pot = chem_pot_interpolation(Te, ne)
    beta = 1 / (1 * ureg.boltzmann_constant * Te)

    fermi_integral_neg1_2 = fermi_neg12_rational_approximation_antia(
        (chem_pot * beta).m_as(ureg.dimensionless)
    )

    therm_wv = jnpu.sqrt(
        (2 * jnp.pi * 1 * ureg.hbar**2)
        / ((1 * ureg.electron_mass) * 1 * ureg.boltzmann_constant * Te)
    )

    k_sq = (
        (q**2)
        / (ureg.epsilon_0 * ureg.boltzmann_constant * Te)
        * (2.0 / therm_wv**3)
        * fermi_integral_neg1_2
    )

    return jnpu.sqrt(k_sq).to(1 / ureg.angstrom)


@partial(jax.jit, static_argnames=["arb_deg"])
def ipd_debye_hueckel(
    Zi: float,
    ne: Quantity,
    ni: Quantity,
    Te: Quantity,
    Ti: Quantity,
    Zbar: float | None = None,
    arb_deg: bool = False,
) -> Quantity:
    """
    The correction to the ionization potential for the m-th ionization stage in
    Debye-Hueckel approximation.

    .. note::

       The Debye-Hueckel approximation is physically meaningful only when
       the coupling parameter << 1, such that Coulomb forces are weak
       perturbations.

    Parameters
    ----------
    Zi
        The charge state of the atom (note that this is the state before the
        ionization).
    ne
        Electron density. Units of 1/[length]**3.
    ni
        Ion density. Units of 1/[length]**3.
    Te
        The electron temperature.
    Ti
        The ion temperature.
    arb_deg: bool, default false
        If ``True`` the Debye screening length is evaluated using
        :py:func:`inverse_screening_length_e`, which includes solving the Fermi
        integral, rather than the classical value.

    Returns
    -------
    Quantity.
        The ipd shift in units of electronvolt.
    """
    if Zbar is None:
        Zbar = Zi

    kappa_i_sq = jnpu.sum(Zbar**2 * ureg.elementary_charge**2 * ni) / (
        1 * ureg.epsilon_0 * ureg.boltzmann_constant * Ti
    )
    if arb_deg:
        kappa_e = jnpu.sum(inverse_screening_length_e(ne, Te))
        kappa_D = jnpu.sqrt(kappa_i_sq + kappa_e**2)
    else:
        kappa_e_sq = jnpu.sum(ureg.elementary_charge**2 * ne) / (
            1 * ureg.epsilon_0 * ureg.boltzmann_constant * Te
        )
        kappa_D = jnpu.sqrt(kappa_i_sq + kappa_e_sq)

    # The ionization potential depression energy shift
    ipd_shift = kappa_D * (
        -(Zi + 1) * ureg.elementary_charge**2 / (4 * jnp.pi * ureg.epsilon_0)
    )

    return ipd_shift.to(ureg.electron_volt)


@jax.jit
def ipd_ion_sphere(
    Zi: Quantity,
    ne: Quantity,
    ni: Quantity,
    C: float | None = None,
    Zbar: float | None = None,
) -> Quantity:
    """
    The correction to the ionization potential for the m-th ionization stage in
    the ion-sphere model. The ion-sphere model considers the ions to be
    strongly correlated. (see also :cite:`Zimmermann.1980`)

    Parameters
    ----------
    Zi
        The charge state of the atom (note that this is the state before the
        ionization)
    ne
        Electron density. Units of 1/[length]**3.
    ni
        Ion density. Units of 1/[length]**3.
    C: float, optional
        A linear scaling factor of it ion Sphere IPD. The default value is 3/2,
        which is consistent with the limit of :py:func:`~.ipd_stewart_pyatt`
        and :cite:`Crowley.2014`. Another typical value would be 9/5,
        introduced by :cite:`Zimmerman.1980`. See also :cite:`Lin.2017` and
        :cite:`Ciricosta.2012`.
    Zbar: float, optional
        The average ionization of the plasma. If not given, Zi is assumed to be
        the average ionization of the plasma.

    Returns
    -------
    Quantity
        The ipd shift in units of electronvolt.
    """
    Zi = jnp.clip(Zi, 1e-6)
    Zbar = Zi if Zbar is None else jnp.clip(Zbar, 1e-6)

    # The ion-sphere radius, determined by the ion density n_i such that the
    # average distance to the nearest neighbor ion is
    # approximately 2 R_0.
    R_0 = (3 * Zbar / (4 * jnp.pi * ne)) ** (1 / 3)

    if C is None:
        C = 3 / 2

    ipd_shift = -(C * (Zi + 1) * 1 * ureg.elementary_charge**2) / (
        R_0 * 4 * jnp.pi * 1 * ureg.epsilon_0
    )

    return ipd_shift.to(ureg.electron_volt)


@partial(jax.jit, static_argnames=["arb_deg"])
def ipd_stewart_pyatt(
    Zi: float,
    ne: Quantity,
    ni: Quantity,
    Te: Quantity,
    Ti: Quantity,
    Zbar: float | None = None,
    arb_deg: bool = False,
) -> Quantity:
    """
    Stewart-Pyatt IPD model, using the formulation which can be found, e.g., in
    :cite:`Calisti.2015` or :cite:`Ropke.2019`.

    .. note::

       The Stewart-Pyatt value is always below both the Debye and ion sphere
       results.

    Parameters
    ----------
    Zi
        The charge state of the atom (note that this is the state before the
        ionization).
    ne
        Electron density. Units of 1/[length]**3.
    ni
        Ion density. Units of 1/[length]**3.
    Te
        The electron temperature.
    Ti
        The ion temperature.
    Zbar: float, optional
        The average ionization of the plasma. If not given, Zi is assumed to be
        the average ionization of the plasma.
    arb_deg: bool, default false
        If ``True`` the Debye screening length is evaluated using
        :py:func:`inverse_screening_length_e`, which includes solving the Fermi
        integral, rather than the classical value.

    Returns
    -------
    Quantity
        The ipd shift in units of electronvolt.
    """
    Zi = jnp.clip(Zi, 1e-6)
    Zbar = Zi if Zbar is None else jnp.clip(Zbar, 1e-6)

    kappa_i_sq = jnpu.sum(Zbar**2 * ureg.elementary_charge**2 * ni) / (
        1 * ureg.epsilon_0 * ureg.boltzmann_constant * Ti
    )
    if arb_deg:
        kappa_e = jnpu.sum(inverse_screening_length_e(ne, Te))
        kappa_D = jnpu.sqrt(kappa_i_sq + kappa_e**2)
    else:
        kappa_e_sq = jnpu.sum(ureg.elementary_charge**2 * ne) / (
            1 * ureg.epsilon_0 * ureg.boltzmann_constant * Te
        )
        kappa_D = jnpu.sqrt(kappa_i_sq + kappa_e_sq)

    R_0 = (3 / (4 * jnp.pi * jnpu.sum(ni))) ** (1 / 3)

    s = 1 / kappa_D / R_0

    ipd_shift = (
        -(3 * (Zi + 1) * ureg.elementary_charge**2)
        / (8 * jnp.pi * ureg.epsilon_0 * R_0)
        * ((1 + s**3) ** (2 / 3) - s**2)
    )

    return ipd_shift.to(ureg.electron_volt)


@partial(jax.jit, static_argnames=["arb_deg", "crowley_correction"])
def ipd_stewart_pyatt_preston(
    Zi: float,
    ne: Quantity,
    ni: Quantity,
    Te: Quantity,
    Ti: Quantity,
    ion_population=None,
    arb_deg: bool = False,
    crowley_correction: bool = False,
) -> Quantity:
    """
    The Stewart Pyatt IPD, as presented by :cite:`Preston.2013`, which closely
    resembles the seminal work of :cite:`Stewart.1966`.

    The IPD is dependent on the quantity
    :math:`z^* = \\frac{\\langle z^2 \\rangle}{\\langle z\\rangle}` which is
    only calculated if an ``ion_population`` argument is given. Otherwise, we
    assume this value was the average ionization state, which is, however, not
    accurate.

    The connection between this formulation and the implementation in
    :py:func:`~.ipd_stewart_pyatt` can be found in Appendix A of
    :cite:`Pain.2022`.

    :cite:`Crowley.2014` pointed out a correction to the formula used by
    Stewart and Pyatt. This can be handled by setting the
    ``crowley_correction`` flag to true

    Parameters
    ----------
    Z
        The charge state of the atom (note that this is the state before the
        ionization).
    ne
        Electron density. Units of 1/[length]**3.
    ni
        Ion density. Units of 1/[length]**3.
    Te
        The electron temperature.
    Ti
        The ion temperature.
    ion_population
        The ion population fractions.
    arb_deg: bool, default false
        If ``True`` the Debye screening length is evaluated using
        :py:func:`inverse_screening_length_e`, which includes solving the Fermi
        integral, rather than the classical value.
    crowley_correction: bool, default False
        If ``True`` apply the correction presented by :cite:`Crowley.2014`.

    Returns
    -------
    Quantity
        The ipd shift in units of electronvolt.
    """

    if ion_population is None:
        Zp = ne / jnpu.sum(ni)
        Zbar = ne / ni
    else:
        Z = jnp.arange(len(ion_population))
        Zbar = jnpu.mean(Z * ion_population)
        Zp = jnpu.mean(Z**2 * ion_population) / Zbar

    cc = 0 if crowley_correction else 1

    kappa_i_sq = jnpu.sum(Zbar**2 * ureg.elementary_charge**2 * ni) / (
        1 * ureg.epsilon_0 * ureg.boltzmann_constant * Ti
    )
    if arb_deg:
        kappa_e = jnpu.sum(inverse_screening_length_e(ne, Te))
        kappa_D = jnpu.sqrt(kappa_i_sq + kappa_e**2)
    else:
        kappa_e_sq = jnpu.sum(ureg.elementary_charge**2 * ne) / (
            1 * ureg.epsilon_0 * ureg.boltzmann_constant * Te
        )
        kappa_D = jnpu.sqrt(kappa_i_sq + kappa_e_sq)
    lambda_D = 1 / kappa_D

    Lambda = (
        3
        * (Zp + cc)
        * (Zi + 1)
        * ureg.elementary_charge**2
        / (
            4
            * ureg.epsilon_0
            * jnp.pi
            * lambda_D
            * ureg.boltzmann_constant
            * Ti
        )
    )

    ipd_shift = (
        -(1 * ureg.boltzmann_constant * Ti)
        / (2 * (Zp + cc))
        * ((1 + Lambda) ** (2 / 3) - 1)
    )

    return ipd_shift.to(ureg.electron_volt)


@partial(jax.jit, static_argnames=["arb_deg"])
def ipd_ecker_kroell(
    Zi: float,
    ne: Quantity,
    ni: Quantity,
    Te: Quantity,
    Ti: Quantity,
    Z_max: Quantity,
    Zbar: float | None = None,
    arb_deg: bool = False,
    C: float | None = None,
) -> Quantity:
    """
    The ionization potential for an atom with charge Zi in the Ecker-Kroell
    model.
    Defines a critical density under which the IPD is identical to
    :py:func:`~.ipd_debye_hueckel`. Above that value, the IPD is given by a
    Ecker Kröll length. If no value ``C`` is given, the latter value is scaled
    to have a continuous IPD. Some studies (e.g. :cite:`Preston.2014` use a
    modified Ecker Kröll model, where a specific value of ``C`` (often 1) is
    set instead. For details see :cite:`EckerKroell.1963`.

    Parameters
    ----------
    Zi
        The charge state of the atom (note that this is the state before the
        ionization)
    ne
        Electron density. Units of 1/[length]**3.
    ni
        Ion density. Units of 1/[length]**3.
    Te
        The electron temperature.
    Ti
        The ion temperature.
    Z_max: Quantity
        Array of maximal charges of the ions. Must be of the same shape as ni.
    Zbar: float, optional
        The average ionization of the plasma. If not given, Zi is assumed to be
        the average ionization of the plasma.
    arb_deg: bool, default false
        If ``True`` the Debye screening length is evaluated using
        :py:func:`inverse_screening_length_e`, which includes solving the Fermi
        integral, rather than the classical value.
    C: float or None, default None
        Multiplicative factor for the high-density part of the EK model. If set
        to None, the factor is chosen to achieve a continuous IPD.

    Returns
    -------
    Quantity
        The ipd shift in units of electronvolt.

    """
    if Zbar is None:
        Zbar = Zi
    # The critical density in the model of Ecker-Kroell
    n_c = (3 / (4 * jnp.pi)) * (
        (4 * jnp.pi * 1 * ureg.epsilon_0)
        * ureg.boltzmann_constant
        * Te
        / (Z_max**2 * ureg.elementary_charge**2)
    ) ** 3

    # Calculating lambda_D at the critical density if arb_deg is given requires
    # to split n_c = n_i + n_e, to that the electronic part can be treated
    # individually.

    ne_crit = n_c / (ne + jnpu.sum(ni)) * ne
    ni_crit = n_c[:, None] / (ne + jnpu.sum(ni)) * ni[None, :]

    kappa_i_sq = jnpu.sum(
        Zbar**2
        * ureg.elementary_charge**2
        * ni
        / (1 * ureg.epsilon_0 * ureg.boltzmann_constant * Ti)
    )
    kappa_i_sq_crit = jnpu.sum(
        Zbar[None, :] ** 2
        * ureg.elementary_charge**2
        * ni_crit
        / (1 * ureg.epsilon_0 * ureg.boltzmann_constant * Ti),
        axis=1,
    )

    if arb_deg:
        kappa_e = jnpu.sum(inverse_screening_length_e(ne, Te))
        kappa_D = jnpu.sqrt(kappa_i_sq + kappa_e**2)

        kappa_e_crit = inverse_screening_length_e(ne_crit, Te)
        kappa_D_crit = jnpu.sqrt(kappa_i_sq_crit + kappa_e_crit**2)
    else:
        kappa_e_sq = jnpu.sum(ureg.elementary_charge**2 * ne) / (
            1 * ureg.epsilon_0 * ureg.boltzmann_constant * Te
        )
        kappa_D = jnpu.sqrt(kappa_i_sq + kappa_e_sq)
        kappa_e_sq_crit = (
            ureg.elementary_charge**2
            * ne_crit
            / (1 * ureg.epsilon_0 * ureg.boltzmann_constant * Te)
        )
        kappa_D_crit = jnpu.sqrt(kappa_i_sq_crit + kappa_e_sq_crit)
    lambda_D = 1 / kappa_D

    # For the definitions, see e.g., :cite:`Preston.2013`.
    R_EK = (3 / (4 * jnp.pi * (ne + ni))) ** (1 / 3)
    R_EK_crit = (3 / (4 * jnp.pi * n_c)) ** (1 / 3)

    if C is None:
        C = (kappa_D_crit * R_EK_crit).m_as(ureg.dimensionless)

    ipd_c1 = (
        -1
        * ureg.elementary_charge**2
        / (4 * jnp.pi * ureg.epsilon_0 * lambda_D)
        * (Zi + 1)
    )
    ipd_c2 = (
        -C
        * ureg.elementary_charge**2
        / (4 * jnp.pi * ureg.epsilon_0 * R_EK)
        * (Zi + 1)
    )

    # The ionization potential depression energy shift
    ipd_shift = jnpu.where((ni + ne) <= n_c, ipd_c1, ipd_c2)

    return ipd_shift.to(ureg.electron_volt)


@jax.jit
def ipd_pauli_blocking(
    Zi: float, ne: Quantity, ni: Quantity, Te: Quantity, Ti: Quantity
) -> Quantity:
    """
    The correction to the ionization potential due to Pauli blocking, as
    described in :cite:`Ropke.2019`.

    Parameters
    ----------
    Zi
        The charge state of the atom (note that this is the state before the
        ionization)
    ne
        Electron density. Units of 1/[length]**3.
    ni
        Ion density. Units of 1/[length]**3.
    Te
        The electron temperature.
    Ti
        The ion temperature.

    Returns
    -------
    Quantity
        The ipd shift in units of electronvolt.
    """
    # This function is not well-defined for Zi==0:
    Zi = jnp.clip(Zi, 1e-6)

    chem_pot = chem_pot_interpolation(Te, ne)

    # Reduced Bohr radius
    a_Z = ureg.a0 / Zi

    @jax.jit
    def integrand(p):
        p /= 1 * ureg.angstrom

        res = (
            p**2
            / (1 + a_Z**2 * p**2) ** 3
            * (
                1
                / (
                    jnpu.exp(
                        (
                            (1 * ureg.hbar * p) ** 2 / (2 * ureg.electron_mass)
                            - chem_pot
                        )
                        / (1 * ureg.boltzmann_constant * Te)
                    )
                    + 1
                )
            )
        )

        return res.m_as(1 / ureg.angstrom**2)

    integral, errl = quad(
        integrand, jnp.array([0, jnp.inf]), epsabs=1e-15, epsrel=1e-15
    )
    integral /= 1 * ureg.angstrom**3
    # The ionization potential depression energy shift
    ipd_shift = -(
        (Zi + 1)
        * ureg.elementary_charge**2
        / (4 * jnp.pi * ureg.epsilon_0)
        * (16 * a_Z**2)
        / jnp.pi
        * integral
    )

    return ipd_shift.to(ureg.electron_volt)
