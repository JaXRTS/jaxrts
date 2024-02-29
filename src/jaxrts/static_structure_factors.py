"""
Static structure factors.
"""

import jax
from jax import numpy as jnp
from jpu import numpy as jnpu

from .units import ureg, Quantity
from .plasma_physics import fermi_energy, wiegner_seitz_radius

jax.config.update("jax_enable_x64", True)


@jax.jit
def T_cf_Greg(T_e: Quantity, n_e: Quantity) -> Quantity:
    """
    The effective temperature to be inserted in the approach by Arkhipov and
    Davletov, see :cite:`Arkhipov.1998` according to :cite:`Gregori.2003`.
    """
    # The fermi temperature
    T_f = fermi_energy(n_e) / ureg.k_B
    dimless_rs = wiegner_seitz_radius(n_e) / ureg.a_0
    T_q = T_f / (1.3251 - 0.1779 * jnpu.sqrt(dimless_rs))
    return jnpu.sqrt(T_e**2 + T_q**2)


@jax.jit
def _lambda_AD(T: Quantity, m1: Quantity, m2: Quantity) -> Quantity:
    """
    :cite:`Gregori.2003` states this as the thermal de Broglie Wavelength, but
    this is not reproducing the known formula for ``m1 == m2``.
    However, both :cite:`Arkhipov.1998` and :cite:`Gregori` use this notation.
    """
    # The 1 * m1 is required so allow this function to handle Units, too.
    mu = (m1 * m2) / (m1 + m2)
    denumerator = 2 * jnp.pi * mu * ureg.k_B * T

    return ureg.hbar / jnpu.sqrt(denumerator)


@jax.jit
def _k_D_AD(T: Quantity, n_e: Quantity, Z_f: float = 1.0) -> Quantity:
    numerator = Z_f * n_e * ureg.elementary_charge**2
    denumerator = ureg.epsilon_0 * ureg.k_B * T
    return jnpu.sqrt(numerator / denumerator)


@jax.jit
def _b_AD(lam_ee: Quantity) -> Quantity:
    return 1 / (lam_ee**2 * jnp.pi * jnp.log(2))


@jax.jit
def _A_AD(T_e: Quantity, b: Quantity) -> Quantity:
    """
    A helper function for the approach by Arkhipov and Davletov, see
    :cite:`Arkhipov.1998` and :cite:`Gregori.2003`.
    """
    return (
        ureg.k_B
        * T_e
        * jnp.log(2)
        * jnp.pi ** (3 / 2)
        * b ** (-3 / 2)
        * ureg.epsilon_0
        / ureg.elementary_charge**2
    )


@jax.jit
def _Delta_AD(
    k: Quantity,
    k_De: Quantity,
    k_Di: Quantity,
    lamee: Quantity,
    lamii: Quantity,
    lamei: Quantity,
    b: Quantity,
    A: Quantity,
) -> Quantity:
    """
    :cite:`Gregori.2003`, Eqn. (11)
    """
    return (
        k**4
        + (k**2 * k_De**2) / (1 + k**2 * lamee**2)
        + (k**2 * k_Di**2) / (1 + k**2 * lamii**2)
        + k_De**2
        * k_Di**2
        * (
            1.0 / ((1 + k**2 * lamee**2) * (1 + k**2 * lamii**2))
            - 1 / (1 + k**2 * lamei**2) ** 2
        )
        + A
        * k**2
        * k_De**2
        * (k**2 + k_Di**2 / (1 + k**2 * lamii**2))
        * jnpu.exp(-(k**2) / (4 * b))
    )


@jax.jit
def _Phi_ee_AD(
    k: Quantity, T_e: Quantity, n_e: Quantity, m_i: Quantity, Z_f: float
) -> Quantity:
    """
    See :cite:`Gregori.2003`, eqn (8) for the coefficient
    :math:`\\Phi_ee(k)` in the approach by Arkhipov and Davletov
    :cite:`Arkhipov.1998`.

    Parameters
    ----------
    k: Quantity
        The scattering vector length (units of 1/[length])
    T_e: Quantity
        The electron temperature in Kelvin. Use :py:func:`~.T_cf_Greg` for the
        effective temperature used in :cite:`Gregori.2003`.
    n_e: Quantity
        The electron density in 1/[volume]
    m_i: Quantity
        The mass of the ion
    Z_f: float
        Number of free electrons per ion.
    """
    # Set up all the variables
    lam_ee = _lambda_AD(T_e, 1 * ureg.electron_mass, 1 * ureg.electron_mass)
    lam_ei = _lambda_AD(T_e, 1 * ureg.electron_mass, m_i)
    lam_ii = _lambda_AD(T_e, m_i, m_i)
    b = _b_AD(lam_ee)
    A = _A_AD(T_e, b)
    k_De = _k_D_AD(T_e, n_e)
    k_Di = _k_D_AD(T_e, n_e, Z_f)
    Delta = _Delta_AD(k, k_De, k_Di, lam_ee, lam_ii, lam_ei, b, A)

    pref = ureg.elementary_charge**2 / (ureg.vacuum_permittivity * Delta)
    sum1 = k**2 / (1 + k**2 * lam_ee**2)
    sum2 = k_Di**2 * (
        1 / ((1 + k**2 * lam_ee**2) * (1 + k**2 * lam_ii**2))
        - 1 / ((1 + k**2 * lam_ei**2) ** 2)
    )
    sum3 = (
        A
        * (k**2 + k_Di**2 / (1 + k**2 * lam_ii**2))
        * k**2
        * jnpu.exp(-(k**2) / (4 * b))
    )
    return (pref * (sum1 + sum2 + sum3)).to_base_units()


@jax.jit
def _Phi_ii_AD(
    k: Quantity, T_e: Quantity, n_e: Quantity, m_i: Quantity, Z_f: float
) -> Quantity:
    """
    See :cite:`Gregori.2003`, eqn (9) for the coefficient
    :math:`\\Phi_ii(k)` in the approach by Arkhipov and Davletov
    :cite:`Arkhipov.1998`.

    Parameters
    ----------
    k: Quantity
        The scattering vector length (units of 1/[length])
    T_e: Quantity
        The electron temperature in Kelvin. Use :py:func:`~.T_cf_Greg` for the
        effective temperature used in :cite:`Gregori.2003`.
    n_e: Quantity
        The electron density in 1/[volume]
    m_i: Quantity
        The mass of the ion
    Z_f: float
        Number of free electrons per ion.
    """
    # Set up all the variables
    lam_ee = _lambda_AD(T_e, 1 * ureg.electron_mass, 1 * ureg.electron_mass)
    lam_ei = _lambda_AD(T_e, 1 * ureg.electron_mass, m_i)
    lam_ii = _lambda_AD(T_e, m_i, m_i)
    b = _b_AD(lam_ee)
    A = _A_AD(T_e, b)
    k_De = _k_D_AD(T_e, n_e)
    k_Di = _k_D_AD(T_e, n_e, Z_f)
    Delta = _Delta_AD(k, k_De, k_Di, lam_ee, lam_ii, lam_ei, b, A)

    pref = (
        Z_f**2 * ureg.elementary_charge**2 / (ureg.vacuum_permittivity * Delta)
    )
    sum1 = k**2 / (1 + k**2 * lam_ii**2)
    sum2 = k_De**2 * (
        1 / ((1 + k**2 * lam_ee**2) * (1 + k**2 * lam_ii**2))
        - 1 / ((1 + k**2 * lam_ei**2) ** 2)
    )
    sum3 = (
        (A * k**2 * k_De**2)
        / (1 + k**2 * lam_ii**2)
        * jnpu.exp(-(k**2) / (4 * b))
    )
    return (pref * (sum1 + sum2 + sum3)).to_base_units()


@jax.jit
def _Phi_ei_AD(
    k: Quantity, T_e: Quantity, n_e: Quantity, m_i: Quantity, Z_f: float
) -> Quantity:
    """
    See :cite:`Gregori.2003`, eqn (10) for the coefficient
    :math:`\\Phi_ee(k)` in the approach by Arkhipov and Davletov
    :cite:`Arkhipov.1998`.

    Parameters
    ----------
    k: Quantity
        The scattering vector length (units of 1/[length])
    T_e: Quantity
        The electron temperature in Kelvin. Use :py:func:`~.T_cf_Greg` for the
        effective temperature used in :cite:`Gregori.2003`.
    n_e: Quantity
        The electron density in 1/[volume]
    m_i: Quantity
        The mass of the ion
    Z_f: float
        Number of free electrons per ion.
    """
    # Set up all the variables
    lam_ee = _lambda_AD(T_e, 1 * ureg.electron_mass, 1 * ureg.electron_mass)
    lam_ei = _lambda_AD(T_e, 1 * ureg.electron_mass, m_i)
    lam_ii = _lambda_AD(T_e, m_i, m_i)
    b = _b_AD(lam_ee)
    A = _A_AD(T_e, b)
    k_De = _k_D_AD(T_e, n_e)
    k_Di = _k_D_AD(T_e, n_e, Z_f)
    Delta = _Delta_AD(k, k_De, k_Di, lam_ee, lam_ii, lam_ei, b, A)

    return (
        -(Z_f * ureg.elementary_charge**2 / (ureg.epsilon_0 * Delta))
        * (k**2 / (1 + k**2 * lam_ei**2))
    ).to_base_units()


@jax.jit
def S_ii_AD(
    k: Quantity, T_e: Quantity, n_e: Quantity, m_i: Quantity, Z_f: float
) -> Quantity:
    """
    The static ion-ion structure factor, in the approach by Arkhipov
    and Davletov :cite:`Arkhipov.1998`, as presented by :cite:`Gregori.2003` in
    equation (7).

    The method is using the Random Phase Approximation, treating the problem
    semi-classically and uses a pseudopotential between charged particles to
    account for quantum diffraction effects and symmetry

    Parameters
    ----------
    k: Quantity
        The scattering vector length (units of 1/[length])
    T_e: Quantity
        The electron temperature in Kelvin. Use :py:func:`~.T_cf_Greg` for the
        effective temperature used in :cite:`Gregori.2003`.
    n_e: Quantity
        The electron density in 1/[volume]
    m_i: Quantity
        The mass of the ion
    Z_f: float
        Number of free electrons per ion.

    Returns
    -------
    Quantity
        See, the static electron electron structure factor
    """
    n_i = n_e / Z_f
    Phi_ii = _Phi_ii_AD(k, T_e, n_e, m_i, Z_f)
    return (1 - n_i / (ureg.k_B * T_e) * Phi_ii).to_base_units()


@jax.jit
def S_ei_AD(
    k: Quantity, T_e: Quantity, n_e: Quantity, m_i: Quantity, Z_f: float
) -> Quantity:
    """
    The static electron-ion structure factor, in the approach by Arkhipov
    and Davletov :cite:`Arkhipov.1998`, as presented by :cite:`Gregori.2003` in
    equation (7).

    The method is using the Random Phase Approximation, treating the problem
    semi-classically and uses a pseudopotential between charged particles to
    account for quantum diffraction effects and symmetry

    Parameters
    ----------
    k: Quantity
        The scattering vector length (units of 1/[length])
    T_e: Quantity
        The electron temperature in Kelvin. Use :py:func:`~.T_cf_Greg` for the
        effective temperature used in :cite:`Gregori.2003`.
    n_e: Quantity
        The electron density in 1/[volume]
    m_i: Quantity
        The mass of the ion
    Z_f: float
        Number of free electrons per ion.

    Returns
    -------
    Quantity
        See, the static electron electron structure factor
    """
    n_i = n_e / Z_f
    Phi_ei = _Phi_ei_AD(k, T_e, n_e, m_i, Z_f)
    return (-jnpu.sqrt(n_i * n_e) / (ureg.k_B * T_e) * Phi_ei).to_base_units()


@jax.jit
def S_ee_AD(
    k: Quantity, T_e: Quantity, n_e: Quantity, m_i: Quantity, Z_f: float
) -> Quantity:
    """
    The static electron-electron structure factor, in the approach by Arkhipov
    and Davletov :cite:`Arkhipov.1998`, as presented by :cite:`Gregori.2003` in
    equation (7).

    The method is using the Random Phase Approximation, treating the problem
    semi-classically and uses a pseudopotential between charged particles to
    account for quantum diffraction effects and symmetry

    Parameters
    ----------
    k: Quantity
        The scattering vector length (units of 1/[length])
    T_e: Quantity
        The electron temperature in Kelvin. Use :py:func:`~.T_cf_Greg` for the
        effective temperature used in :cite:`Gregori.2003`.
    n_e: Quantity
        The electron density in 1/[volume]
    m_i: Quantity
        The mass of the ion
    Z_f: float
        Number of free electrons per ion.

    Returns
    -------
    Quantity
        See, the static electron electron structure factor
    """
    Phi_ee = _Phi_ee_AD(k, T_e, n_e, m_i, Z_f)
    return (1 - n_e / (ureg.k_B * T_e) * Phi_ee).to_base_units()
