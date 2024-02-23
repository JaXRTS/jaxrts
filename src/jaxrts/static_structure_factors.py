"""
Static structure factors.
"""

from jax import numpy as jnp
from jpu import numpy as jnpu

from .units import ureg, Quantity
from .plasma_physics import fermi_energy, wiegner_seitz_radius


def _T_cf_AD(T_e: Quantity, n_e: Quantity) -> Quantity:
    """
    The effective temperature in the approach by Arkhipov and Davletov, see
    :cite:`Arkhipov.1998` and :cite:`Gregori.2003`.
    """
    # The fermi temperature
    T_f = fermi_energy(n_e) / ureg.k_B
    dimless_rs = wiegner_seitz_radius(n_e) / ureg.a_0
    T_q = T_f / (1.3251 - 0.1779 * jnpu.sqrt(dimless_rs))
    return jnpu.sqrt(T_e**2 + T_q**2)


def _lambda_AD(T: Quantity, m1: Quantity, m2: Quantity) -> Quantity:
    """
    :cite:`Gregori.2003` states this as the thermal de Broglie Wavelength, but
    this is not reproducing the known formula for ``m1 == m2``.
    However, both :cite:`Arkhipov.1998` and :cite:`Gregori` use this notation.
    """
    # The 1 * m1 is required so allow this function to handle Units, too.
    mu = (m1 * m2) / (1 * m1 + 1 * m2)
    denumerator = 2 * jnp.pi * mu * ureg.k_B * T
    return ureg.hbar / jnpu.sqrt(denumerator)


def _k_De_AD(T: Quantity, n_e: Quantity) -> Quantity:
    numerator = n_e * ureg.elementary_charge**2
    denumerator = ureg.epsilon_0 * ureg.k_B * T
    return jnpu.sqrt(numerator / denumerator)


def _k_Di_AD(T: Quantity, n_e: Quantity, Zf: float) -> Quantity:
    numerator = Zf * n_e * ureg.elementary_charge**2
    denumerator = ureg.epsilon_0 * ureg.k_B * T
    return jnpu.sqrt(numerator / denumerator)


def _b_AD(lam_ee: Quantity) -> Quantity:
    return 1 / (lam_ee**2 * jnp.pi * jnp.log(2))


def _A_AD(T_cf: Quantity, b: Quantity) -> Quantity:
    """
    A helper function for the approach by Arkhipov and Davletov, see
    :cite:`Arkhipov.1998` and :cite:`Gregori.2003`.
    """
    return (
        ureg.k_B
        * T_cf
        * jnp.log(2)
        * jnp.pi ** (3 / 2)
        * b ** (-3 / 2)
        * ureg.epsilon_0
        / ureg.elementary_charge**2
    )


def _Delta_AD(
    k: Quantity,
    kDe: Quantity,
    kDi: Quantity,
    lamee: Quantity,
    lamii: Quantity,
    lamei: Quantity,
    b: Quantity,
    A: Quantity,
) -> Quantity:
    """
    :cite:`Gregori.2003`, Eqn. (11)
    """
    # split the sum in 5 parts
    p1 = k**4

    p2 = (k**2 * kDe**2) / (1 + k**2 * lamee**2)

    p3 = (k**2 * kDi**2) / (1 + k**2 * lamii**2)

    p4_denom1 = (1 + k**2 * lamee**2) * (1 + k**2 * lamii**2)
    p4_denom2 = (1 + k**2 * lamei**2) ** 2
    p4 = kDe**2 * kDi**2 * (1 / p4_denom1 - 1 / p4_denom2)

    p5 = (
        (A * k**2 * kDe**2)
        * (k**2 + kDi**2 / (1 * k**2 * lamii**2))
        * jnpu.exp(-(k**2) / (4 * b))
    )

    return p1 + p2 + p3 + p4 + p5


def _Phi_ee_AD(
    k: Quantity, T_e: Quantity, n_e: Quantity, m_i: Quantity, Zf: float
) -> Quantity:
    """
    See :cite:`Gregori.2003`, eqn (8) for the coefficient
    :math:`\\Phi_ee(k)` in the approach by Arkhipov and Davletov
    :cite:`Arkhipov.1998`.
    """
    # Set up all the variables
    T_cf = _T_cf_AD(T_e, n_e)
    lam_ee = _lambda_AD(T_cf, ureg.electron_mass, ureg.electron_mass)
    lam_ei = _lambda_AD(T_cf, ureg.electron_mass, m_i)
    lam_ii = _lambda_AD(T_cf, m_i, m_i)
    b = _b_AD(lam_ee)
    A = _A_AD(T_cf, b)
    k_De = _k_De_AD(T_e, n_e)
    k_Di = _k_Di_AD(T_e, n_e, Zf)
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
    return pref * (sum1 + sum2 + sum3).to_base_units()


def _Phi_ii_AD(
    k: Quantity, T_e: Quantity, n_e: Quantity, m_i: Quantity, Zf: float
) -> Quantity:
    """
    See :cite:`Gregori.2003`, eqn (9) for the coefficient
    :math:`\\Phi_ii(k)` in the approach by Arkhipov and Davletov
    :cite:`Arkhipov.1998`.
    """
    # Set up all the variables
    T_cf = _T_cf_AD(T_e, n_e)
    lam_ee = _lambda_AD(T_cf, ureg.electron_mass, ureg.electron_mass)
    lam_ei = _lambda_AD(T_cf, ureg.electron_mass, m_i)
    lam_ii = _lambda_AD(T_cf, m_i, m_i)
    b = _b_AD(lam_ee)
    A = _A_AD(T_cf, b)
    k_De = _k_De_AD(T_e, n_e)
    k_Di = _k_Di_AD(T_e, n_e, Zf)
    Delta = _Delta_AD(k, k_De, k_Di, lam_ee, lam_ii, lam_ei, b, A)

    pref = (
        Zf**2 * ureg.elementary_charge**2 / (ureg.vacuum_permittivity * Delta)
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
    return pref * (sum1 + sum2 + sum3).to_base_units()


def _Phi_ei_AD(
    k: Quantity, T_e: Quantity, n_e: Quantity, m_i: Quantity, Zf: float
) -> Quantity:
    """
    See :cite:`Gregori.2003`, eqn (10) for the coefficient
    :math:`\\Phi_ee(k)` in the approach by Arkhipov and Davletov
    :cite:`Arkhipov.1998`.
    """
    # Set up all the variables
    T_cf = _T_cf_AD(T_e, n_e)
    lam_ee = _lambda_AD(T_cf, ureg.electron_mass, ureg.electron_mass)
    lam_ei = _lambda_AD(T_cf, ureg.electron_mass, m_i)
    lam_ii = _lambda_AD(T_cf, m_i, m_i)
    b = _b_AD(lam_ee)
    A = _A_AD(T_cf, b)
    k_De = _k_De_AD(T_e, n_e)
    k_Di = _k_Di_AD(T_e, n_e, Zf)
    Delta = _Delta_AD(k, k_De, k_Di, lam_ee, lam_ii, lam_ei, b, A)

    return -(Zf * ureg.elementary_charge**2 / (ureg.epsilon_0 * Delta)) * (
        k**2 / (1 + k**2 * lam_ei**2)
    ).to_base_units()


def S_ii_AD(
    k: Quantity, T_e: Quantity, n_e: Quantity, m_i: Quantity, Zf: float
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
        The electron temperature in Kelvin
    n_e: Quantity
        The electron density in 1/[volume]
    m_i: Quantity
        The mass of the ion
    Zf: float
        Number of free electrons per ion.

    Returns
    -------
    Quantity
        See, the static electron electron structure factor
    """
    T_cf = _T_cf_AD(T_e, n_e)
    n_i = Zf * n_e
    Phi_ii = _Phi_ii_AD(k, T_e, n_e, m_i, Zf)
    return 1 - n_i / (ureg.k_B * T_cf) * Phi_ii


def S_ei_AD(
    k: Quantity, T_e: Quantity, n_e: Quantity, m_i: Quantity, Zf: float
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
        The electron temperature in Kelvin
    n_e: Quantity
        The electron density in 1/[volume]
    m_i: Quantity
        The mass of the ion
    Zf: float
        Number of free electrons per ion.

    Returns
    -------
    Quantity
        See, the static electron electron structure factor
    """
    T_cf = _T_cf_AD(T_e, n_e)
    n_i = Zf * n_e
    Phi_ei = _Phi_ei_AD(k, T_e, n_e, m_i, Zf)
    return -jnpu.sqrt(n_i * n_e) / (ureg.k_B * T_cf) * Phi_ei


def S_ee_AD(
    k: Quantity, T_e: Quantity, n_e: Quantity, m_i: Quantity, Zf: float
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
        The electron temperature in Kelvin
    n_e: Quantity
        The electron density in 1/[volume]
    m_i: Quantity
        The mass of the ion
    Zf: float
        Number of free electrons per ion.

    Returns
    -------
    Quantity
        See, the static electron electron structure factor
    """
    T_cf = _T_cf_AD(T_e, n_e)
    Phi_ee = _Phi_ee_AD(k, T_e, n_e, m_i, Zf)
    return 1 - n_e / (ureg.k_B * T_cf) * Phi_ee
