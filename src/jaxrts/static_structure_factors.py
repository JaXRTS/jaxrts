"""
Static structure factors.
"""

import logging

import jax
from jax import numpy as jnp
from jpu import numpy as jnpu
from quadax import quadts as quad

from .plasma_physics import fermi_energy, wiegner_seitz_radius
from .units import Quantity, ureg

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger("__name__")


@jax.jit
def T_cf_Greg(T_e: Quantity, n_e: Quantity) -> Quantity:
    """
    T_cf_Greg(T_e: Quantity, n_e: Quantity) -> Quantity

    The effective temperature to be inserted in the approach by Arkhipov and
    Davletov, see :cite:`Arkhipov.1998` according to :cite:`Gregori.2003`.
    """
    # The fermi temperature
    T_f = fermi_energy(n_e) / ureg.k_B
    dimless_rs = wiegner_seitz_radius(n_e) / ureg.a_0
    T_q = T_f / (1.3251 - 0.1779 * jnpu.sqrt(dimless_rs))
    return jnpu.sqrt(T_e**2 + T_q**2)


@jax.jit
def T_i_eff_Greg(T_i: Quantity, T_D: Quantity) -> Quantity:
    """
    T_i_eff_Greg(T_i: Quantity, T_D: Quantity) -> Quantity

    The effective ion temperature as it is proposed by :cite:`Gregori.2006`.

    Parameters
    ----------
    T_i: Quantity
        Ion temperature.
    T_D: Quantity
        The Debye temperature.
    """
    y0 = 3 / (2 * jnp.pi**2)
    return jnpu.sqrt(T_i**2 + y0 * T_D**2)


@jax.jit
def T_Debye_Bohm_Staver(
    T_e: Quantity, n_e: Quantity, m_i: Quantity, Z_f: float
) -> Quantity:
    """
    Bohm Staver relation, as presented in eqn (3) of :cite:`Gregori.2006`. An
    approximation function for the Debye temperature of 'simple metals'.

    Parameters
    ----------
    T_e: Quantity
        The electron temperature in Kelvin. Use :py:func:`~.T_cf_Greg` for the
        effective temperature used in :cite:`Gregori.2003`.
    n_e: Quantity
        The electron density in 1/[volume].
    m_i: Quantity
        The mass of the ion.
    Z_f: float
        Number of free electrons per ion.
    """
    kmax = (6 * jnp.pi**2 * n_e / Z_f) ** (1 / 3)
    k_De = _k_D_AD(T_e, n_e)
    omega = jnpu.sqrt(
        (Z_f * ureg.elementary_charge**2 * n_e) / (ureg.epsilon_0 * m_i)
    )
    Omega = jnpu.sqrt(omega**2 / (1 + k_De**2 / kmax**2))
    return ureg.hbar / ureg.k_B * Omega


@jax.jit
def _lambda_AD(T: Quantity, m1: Quantity, m2: Quantity) -> Quantity:
    """
    _lambda_AD(T: Quantity, m1: Quantity, m2: Quantity) -> Quantity

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
    """
    _k_D_AD(T: Quantity, n_e: Quantity, Z_f: float = 1.0) -> Quantity
    """
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
    See :cite:`Gregori.2003`, Eqn. (11) or :cite:`Arkhipov.2000` (Eqn 10, which
    is this equation, divided by :math:`k^4`.
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
        + A * k**2 * k_De**2
        # The original paper :cite:`Arkhipov.1998` differs in the subsequent
        # line. This has been rectified by the authors their paper from 2000.
        * (k**2 + k_Di**2 / (1 + k**2 * lamii**2))
        * jnpu.exp(-(k**2) / (4 * b))
    )


@jax.jit
def _T_rs_Greg2006(
    T_r: Quantity, T_s: Quantity, m_r: Quantity, m_s: Quantity
) -> Quantity:
    """
    Calculate the effective temperature between an interacting pair of species
    ``r`` and ``s`` (ions and / or electrons). See eqn 4 in
    :cite:`Gregori.2006`.

    Parameters
    ----------
    T_r: Quantity
        Temperature of species ``r``
    T_s: Quantity
        Temperature of species ``s``
    m_r: Quantity
        Mass of species ``r``
    m_s: Quantity
        Mass of species ``s``

    Returns
    -------
    Quantity
        The effective temperature. If both temperatures are identical, than the
        result is this temperature.
    """
    return (m_s * T_r + m_r * T_s) / (m_r + m_s)


@jax.jit
def _Phi_ee_AD(
    k: Quantity,
    T_e: Quantity,
    T_i: Quantity,
    n_e: Quantity,
    m_i: Quantity,
    Z_f: float,
) -> Quantity:
    """
    See :cite:`Gregori.2003`, eqn (8) for the coefficient
    :math:`\\Phi_ee(k)` in the approach by Arkhipov and Davletov
    :cite:`Arkhipov.1998`.

    While the seminal papers treated the electron- and ion temperature to be
    equal, we follow the work of :cite:`Gregori.2006` to allow for different
    temperatures of the two components. The results of :cite:`Arkhipov.1998`
    and :cite:`Gregori.2003` can be obtained by setting ``T_e == T_i``

    Parameters
    ----------
    k: Quantity
        The scattering vector length (units of 1/[length])
    T_e: Quantity
        The electron temperature in Kelvin. Use :py:func:`~.T_cf_Greg` for the
        effective temperature used in :cite:`Gregori.2003`.
    T_i: Quantity
        The ion temperature in Kelvin.
    n_e: Quantity
        The electron density in 1/[volume]
    m_i: Quantity
        The mass of the ion
    Z_f: float
        Number of free electrons per ion.
    """
    # Set up all the variables
    T_ei = _T_rs_Greg2006(T_e, T_i, 1 * ureg.electron_mass, m_i)
    lam_ee = _lambda_AD(T_e, 1 * ureg.electron_mass, 1 * ureg.electron_mass)
    lam_ei = _lambda_AD(T_ei, 1 * ureg.electron_mass, m_i)
    lam_ii = _lambda_AD(T_i, m_i, m_i)
    b = _b_AD(lam_ee)
    A = _A_AD(T_e, b)
    k_De = _k_D_AD(T_e, n_e)
    k_Di = _k_D_AD(T_i, n_e, Z_f)
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
    k: Quantity,
    T_e: Quantity,
    T_i: Quantity,
    n_e: Quantity,
    m_i: Quantity,
    Z_f: float,
) -> Quantity:
    """
    See :cite:`Gregori.2003`, eqn (9) for the coefficient
    :math:`\\Phi_ii(k)` in the approach by Arkhipov and Davletov
    :cite:`Arkhipov.1998`.

    While the seminal papers treated the electron- and ion temperature to be
    equal, we follow the work of :cite:`Gregori.2006` to allow for different
    temperatures of the two components. The results of :cite:`Arkhipov.1998`
    and :cite:`Gregori.2003` can be obtained by setting ``T_e == T_i``

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
    T_ei = _T_rs_Greg2006(T_e, T_i, 1 * ureg.electron_mass, m_i)
    lam_ee = _lambda_AD(T_e, 1 * ureg.electron_mass, 1 * ureg.electron_mass)
    lam_ei = _lambda_AD(T_ei, 1 * ureg.electron_mass, m_i)
    lam_ii = _lambda_AD(T_i, m_i, m_i)
    b = _b_AD(lam_ee)
    A = _A_AD(T_e, b)
    k_De = _k_D_AD(T_e, n_e)
    k_Di = _k_D_AD(T_i, n_e, Z_f)
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
    k: Quantity,
    T_e: Quantity,
    T_i: Quantity,
    n_e: Quantity,
    m_i: Quantity,
    Z_f: float,
) -> Quantity:
    """
    See :cite:`Gregori.2003`, eqn (10) for the coefficient
    :math:`\\Phi_ee(k)` in the approach by Arkhipov and Davletov
    :cite:`Arkhipov.1998`.

    While the seminal papers treated the electron- and ion temperature to be
    equal, we follow the work of :cite:`Gregori.2006` to allow for different
    temperatures of the two components. The results of :cite:`Arkhipov.1998`
    and :cite:`Gregori.2003` can be obtained by setting ``T_e == T_i``

    Parameters
    ----------
    k: Quantity
        The scattering vector length (units of 1/[length])
    T_e: Quantity
        The electron temperature in Kelvin. Use :py:func:`~.T_cf_Greg` for the
        effective temperature used in :cite:`Gregori.2003`.
    T_i: Quantity
        The ion temperature in Kelvin.
    n_e: Quantity
        The electron density in 1/[volume]
    m_i: Quantity
        The mass of the ion
    Z_f: float
        Number of free electrons per ion.
    """
    # Set up all the variables
    T_ei = _T_rs_Greg2006(T_e, T_i, 1 * ureg.electron_mass, m_i)
    lam_ee = _lambda_AD(T_e, 1 * ureg.electron_mass, 1 * ureg.electron_mass)
    lam_ei = _lambda_AD(T_ei, 1 * ureg.electron_mass, m_i)
    lam_ii = _lambda_AD(T_i, m_i, m_i)
    b = _b_AD(lam_ee)
    A = _A_AD(T_e, b)
    k_De = _k_D_AD(T_e, n_e)
    k_Di = _k_D_AD(T_i, n_e, Z_f)
    Delta = _Delta_AD(k, k_De, k_Di, lam_ee, lam_ii, lam_ei, b, A)

    return (
        -(Z_f * ureg.elementary_charge**2 / (ureg.epsilon_0 * Delta))
        * (k**2 / (1 + k**2 * lam_ei**2))
    ).to_base_units()


@jax.jit
def S_ii_AD(
    k: Quantity,
    T_e: Quantity,
    T_i: Quantity,
    n_e: Quantity,
    m_i: Quantity,
    Z_f: float,
) -> Quantity:
    """
    The static ion-ion structure factor, in the approach by Arkhipov
    and Davletov :cite:`Arkhipov.1998`, as presented by :cite:`Gregori.2003` in
    equation (7).

    The method is using the Random Phase Approximation, treating the problem
    semi-classically and uses a pseudopotential between charged particles to
    account for quantum diffraction effects and symmetry.

    Parameters
    ----------
    k: Quantity
        The scattering vector length (units of 1/[length])
    T_e: Quantity
        The electron temperature in Kelvin. Use :py:func:`~.T_cf_Greg` for the
        effective temperature used in :cite:`Gregori.2003`.
    T_i: Quantity
        The ion temperature in Kelvin.
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
    Phi_ii = _Phi_ii_AD(k, T_e, T_i, n_e, m_i, Z_f)
    return (1 - n_i / (ureg.k_B * T_i) * Phi_ii).to_base_units()


@jax.jit
def S_ei_AD(
    k: Quantity,
    T_e: Quantity,
    T_i: Quantity,
    n_e: Quantity,
    m_i: Quantity,
    Z_f: float,
) -> Quantity:
    """
    The static electron-ion structure factor, in the approach by Arkhipov
    and Davletov :cite:`Arkhipov.1998`, as presented by :cite:`Gregori.2003` in
    equation (7).

    The method is using the Random Phase Approximation, treating the problem
    semi-classically and uses a pseudopotential between charged particles to
    account for quantum diffraction effects and symmetry

    While the seminal papers treated the electron- and ion temperature to be
    equal, we follow the work of :cite:`Gregori.2006` to allow for different
    temperatures of the two components. The results of :cite:`Arkhipov.1998`
    and :cite:`Gregori.2003` can be obtained by setting ``T_e == T_i``

    Parameters
    ----------
    k: Quantity
        The scattering vector length (units of 1/[length])
    T_e: Quantity
        The electron temperature in Kelvin. Use :py:func:`~.T_cf_Greg` for the
        effective temperature used in :cite:`Gregori.2003`.
    T_i: Quantity
        The ion temperature in Kelvin.
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
    T_ei = _T_rs_Greg2006(T_e, T_i, 1 * ureg.electron_mass, m_i)
    n_i = n_e / Z_f
    Phi_ei = _Phi_ei_AD(k, T_e, T_i, n_e, m_i, Z_f)
    return (-jnpu.sqrt(n_i * n_e) / (ureg.k_B * T_ei) * Phi_ei).to_base_units()


@jax.jit
def S_ee_AD(
    k: Quantity,
    T_e: Quantity,
    T_i: Quantity,
    n_e: Quantity,
    m_i: Quantity,
    Z_f: float,
) -> Quantity:
    """
    The static electron-electron structure factor, in the approach by Arkhipov
    and Davletov :cite:`Arkhipov.1998`, as presented by :cite:`Gregori.2003` in
    equation (7).

    The method is using the Random Phase Approximation, treating the problem
    semi-classically and uses a pseudopotential between charged particles to
    account for quantum diffraction effects and symmetry

    While the seminal papers treated the electron- and ion temperature to be
    equal, we follow the work of :cite:`Gregori.2006` to allow for different
    temperatures of the two components. The results of :cite:`Arkhipov.1998`
    and :cite:`Gregori.2003` can be obtained by setting ``T_e == T_i``

    The function is amended compared to the earlier paper by a summand given in
    :Gregori.2006`, which cite Seuferling at al. For both temperatures being
    equal, this second term vanishes.

    Parameters
    ----------
    k: Quantity
        The scattering vector length (units of 1/[length])
    T_e: Quantity
        The electron temperature in Kelvin. Use :py:func:`~.T_cf_Greg` for the
        effective temperature used in :cite:`Gregori.2003`.
    T_i: Quantity
        The ion temperature in Kelvin.
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
    Phi_ee = _Phi_ee_AD(k, T_e, T_i, n_e, m_i, Z_f)
    S_ee_AD = (1 - n_e / (ureg.k_B * T_e) * Phi_ee).to_base_units()
    # This the addition by Gregori.2006:
    S_ei = S_ei_AD(k, T_e, T_i, n_e, m_i, Z_f)
    S_ii = S_ii_AD(k, T_e, T_i, n_e, m_i, Z_f)
    q = jnpu.sqrt(Z_f) * S_ei / S_ii
    S_ee_Greg_addition = (
        (T_e / T_i - 1) * jnp.abs(q.m_as(ureg.dimensionless)) ** 2 / Z_f * S_ii
    )
    return S_ee_AD - S_ee_Greg_addition


@jax.jit
def g_ee_ABD(
    r: Quantity,
    T_e: Quantity,
    T_i: Quantity,
    n_e: Quantity,
    m_i: Quantity,
    Z_f: float,
) -> Quantity:
    """
    The radial electron-electron distribution function, in the approach by
    Arkhipov, Baimbetov and Davletov :cite:`Arkhipov.2000` (Eqn. 17).

    The method is using the Random Phase Approximation, treating the problem
    semi-classically and uses a pseudopotential between charged particles to
    account for quantum diffraction effects and symmetry

    While the seminal papers treated the electron- and ion temperature to be
    equal, we follow the work of :cite:`Gregori.2006` to allow for different
    temperatures of the two components. The results of :cite:`Arkhipov.1998`
    and :cite:`Gregori.2003` can be obtained by setting ``T_e == T_i``

    Parameters
    ----------
    r: Quantity
        Distance in [length] units.
    T_e: Quantity
        The electron temperature in Kelvin. Use :py:func:`~.T_cf_Greg` for the
        effective temperature used in :cite:`Gregori.2003`.
    T_i: Quantity
        The ion temperature in Kelvin.
    n_e: Quantity
        The electron density in 1/[volume]
    m_i: Quantity
        The mass of the ion
    Z_f: float
        Number of free electrons per ion.

    Returns
    -------
    Quantity
        The radial electron-electron distribution function in the pair
        correlation approximation.
    """

    def to_integrate(k):
        k = k / (1 * ureg.angstrom)
        Phi_ee_k = _Phi_ee_AD(k, T_e, T_i, n_e, m_i, Z_f)
        fac = jnpu.sin(k * r[:, jnp.newaxis]) * k
        return (Phi_ee_k * fac).m_as(ureg.joule * ureg.angstrom**2)

    integ, err = quad(to_integrate, [0, jnp.inf], epsabs=1e-20, epsrel=1e-20)

    integ *= 1 * ureg.joule * ureg.angstrom
    integ /= (2 * jnp.pi) ** 3

    Phi_ee_r = 4 * jnp.pi / r * integ

    return jnpu.exp(-Phi_ee_r / (ureg.k_B * T_e)) * (1 * ureg.dimensionless)


@jax.jit
def g_ii_ABD(
    r: Quantity,
    T_e: Quantity,
    T_i: Quantity,
    n_e: Quantity,
    m_i: Quantity,
    Z_f: float,
) -> Quantity:
    """
    The radial ion-ion distribution function, in the approach by Arkhipov,
    Baimbetov and Davletov :cite:`Arkhipov.2000` (Eqn. 17).

    The method is using the Random Phase Approximation, treating the problem
    semi-classically and uses a pseudopotential between charged particles to
    account for quantum diffraction effects and symmetry

    While the seminal papers treated the electron- and ion temperature to be
    equal, we follow the work of :cite:`Gregori.2006` to allow for different
    temperatures of the two components. The results of :cite:`Arkhipov.1998`
    and :cite:`Gregori.2003` can be obtained by setting ``T_e == T_i``

    Parameters
    ----------
    r: Quantity
        Distance in [length] units.
    T_e: Quantity
        The electron temperature in Kelvin. Use :py:func:`~.T_cf_Greg` for the
        effective temperature used in :cite:`Gregori.2003`.
    T_i: Quantity
        The ion temperature in Kelvin.
    n_e: Quantity
        The electron density in 1/[volume]
    m_i: Quantity
        The mass of the ion
    Z_f: float
        Number of free electrons per ion.

    Returns
    -------
    Quantity
        The radial ion-ion distribution function in the pair correlation
        approximation.
    """

    def to_integrate(k):
        k = k / (1 * ureg.angstrom)
        Phi_ii_k = _Phi_ii_AD(k, T_e, T_i, n_e, m_i, Z_f)
        fac = jnpu.sin(k * r[:, jnp.newaxis]) * k
        return (Phi_ii_k * fac).m_as(ureg.joule * ureg.angstrom**2)

    integ, err = quad(to_integrate, [0, jnp.inf], epsabs=1e-20, epsrel=1e-20)

    integ *= 1 * ureg.joule * ureg.angstrom
    integ /= (2 * jnp.pi) ** 3

    Phi_ii_r = 4 * jnp.pi / r * integ

    return jnpu.exp(-Phi_ii_r / (ureg.k_B * T_i)) * (1 * ureg.dimensionless)


@jax.jit
def g_ei_ABD(
    r: Quantity,
    T_e: Quantity,
    T_i: Quantity,
    n_e: Quantity,
    m_i: Quantity,
    Z_f: float,
) -> Quantity:
    """
    The radial electron-ion distribution function, in the approach by Arkhipov,
    Baimbetov and Davletov :cite:`Arkhipov.2000` (Eqn. 17)

    The method is using the Random Phase Approximation, treating the problem
    semi-classically and uses a pseudopotential between charged particles to
    account for quantum diffraction effects and symmetry.

    While the seminal papers treated the electron- and ion temperature to be
    equal, we follow the work of :cite:`Gregori.2006` to allow for different
    temperatures of the two components. The results of :cite:`Arkhipov.1998`
    and :cite:`Gregori.2003` can be obtained by setting ``T_e == T_i``

    Parameters
    ----------
    r: Quantity
        Distance in [length] units.
    T_e: Quantity
        The electron temperature in Kelvin. Use :py:func:`~.T_cf_Greg` for the
        effective temperature used in :cite:`Gregori.2003`.
    T_i: Quantity
        The ion temperature in Kelvin.
    n_e: Quantity
        The electron density in 1/[volume]
    m_i: Quantity
        The mass of the ion
    Z_f: float
        Number of free electrons per ion.

    Returns
    -------
    Quantity
        The radial electron-ion distribution function in the pair correlation
        approximation.
    """
    T_ei = _T_rs_Greg2006(T_e, T_i, 1 * ureg.electron_mass, m_i)

    def to_integrate(k):
        k = k / (1 * ureg.angstrom)
        Phi_ei_k = _Phi_ei_AD(k, T_e, T_i, n_e, m_i, Z_f)
        fac = jnpu.sin(k * r[:, jnp.newaxis]) * k
        return (Phi_ei_k * fac).m_as(ureg.joule * ureg.angstrom**2)

    integ, err = quad(to_integrate, [0, jnp.inf], epsabs=1e-20, epsrel=1e-20)

    integ *= 1 * ureg.joule * ureg.angstrom
    integ /= (2 * jnp.pi) ** 3

    Phi_ei_r = 4 * jnp.pi / r * integ

    return jnpu.exp(-Phi_ei_r / (ureg.k_B * T_ei)) * (1 * ureg.dimensionless)
