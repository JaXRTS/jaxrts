"""
This submodule is dedicated to the calculation of the ion-feature.
"""

from .units import ureg, Quantity
from .electron_feature import dielectric_function_salpeter
from typing import List

import jax
from jax import jit
import jax.numpy as jnp
import numpy as onp

import logging

logger = logging.getLogger(__name__)

import jpu

jax.config.update("jax_enable_x64", True)


@jit
def k_D(n_e: Quantity, T_e: Quantity, Z_f: float = 1.0):

    T_q = 0.0
    T_cf = jpu.numpy.sqrt(T_e**2 + T_q**2)

    return jpu.numpy.sqrt(
        (Z_f * n_e * (1 * ureg.elementary_charge) ** 2)
        / ((1 * ureg.vacuum_permittivity) * ureg.boltzmann_constant * T_cf)
    ).to_base_units()


@jit
def reduced_mass(m1: Quantity, m2: Quantity) -> Quantity:
    """
    Calculates the reduced mass of two interacting particles.

    Parameters
    ----------
    m1  :   Quantity
            The mass of the first particle.
    m2  :   Quantity
            The mass of the second particle.

    Returns
    -------
    mu:    Quantity
           The reduced mass of the interacting pair of particles.
    """
    return m1 * m2 / (m1 + m2)


@jit
def Delta(
    k: Quantity,
    m_ion: Quantity,
    n_e: Quantity,
    T_e: Quantity,
    T_cf: Quantity,
    Z_f: float,
) -> Quantity:

    k_De = k_D(n_e, T_e)
    k_Di = k_D(n_e, T_e, Z_f=Z_f)

    therm_dbwl_ee = (
        (1 * ureg.planck_constant / (2 * jnp.pi))
        / jpu.numpy.sqrt(
            2
            * jnp.pi
            * reduced_mass(1 * ureg.electron_mass, 1 * ureg.electron_mass)
            * 1
            * ureg.boltzmann_constant
            * T_cf
        )
    ).to_base_units()
    therm_dbwl_ii = (
        (1 * ureg.planck_constant / (2 * jnp.pi))
        / jpu.numpy.sqrt(
            2
            * jnp.pi
            * reduced_mass(m_ion, m_ion)
            * 1
            * ureg.boltzmann_constant
            * T_cf
        )
    ).to_base_units()
    therm_dbwl_ei = (
        (1 * ureg.planck_constant / (2 * jnp.pi))
        / jpu.numpy.sqrt(
            2
            * jnp.pi
            * reduced_mass(1 * ureg.electron_mass, m_ion)
            * 1
            * ureg.boltzmann_constant
            * T_cf
        )
    ).to_base_units()

    b = 1.0 / (therm_dbwl_ee**2 * jnp.pi * jnp.log(2))
    A = (
        ureg.boltzmann_constant
        * T_cf
        * jnp.log(2)
        * (jnp.pi ** (3.0 / 2.0))
        * b ** (-3.0 / 2.0)
        * (1 * ureg.vacuum_permittivity)
        / (1 * ureg.elementary_charge) ** 2
    ).to_base_units()

    return (
        k**4
        + (k**2 * k_De**2) / (1 + k**2 * therm_dbwl_ee**2)
        + (k**2 * k_Di**2) / (1 + k**2 * therm_dbwl_ii**2)
        + k_De**2
        * k_Di**2
        * (
            1.0
            / ((1 + k**2 * therm_dbwl_ee**2) * (1 + k**2 * therm_dbwl_ii**2))
            - 1 / (1 + k**2 * therm_dbwl_ei**2) ** 2
        )
        + A
        * k**2
        * k_De**2
        * (k**2 + k_Di**2 / (1 + k**2 * therm_dbwl_ii**2))
        * jpu.numpy.exp(-(k**2) / (4 * b))
    )


@jit
def phi_ee(
    k: Quantity,
    m_ion: Quantity,
    n_e: Quantity,
    T_e: Quantity,
    T_cf: Quantity,
    Z_f: float,
) -> Quantity:
    """
    Calculates the coefficient for the electron-electron static structure and corresponds to
    the Fourier transform of the screened Coulomb potential.

    Parameters
    ----------
    k :     Quantity
            Length of the scattering number (given by the scattering angle and the
            energies of the incident photons (unit: 1 / [length]).
    m_ion : Quantity
            The mass of the ion for which the ion feature is calculated.
    n_e:    Quantity
            The electron number density.
    T_e:    Quantity
            The electron temperature.
    T_cf:   Quantity
            The effective temperature at which the correlation properties are calculated.
            This corrected temperature is chosen such that the temperature of an electron liquid
            obeying classical statistics exactly gives the same correlation energy of a degenerate quantum fluid at T_e = 0,
            obtained from QMC simulations (Gregori.2004)
    Z_f:    float
            The number of electrons not tightly bound to the atom = valence electrons

    Returns
    -------
    phi_ee: Quantity
            The electron-electron static structure coefficient.
    """

    k_De = k_D(n_e, T_e)
    k_Di = k_D(n_e, T_e, Z_f=Z_f)

    therm_dbwl_ee = (
        (1 * ureg.planck_constant / (2 * jnp.pi))
        / jpu.numpy.sqrt(
            2
            * jnp.pi
            * reduced_mass(1 * ureg.electron_mass, 1 * ureg.electron_mass)
            * 1
            * ureg.boltzmann_constant
            * T_cf
        )
    ).to_base_units()
    therm_dbwl_ii = (
        (1 * ureg.planck_constant / (2 * jnp.pi))
        / jpu.numpy.sqrt(
            2
            * jnp.pi
            * reduced_mass(m_ion, m_ion)
            * 1
            * ureg.boltzmann_constant
            * T_cf
        )
    ).to_base_units()
    therm_dbwl_ei = (
        (1 * ureg.planck_constant / (2 * jnp.pi))
        / jpu.numpy.sqrt(
            2
            * jnp.pi
            * reduced_mass(1 * ureg.electron_mass, m_ion)
            * 1
            * ureg.boltzmann_constant
            * T_cf
        )
    ).to_base_units()

    b = 1.0 / (therm_dbwl_ee**2 * jnp.pi * jnp.log(2))
    A = (
        ureg.boltzmann_constant
        * T_cf
        * jnp.log(2)
        * (jnp.pi ** (3.0 / 2.0))
        * b ** (-3.0 / 2.0)
        * (1 * ureg.vacuum_permittivity)
        / (1 * ureg.elementary_charge) ** 2
    ).to_base_units()

    return (
        ((1 * ureg.elementary_charge) ** 2)
        / (
            (1 * ureg.vacuum_permittivity).to_base_units()
            * Delta(k, m_ion, n_e, T_e, T_cf, Z_f)
        )
    ) * (
        k**2 / (1 + k**2 * therm_dbwl_ee**2)
        + k_Di**2
        * (
            1 / ((1 + k**2 * therm_dbwl_ee**2) * (1 + k**2 * therm_dbwl_ii**2))
            - 1 / (1 + k**2 * therm_dbwl_ei**2) ** 2
        )
        + A
        * k**2
        * jpu.numpy.exp(-(k**2) / (4 * b))
        * (k**2 + k_Di**2 / (1 + k**2 * therm_dbwl_ii**2))
    )


@jit
def phi_ii(
    k: Quantity,
    m_ion: Quantity,
    n_e: Quantity,
    T_e: Quantity,
    T_cf: Quantity,
    Z_f: float,
) -> Quantity:
    """
    Calculates the coefficient for the ion-ion static structure and corresponds to
    the Fourier transform of the screened Coulomb potential.

    Parameters
    ----------
    k :     Quantity
            Length of the scattering number (given by the scattering angle and the
            energies of the incident photons (unit: 1 / [length]).
    m_ion : Quantity
            The mass of the ion for which the ion feature is calculated.
    n_e:    Quantity
            The electron number density.
    T_e:    Quantity
            The electron temperature.
    T_cf:   Quantity
            The effective temperature at which the correlation properties are calculated.
            This corrected temperature is chosen such that the temperature of an electron liquid
            obeying classical statistics exactly gives the same correlation energy of a degenerate quantum fluid at T_e = 0,
            obtained from QMC simulations (Gregori.2004)
    Z_f:    float
            The number of electrons not tightly bound to the atom = valence electrons

    Returns
    -------
    phi_ei: Quantity
            The electron-ion static structure coefficient.
    """

    k_De = k_D(n_e, T_e)
    k_Di = k_D(n_e, T_e, Z_f=Z_f)

    therm_dbwl_ee = (
        (1 * ureg.planck_constant / (2 * jnp.pi))
        / jpu.numpy.sqrt(
            2
            * jnp.pi
            * reduced_mass(1 * ureg.electron_mass, 1 * ureg.electron_mass)
            * 1
            * ureg.boltzmann_constant
            * T_cf
        )
    ).to_base_units()
    therm_dbwl_ii = (
        (1 * ureg.planck_constant / (2 * jnp.pi))
        / jpu.numpy.sqrt(
            2
            * jnp.pi
            * reduced_mass(m_ion, m_ion)
            * 1
            * ureg.boltzmann_constant
            * T_cf
        )
    ).to_base_units()
    therm_dbwl_ei = (
        (1 * ureg.planck_constant / (2 * jnp.pi))
        / jpu.numpy.sqrt(
            2
            * jnp.pi
            * reduced_mass(1 * ureg.electron_mass, m_ion)
            * 1
            * ureg.boltzmann_constant
            * T_cf
        )
    ).to_base_units()

    b = 1.0 / (therm_dbwl_ee**2 * jnp.pi * jnp.log(2))
    A = (
        ureg.boltzmann_constant
        * T_cf
        * jnp.log(2)
        * (jnp.pi ** (3.0 / 2.0))
        * b ** (-3.0 / 2.0)
        * (1 * ureg.vacuum_permittivity)
        / (1 * ureg.elementary_charge) ** 2
    ).to_base_units()

    return (
        (Z_f**2 * (1 * ureg.elementary_charge) ** 2)
        / (
            (1 * ureg.vacuum_permittivity).to_base_units()
            * Delta(k, m_ion, n_e, T_e, T_cf, Z_f)
        )
    ) * (
        k**2 / (1 + k**2 * therm_dbwl_ii**2)
        + k_De**2
        * (
            1 / ((1 + k**2 * therm_dbwl_ee**2) * (1 + k**2 * therm_dbwl_ii**2))
            - 1 / ((1 + k**2 * therm_dbwl_ei**2) ** 2)
        )
        + A
        * k**2
        * k_De**2
        * jpu.numpy.exp(-(k**2) / (4 * b))
        / (1 + k**2 * therm_dbwl_ii**2)
    )


@jit
def phi_ei(
    k: Quantity,
    m_ion: Quantity,
    n_e: Quantity,
    T_e: Quantity,
    T_cf: Quantity,
    Z_f: float,
) -> Quantity:
    """
    Calculates the coefficient for the electro-ion static structure and corresponds to
    the Fourier transform of the screened Coulomb potential.

    Parameters
    ----------
    k :     Quantity
            Length of the scattering number (given by the scattering angle and the
            energies of the incident photons (unit: 1 / [length]).
    m_ion : Quantity
            The mass of the ion for which the ion feature is calculated.
    n_e:    Quantity
            The electron number density.
    T_e:    Quantity
            The electron temperature.
    T_cf:   Quantity
            The effective temperature at which the correlation properties are calculated.
            This corrected temperature is chosen such that the temperature of an electron liquid
            obeying classical statistics exactly gives the same correlation energy of a degenerate quantum fluid at T_e = 0,
            obtained from QMC simulations (Gregori.2004)
    Z_f:    float
            The number of electrons not tightly bound to the atom = valence electrons

    Returns
    -------
    phi_ii: Quantity
            The ion-ion static structure coefficient.
    """

    therm_dbwl_ei = (
        (1 * ureg.planck_constant / (2 * jnp.pi))
        / jpu.numpy.sqrt(
            2
            * jnp.pi
            * reduced_mass(1 * ureg.electron_mass, m_ion)
            * 1
            * ureg.boltzmann_constant
            * T_cf
        )
    ).to_base_units()

    return (
        -Z_f
        * (1 * ureg.elementary_charge) ** 2
        / (
            (1 * ureg.vacuum_permittivity).to_base_units()
            * Delta(k, m_ion, n_e, T_e, T_cf, Z_f)
        )
    ) * (k**2 / (1 + k**2 * therm_dbwl_ei**2))


@jit
def S_ee(
    k: Quantity, m_ion: Quantity, n_e: Quantity, T_e: Quantity, Z_f: float
) -> Quantity:
    """
    Calculates the electro-electron static structure factor.

    Parameters
    ----------
    k :     Quantity
            Length of the scattering number (given by the scattering angle and the
            energies of the incident photons (unit: 1 / [length]).
    m_ion : Quantity
            The mass of the ion for which the ion feature is calculated.
    n_e:    Quantity
            The electron number density.
    T_e:    Quantity
            The electron temperature.
    Z_f:    float
            The number of electrons not tightly bound to the atom = valence electrons

    Returns
    -------
    S_ee:   Quantity
            The electron-electron static structure factor.
    """

    T_F = (
        (1 * ureg.planck_constant / (2 * jnp.pi)) ** 2
        * (3 * jnp.pi**2 * n_e) ** (2.0 / 3.0)
        / (2 * ureg.boltzmann_constant * 1 * ureg.electron_mass)
    )
    d = (3 / (4 * jnp.pi * n_e)) ** (1.0 / 3.0)
    r_s = (d / (1 * ureg.bohr_radius)).to_base_units()
    T_q = T_F / (1.3251 - 0.1779 * jpu.numpy.sqrt(r_s))
    T_cf = jpu.numpy.sqrt(T_e**2 + T_q**2)

    return (1.0 - ((n_e) / (1 * ureg.boltzmann_constant * T_cf)) * phi_ee(
        k, m_ion, n_e, T_e, T_cf, Z_f
    )).to_base_units()


@jit
def S_ii(
    k: Quantity, m_ion: Quantity, n_e: Quantity, T_e: Quantity, Z_f: float
) -> Quantity:
    """
    Calculates the ion-ion static structure factor.

    Parameters
    ----------
    k :     Quantity
            Length of the scattering number (given by the scattering angle and the
            energies of the incident photons (unit: 1 / [length]).
    m_ion : Quantity
            The mass of the ion for which the ion feature is calculated.
    n_e:    Quantity
            The electron number density.
    T_e:    Quantity
            The electron temperature.
    Z_f:    float
            The number of electrons not tightly bound to the atom = valence electrons

    Returns
    -------
    S_ii:   Quantity
            The ion-ion static structure factor.
    """

    T_F = (
        (1 * ureg.planck_constant / (2 * jnp.pi)) ** 2
        * (3 * jnp.pi**2 * n_e) ** (2.0 / 3.0)
        / (2 * ureg.boltzmann_constant * 1 * ureg.electron_mass)
    )
    d = (3 / (4 * jnp.pi * n_e)) ** (1.0 / 3.0)
    r_s = (d / (1 * ureg.bohr_radius)).to_base_units()
    T_q = T_F / (1.3251 - 0.1779 * jpu.numpy.sqrt(r_s))
    T_cf = jpu.numpy.sqrt(T_e**2 + T_q**2)

    n_i = n_e / Z_f

    return (1.0 - ((n_i) / (1 * ureg.boltzmann_constant * T_cf)) * phi_ii(
        k, m_ion, n_e, T_e, T_cf, Z_f
    )).to_base_units()


@jit
def S_ei(
    k: Quantity, m_ion: Quantity, n_e: Quantity, T_e: Quantity, Z_f: float
) -> Quantity:
    """
    Calculates the electro-ion static structure factor.

    Parameters
    ----------
    k :     Quantity
            Length of the scattering number (given by the scattering angle and the
            energies of the incident photons (unit: 1 / [length]).
    m_ion : Quantity
            The mass of the ion for which the ion feature is calculated.
    n_e:    Quantity
            The electron number density.
    T_e:    Quantity
            The electron temperature.
    Z_f:    float
            The number of electrons not tightly bound to the atom = valence electrons

    Returns
    -------
    S_ei:   Quantity
            The electron-ion static structure factor.
    """

    T_F = (
        (1 * ureg.planck_constant / (2 * jnp.pi)) ** 2
        * (3 * jnp.pi**2 * n_e) ** (2.0 / 3.0)
        / (2 * ureg.boltzmann_constant * 1 * ureg.electron_mass)
    )
    d = (3 / (4 * jnp.pi * n_e)) ** (1.0 / 3.0)
    r_s = (d / (1 * ureg.bohr_radius)).to_base_units()
    T_q = T_F / (1.3251 - 0.1779 * jpu.numpy.sqrt(r_s))
    T_cf = jpu.numpy.sqrt(T_e**2 + T_q**2)

    n_i = n_e / Z_f

    return ((
        jpu.numpy.sqrt(n_i * n_e) / (1 * ureg.boltzmann_constant * T_cf)
    ) * phi_ei(k, m_ion, n_e, T_e, T_cf, Z_f)).to_base_units()


@jit
def q(
    k: Quantity, m_ion: Quantity, n_e: Quantity, T_e: Quantity, Z_f: float
) -> Quantity:
    """
    Calculates the screening charge.

    Parameters
    ----------
    k :     Quantity
            Length of the scattering number (given by the scattering angle and the
            energies of the incident photons (unit: 1 / [length]).
    m_ion : Quantity
            The mass of the ion for which the ion feature is calculated.
    n_e:    Quantity
            The electron number density.
    T_e:    Quantity
            The electron temperature.
    Z_f:    float
            The number of electrons not tightly bound to the atom = valence electrons

    Returns
    -------
    q(k):  Quantity
           The screening charge.
    """

    # Way to calculate it given by Gregori.2004:
    
    C_ei = (jpu.numpy.sqrt(Z_f) * S_ei(k, m_ion, n_e, T_e, Z_f)) / (
        S_ee(k, m_ion, n_e, T_e, Z_f) * S_ii(k, m_ion, n_e, T_e, Z_f)
        - S_ei(k, m_ion, n_e, T_e, Z_f) ** 2
    )

    return (C_ei / (dielectric_function_salpeter(k, T_e=T_e, n_e=n_e, E=0 * ureg.electron_volts))).to_base_units()

    # This is the version as given by Gregori.2003 giving different results ...
    # return (
    #     jpu.numpy.sqrt(Z_f)
    #     * S_ei(k, m_ion, n_e, T_e, Z_f)
    #     / S_ii(k, m_ion, n_e, T_e, Z_f)
    # ).to_base_units()


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as onp
    import scienceplots

    import jax.numpy as jnp

    plt.style.use("science")

    lambda_0 = 4.13 * ureg.nanometer
    theta = 160
    theta_int = jnp.linspace(0, 180, 400)
    k = (4 * jnp.pi / lambda_0) * jnp.sin(jnp.deg2rad(theta) / 2.0)
    k_int = (4 * jnp.pi / lambda_0) * jnp.sin(jnp.deg2rad(theta_int) / 2.0)
    E = jnp.linspace(-10, 10, 500) * ureg.electron_volts

    for Z_f in [1, 2, 3]:
        plt.plot(
            k_int,
            q(
                k_int,
                2e-26 * ureg.kilogram,
                1e21 / ureg.centimeter**3,
                8 * ureg.electron_volts / (1 * ureg.boltzmann_constant),
                Z_f=Z_f,
            ),
        )

    plt.xlabel(r"$k$ [1/nm]")
    plt.ylabel(r"$q(k)$")

    plt.legend()
    plt.tight_layout()
    plt.show()
