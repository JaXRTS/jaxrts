from units import ureg, Quantity

from typing import List

from quadax import quadts as quad

import jax
from jax import jit
from jax import numpy as jnp
from jpu import numpy as jnpu
import logging

logger = logging.getLogger(__name__)

import jpu


def ipd_debye_hueckel(Z_f : float, T : Quantity, n_e : Quantity) -> Quantity:
    
    """
    The correction to the ionization potential for the m-th ionization stage in Debye-Hueckel approximation.
    Note: The Debye-Hueckel approximation is physically meaningful only when the coupling parameter << 1, such that
    Coulomb forces are weak perturbations.
    """
    kappa = jpu.numpy.sqrt((ureg.elementary_charge ** 2) / (ureg.epsilon_0 * ureg.boltzmann_constant * T) * (jpu.numpy.sum(Z_f * n_e) + n_e))
    ipd_shift = kappa * (Z_f + 1) * ureg.elementary_charge ** 2 / (4 * jnp.pi * ureg.epsilon_0)
    
    return ipd_shift.to(ureg.electron_volt)

def ipd_ion_sphere(Z_f : float, Z_eff : int, T : Quantity, n_e : Quantity) -> Quantity:
    
    """
    The correction to the ionization potential for the m-th ionization stage in the ion-sphere model.
    The ion-sphere model considers the ions to be strongly correlated.
    (see also More 1981)
    
    Parameters
    ----------
    
    """
    
    # The ion-sphere radius, determined by the ion density n_i such that the average distance to the nearest neighbor ion is
    # approximately 2 R_0.
    R_0 = (3 / (4 * jnp.pi * n_e / Z_f)) ** (1/3)
    
    ipd_shift = -(3/2) * Z_eff ** (1/3) * ((Z_f ** (2/3) * ureg.elementary_charge**2)/(ureg.epsilon_0 * R_0))
    
    return ipd_shift.to(ureg.electron_volt)

def ipd_steward_pyatt(Z_m : float, Z_f : float, Z_p : float, T : Quantity, n_e : Quantity) -> Quantity:
    
    """
    The correction to the ionization potential for the m-th ionization stage in the Stewart-Pyatt model.
    This model is founded on the Thomas-Fermi Model for the electrons and extends it to include ions in the vicinity of a given nucleus.
    """
    lambda_Di = jpu.numpy.sqrt(ureg.epsilon_0 * ureg.boltzmann_constant * T / (n_e * ureg.elementary_charge ** 2))
    R_0 = (3 / (4 * jnp.pi * n_e / Z_f)) ** (1/3)
    R_m = (Z_m / Z_f) ** (1/3) * R_0
    ipd_shift = -(3/2) * (Z_m / Z_p) ** (2/3) * (Z_p * ureg.elementary_charge ** 2 / (ureg.epsilon_0 * R_0)) * ((1 + (lambda_Di / R_m)**3) ** (2/3) - (lambda_Di / R_m) ** 2)
    
    return ipd_shift.to(ureg.electron_volt)


def ipd_ecker_kroell(Z_m : float, Z_f : float, T : Quantity, n_e : Quantity) -> Quantity:
    
    """
    The correction to the ionization potential for the m-th ionization stage in the Ecker-Kroell model.
    This model is similar to the model of Steward-Pyatt and divided the radial dimension into three regions.
    For details see Ecker&Kroell 1963.
    """
    lambda_Di = jpu.numpy.sqrt(ureg.epsilon_0 * ureg.boltzmann_constant * T / (n_e * ureg.elementary_charge ** 2))

    R_0 = (3 / (4 * jnp.pi * n_e / Z_f)) ** (1/3)
    
    # The critical density in the model of Ecker-Kroell
    
    n_c = (3 / (4 * jnp.pi)) * (ureg.boltzmann_constant * T / ureg.elementary_charge ** 2) ** 3
    
    # The constant in Ecker-Kroells model, which is determined from the continuity of the potential across the critical density.
    
    C = 2.2 * jpu.numpy.sqrt(ureg.elementary_charge ** 2 / (ureg.boltzmann_constant * T)) * n_c ** (1/6)
    
    ipd_c1 = -1 * ureg.elementary_charge ** 2 / (ureg.epsilon_0 * lambda_Di) * Z_m
    ipd_c2 = -C * ureg.elementary_charge ** 2 / (ureg.epsilon_0 * R_0) * Z_m
    
    ipd_shift = jnp.where(n_e / Z_f <= n_c, ipd_c1, ipd_c2)
    
    return ipd_shift.to(ureg.electron_volt)

def ipd_pauli_blocking(Z_m : float, chem_pot : Quantity, T : Quantity) -> Quantity:

    """
    The correction to the ionization potential due to Pauli blocking, as described in Röpke et al 2019.
    """
    
    # Reduced bohr radius    
    a_Z = ureg.bohr_radius / Z_m
    
    def integrand(p):
        p *= (1 * ureg.kilogram / ureg.meter / ureg.second)
        res = (p ** 2 / (1 + a_Z ** 2 * (p/(1*ureg.planck_constant)) ** 2) ** 3) * (1 / (jnp.exp((p**2 / (2 * ureg.electron_mass))/(ureg.boltzmann_constant * T) - chem_pot) + 1)).to_base_units()
        print(res)
        return (p ** 2 / (1 + a_Z ** 2 * (p/(1*ureg.planck_constant)) ** 2) ** 3) * (1 / (jnp.exp((p**2 / (2 * ureg.electron_mass))/(ureg.boltzmann_constant * T) - chem_pot) + 1))
    
    integral, errl = quad(integrand, [0, jnp.inf])
    
    ipd_shift = Z_m * ureg.elementary_charge ** 2 / (4 * jnp.pi * ureg.epsilon_0) * (16 * a_Z ** 2) / jnp.pi * integral
    
    return ipd_shift.to(ureg.electron_volt)


def chem_pot_interpolation(T: Quantity, n_e: Quantity) -> Quantity:
    """
    Interpolation function for the chemical potential between the classical and
    quantum region, given in :cite:`Gregori.2003`, eqn. (19).

    Parameters
    ----------
    T
        The plasma temperature in Kelvin.
    n_e
        Electron density. Units of 1/[length]**3.

    Returns
    -------
    Quantity
        Chemical potential
    """
    A = 0.25945
    B = 0.072
    b = 0.858

    E_f = ureg.hbar**2 / (2 * ureg.m_e) * (3 * jnp.pi**2 * n_e) ** (2 / 3)
    Theta = (ureg.k_B * T / E_f).to_base_units()
    f = (
        (-3 / 2 * jnpu.log(Theta))
        + (jnpu.log(4 / (3 * jnp.sqrt(jnp.pi))))
        + (
            (A * Theta ** (-b - 1) + B * Theta ** (-(b + 1) / 2))
            / (1 + A * Theta ** (-b))
        )
    )
    return f * ureg.k_B * T

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    # lambda_0 = 4.13 * ureg.nanometer
    # theta = 160
    # k = (4 * onp.pi / lambda_0) * onp.sin(onp.deg2rad(theta) / 2.0)
    # n_e = 1e21 / ureg.centimeter**3
    
    n_eplot = jnp.linspace(1E23, 1E27, 5000) / (1 * ureg.centimeter) ** 3
    T = 100 * ureg.electron_volt / ureg.boltzmann_constant
    
    mu = chem_pot_interpolation(
            T, n_eplot
        )
    
    plt.plot(n_eplot, ipd_debye_hueckel(1, T, n_eplot))
    plt.plot(n_eplot, ipd_ion_sphere(1, 1, T, n_eplot))
    plt.plot(n_eplot, ipd_steward_pyatt(1, 1, 1, T, n_eplot))
    plt.plot(n_eplot, ipd_pauli_blocking(1, mu, T))
    
    plt.ylabel(r"$\Delta E_I$ [eV]")
    plt.xlabel(r"$n_e$ [cm$^{-3}$]")
    plt.show()
    
    
    
    
    
    