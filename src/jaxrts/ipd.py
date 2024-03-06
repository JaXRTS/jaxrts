from .units import ureg, Quantity

from typing import List

from quadax import quadts as quad

import jax
from jax import jit
from jax import numpy as jnp
from jpu import numpy as jnpu
import logging

logger = logging.getLogger(__name__)

import jpu


def ipd_debye_hueckel(Z_m : float, lambda_Di : List | Quantity, T : Quantity, n_e : Quantity) -> Quantity:
    
    """
    The correction to the ionization potential for the m-th ionization stage in Debye-Hueckel approximation.
    Note: The Debye-Hueckel approximation is physically meaningful only when the coupling parameter << 1, such that
    Coulomb forces are weak perturbations.
    """
    
    ipd_shift = -Z_m * ureg.elementary_charge ** 2 / lambda_Di
    
    return ipd_shift

def ipd_ion_sphere(Z_m : float, Z_eff : int, T : Quantity, n_i : Quantity) -> Quantity:
    
    """
    The correction to the ionization potential for the m-th ionization stage in the ion-sphere model.
    The ion-sphere model considers the ions to be strongly correlated.
    (see also More 1981)
    
    Parameters
    ----------
    
    """
    
    # The ion-sphere radius, determined by the ion density n_i such that the average distance to the nearest neighbor ion is
    # approximately 2 R_0.
    R_0 = (3 / (4 * jnp.pi * n_i)) ** (1/3)
    
    ipd_shift = -(3/2) * Z_eff ** (1/3) * ((Z_m ** (2/3) * ureg.elementary_charge**2)/(R_0))
    
    return ipd_shift

def ipd_steward_pyatt(Z_m : float, Z_eff : float, Z_p : float, lambda_Di : List | Quantity, T : Quantity, n_i : Quantity) -> Quantity:
    
    """
    The correction to the ionization potential for the m-th ionization stage in the Stewart-Pyatt model.
    This model is founded on the Thomas-Fermi Model for the electrons and extends it to include ions in the vicinity of a given nucleus.
    """
    R_0 = (3 / (4 * jnp.pi * n_i)) ** (1/3)
    R_m = (Z_m / Z_eff) ** (1/3) * R_0
    ipd_shift = -(3/2) * (Z_m / Z_p) ** (2/3) * (Z_p * ureg.elementary_charge ** 2 / R_0) * ((1 + (lambda_Di / R_m)**3) ** (2/3) - (lambda_Di / R_m) ** 2)
    
    return ipd_shift


def ipd_ecker_kroell(Z_m : float, Z_eff : float, Z_p : float, lambda_Di : List | Quantity, T : Quantity, n_i : Quantity) -> Quantity:
    
    """
    The correction to the ionization potential for the m-th ionization stage in the Ecker-Kroell model.
    This model is similar to the model of Steward-Pyatt and divided the radial dimension into three regions.
    For details see Ecker&Kroell 1963.
    """
    
    R_0 = (3 / (4 * jnp.pi * n_i)) ** (1/3)
    
    # The critical density in the model of Ecker-Kroell
    
    n_c = (3 / (4 * jnp.pi)) * (ureg.boltzmann_constant * T / ureg.elementary_charge ** 2) ** 3
    
    # The constant in Ecker-Kroells model, which is determined from the continuity of the potential across the critical density.
    
    C = 2.2 * jnp.sqrt(ureg.elementary_charge ** 2 / (ureg.boltzmann_constant * T)) * n_c ** (1/6)
    
    cases = jnp.ones(2)
    
    
    ipd_c1 = -ureg.elementary_charge ** 2 / (ureg.epsilon_0 * lambda_Di) * Z_m
    ipd_c2 = -C * ureg.elementary_charge ** 2 / (ureg.epsilon_0 * R_0) * Z_m
    
    ipd_shift = jnp.sum(jnp.where(cases <= n_c, ipd_c1, 0) + jnp.where(cases > n_c, ipd_c2, 0))
    
    return ipd_shift

def ipd_pauli_blocking(Z : float, chem_pot : Quantity, T : Quantity) -> Quantity:

    """
    The correction to the ionization potential due to Pauli blocking, as described in Röpke et al 2019.
    """

    
    # Reduced bohr radius    
    a_Z = ureg.bohr_radius / Z
    
    def integrand(p):
        
        return (p ** 2 / (1 + a_Z ** 2 * p ** 2) ** 3) * (1 / (jnp.exp((p**2 / (2 * ureg.electron_mass))/(ureg.boltzmann_constant * T) - chem_pot) + 1))
    
    integral, errl = quad(integrand, [0, jnp.inf])
    
    ipd_shift = Z * ureg.elementary_charge ** 2 / (4 * jnp.pi * ureg.epsilon_0) * (16 * a_Z ** 2) / jnp.pi * integral
    
    return ipd_shift