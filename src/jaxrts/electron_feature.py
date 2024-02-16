"""
This submodule is dedicated to the calculation of the free electron dynamic structure
"""

from .units import ureg, Quantity
from typing import List

import jax
from jax import jit
import jax.numpy as jnp
import numpy as onp

import logging
import jpu

jax.config.update("jax_enable_x64", True)

@jit
def W(x : jnp.ndarray | float) -> jnp.ndarray:
    
    """
    Convenience function for the electron dielectric response function as defined in Gregori.2003.
    
    Parameters
    ----------
    x : jnp.ndarray | float
    
    Returns
    -------
    W:  jnp.ndarray
        The value of the convenience function.
        
    """

    x_v = jnp.linspace(0, x.magnitude, 3000).T

    y_v = jpu.numpy.exp(x_v**2)

    res = (
        1
        - 2
        * x
        * jpu.numpy.exp(-(x**2))
        * jax.scipy.integrate.trapezoid(y_v, x_v, axis=1)
        + (jpu.numpy.sqrt(jnp.pi) * x * jpu.numpy.exp(-(x**2))) * 1j
    ).to_base_units()

    return res


@jit
def eps_k_w(
    k: Quantity, T_e: Quantity, n_e: Quantity, E: Quantity | List
) -> jnp.ndarray:

    """
    
    Implementation of the quantum corrected Salpeter approximation of the electron dielectric response function.
    
    Parameters
    ----------
    k :  Quantity
         Length of the scattering number (given by the scattering angle and the
         energies of the incident photons (unit: 1 / [length]).
    E :  Quantity | List
         The energy shift for which the free electron dynamic structure is calculated. 
         Can be an interval of values.
    T_e: Quantity
         The electron temperature. 
    n_e: Quantity
         The electron number density.
    
    Returns
    -------
    eps: jnp.ndarray
         The electron dielectric response function.
    """
    
    omega = (E / (1 * ureg.planck_constant / (2 * jnp.pi))).to_base_units()

    v_t = jpu.numpy.sqrt(
        ((1 * ureg.boltzmann_constant) * T_e) / (1 * ureg.electron_mass)
    ).to_base_units()
    
    x_e = (omega / (jpu.numpy.sqrt(2) * k * v_t)).to_base_units()

    kappa = (
        (1 * ureg.planck_constant / (2 * jnp.pi))
        * k
        / (2 * jpu.numpy.sqrt(2) * (1 * ureg.electron_mass) * v_t)
    ).to_base_units()

    # The electron plasma frequency
    w_p_sq = (
        n_e
        * (1 * ureg.elementary_charge) ** 2
        / ((1 * ureg.electron_mass) * (1 * ureg.vacuum_permittivity))
    )

    return 1 + ((w_p_sq) / (k**2 * v_t**2)) * (1 / (4 * kappa)) * (
        (1 - W((x_e + kappa).to_base_units())) / (x_e + kappa) - (1 - W((x_e - kappa).to_base_units())) / (x_e - kappa)
    )

@jit
def S0_ee(k: Quantity, T_e: Quantity, n_e: Quantity, E: Quantity | List) -> jnp.ndarray:

    """
    Calculates the free electron dynamics structure using the quantum corrected Salpeter 
    approximation of the electron dielectric response function.
    
    Parameters
    ----------
    k :  Quantity
         Length of the scattering number (given by the scattering angle and the
         energies of the incident photons (unit: 1 / [length]).
    E :  Quantity | List
         The energy shift for which the free electron dynamic structure is calculated. 
         Can be an interval of values.
    T_e: Quantity
         The electron temperature. 
    n_e: Quantity
         The electron number density.
    
    Returns
    -------
    S0_ee: jnp.ndarray
           The free electron dynamic structure.
    """
    
    return -(
        (1 * ureg.planck_constant / (2 * jnp.pi))
        / (1 - jpu.numpy.exp(-(E / (1 * ureg.boltzmann_constant * T_e))))
        * (
            ((1 * ureg.vacuum_permittivity) * k**2)
            / (jnp.pi * (1 * ureg.elementary_charge) ** 2 * n_e)
        )
        * jnp.imag(1 / eps_k_w(k, T_e, n_e, E).magnitude)
    ).to_base_units()
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as onp
    import scienceplots

    import jax.numpy as jnp

    plt.style.use("science")

    lambda_0 = 4.13 * ureg.nanometer
    theta = 160
    k = (4 * jnp.pi / lambda_0) * jnp.sin(jnp.deg2rad(theta) / 2.0)
        
    count = 0
    norm = 1.0
    for T in [0.5 * ureg.electron_volts, 2.0 * ureg.electron_volts, 13.0 * ureg.electron_volts]:
        E = jnp.linspace(-10, 10, 500) * ureg.electron_volts
        vals = S0_ee(k, T_e = T / (1 * ureg.boltzmann_constant), n_e = 1e21 / ureg.centimeter**3, E = E)
        count += 1
        if(count == 1):
            norm = onp.max(vals)
        plt.plot(E, vals / norm, label = 'T = ' + str(T.magnitude) + " eV")

    plt.xlabel(r"$\omega$ [eV]")
    plt.ylabel(r"$S^0_{\text{ee}}$ [arb. units]")

    plt.legend()
    plt.tight_layout()
    plt.show()