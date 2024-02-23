"""
Plot static structure Factors in the Approximation by Arkhipov and Devletov
===========================================================================

This scripts plots :math:`S_{ee}`, :math:`S_{ei}`, and :math:`S_{ii}` for 
Carbon with an ionization of 4.5 at :math:`T = 13\\text{eV}` and 
:math:`n_e = 1\\times 10^{23} 1 / \\text{cm}^3`.
"""

import matplotlib.pyplot as plt
import scienceplots

import jaxrts

import jax.numpy as jnp

ureg = jaxrts.units.ureg

plt.style.use("science")

k = jnp.linspace(0.5, 6) / (1 * ureg.angstrom)
n_e = 1e23 / ureg.centimeter**3
T_e = 13.0 * ureg.electron_volts / ureg.k_B
Zf = 4.5
m_i = 12 * ureg.atomic_mass_constant

See = jaxrts.static_structure_factors.S_ee_AD(k, T_e, n_e, m_i, Zf)
Sei = jaxrts.static_structure_factors.S_ei_AD(k, T_e, n_e, m_i, Zf)
Sii = jaxrts.static_structure_factors.S_ii_AD(k, T_e, n_e, m_i, Zf)

plt.plot(k.m_as(1/ureg.angstrom), See, label="$S_{ee}$")
plt.plot(k.m_as(1/ureg.angstrom), Sei, label="$S_{ei}$")
plt.plot(k.m_as(1/ureg.angstrom), Sii, label="$S_{ii}$")

plt.xlabel(r"$k$ [1/$\AA$]")
plt.ylabel(r"Static stucture factor [arb. units]")

plt.legend()
plt.tight_layout()
plt.show()
