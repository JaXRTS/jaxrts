"""
Plot static structure Factors in the Approximation by Arkhipov and Devletov
===========================================================================

This scripts plots :math:`S_{ee}`, :math:`S_{ei}`, and :math:`S_{ii}` for 
Carbon with an ionization of 4.5 at :math:`T = 13\\text{eV}` and 
:math:`n_e = 1\\times 10^{22} 1 / \\text{cm}^3`, in the approach presented
by :cite:`Arkhipov.1998`.
"""

import matplotlib.pyplot as plt
import scienceplots

import jaxrts

import jax.numpy as jnp

ureg = jaxrts.units.ureg

plt.style.use("science")

k = jnp.linspace(0.0, 6, 200) / (1 * ureg.angstrom)
n_e = 1e22 / ureg.centimeter**3
T_e = 13.0 * ureg.electron_volts / ureg.k_B
T_cf = jaxrts.static_structure_factors.T_cf_Greg(T_e, n_e)
Z_f = 4.5
m_i = 12 * ureg.atomic_mass_constant

See = jaxrts.static_structure_factors.S_ee_AD(k, T_cf, n_e, m_i, Z_f)
Sei = jaxrts.static_structure_factors.S_ei_AD(k, T_cf, n_e, m_i, Z_f)
Sii = jaxrts.static_structure_factors.S_ii_AD(k, T_cf, n_e, m_i, Z_f)

plt.plot(
    k.m_as(1 / ureg.angstrom),
    See.m_as(ureg.dimensionless),
    label="$S_{ee}$",
    color="C0",
)
plt.plot(
    k.m_as(1 / ureg.angstrom),
    Sii.m_as(ureg.dimensionless),
    label="$S_{ii}$",
    color="C1",
)
plt.plot(
    k.m_as(1 / ureg.angstrom),
    Sei.m_as(ureg.dimensionless),
    label="$S_{ei}$",
    color="C2",
)

plt.xlabel(r"$k$ [1/$\AA$]")
plt.ylabel(r"Static stucture factor [arb. units]")

plt.legend()
plt.tight_layout()
plt.show()
