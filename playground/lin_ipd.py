from pathlib import Path
import matplotlib.pyplot as plt

import jpu.numpy as jnpu
import numpy as np
import jax
from jax import numpy as jnp
from jax.tree_util import Partial

import jaxrts
from jaxrts.ipd import (
    ipd_debye_hueckel,
    ipd_ecker_kroell,
    ipd_ion_sphere,
    ipd_stewart_pyatt,
    ipd_lin,
)
from jaxrts.units import to_array

ureg = jaxrts.ureg

T = ureg("600 eV") / ureg.k_B
Zi = 11.0
element = jaxrts.Element("Al")

rho, ipd_lit = np.genfromtxt(
    "tests/data/Lin2017/Fig1/ipd_sf_lin.csv", unpack=True, delimiter=","
)
plt.plot(rho, ipd_lit, label="literature")
rho *= ureg.gram / ureg.centimeter**3
# rho = jnp.logspace(-3, 2, 100) * ureg.gram / ureg.centimeter**3
ni = (rho / element.atomic_mass).to(1 / ureg.centimeter**3)


def lin_IPD(n):
    @Partial
    def S_iiZZ(k):
        T_e_prime = jaxrts.static_structure_factors.T_cf_Greg(T, Zi * n)
        T_D = jaxrts.static_structure_factors.T_Debye_Bohm_Staver(
            T_e_prime, Zi * n, element.atomic_mass, Zi
        )
        T_i_prime = jaxrts.static_structure_factors.T_i_eff_Greg(T, T_D)

        lfc = jaxrts.ee_localfieldcorrections.eelfc_interp_gregori2007(
            k, T_e_prime, Zi * n
        )
        Sii = jaxrts.static_structure_factors.S_ii_CHS(
            k, T_e_prime, T_i_prime, Zi * n, element.atomic_mass, Zi, n, lfc
        )
        k_De = jaxrts.static_structure_factors._k_D_AD(T_e_prime, Zi * n)
        S_ee0 = k**2 / (k**2 + k_De**2 * (1 - lfc))
        q = Zi * (k_De / k) ** (2) * S_ee0
        return ((1 - q / Zi) ** 2 * Sii).m_as(ureg.dimensionless)

    return ipd_lin(Zi, n, T, S_iiZZ)


# import matplotlib.pyplot as plt
#
# k0 = jnp.linspace(1e-12, 5, 500) * ureg.dimensionless
# for i, n in enumerate(ni):
#     plt.plot(
#         k0,
#         [
#             (
#                 S_iiZZ(k * jaxrts.plasma_physics.fermi_wavenumber(n), n)
#                 / (1 * ureg.dimensionless)
#             ).m_as(ureg.dimensionless)
#             for k in k0
#         ],
#         color=f"C{i}",
#         label=n,
#     )
#     plt.plot(
#         k0,
#         [
#             (
#                 S_iiZZ(k * jaxrts.plasma_physics.fermi_wavenumber(n), n)
#                 / (k**2)
#             ).m_as(ureg.dimensionless)
#             for k in k0
#         ],
#         color=f"C{i}",
#         ls="dotted",
#     )
# plt.plot(
#     k0,
#     (1 / k0**2).m_as(ureg.dimensionless),
#     ls="dashed",
#     color=f"C{i}",
# )
# plt.legend()
# plt.show()
# exit()


ipd = jax.vmap(lin_IPD)(ni)
print(ipd)


plt.plot(
    rho.m_as(ureg.gram / ureg.centimeter**3),
    ipd.m_as(ureg.electron_volt),
    label="jaxrts",
)
plt.legend()
plt.show()
