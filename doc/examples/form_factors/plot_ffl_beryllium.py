"""
Plot form factor lowering effect for Beryllium
==============================================

"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import scienceplots  # noqa:F401

import jaxrts

ureg = jaxrts.ureg

plt.style.use("science")

Z_A = 4
Z_C = 0.5

k = jnp.linspace(0, 17, 100) / (1 * ureg.angstrom)


fig, ax = plt.subplots()
for IPD in jnp.array([0, 50, 100, 150]) * ureg.electron_volt:

    binding_E = jaxrts.Element(Z_A).ionization.energies[::-1][:2] - IPD
    f_ffl = jaxrts.form_factors.form_factor_lowering_10(
        k, binding_E, Z_C, Z_A, Z_squared_correction=True
    )

    ax.plot(
        k.m_as(1 / ureg.angstrom),
        f_ffl.m_as(ureg.dimensionless),
        label=f"IPD = {IPD.m_as(ureg.electron_volt)}eV",
    )
ax.set_ylabel("$f_{1s}$")
ax.set_xlabel("$k$ [1/angstrom]")

ax.legend()
plt.tight_layout()
plt.title("$f_{1s}$ for beryllium ${3.5+}$")
plt.show()
