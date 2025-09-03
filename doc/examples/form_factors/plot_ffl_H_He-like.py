"""
Zero IPD limit form factor lowering
===================================

This examples compares the limits of the Form Factor lowering reported by
:cite:`Doppner.2023` (:py:class:`jaxrts.models.FormFactorLowering`) for zero
IPD to the form factors by :cite:`Pauling.1932`.

We compare the form factors for a beryllium ion. Due to screening, there are
light differences between Hydrogen- and Helium-like ions. For a fractional
charge, we interpolate linearly between these two states.
"""
import jax.numpy as jnp

import matplotlib.pyplot as plt
import scienceplots  # noqa:F401

import jaxrts

ureg = jaxrts.ureg

plt.style.use("science")

Z_A = 4
k = jnp.linspace(0, 17, 100) / (1 * ureg.angstrom)


fig, ax = plt.subplots()
for idx, Z in enumerate([1.5, 2.0, 2.5, 3.0, 3.5]):
    Z_C = Z_A - Z

    # Pauling
    # =======
    Zeff_pauling = (
        Z_A - jaxrts.form_factors.pauling_size_screening_constants(Z_C)[0]
    )
    f_pauling = jaxrts.form_factors.pauling_f10(k, Zeff_pauling)

    # Form Factor lowering
    # ====================

    binding_E = jaxrts.Element(Z_A).ionization.energies[::-1][:2]
    f_ffl = jaxrts.form_factors.form_factor_lowering_10(
        k, binding_E, Z_C, Z_A, Z_squared_correction=True
    )

    ax.plot(
        k.m_as(1 / ureg.angstrom),
        f_pauling.m_as(ureg.dimensionless),
        color=f"C{idx}",
        ls="dotted",
        lw=2 if idx in [1, 2, 3] else 1,
        label=f"Pauling, Z={Z}" if idx == 0 else f"Z={Z}",
    )
    ax.plot(
        k.m_as(1 / ureg.angstrom),
        f_ffl.m_as(ureg.dimensionless),
        color=f"C{idx}",
        ls="dashed",
        lw=2 if idx in [1, 2, 3] else 1,
        label="FFL" if idx == 0 else None,
    )
ax.set_ylabel("$f_{1s}$")
ax.set_xlabel("$k$ [1/angstrom]")

ax.legend()
plt.tight_layout()
plt.title("$f_{1s}$ for beryllium")
plt.show()
