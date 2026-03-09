"""
Zero IPD limit for effective charge when using form factor lowering
===================================================================

This examples compares the limits of the Form Factor lowering reported by
:cite:`Doppner.2023` (:py:class:`jaxrts.models.FormFactorLowering`) for zero
IPD to the form factors by :cite:`Pauling.1932`.

Since the form of the 1s form factor is identical, and the only change is the
effective change, this value is plotted.
"""

import jax.numpy as jnp
import jpu.numpy as jnpu
import matplotlib.pyplot as plt
import scienceplots  # noqa:F401

import jaxrts

ureg = jaxrts.ureg

plt.style.use("science")

Z_A = jnp.arange(2, 36)


# Pauling
# =======

Zeff_pauling_H_like = Z_A
Zeff_pauling_He_like = (
    Z_A - jaxrts.form_factors.pauling_size_screening_constants(Z_A)[0]
)

# Form Factor lowering
# ====================

Zeff_ffl_H_like = []
Zeff_ffl_He_like = []

Zeff_ffl_H_like_corr = []
Zeff_ffl_He_like_corr = []
for Zs in Z_A:
    binding_E = jnpu.sort(jaxrts.Element(Zs).ionization.energies)[-2:][::-1]
    H_like, He_like = jaxrts.form_factors.form_factor_lowering_Zeff_10(
        binding_E, Zs, Z_squared_correction=False
    )
    Zeff_ffl_H_like.append(H_like)
    Zeff_ffl_He_like.append(He_like)

    H_like, He_like = jaxrts.form_factors.form_factor_lowering_Zeff_10(
        binding_E, Zs, Z_squared_correction=True
    )
    Zeff_ffl_H_like_corr.append(H_like)
    Zeff_ffl_He_like_corr.append(He_like)

Zeff_ffl_H_like = jnp.array(Zeff_ffl_H_like)
Zeff_ffl_He_like = jnp.array(Zeff_ffl_He_like)

Zeff_ffl_H_like_corr = jnp.array(Zeff_ffl_H_like_corr)
Zeff_ffl_He_like_corr = jnp.array(Zeff_ffl_He_like_corr)


fig, ax = plt.subplots()


ax.plot(
    Z_A,
    Zeff_ffl_H_like - Zeff_pauling_H_like,
    color="C0",
    ls="none",
    marker="o",
    label="ff lowering - Pauling",
    markerfacecolor="none",
    markeredgewidth=2,
)
ax.plot(
    Z_A,
    Zeff_ffl_H_like_corr - Zeff_pauling_H_like,
    color="C0",
    ls="none",
    marker="s",
    label="ff lowering (corr.) - Pauling",
)

ax.plot(
    Z_A,
    Zeff_ffl_He_like - Zeff_pauling_He_like,
    color="C1",
    ls="none",
    marker="o",
    markerfacecolor="none",
    markeredgewidth=2,
)
ax.plot(
    Z_A,
    Zeff_ffl_He_like_corr - Zeff_pauling_He_like,
    color="C1",
    ls="none",
    marker="s",
)
ax.plot(Z_A, jnp.zeros_like(Z_A), color="black", ls="dashed")

ax.set_ylabel("$\\Delta Z_{eff}$")
ax.set_xlabel("$Z$")
ax.set_title("blue: H-like, green: He-like")

ax.legend()
plt.tight_layout()
plt.show()
