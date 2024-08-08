"""
Introduce LFC to Born Mermin Calculations
=========================================

This examples shows how local field corrections can be very relevant and might
change the results also for Born Mermin calculations with include the
electron-ion collision frequency.

We reproduces Figures from :cite:`Fortmann.2010`, who are plotting the position
:math:`\\omega` at which :math:`S_{ee}` is maximal.

"""

import jax
import jax.numpy as jnp
import jpu.numpy as jnpu
import matplotlib.pyplot as plt

import jaxrts

ureg = jaxrts.ureg


@jax.tree_util.Partial
def S_ii(q):
    return jnpu.ones_like(q)


def calculate_fwhm(data, x):
    peak_value = jnp.max(data)

    # Find indices where the data crosses the half maximum
    idx = jnp.where(data >= peak_value / 2.0, jnp.arange(len(data)), jnp.nan)

    left_idx = jnp.nanmin(idx)
    right_idx = jnp.nanmax(idx)

    # Interpolate lineary between the points found
    left_x = jnp.interp(
        peak_value / 2,
        jnp.array(
            [data[left_idx.astype(int) - 1], data[left_idx.astype(int)]]
        ),
        jnp.array([x[left_idx.astype(int) - 1], x[left_idx.astype(int)]]),
    )
    right_x = jnp.interp(
        peak_value / 2,
        jnp.array(
            [data[right_idx.astype(int)], data[right_idx.astype(int) + 1]]
        ),
        jnp.array([x[right_idx.astype(int)], x[right_idx.astype(int) + 1]]),
    )
    fwhm = right_x - left_x
    return jnpu.absolute(fwhm)


# :cite:`Fortmann.2010` calculated these values at zero kelvin. This is
# currently not implemented, in our code. We therefore use a finite
# temperature, instead, accepting that we might deviate slightly from the
# published result
T = 1.0 * ureg.electron_volt / ureg.k_B
r_s = 2
Zf = 1.0


@jax.tree_util.Partial
def V_eiS(q):
    return jaxrts.plasma_physics.coulomb_potential_fourier(
        Zf, -1, q
    ) / jaxrts.free_free.dielectric_function_RPA_0K(
        q, 0 * ureg.electron_volt, n_e
    )


n_e = 3 / (4 * jnp.pi * (r_s * ureg.a0) ** 3)

k_f = jaxrts.plasma_physics.fermi_wavenumber(n_e)
E_f = jaxrts.plasma_physics.fermi_energy(n_e)
k = jnp.linspace(0, 2) * k_f
E = jnp.linspace(-10, 1, 1000) * E_f
mu = jaxrts.plasma_physics.chem_pot_interpolationIchimaru(T, n_e)

sLFC = jaxrts.ee_localfieldcorrections.eelfc_interpolationgregori_farid(
    k[:, jnp.newaxis], T, n_e
)

S_ee_noLFC = jaxrts.free_free.S0_ee_BMA_Fortmann(
    k[:, jnp.newaxis], T, mu, S_ii, V_eiS, n_e, Zf, E[jnp.newaxis, :], 0.0
)
S_ee_sLFC = jaxrts.free_free.S0_ee_BMA_Fortmann(
    k[:, jnp.newaxis], T, mu, S_ii, V_eiS, n_e, Zf, E[jnp.newaxis, :], sLFC
)

plt.style.use("science")
fig, ax = plt.subplots(2, figsize=(5, 5))
for S_ee, label in [(S_ee_noLFC, "no LFC"), (S_ee_sLFC, "sLFC")]:
    idx = jnpu.argmax(S_ee, axis=1)
    ax[0].plot(
        (k / k_f).m_as(ureg.dimensionless),
        jnp.where(idx > 0, (-E[idx] / E_f).m_as(ureg.dimensionless), jnp.nan),
        label=label,
    )
    # Calculate the FWHM
    FWHM = jax.vmap(calculate_fwhm, in_axes=(0, None))(
        S_ee.m_as(ureg.second), (E / E_f).m_as(ureg.dimensionless)
    )
    ax[1].plot(
        (k / k_f).m_as(ureg.dimensionless),
        jnp.where(idx > 0, FWHM, jnp.nan),
    )

for axis in ax:
    axis.set_xlabel("$k / k_f$")
    axis.set_ylabel("$E / E_f$")
ax[0].set_title("Plasmon position (maximum of $S_{ee}$)")
ax[1].set_title("Plasmon width (width of $S_{ee}$)")

plt.tight_layout()
plt.show()
