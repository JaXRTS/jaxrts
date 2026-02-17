"""
LFC comparison
==============

This example compares different static (and quasi-static) local field
corrections that are available in jaxrts.

The model by Hubbard :cite:`Hubbard.1957` is a simple formula, and the first
attempt to formulate a LFC, but does not obey the correct limiting cases.

The dotted curves showing the models by :cite:`UtsumiIchimaru.1982` and
:cite:`Farid.1993` show no temperature dependence as they are only implemented
for zero-temperature, whey they are the respective limits for the interpolation
formulas.

Dornheim et al. :cite:`Dornheim.2020` argue, that while the :math:`k^2` scaling
for large :math:`k` as presented by Farid is correct for the static LFC, in
practice, rather an effective static LFC should be used that obeys the correct
limiting case for a full, :math:`\\omega` dependent LCF. An analytical function
fitting their datapoints is provided in :cite:`Dornheim.2021`.
"""

import jax.numpy as jnp
import jpu.numpy as jnpu
import matplotlib.pyplot as plt
import scienceplots

import jaxrts

ureg = jaxrts.ureg

plt.style.use("science")


temp = (
    jnp.array([0.1, 50, 1000]) * ureg.electron_volt / ureg.boltzmann_constant
)

n_e = 2.5e23 / ureg.centimeter**3
kf = jaxrts.plasma_physics.fermi_wavenumber(n_e)
k = jnpu.linspace(0 * kf, 7 * kf, 5000)

fig, ax = plt.subplots(len(temp), sharex=True, figsize=(6, 9))


for T, axis in zip(temp, ax):
    theta = (T / jaxrts.plasma_physics.fermi_temperature(n_e)).m_as(
        ureg.dimensionless
    )
    print(f"theta = {theta:.2}")
    if theta < 4:
        G_dornheim = [
            jaxrts.ee_localfieldcorrections.eelfc_dornheim2021(
                _k, T, n_e
            )
            for _k in k
        ]
        axis.plot(
            (k / kf).m_as(ureg.dimensionless),
            G_dornheim,
            label="Dornheim.2021 (QSA)",
            color="C3",
        )
    else:
        axis.text(0, 2, "outside Dornheim.2021 validity", color="C3")
    G_hubbard = jaxrts.ee_localfieldcorrections.eelfc_hubbard(k, T, n_e)
    G_geldart = jaxrts.ee_localfieldcorrections.eelfc_geldartvosko(k, T, n_e)
    G_gregori = jaxrts.ee_localfieldcorrections.eelfc_interpolationgregori2007(
        k, T, n_e
    )
    G_farid = jaxrts.ee_localfieldcorrections.eelfc_farid(k, T, n_e)
    G_utsumi = jaxrts.ee_localfieldcorrections.eelfc_utsumiichimaru(k, T, n_e)
    G_fortmann = (
        jaxrts.ee_localfieldcorrections.eelfc_interpolationgregori_farid(
            k, T, n_e
        )
    )
    G_um = jaxrts.ee_localfieldcorrections.eelfc_interpolationgregori_farid(
        k, T, n_e
    )
    axis.plot(
        (k / kf).m_as(ureg.dimensionless),
        G_gregori.m_as(ureg.dimensionless),
        label="Interp. Gregori.2007",
    )
    axis.plot(
        (k / kf).m_as(ureg.dimensionless),
        G_fortmann.m_as(ureg.dimensionless),
        label="Interp. Fortmann.2010",
    )
    axis.plot(
        (k / kf).m_as(ureg.dimensionless),
        G_hubbard.m_as(ureg.dimensionless),
        label="Hubbard.1957",
    )
    axis.plot(
        (k / kf).m_as(ureg.dimensionless),
        G_utsumi.m_as(ureg.dimensionless),
        label="Utsumi \\& Ichimaru (zero T)",
        ls="dotted",
        color="gray",
    )
    axis.plot(
        (k / kf).m_as(ureg.dimensionless),
        G_farid,
        label="Farid (zero T)",
        ls="dotted",
        color="black",
    )
    axis.plot(
        (k / kf).m_as(ureg.dimensionless),
        G_geldart.m_as(ureg.dimensionless),
        label="Geldart \\& Vosko (high T)",
        ls="dashed",
        color="black",
    )
    axis.set_title(
        f"$k_BT=${(T * ureg.boltzmann_constant).m_as(ureg.electron_volt):.0e} eV"  # noqa:E501
    )
    axis.set_ylabel("$G$")

ax[0].legend(ncols=2)
ax[-1].set_xlabel("$k / k_f$")

plt.tight_layout()
plt.show()
