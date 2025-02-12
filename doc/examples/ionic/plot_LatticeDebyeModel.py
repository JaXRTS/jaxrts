"""
LatticeDebyeModel for approximating diffuse scattering in crystals
==================================================================

This example showcases the increased diffuse scattering in heated crystals by
the Debye Waller Effect.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import scienceplots

import jaxrts

ureg = jaxrts.ureg
plt.style.use("science")


theta_Debye = ureg("692K")

rho0 = ureg("2.329085g/cc")
# We don't compress, so rhos is just rho0
rhos = rho0
a0 = ureg("543.0986 pm")
a = (rho0 / rhos) ** (1 / 3) * a0


def peak_function(k):
    return jnp.array(
        [
            [
                (
                    (1 / ureg.angstrom)
                    * jaxrts.instrument_function.instrument_gaussian(
                        k, sigma=0.02 / ureg.angstrom
                    )
                ).m_as(ureg.dimensionless)
            ]
        ]
    )


k_pos = []
intensities = []

# Calculate the peak positions and intensities
for h, k, l in [
    (1, 1, 1),
    (2, 2, 0),
    (3, 1, 1),
    (4, 0, 0),
    (3, 3, 1),
    (4, 2, 2),
    (5, 1, 1),
    (3, 3, 3),
    (4, 4, 0),
    (5, 3, 1),
    (6, 2, 0),
    (5, 3, 3),
]:
    d_hkl = a / (jnp.sqrt(h**2 + k**2 + l**2))
    k_pos.append(2 * jnp.pi / d_hkl)
    # Calculate the form factors / forbidden peaks
    I = (
        jnp.exp(2j * jnp.pi * 0)
        + jnp.exp(2j * jnp.pi * (h / 4 + k / 4 + l / 4))
        + jnp.exp(2j * jnp.pi * (k / 2 + l / 2))
        + jnp.exp(2j * jnp.pi * (h / 2 + l / 2))
        + jnp.exp(2j * jnp.pi * (h / 2 + k / 2))
        + jnp.exp(2j * jnp.pi * (1 / 4 * h + 3 / 4 * k + 3 / 4 * l))
        + jnp.exp(2j * jnp.pi * (3 / 4 * h + 1 / 4 * k + 3 / 4 * l))
        + jnp.exp(2j * jnp.pi * (3 / 4 * h + 3 / 4 * k + 1 / 4 * l))
    )
    intensities.append(jnp.real(jnp.sqrt(I**2)))

k_pos = jnp.array([k.m_as(1 / ureg.angstrom) for k in k_pos]) / (
    1 * ureg.angstrom
)
intensities = jnp.array(intensities)

PowderModel = jaxrts.models.PeakCollection(k_pos, intensities, peak_function)

# Set the plasma model to 1
S_plasmaModel = jaxrts.models.FixedSii(jnp.array([[1]]))

# Use a fixed Debye temperature
DebyeSolid = jaxrts.models.DebyeWallerSolid(S_plasmaModel, PowderModel)

probe_k = jnp.linspace(1, 8, 3000) / (1 * ureg.angstrom)

# This is a dummy setup
_setup = jaxrts.Setup(ureg("90°"), None, None, lambda x: x)

state = jaxrts.PlasmaState([jaxrts.Element("Si")], [0], [rhos], ureg("300K"))
state["Debye temperature"] = jaxrts.models.ConstantDebyeTemp(theta_Debye)

for T in jnp.array([300, 800, 1300]) * ureg.kelvin:
    Sii = []
    state.T_e = T
    state.T_i = jnp.array([T.m_as(ureg.kelvin)]) * ureg.kelvin
    for k in probe_k:
        setup = jaxrts.setup.get_probe_setup(k, _setup)
        Sii_out = DebyeSolid.S_ii(state, setup)
        print(Sii_out)
        Sii.append(Sii_out[0, 0])

    Sii = jnp.array(Sii)
    plt.plot(
        probe_k.m_as(1 / ureg.angstrom), Sii, label=f"{T.m_as(ureg.kelvin)}K"
    )

plt.yscale("log")

plt.ylabel("$S_{ii}(k)$")
plt.xlabel("$k$ [1/\\AA]")

plt.legend()
plt.show()
