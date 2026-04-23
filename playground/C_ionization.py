import jaxrts
import jax.numpy as jnp
import matplotlib.pyplot as plt

ureg = jaxrts.ureg

n_e = []
Z_saha = []
Z_gen_saha = []
Z_BU = []

for dens in jnp.logspace(-1, 3, 350):
    state = jaxrts.PlasmaState(
        [jaxrts.Element("C")],
        [1],
        [dens] * ureg.gram / ureg.centimeter**3,
        100 * ureg.electron_volt / ureg.k_B,
    )
    state["ipd"] = jaxrts.models.StewartPyattIPD(arb_deg=True)
    state["chemical potential"] = jaxrts.models.IchimaruChemPotential()
    n_e.append(state.n_e.m_as(ureg.centimeter ** (-3)))

    charge_distribution, Z_mean = jaxrts.ionization.calculate_mean_free_charge_saha(
        state, True
    )
    Z_saha.append(Z_mean)

    charge_distribution, Z_mean = jaxrts.ionization.calculate_mean_free_charge_saha(
        state, True, True
    )
    Z_gen_saha.append(Z_mean)

    charge_distribution, Z_mean = jaxrts.ionization.calculate_mean_free_charge_BU(
        state,
        True,
    )
    Z_BU.append(Z_mean)

plt.xlabel("$n_e$ (1/cc)")
plt.ylabel("$Z$")
plt.title("C, SP, 100eV")

plt.plot(n_e, Z_saha, label="Saha")
plt.plot(n_e, Z_gen_saha, label="gen Saha")
plt.plot(n_e, Z_BU, label="BU")
plt.xscale("log")
plt.legend()
plt.show()
