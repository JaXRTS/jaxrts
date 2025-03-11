from copy import deepcopy
import jaxrts
import jax
import jax.numpy as jnp
import jpu.numpy as jnpu
import numpy as onp
import pathlib
import matplotlib.pyplot as plt
import time

from model import OneComponentNNModel

ureg = jaxrts.ureg


@jax.jit
def calculate(plasma_state, setup):
    return (
        plasma_state["ionic scattering"]
        .S_ii(plasma_state, setup)
        .m_as(ureg.dimensionless)
    )


@jax.jit
def prep_state(state, setup, rho=None, temp=None, Z=None, k=None):
    if rho is not None:
        state.mass_density = jnp.array([rho]) * ureg.gram / ureg.centimeter**3
    if temp is not None:
        state.T_i = jnp.array([temp]) * ureg.electron_volt / ureg.k_B
        state.T_e = temp * ureg.electron_volt / ureg.k_B
    if Z is not None:
        state.Z_free = jnp.array([Z])
    if k is not None:
        setup = jaxrts.setup.get_probe_setup(k / (1 * ureg.angstrom), setup)
    return state, setup


@jax.jit
def calc_Sii(plasma_state, setup, rho=None, temp=None, Z=None, k=None):
    prepped_state, prepped_setup = prep_state(
        plasma_state, setup, rho=rho, temp=temp, Z=Z, k=k
    )
    out = calculate(prepped_state, prepped_setup)
    return out


setup = jaxrts.Setup(ureg("15°"), ureg("10keV"), None, lambda x: x)

plasma_state = jaxrts.PlasmaState(
    ions=[jaxrts.Element("C")],
    Z_free=jnp.array([2]),
    mass_density=jnp.array([3.5]) * ureg.gram / ureg.centimeter**3,
    T_e=80 * ureg.electron_volt / ureg.k_B,
)

plasma_state["free-free scattering"] = jaxrts.models.Neglect()
plasma_state["bound-free scattering"] = jaxrts.models.Neglect()
plasma_state["free-bound scattering"] = jaxrts.models.Neglect()
plasma_state["screening length"] = (
    jaxrts.models.ArbitraryDegeneracyScreeningLength()
)
plasma_state["ion-ion Potential"] = (
    jaxrts.hnc_potentials.DebyeHueckelPotential()
)


calculate_state = deepcopy(plasma_state)
calculate_state["ionic scattering"] = jaxrts.models.OnePotentialHNCIonFeat(
    mix=0.8
)
predict_state = deepcopy(plasma_state)
predict_state["ionic scattering"] = OneComponentNNModel(
    pathlib.Path(__file__).parent / "checkpoints/C/"
)

predict_state, setup = prep_state(predict_state, setup, temp=50)
calculate_state, setup = prep_state(calculate_state, setup, temp=50)
t0 = time.time()
net_out = onp.array(calculate(predict_state, setup))
t1 = time.time()
calc_out = onp.array(calculate(calculate_state, setup))
t2 = time.time()
print(f"Net:  {net_out[0][0]:.4f} ({t1-t0:.4f}s)")
print(f"Calc: {calc_out[0][0]:.4f} ({t2-t1:.4f}s)")
print("======================")

t0 = time.time()
net_out = onp.array(calculate(predict_state, setup))
t1 = time.time()
calc_out = onp.array(calculate(calculate_state, setup))
t2 = time.time()
print(f"Net:  {net_out[0][0]:.4f} ({t1-t0:.4f}s)")
print(f"Calc: {calc_out[0][0]:.4f} ({t2-t1:.4f}s)")
print("======================")

predict_state, setup = prep_state(predict_state, setup, temp=30)
calculate_state, setup = prep_state(calculate_state, setup, temp=30)
t0 = time.time()
net_out = onp.array(calculate(predict_state, setup))
t1 = time.time()
calc_out = onp.array(calculate(calculate_state, setup))
t2 = time.time()
print(f"Net:  {net_out[0][0]:.4f} ({t1-t0:.4f}s)")
print(f"Calc: {calc_out[0][0]:.4f} ({t2-t1:.4f}s)")
print("======================")

# Show the net prediction vs calculated data:

rho1 = jnp.linspace(0.1, 10, 10)
rho2 = jnp.linspace(0.1, 10, 200)
scatv1 = jnp.linspace(0.1, 10, 10)
scatv2 = jnp.linspace(0.1, 10, 200)
temp1 = jnp.linspace(5, 100, 10)
temp2 = jnp.linspace(5, 100, 200)

r1, t1 = jnpu.meshgrid(rho1, temp1)
r2, t2 = jnpu.meshgrid(rho2, temp2)

t0 = time.time()
calculated_Sii = onp.array(
    jax.vmap(calc_Sii, in_axes=(None, None, 0, 0, None, None))(
        calculate_state, setup, r1.flatten(), t1.flatten(), None, 1.5
    ).reshape(r1.shape)
)
print(time.time() - t0)
t0 = time.time()
predicted_Sii = onp.array(
    jax.vmap(calc_Sii, in_axes=(None, None, 0, 0, None, None))(
        predict_state, setup, r2.flatten(), t2.flatten(), None, 1.5
    ).reshape(r2.shape)
)
print(time.time() - t0)
print("======================")

ax = plt.figure().add_subplot(projection="3d")
ax.scatter(r1, t1, calculated_Sii, marker="x", color="C0", label="calc")
ax.plot_wireframe(
    r2,
    t2,
    predicted_Sii,
    color="C1",
    label="NN",
    alpha=0.8,
)
ax.set_xlabel("$\\rho$ [g/cc]")
ax.set_ylabel("$T$ [eV]")
ax.set_zlabel("$S_{CC}$")

plt.legend()

plt.savefig("rho-T.png")
plt.close()


r1, k1 = jnpu.meshgrid(rho1, scatv1)
r2, k2 = jnpu.meshgrid(rho2, scatv2)

t0 = time.time()
calculated_Sii = onp.array(
    jax.vmap(calc_Sii, in_axes=(None, None, 0, None, None, 0))(
        calculate_state, setup, r1.flatten(), 30, None, k1.flatten()
    ).reshape(r1.shape)
)
print(time.time() - t0)
t0 = time.time()
predicted_Sii = onp.array(
    jax.vmap(calc_Sii, in_axes=(None, None, 0, None, None, 0))(
        predict_state, setup, r2.flatten(), 30, None, k2.flatten()
    ).reshape(r2.shape)
)
print(time.time() - t0)
print("======================")

ax = plt.figure().add_subplot(projection="3d")
ax.scatter(r1, k1, calculated_Sii, marker="x", color="C0", label="calc")
ax.plot_wireframe(
    r2,
    k2,
    predicted_Sii,
    color="C1",
    label="NN",
    alpha=0.8,
)
ax.set_xlabel("$\\rho$ [g/cc]")
ax.set_ylabel("$k$ [1/Angstrom]")
ax.set_zlabel("$S_{CC}$")

plt.legend()

plt.savefig("rho-k.png")
plt.close()

t1, k1 = jnpu.meshgrid(temp1, scatv1)
t2, k2 = jnpu.meshgrid(temp2, scatv2)

t0 = time.time()
calculated_Sii = onp.array(
    jax.vmap(calc_Sii, in_axes=(None, None, None, 0, None, 0))(
        calculate_state, setup, 3.5, t1.flatten(), None, k1.flatten()
    ).reshape(t1.shape)
)
print(time.time() - t0)
t0 = time.time()
predicted_Sii = onp.array(
    jax.vmap(calc_Sii, in_axes=(None, None, None, 0, None, 0))(
        predict_state, setup, 3.5, t2.flatten(), None, k2.flatten()
    ).reshape(t2.shape)
)
print(time.time() - t0)
print("======================")

ax = plt.figure().add_subplot(projection="3d")
ax.scatter(t1, k1, calculated_Sii, marker="x", color="C0", label="calc")
ax.plot_wireframe(
    t2,
    k2,
    predicted_Sii,
    color="C1",
    label="NN",
    alpha=0.8,
)
ax.set_xlabel("$T$ [eV]")
ax.set_ylabel("$k$ [1/Angstrom]")
ax.set_zlabel("$S_{CC}$")

plt.legend()

plt.savefig("T-k.png")
plt.close()
