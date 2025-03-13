from copy import deepcopy
import jaxrts
import jax
import jax.numpy as jnp
import jpu.numpy as jnpu
import numpy as onp
import pathlib
import matplotlib.pyplot as plt
import time

from model import OneComponentNNModel, TwoComponentNNModel
from generate_data import plasma_state

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
        state.mass_density = (
            rho
            * jaxrts.helpers.mass_from_number_fraction(
                state.number_fraction, state.ions
            )
            * ureg.gram
            / ureg.centimeter**3
        )
    if temp is not None:
        state.T_i = (
            jnp.ones(state.nions) * temp * ureg.electron_volt / ureg.k_B
        )
        state.T_e = temp * ureg.electron_volt / ureg.k_B
    if Z is not None:
        state.Z_free = Z
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


calculate_state = deepcopy(plasma_state)
calculate_state["ionic scattering"] = jaxrts.models.OnePotentialHNCIonFeat(
    mix=0.8
)
predict_state = deepcopy(plasma_state)
predict_state["ionic scattering"] = TwoComponentNNModel(
    pathlib.Path(__file__).parent / "checkpoints/epoch_190/"
)

predict_state, setup = prep_state(predict_state, setup, temp=50)
calculate_state, setup = prep_state(calculate_state, setup, temp=50)
t0 = time.time()
net_out = onp.array(calculate(predict_state, setup))
t1 = time.time()
calc_out = onp.array(calculate(calculate_state, setup))
t2 = time.time()
print(f"Net:  {net_out[1, 1]:.4f} ({t1-t0:.4f}s)")
print(f"Calc: {calc_out[1, 1]:.4f} ({t2-t1:.4f}s)")
print("======================")

t0 = time.time()
net_out = onp.array(calculate(predict_state, setup))
t1 = time.time()
calc_out = onp.array(calculate(calculate_state, setup))
t2 = time.time()
print(f"Net:  {net_out[1, 1]:.4f} ({t1-t0:.4f}s)")
print(f"Calc: {calc_out[1, 1]:.4f} ({t2-t1:.4f}s)")
print("======================")

predict_state, setup = prep_state(predict_state, setup, temp=30)
calculate_state, setup = prep_state(calculate_state, setup, temp=30)
t0 = time.time()
net_out = onp.array(calculate(predict_state, setup))
t1 = time.time()
calc_out = onp.array(calculate(calculate_state, setup))
t2 = time.time()
print(f"Net:  {net_out[1, 1]:.4f} ({t1-t0:.4f}s)")
print(f"Calc: {calc_out[1, 1]:.4f} ({t2-t1:.4f}s)")
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
    )
)
print("======================")
print(time.time() - t0)
t0 = time.time()
predicted_Sii = onp.array(
    jax.vmap(calc_Sii, in_axes=(None, None, 0, 0, None, None))(
        predict_state, setup, r2.flatten(), t2.flatten(), None, 1.5
    )
)
print(time.time() - t0)
print("======================")
predicted_Sii_small = onp.array(
    jax.vmap(calc_Sii, in_axes=(None, None, 0, 0, None, None))(
        predict_state, setup, r1.flatten(), t1.flatten(), None, 1.5
    )
)

fig = plt.figure()
no_of_Sii = int(plasma_state.nions * (plasma_state.nions + 1) / 2)
fig, ax = plt.subplots(
    1,
    no_of_Sii,
    subplot_kw={"projection": "3d"},
    figsize=(5 * plasma_state.nions, 5),
)

i = 0
for k in range(calculated_Sii.shape[1]):
    for l in range(k + 1):
        print(k, l)
        calculated_S11 = calculated_Sii[:, k, l].reshape(r1.shape)
        predicted_S11 = predicted_Sii[:, k, l].reshape(r2.shape)
        predicted_S11_small = predicted_Sii_small[:, k, l].reshape(r1.shape)
        err = calculated_S11 - predicted_S11_small
        print(jnp.mean(jnp.abs(err)), jnp.max(jnp.abs(err)))
        ax[i].scatter(
            r1, t1, calculated_S11, marker="x", color="C0", label="calc"
        )
        ax[i].plot_wireframe(
            r2,
            t2,
            predicted_S11,
            color="C1",
            label="NN",
            alpha=0.8,
        )
        ax[i].set_xlabel("$\\rho$ [g/cc]")
        ax[i].set_ylabel("$T$ [eV]")
        ax[i].set_zlabel(
            "$S_{"
            + plasma_state.ions[k].symbol
            + plasma_state.ions[l].symbol
            + "}$"
        )
        i += 1

plt.legend()

plt.savefig("rho-T.png")
plt.close()


r1, k1 = jnpu.meshgrid(rho1, scatv1)
r2, k2 = jnpu.meshgrid(rho2, scatv2)

t0 = time.time()
calculated_Sii = onp.array(
    jax.vmap(calc_Sii, in_axes=(None, None, 0, None, None, 0))(
        calculate_state, setup, r1.flatten(), 30, None, k1.flatten()
    )
)
print("======================")
print(time.time() - t0)
t0 = time.time()
predicted_Sii = onp.array(
    jax.vmap(calc_Sii, in_axes=(None, None, 0, None, None, 0))(
        predict_state, setup, r2.flatten(), 30, None, k2.flatten()
    )
)
print(time.time() - t0)
print("======================")

predicted_Sii_small = onp.array(
    jax.vmap(calc_Sii, in_axes=(None, None, 0, None, None, 0))(
        predict_state, setup, r1.flatten(), 30, None, k1.flatten()
    )
)

fig = plt.figure()
no_of_Sii = int(plasma_state.nions * (plasma_state.nions + 1) / 2)
fig, ax = plt.subplots(
    1,
    no_of_Sii,
    subplot_kw={"projection": "3d"},
    figsize=(5 * plasma_state.nions, 5),
)

i = 0
for k in range(calculated_Sii.shape[1]):
    for l in range(k + 1):
        print(k, l)
        calculated_S11 = calculated_Sii[:, k, l].reshape(r1.shape)
        predicted_S11 = predicted_Sii[:, k, l].reshape(r2.shape)
        predicted_S11_small = predicted_Sii_small[:, k, l].reshape(r1.shape)
        err = calculated_S11 - predicted_S11_small
        print(jnp.mean(jnp.abs(err)), jnp.max(jnp.abs(err)))
        ax[i].scatter(
            r1, k1, calculated_S11, marker="x", color="C0", label="calc"
        )
        ax[i].plot_wireframe(
            r2,
            k2,
            predicted_S11,
            color="C1",
            label="NN",
            alpha=0.8,
        )
        ax[i].set_xlabel("$\\rho$ [g/cc]")
        ax[i].set_ylabel("$k$ 1/[Angstrom]")
        ax[i].set_zlabel(
            "$S_{"
            + plasma_state.ions[k].symbol
            + plasma_state.ions[l].symbol
            + "}$"
        )
        i += 1


plt.legend()

plt.savefig("rho-k.png")
plt.close()

t1, k1 = jnpu.meshgrid(temp1, scatv1)
t2, k2 = jnpu.meshgrid(temp2, scatv2)

t0 = time.time()
calculated_Sii = onp.array(
    jax.vmap(calc_Sii, in_axes=(None, None, None, 0, None, 0))(
        calculate_state, setup, 3.5, t1.flatten(), None, k1.flatten()
    )
)
print("======================")
print(time.time() - t0)
t0 = time.time()
predicted_Sii = onp.array(
    jax.vmap(calc_Sii, in_axes=(None, None, None, 0, None, 0))(
        predict_state, setup, 3.5, t2.flatten(), None, k2.flatten()
    )
)
print(time.time() - t0)
print("======================")

predicted_Sii_small = onp.array(
    jax.vmap(calc_Sii, in_axes=(None, None, None, 0, None, 0))(
        predict_state, setup, 3.5, t1.flatten(), None, k1.flatten()
    )
)

fig = plt.figure()
no_of_Sii = int(plasma_state.nions * (plasma_state.nions + 1) / 2)
fig, ax = plt.subplots(
    1,
    no_of_Sii,
    subplot_kw={"projection": "3d"},
    figsize=(5 * plasma_state.nions, 5),
)

i = 0
for k in range(calculated_Sii.shape[1]):
    for l in range(k + 1):
        print(k, l)
        calculated_S11 = calculated_Sii[:, k, l].reshape(t1.shape)
        predicted_S11 = predicted_Sii[:, k, l].reshape(t2.shape)
        predicted_S11_small = predicted_Sii_small[:, k, l].reshape(t1.shape)
        err = calculated_S11 - predicted_S11_small
        print(jnp.mean(jnp.abs(err)), jnp.max(jnp.abs(err)))
        ax[i].scatter(
            t1, k1, calculated_S11, marker="x", color="C0", label="calc"
        )
        ax[i].plot_wireframe(
            t2,
            k2,
            predicted_S11,
            color="C1",
            label="NN",
            alpha=0.8,
        )
        ax[i].set_xlabel("$T$ [eV]")
        ax[i].set_ylabel("$k$ 1/[Angstrom]")
        ax[i].set_zlabel(
            "$S_{"
            + plasma_state.ions[k].symbol
            + plasma_state.ions[l].symbol
            + "}$"
        )
        i += 1


plt.legend()

plt.savefig("T-k.png")
plt.close()
