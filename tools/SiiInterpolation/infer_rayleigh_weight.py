import multiprocessing as mp
import multiprocessing
from copy import deepcopy
import jaxrts
import jax
import jax.numpy as jnp
import numpy as onp
import pathlib
import matplotlib.pyplot as plt
from typing import Tuple
import tqdm
import json
from jaxrts.experimental.SiiNN import NNSiiModel
import time

"""
Script to compare the performance of the NN to HNC for randomally drawn Samples within the set range of the input parameters.
Make sure the NN is trained for the same parameter size or a larger one -- extrapolation typically dont work out well!
Usually 2000 samples are suffiecient to test out if a NN is good at interpolating the HNC outputs --> smaller errors.
The script starts the calculation of num_compares on several CPU cores to as fast as possible calculate 
"""

plt.style.use("my_style")
ureg = jaxrts.ureg
file_path = "train_data/1.0C1.0H_200000.json"
NN_name = f"CH_e700_expanded_64_128"

# number of Samples for training dataset.
num_compares: int = int(2000)

# Set limits of parameters k, rho and T that defines the parameter space
k_lower: float = 0.1  # Units in 1 / Angström
k_upper: float = 20  # Units in 1 / Angström

T_lower: float = 1.0  # Units in eV
T_upper: float = 300.0  # Units in eV

rho_lower: float = 0.01  # Units in g/cm^3
rho_upper: float = 100  # Units in g/cm^3

expanded = True
num_cores = multiprocessing.cpu_count() - 2

with open(file_path, "r") as json_file:
    plasma_state: jaxrts.PlasmaState = jaxrts.saving.load(
        fp=json_file, unit_reg=ureg
    )

state_hnc = deepcopy(plasma_state)
state_hnc["ionic scattering"] = jaxrts.models.OnePotentialHNCIonFeat(mix=0.25)

state_net = deepcopy(plasma_state)
state_net["ionic scattering"] = NNSiiModel(
    pathlib.Path(__file__).parent.parent / f"trained_NNs/{NN_name}"
)

setup_start = jaxrts.Setup(ureg("15°"), ureg("10keV"), None, lambda x: x)


@jax.jit
def prep_state(
    state, setup, rho=None, temp=None, Z=None, k=None
) -> Tuple[jaxrts.PlasmaState, jaxrts.Setup]:
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
def calculate_WR(plasma_state, setup):
    return plasma_state["ionic scattering"].Rayleigh_weight(
        plasma_state, setup
    )


@jax.jit
def calculate_WR_HNC_non_expanded(Z=None, temp=None, scatv=None, rho=None):
    state = deepcopy(state_hnc)
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
    if scatv is not None:
        setup = jaxrts.setup.get_probe_setup(
            scatv / (1 * ureg.angstrom), setup_start
        )

    return state["ionic scattering"].Rayleigh_weight(state, setup)


@jax.jit
def calculate_WR_HNC_expanded(Z=None, temp=None, scatv=None, rho=None):
    state = deepcopy(state_hnc)
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
    if scatv is not None:
        setup = jaxrts.setup.get_probe_setup(
            scatv / (1 * ureg.angstrom), setup_start
        )
    expanded_state = state.expand_integer_ionization_states()
    return expanded_state["ionic scattering"].Rayleigh_weight(
        expanded_state, setup
    )


def worker_wrapper_expanded(args):
    # This unpacks the tuple provided by the zip() in 'tasks'
    return calculate_WR_HNC_expanded(*args)


def worker_wrapper_non_expanded(args):
    # This unpacks the tuple provided by the zip() in 'tasks'
    return calculate_WR_HNC_non_expanded(*args)


if __name__ == "__main__":
    t0 = time.time()
    print(f"--- Testing the uncertainty of the Neural Net {NN_name} ---")

    rng = jax.random.PRNGKey(10404034513213)
    key1, key2, key3, key4 = jax.random.split(rng, 4)
    Z = (
        jax.random.uniform(
            key1,
            (num_compares, len(plasma_state.ions)),
            minval=0,
            maxval=1,
        )
        * jnp.array([i.Z for i in plasma_state.ions])[jnp.newaxis, :]
    )
    T = jax.random.uniform(
        key2, (num_compares,), minval=T_lower, maxval=T_upper
    )
    k = jax.random.uniform(
        key3, (num_compares,), minval=k_lower, maxval=k_upper
    )
    rho = jax.random.uniform(
        key4, (num_compares,), minval=rho_lower, maxval=rho_upper
    )

    # Prepare inputs as a list of tuples for the pool
    tasks = list(zip(onp.array(Z), onp.array(T), onp.array(k), onp.array(rho)))
    worker_wrapper = (
        worker_wrapper_expanded if expanded else worker_wrapper_non_expanded
    )

    results = []
    print(f"Launching {len(tasks)} tasks on {num_cores} cores:")

    # Start Multiprocess
    with mp.Pool(processes=num_cores) as pool:
        for res in tqdm.tqdm(
            pool.imap(worker_wrapper, tasks), total=len(tasks)
        ):
            results.append(res)

    # flatten results array to match shape for plotting
    weight_HNC = jnp.array(results).flatten()
    print(f"Total time taken = {time.time() - t0:.2f} s")

    WR_error = jnp.zeros(num_compares)
    rayleigh_weight_dict: dict = {}
    total_error = 0
    i = 0
    for ioni, temp, scat_vec, dens in zip(Z, T, k, rho):
        predict_state, setup = prep_state(
            state_net, setup_start, rho=dens, temp=temp, Z=ioni, k=scat_vec
        )

        if expanded:
            predict_state = predict_state.expand_integer_ionization_states()

        weight_net = calculate_WR(predict_state, setup)[0]
        absolute_error = abs(weight_net - weight_HNC[i])
        WR_error = WR_error.at[i].set(absolute_error)
        rayleigh_weight_dict[i] = (
            onp.array(ioni).tolist(),
            float(temp),
            float(scat_vec),
            float(dens),
            float(absolute_error),
        )
        total_error += absolute_error
        i += 1
    average_error = total_error / num_compares

    with open(f"trained_NNs/{NN_name}/Rayleigh_weight_error.json", "w") as f:
        json.dump(rayleigh_weight_dict, f, indent=4)

    nr_array = jnp.arange(num_compares)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(
        0.05,
        0.95,
        s=f"Avg error = {average_error:.4f}",
        transform=ax.transAxes,
    )
    ax.scatter(nr_array, WR_error, marker="X", s=10, alpha=0.8)
    ax.set_ylabel("Rayleigh weight error NN - HNC")
    ax.set_xlabel("Number")
    fig.savefig(f"trained_NNs/{NN_name}/Rayleigh_weight_error.png", dpi=300)
    plt.show()
    plt.close()

    with open(f"trained_NNs/{NN_name}/Rayleigh_weight_error.json", "r") as f:
        rayleigh_weight_dict = json.load(f)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    for value in rayleigh_weight_dict.values():

        test_state, _ = prep_state(
            state_net,
            setup_start,
            rho=value[3],
            temp=value[1],
            Z=jnp.array(value[0]),
            k=value[2],
        )
        test_state: jaxrts.PlasmaState = (
            test_state.expand_integer_ionization_states()
        )
        ax1.scatter(value[2], value[-1], c="C0", marker="X", s=10, alpha=0.2)

    ax1.set_xlabel("k [1/Angström]")
    ax1.set_ylabel("Absolute Error WR NN-HNC")
    fig.suptitle("Comparison Error HNC-NN over scattering vector")
    fig.tight_layout()
    fig.savefig(
        f"trained_NNs/{NN_name}/Rayleigh_weight_error_over_k.png", dpi=300
    )
    plt.show()
    plt.close()
