import multiprocessing as mp
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
Make sure the NN is trained for the same parameter size or a larger one -- extrapolations typically dont work out well!
Usually 2000 samples are sufficient to test out if a NN is good at interpolating the HNC outputs --> smaller errors.
The script starts the calculation of num_compares on several CPU cores. The metric to determine how well the NN interpolates
the Sii's from HNC is the mean squared error (MSE).
"""

ureg = jaxrts.ureg
# Set filepath to PlasmaState json file
file_path = "train_data/1.0C1.0H_200000.json"
expanded = False

# Set the names of the Neural nets to test and compare to HNC
# Must be placed in the trained_NNs dir!
NN_names = [
    # f"CH_e100_expanded_64_128_128_64",
    # f"CH_e200_expanded_64_128_128_64",
    # f"CH_e300_expanded_64_128_128_64",
    # f"CH_e400_expanded_64_128_128_64",
    # f"CH_e500_expanded_64_128_128_64",
    # f"CH_e600_expanded_64_128_128_64",
    # f"CH_e700_expanded_64_128_128_64",
    f"CH_e800_64_128_128_64",
    # f"CH_e900_expanded_64_128_128_64",
    # f"CH_e1000_expanded_64_128_128_64",
    # f"CH_e1100_expanded_64_128_128_64",
    # f"CH_e1200_expanded_64_128_128_64",
    # f"CH_e1300_expanded_64_128_128_64",
    # f"CH_e1400_expanded_64_128_128_64",
    # f"CH_e1500_expanded_64_128_128_64",
    # f"CH_e1600_expanded_64_128_128_64",
    # f"CH_e1700_expanded_64_128_128_64",
    # f"CH_e1800_expanded_64_128_128_64",
    # f"CH_e1900_expanded_64_128_128_64",
    # f"CH_e2000_expanded_64_128_128_64",
]

# number of Samples for training dataset.
num_compares: int = int(2000)

# Set limits of parameters k, rho and T that defines the parameter space
k_lower: float = 0.1  # Units in 1 / Angström
k_upper: float = 20  # Units in 1 / Angström

T_lower: float = 1.0  # Units in eV
T_upper: float = 300.0  # Units in eV

rho_lower: float = 0.01  # Units in g/cm^3
rho_upper: float = 100  # Units in g/cm^3

# Print out plasma conditions for which the squared error is greater than given value
print_out_error_threshold: float | None = None
print(f"Printout error threshold is set to {print_out_error_threshold}!")

num_cores: int = mp.cpu_count() - 2

# Want to show all individual plots, set to True, figures are always saved in "trained_NNs/<NN name>/"
show_figures: bool = False

with open(file_path, "r") as json_file:
    plasma_state: jaxrts.PlasmaState = jaxrts.saving.load(
        fp=json_file, unit_reg=ureg
    )

state_hnc = deepcopy(plasma_state)
state_hnc["ionic scattering"] = jaxrts.models.OnePotentialHNCIonFeat(mix=0.25)

state_net = deepcopy(plasma_state)
state_net["ionic scattering"] = NNSiiModel(
    pathlib.Path(__file__).parent.parent / f"trained_NNs/{NN_names[0]}"
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
def calculate_Sii(plasma_state, setup):
    return plasma_state["ionic scattering"].S_ii(plasma_state, setup)


@jax.jit
def calculate_Sii_HNC_non_expanded(Z=None, temp=None, scatv=None, rho=None):
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

    return state["ionic scattering"].S_ii(state, setup)


@jax.jit
def calculate_Sii_HNC_expanded(Z=None, temp=None, scatv=None, rho=None):
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
    return expanded_state["ionic scattering"].S_ii(expanded_state, setup)


def worker_wrapper_expanded(args):
    # This unpacks the tuple provided by the zip() in 'tasks'
    return calculate_Sii_HNC_expanded(*args)


def worker_wrapper_non_expanded(args):
    # This unpacks the tuple provided by the zip() in 'tasks'
    return calculate_Sii_HNC_non_expanded(*args)


if __name__ == "__main__":
    t0 = time.time()
    print(f"--- Calculating the HNC output for comparison ---")

    rng = jax.random.PRNGKey(10404034513215)
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

    Sii_from_HNC = []
    print(f"Launching {len(tasks)} tasks on {num_cores} cores:")

    # Start Multiprocess
    with mp.Pool(processes=num_cores) as pool:
        for res in tqdm.tqdm(
            pool.imap(worker_wrapper, tasks), total=len(tasks)
        ):
            Sii_from_HNC.append(res.m_as(ureg.dimensionless))

    Sii_res_HNC = jnp.asarray(Sii_from_HNC)
    print(f"Total time taken = {time.time() - t0:.2f} s")

    ### Calculating output from Sii and the error compared to HNC
    for NN_name in NN_names:
        print(f"--- Calculating the Output for NN: {NN_name} the output ---")
        state_net["ionic scattering"] = NNSiiModel(
            pathlib.Path(__file__).parent.parent / f"trained_NNs/{NN_name}"
        )
        Sii_error_net = jnp.zeros(num_compares)
        Sii_error_dict: dict = {}
        total_squared_error = 0
        i = 0
        for ioni, temp, scat_vec, dens in zip(Z, T, k, rho):
            predict_state, setup = prep_state(
                state_net, setup_start, rho=dens, temp=temp, Z=ioni, k=scat_vec
            )

            if expanded:
                predict_state = (
                    predict_state.expand_integer_ionization_states()
                )

            Sii_res_net = calculate_Sii(predict_state, setup).m_as(
                ureg.dimensionless
            )
            squared_error = jnp.sum((Sii_res_net - Sii_res_HNC[i]) ** 2)

            if print_out_error_threshold is not None:
                if squared_error >= print_out_error_threshold:
                    print(
                        f"Condition {i}: Z = {ioni}, T = {temp:.3f} eV, k = {scat_vec:.3f} inv. Angström, rho = {dens:.3f} g/cc"
                    )
                    print(f"     Squared error = {squared_error:.3e}")

            Sii_error_net = Sii_error_net.at[i].set(squared_error)
            Sii_error_dict[i] = (
                onp.array(ioni).tolist(),
                float(temp),
                float(scat_vec),
                float(dens),
                float(squared_error),
            )
            total_squared_error += squared_error
            i += 1
        mean_squared_error = total_squared_error / num_compares

        with open(f"trained_NNs/{NN_name}/Sii_error.json", "w") as f:
            json.dump(Sii_error_dict, f, indent=4)

        print(
            f"The MeanSquaredError of all Sii interpolations is: {mean_squared_error:.3e}"
        )

        nr_array = jnp.arange(num_compares)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.05,
            0.95,
            s=f"MSE = {mean_squared_error:.3e}",
            transform=ax.transAxes,
        )
        ax.scatter(nr_array, Sii_error_net, marker="X", s=10, alpha=0.8)
        ax.set_ylabel(r"Sqaured error of Sii (NN - HNC)$^2$")
        ax.set_xlabel("Number")
        ax.set_yscale("log")
        fig.savefig(f"trained_NNs/{NN_name}/Sii_error.png", dpi=300)

        if show_figures:
            plt.show()

        plt.close()

        with open(f"trained_NNs/{NN_name}/Sii_error.json", "r") as f:
            Sii_error_dict = json.load(f)

        fig, ax1 = plt.subplots(figsize=(8, 6))
        for value in Sii_error_dict.values():

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
            ax1.scatter(
                value[2], value[-1], c="C0", marker="X", s=10, alpha=0.2
            )

        ax1.text(
            0.05,
            0.95,
            s=f"MSE = {mean_squared_error:.3e}",
            transform=ax1.transAxes,
        )
        ax1.set_xlabel("k [1/Angström]")
        ax1.set_ylabel(r"Sqaured error of Sii (NN - HNC)$^2$")
        ax1.set_yscale("log")
        fig.suptitle("Comparison Error HNC-NN over scattering vector")
        fig.tight_layout()
        fig.savefig(f"trained_NNs/{NN_name}/Sii_error_over_k.png", dpi=300)

        if show_figures:
            plt.show()

        plt.close()
