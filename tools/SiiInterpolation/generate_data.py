if __name__ == "__main__":
    import multiprocessing as mp

    # mp.freeze_support()
    mp.set_start_method("spawn")

import os

import multiprocessing
from functools import partial

import jaxrts
import jax.numpy as jnp
import jax
import tqdm
from tqdm.contrib.concurrent import thread_map
import h5py
import numpy as onp

ureg = jaxrts.ureg

"""
This code will generate data_no amount of training data used to train a neural net using the train.oy script later on.
The train data is saved in the train_data directory, alongside the plasmastate used to generate the data.
The notation reeals the relative abundance of the elements in the plasma state, the data_no and 
if the ionization state is expanded into integer values using the 'expanded' flag, e.g. 1.0C1.0H_100000_expanded.h5
"""


# number of Samples for training dataset.
data_no: int = int(4e5)

# Set limits of parameters k, rho and T that defines the parameter space
k_lower: float = 0.1  # Units in 1 / Angström
k_upper: float = 20  # Units in 1 / Angström

T_lower: float = 1.0  # Units in eV
T_upper: float = 300.0  # Units in eV

rho_lower: float = 0.01  # Units in g/cm^3
rho_upper: float = 100  # Units in g/cm^3


# Set plasmastate for which the trainingdata is generated
ions = [
    jaxrts.Element("C"),
    # jaxrts.Element("H"),
    # jaxrts.Element("O"),
]
# number_fraction = jnp.array([1 / 2, 1 / 2])
number_fraction = jnp.array([1])
# pet_number_fraction = [5 / 11, 4 / 11, 2 / 11]

# Expand into integer ionization state or use average atom ionization
expand_integer_ionization_state: bool = False


setup = jaxrts.Setup(ureg("45°"), ureg("10keV"), None, lambda x: x)
mass_fraction = jaxrts.helpers.mass_from_number_fraction(number_fraction, ions)
plasma_state = jaxrts.PlasmaState(
    ions=ions,
    Z_free=jnp.ones(len(ions)),
    mass_density=mass_fraction
    * jnp.array([1.0])
    * ureg.gram
    / ureg.centimeter**3,
    T_e=50 * ureg.electron_volt / ureg.k_B,
)

plasma_state["chemical potential"] = jaxrts.models.IchimaruChemPotential()
plasma_state["ee-lfc"] = jaxrts.models.SLFCInterpFortmann2010()
plasma_state["screening length"] = (
    jaxrts.models.ArbitraryDegeneracyScreeningLength()
)
plasma_state["electron-ion Potential"] = (
    jaxrts.hnc_potentials.CoulombPotential()
)
plasma_state["screening"] = jaxrts.models.FiniteWavelengthScreening()
plasma_state["ion-ion Potential"] = (
    jaxrts.hnc_potentials.DebyeHueckelPotential()
)
plasma_state["form-factors"] = jaxrts.models.PaulingFormFactors()
plasma_state["ionic scattering"] = jaxrts.models.OnePotentialHNCIonFeat(
    mix=0.25
)


@jax.jit
def draw_data_non_expanded(T, Z, rho, k):
    """
    Sets plasma state parameters and returns Sii's
    """
    plasma_state.mass_density = (
        mass_fraction * rho * ureg.gram / ureg.centimeter**3
    )
    plasma_state.Z_free = jnp.array(Z)
    probe_setup = jaxrts.setup.get_probe_setup(k / (1 * ureg.angstrom), setup)

    plasma_state.T_e = T * ureg.eV / ureg.k_B
    plasma_state.T_i = (
        jnp.ones(
            len(ions),
        )
        * T
        * ureg.eV
        / ureg.k_B
    )
    return plasma_state["ionic scattering"].S_ii(plasma_state, probe_setup)


@jax.jit
def draw_data_expanded(T, Z, rho, k):
    """
    Sets plasma state parameters and returns Sii's
    """
    plasma_state.mass_density = (
        mass_fraction * rho * ureg.gram / ureg.centimeter**3
    )
    plasma_state.Z_free = jnp.array(Z)
    probe_setup = jaxrts.setup.get_probe_setup(k / (1 * ureg.angstrom), setup)

    plasma_state.T_e = T * ureg.eV / ureg.k_B
    plasma_state.T_i = (
        jnp.ones(
            len(ions),
        )
        * T
        * ureg.eV
        / ureg.k_B
    )
    new_state = plasma_state.expand_integer_ionization_states()
    return new_state["ionic scattering"].S_ii(new_state, probe_setup)


def compute_S_values(i, T, Z, rho, k, expanded: bool):
    """
    This function calculates S_ii's for a given index i
    """
    if expanded:
        Sii = draw_data_expanded(T[i], Z[i, :], rho[i], k[i])
    else:
        Sii = draw_data_non_expanded(T[i], Z[i, :], rho[i], k[i])

    Sii_out = []
    for i in range(Sii.shape[0]):
        for j in range(i + 1):
            Sii_out.append(Sii[i, j].m_as(ureg.dimensionless))
    return Sii_out


def parallel_computation(data_no, T, Z, rho, k, expanded: bool, cpu_count: int):
    """
    Start Multiprocessing to generate HNC output.
    """
    with multiprocessing.Pool(
        processes=cpu_count
    ) as pool:
        func = partial(
            compute_S_values,
            T=T,
            Z=Z,
            rho=rho,
            k=k,
            expanded=expanded,
        )

        # Use tqdm with pool.map to distribute the tasks and show progress
        results = list(
            tqdm.tqdm(pool.imap(func, range(data_no)), total=data_no)
        )

    return results


if __name__ == "__main__":
    cpu_count = multiprocessing.cpu_count() - 2
    print(f"I am using {cpu_count} cpu cores for this process!")

    expanded = ""
    if expand_integer_ionization_state:
        expanded = "_expanded"

    # Create random key
    rng = jax.random.PRNGKey(10404034513213)
    key1, key2, key3, key4 = jax.random.split(rng, 4)

    # Create randomly drawn plasma state parameters
    T = jax.random.uniform(
        key1, (data_no,), minval=T_lower, maxval=T_upper
    )  # in Units eV
    Z = (
        jax.random.uniform(
            key2, (data_no, len(plasma_state.ions)), minval=0, maxval=1
        )
        * jnp.array([i.Z for i in ions])[jnp.newaxis, :]
    )
    ### round some entries of Z to full integer values to ensure also integer
    ### integer ionization values are present in the Z dataset --> better coverage
    Z = Z.at[: int(0.01 * data_no), :].set(
        jnp.round(Z[: int(0.01 * data_no), :])
    )

    pos = 0
    shift = 0.05 / len(plasma_state.ions)
    for idx in range(len(plasma_state.ions)):
        Z = Z.at[
            int((0.01 + pos) * data_no) : int((0.01 + pos + shift) * data_no),
            idx,
        ].set(
            jnp.round(
                Z[
                    int((0.01 + pos) * data_no) : int(
                        (0.01 + pos + shift) * data_no
                    ),
                    idx,
                ]
            )
        )
        pos += shift

    rho = jax.random.uniform(
        key3, (data_no,), minval=rho_lower, maxval=rho_upper
    )  # In units g/cc
    k = jax.random.uniform(
        key4, (data_no,), minval=k_lower, maxval=k_upper
    )  # In units 1/Angström

    # start parallel computation of Sii's using the HNC algorithm
    Sii = parallel_computation(
        data_no,
        T,
        Z,
        rho,
        k,
        expand_integer_ionization_state,
        cpu_count
    )

    # Create the element string for labelling the saved dataset
    element_string = ""
    for x, ion in zip(plasma_state.number_fraction, plasma_state.ions):
        x /= jnp.min(plasma_state.number_fraction)
        element_string += f"{x:.1f}{ion.symbol}"

    # saving file paths
    if not os.path.exists("train_data"):
        os.makedirs("train_data")

    hdf5_file_path = f"train_data/{element_string}_{data_no}{expanded}.h5"
    plasma_state_file_path = (
        f"train_data/{element_string}_{data_no}{expanded}.json"
    )
    # save plasma state used for creating HNC output
    with open(plasma_state_file_path, "w") as f:
        jaxrts.saving.dump(plasma_state, f, indent=2)

    # Save the NumPy array to an HDF5 file
    with h5py.File(hdf5_file_path, "w") as hdf5_file:
        inputs = hdf5_file.create_group("inputs")
        inputs.create_dataset("T", data=onp.array(T))
        for i, ion in enumerate(ions):
            inputs.create_dataset(f"Z_{ion.symbol}", data=onp.array(Z[:, i]))
        inputs.create_dataset("rho", data=onp.array(rho))
        inputs.create_dataset("k", data=onp.array(k))
        outputs = hdf5_file.create_group("outputs")

        num = 0
        # If expanded integer ionization state is used there are more ion species
        # and entries in S_ii
        if expand_integer_ionization_state:
            expanded_state = plasma_state.expand_integer_ionization_states()
            expanded_ions = []
            for i, ion in enumerate(expanded_state.ions):
                if i % 2 == 1:
                    expanded_ions.append(ion.symbol + "+")
                else:
                    expanded_ions.append(ion.symbol)

            for i in range(len(expanded_ions)):
                for k in range(i + 1):
                    outputs.create_dataset(
                        f"S_{expanded_ions[i]}{expanded_ions[k]}",
                        data=onp.array(Sii)[:, num],
                    )
                    num += 1

        else:
            for i in range(len(ions)):
                for k in range(i + 1):
                    outputs.create_dataset(
                        f"S_{ions[i].symbol}{ions[k].symbol}",
                        data=onp.array(Sii)[:, num],
                    )
                    num += 1
