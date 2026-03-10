import os

# Allow jax to use 4 CPUs, see
# https://astralord.github.io/posts/exploring-parallel-strategies-with-jax/
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    mp.set_start_method("spawn")

import multiprocessing
from functools import partial

import jaxrts
import pathlib
import jax.numpy as jnp
import jax
import jpu.numpy as jnpu
import tqdm
from tqdm.contrib.concurrent import thread_map
import h5py
import numpy as onp

ureg = jaxrts.ureg


data_no = int(1e4)

setup = jaxrts.Setup(ureg("45°"), ureg("10keV"), None, lambda x: x)

# ions = [jaxrts.Element("C"), jaxrts.Element("H"), jaxrts.Element("O")]
# number_fraction = jnp.array([0.3590, 0.4615, 0.1795])
ions = [
    # jaxrts.Element("Be"),
    jaxrts.Element("H"),
    jaxrts.Element("O"),
]
# number_fraction = jnp.array([0.3376, 0.4242, 0.2150, 0.02])
# number_fraction = jnp.array([0.3590, 0.4615, 0.1795])
number_fraction = jnp.array([1 / 3, 2 / 3])
# pet_number_fraction = [5 / 11, 4 / 11, 2 / 11]
mass_fraction = jaxrts.helpers.mass_from_number_fraction(number_fraction, ions)

plasma_state = jaxrts.PlasmaState(
    ions=ions,
    Z_free=jnp.array([0.5, 4.0]),
    mass_density=mass_fraction
    * jnp.array([1.0])
    * ureg.gram
    / ureg.centimeter**3,
    T_e=40 * ureg.electron_volt / ureg.k_B,
)

E_f = jaxrts.plasma_physics.fermi_energy(plasma_state.n_e)

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
def draw_data_non_expanded(theta, Z, rho, k_over_qk):
    plasma_state.mass_density = (
        mass_fraction * rho * ureg.gram / ureg.centimeter**3
    )
    plasma_state.Z_free = jnp.array(Z)

    E_f = jaxrts.plasma_physics.fermi_energy(plasma_state.n_e)
    q_k = jnpu.sqrt(2 * ureg.electron_mass * E_f)

    k = k_over_qk * q_k / (1 * ureg.hbar)
    probe_setup = jaxrts.setup.get_probe_setup(k, setup)

    plasma_state.T_e = theta * E_f / ureg.k_B
    plasma_state.T_i = (
        jnp.ones(
            len(ions),
        )
        * theta
        * E_f
        / ureg.k_B
    )
    # new_state = plasma_state.expand_integer_ionization_states()
    # return new_state["ionic scattering"].S_ii(new_state, probe_setup)

    return plasma_state["ionic scattering"].S_ii(plasma_state, probe_setup)


@jax.jit
def draw_data_expanded(theta, Z, rho, k_over_qk):
    plasma_state.mass_density = (
        mass_fraction * rho * ureg.gram / ureg.centimeter**3
    )
    plasma_state.Z_free = jnp.array(Z)

    E_f = jaxrts.plasma_physics.fermi_energy(plasma_state.n_e)
    q_k = jnpu.sqrt(2 * ureg.electron_mass * E_f)

    k = k_over_qk * q_k / (1 * ureg.hbar)
    probe_setup = jaxrts.setup.get_probe_setup(k, setup)

    plasma_state.T_e = theta * E_f / ureg.k_B
    plasma_state.T_i = (
        jnp.ones(
            len(ions),
        )
        * theta
        * E_f
        / ureg.k_B
    )
    new_state = plasma_state.expand_integer_ionization_states()
    return new_state["ionic scattering"].S_ii(new_state, probe_setup)


def compute_S_values(i, theta, Z, rho, k_over_qk, expanded: bool):
    # This function calculates S_HH, S_HO, S_OO for a given index i
    if expanded:
        Sii = draw_data_expanded(theta[i], Z[i, :], rho[i], k_over_qk[i])
    else:
        Sii = draw_data_non_expanded(theta[i], Z[i, :], rho[i], k_over_qk[i])

    Sii_out = []
    for i in range(Sii.shape[0]):
        for k in range(i + 1):
            Sii_out.append(Sii[i, k].m_as(ureg.dimensionless))
    return Sii_out


def parallel_computation(data_no, theta, Z, rho, k_over_qk, expanded: bool):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        func = partial(
            compute_S_values,
            theta=theta,
            Z=Z,
            rho=rho,
            k_over_qk=k_over_qk,
            expanded=expanded,
        )

        # Use tqdm with pool.map to distribute the tasks and show progress
        results = list(
            tqdm.tqdm(pool.imap(func, range(data_no)), total=data_no)
        )

    return results


if __name__ == "__main__":
    # Expand into integer ionization state or use average atom ionization
    expand_integer_ionization_state = False
    expanded = ""
    if expand_integer_ionization_state:
        expanded = "_expanded"

    rng = jax.random.PRNGKey(10404034513213)
    key1, key2, key3, key4 = jax.random.split(rng, 4)

    theta = 10 ** jax.random.uniform(key1, (data_no,), minval=-2, maxval=2)
    Z = (
        jax.random.uniform(
            key2, (data_no, len(plasma_state.ions)), minval=0, maxval=1
        )
        * jnp.array([i.Z for i in ions])[jnp.newaxis, :]
    )
    rho = jax.random.uniform(key3, (data_no,), minval=0.001, maxval=50)
    k_over_qk = 10 ** jax.random.uniform(key4, (data_no,), minval=-1, maxval=1)

    Sii = parallel_computation(
        data_no,
        theta,
        Z,
        rho,
        k_over_qk,
        expand_integer_ionization_state,
    )

    element_string = ""
    for x, ion in zip(plasma_state.number_fraction, plasma_state.ions):
        x /= jnp.min(plasma_state.number_fraction)
        element_string += f"{x:.1f}{ion.symbol}"

    # saving file paths
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
        inputs.create_dataset("theta", data=onp.array(theta))
        for i, ion in enumerate(ions):
            inputs.create_dataset(f"Z_{ion.symbol}", data=onp.array(Z[:, i]))
        inputs.create_dataset("rho", data=onp.array(rho))
        inputs.create_dataset("k_over_qk", data=onp.array(k_over_qk))
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
