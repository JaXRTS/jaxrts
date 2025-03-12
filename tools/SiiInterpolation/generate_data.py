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


data_no = int(10)

setup = jaxrts.Setup(ureg("45°"), ureg("10keV"), None, lambda x: x)

ions = [jaxrts.Element("H"), jaxrts.Element("O")]
number_fraction = jnp.array([2 / 3, 1 / 3])
mass_fraction = jaxrts.helpers.mass_from_number_fraction(number_fraction, ions)

plasma_state = jaxrts.PlasmaState(
    ions=ions,
    Z_free=jnp.array([2, 2]),
    mass_density=mass_fraction
    * jnp.array([3.5])
    * ureg.gram
    / ureg.centimeter**3,
    T_e=jnp.array([80, 80]) * ureg.electron_volt / ureg.k_B,
)
E_f = jaxrts.plasma_physics.fermi_energy(plasma_state.n_e)

plasma_state["screening length"] = (
    jaxrts.models.ArbitraryDegeneracyScreeningLength()
)
plasma_state["screening"] = jaxrts.models.Gregori2004Screening()
plasma_state["free-free scattering"] = jaxrts.models.Neglect()
plasma_state["bound-free scattering"] = jaxrts.models.Neglect()
plasma_state["free-bound scattering"] = jaxrts.models.Neglect()
plasma_state["ion-ion Potential"] = (
    jaxrts.hnc_potentials.DebyeHueckelPotential()
)
plasma_state["form-factors"] = jaxrts.models.PaulingFormFactors()
plasma_state["ionic scattering"] = jaxrts.models.OnePotentialHNCIonFeat(
    mix=0.8
)


@jax.jit
def draw_data(theta, Z, rho, k_over_qk):
    plasma_state.mass_density = (
        mass_fraction * rho * ureg.gram / ureg.centimeter**3
    )
    plasma_state.Z_free = jnp.array(Z)

    E_f = jaxrts.plasma_physics.fermi_energy(plasma_state.n_e)
    q_k = jnpu.sqrt(2 * ureg.electron_mass * E_f)

    k = k_over_qk * q_k / (1 * ureg.hbar)
    probe_setup = jaxrts.setup.get_probe_setup(k, setup)

    plasma_state.T_e = theta * E_f / ureg.k_B
    plasma_state.T_i = jnp.ones(len(ions),) * theta * E_f / ureg.k_B

    return plasma_state["ionic scattering"].S_ii(plasma_state, probe_setup)


def compute_S_values(i, theta, Z, rho, k_over_qk):
    # This function calculates S_HH, S_HO, S_OO for a given index i
    Sii = draw_data(theta[i], Z[i, :], rho[i], k_over_qk[i])
    Sii_out = []
    for i in range(Sii.shape[0]):
        for k in range(i + 1):
            Sii_out.append(Sii[i, k].m_as(ureg.dimensionless))

    return Sii_out


def parallel_computation(data_no, theta, Z, rho, k_over_qk):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        func = partial(
            compute_S_values,
            theta=theta,
            Z=Z,
            rho=rho,
            k_over_qk=k_over_qk,
        )

        # Use tqdm with pool.map to distribute the tasks and show progress
        results = list(
            tqdm.tqdm(pool.imap(func, range(data_no)), total=data_no)
        )

    return results


if __name__ == "__main__":
    rng = jax.random.PRNGKey(10404020250306)
    key1, key2, key3, key4, key5 = jax.random.split(rng, 5)

    theta = 10 ** jax.random.uniform(key1, (data_no,), minval=-2, maxval=2)
    Z = jnp.array(
        [
            jax.random.uniform(key2, (data_no,), minval=0, maxval=i.Z)
            for i in ions
        ]
    ).T
    rho = jax.random.uniform(key4, (data_no,), minval=0.1, maxval=10)
    k_over_qk = 10 ** jax.random.uniform(key5, (data_no,), minval=-2, maxval=1)

    Sii = parallel_computation(data_no, theta, Z, rho, k_over_qk)
    print(Sii)

    hdf5_file_path = "water.h5"

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
        for i in range(len(ions)):
            for k in range(i + 1):
                outputs.create_dataset(
                    f"S_{ions[i].symbol}{ions[k].symbol}",
                    data=onp.array(Sii)[:, num],
                )
                num += 1
