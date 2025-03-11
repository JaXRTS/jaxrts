import jaxrts
import pathlib
import jax.numpy as jnp
import jax
import jpu.numpy as jnpu
import tqdm
import h5py
import numpy as onp

ureg = jaxrts.ureg


data_no = 30000

setup = jaxrts.Setup(ureg("45°"), ureg("10keV"), None, lambda x: x)

plasma_state = jaxrts.PlasmaState(
    ions=[jaxrts.Element("C")],
    Z_free=jnp.array([2]),
    mass_density=jnp.array([3.5]) * ureg.gram / ureg.centimeter**3,
    T_e=jnp.array([80]) * ureg.electron_volt / ureg.k_B,
)
E_f = jaxrts.plasma_physics.fermi_energy(plasma_state.n_e)
print(E_f.to(ureg.electron_volt))

plasma_state["ionic scattering"] = jaxrts.models.OnePotentialHNCIonFeat(
    mix=0.8
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


@jax.jit
def draw_data(theta, Z, rho, k_over_qk):
    plasma_state.mass_density = (
        jnp.array([rho]) * ureg.gram / ureg.centimeter**3
    )
    plasma_state.Z_free = jnp.array([Z])

    E_f = jaxrts.plasma_physics.fermi_energy(plasma_state.n_e)
    q_k = jnpu.sqrt(2 * ureg.electron_mass * E_f)

    k = k_over_qk * q_k / (1 * ureg.hbar)
    probe_setup = jaxrts.setup.get_probe_setup(k, setup)

    plasma_state.T_e = theta * E_f / ureg.k_B
    plasma_state.T_i = jnp.array([theta]) * E_f / ureg.k_B

    return plasma_state["ionic scattering"].S_ii(plasma_state, probe_setup)


rng = jax.random.PRNGKey(10404020250306)
key1, key2, key3, key4 = jax.random.split(rng, 4)

theta = 10 ** jax.random.uniform(key1, (data_no,), minval=-2, maxval=2)
Z = jax.random.uniform(key2, (data_no,), minval=0, maxval=6)
rho = jax.random.uniform(key3, (data_no,), minval=0.1, maxval=10)
k_over_qk = 10 ** jax.random.uniform(key4, (data_no,), minval=-2, maxval=1)

S_CC = []
for i in tqdm.tqdm(range(data_no)):
    S_CC.append(
        draw_data(theta[i], Z[i], rho[i], k_over_qk[i])[0, 0].m_as(
            ureg.dimensionless
        )
    )


hdf5_file_path = "data.h5"


# Save the NumPy array to an HDF5 file
with h5py.File(hdf5_file_path, "w") as hdf5_file:
    inputs = hdf5_file.create_group("inputs")
    inputs.create_dataset("theta", data=onp.array(theta))
    inputs.create_dataset("Z", data=onp.array(Z))
    inputs.create_dataset("rho", data=onp.array(rho))
    inputs.create_dataset("k_over_qk", data=onp.array(k_over_qk))
    outputs = hdf5_file.create_group("outputs")
    outputs.create_dataset("S_CC", data=onp.array(S_CC))

with h5py.File(hdf5_file_path, "r") as hdf5_file:
    print(hdf5_file["outputs"]["S_CC"][:])
