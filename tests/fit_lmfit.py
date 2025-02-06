import time
import pathlib

import os

# Allow jax to use 4 CPUs, see
# https://astralord.github.io/posts/exploring-parallel-strategies-with-jax/
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn")

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.2)

from jax import numpy as jnp
from jpu import numpy as jnpu
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

import lmfit

import jaxrts

from jaxrts.saha import calculate_mean_free_charge_saha

ureg = jaxrts.ureg


def load_lineout(path, th=-10):
    E, intens = np.genfromtxt(path, unpack=True, delimiter=" ", skip_header=1)
    # Filter out vcalculate_xrts(T_e, T_i, mass_dens, Z_f, FXRTS_params, BXRTS_paramsiery negative values
    mask = np.where(intens > th)
    return E[mask], intens[mask]


def simulation_energies(measured_energy):
    """
    We find that most of the energies are equi-distant; but some energies are
    missing due to exclusion of faulty pixels or detector gaps. Create an
    equidistant array and a mask to filter these energies that are not in the
    observed energies list.
    """
    # Create an array with all the equidistant energies
    dist = np.min(np.diff(measured_energy))
    min_val = np.min(measured_energy)
    # Add a small value
    max_val = np.max(measured_energy) + 1e-10
    E = np.arange(min_val, max_val, dist)
    mask = np.array([np.isclose(e, measured_energy).any() for e in E])
    return E, mask


parent_dir = pathlib.Path(__file__).parent

fE, fxrts = None, None
# Load the file
# fE, fxrts = load_lineout(parent_dir / "lineouts/Dia1007_-15_lineout_JF4.txt")
# cut the spectra
# fE = fE[400:-109]
# fxrts = fxrts[400:-109]
# Norm the spectra
# fxrts /= np.max(fxrts)

# Initialize a plasma_state (The initial conditions are not relevant for
# all the parameter that we vary)

# Create the probing setups

# Set up the energies
# sim_fxrts_E, fxrts_mask = simulation_energies(fE)
# _sim_fxrts_E, _fxrts_mask = simulation_energies(fE)

# sharding = jax.sharding.PositionalSharding(jax.devices())
# sim_fxrts_E = jax.device_put(_sim_fxrts_E, sharding)
# sharding1 = jax.sharding.PositionalSharding(jax.devices())
# fxrts_mask = jax.device_put(_fxrts_mask, sharding1)


# The instrument function:
# fxrts_inst_e, fxrts_inst = load_lineout(
#     parent_dir / "instrument/instrument_JF4_Handdrawn.csv"
# )

# fxrts_inst_e -= fxrts_inst_e[np.argmax(fxrts_inst)]
# fxrts_inst /= np.trapz(fxrts_inst, fxrts_inst_e)
# fxrts_inst_e = jnp.array(fxrts_inst_e)
# fxrts_inst = jnp.array(fxrts_inst)

bE, bxrts = load_lineout("260 X2PP 40 0_15_bxrts.txt")
bE *= 1000  # Convert to eV

# CAREFULL WITH THIS
bE += 7 * jnp.ones_like(bE)

# cut the spectra
# bxrts = bxrts[bE > 7850]
# bE = bE[bE > 7850]


dbE = np.diff(bE)[0]
extend_bE_min = -(np.arange(20)+1)[::-1] * dbE + bE[0]
extend_bE_max = (np.arange(20)+1) * dbE + bE[-1]
extend_bE = np.array([*extend_bE_min, *bE, *extend_bE_max])

# Norm the spectrum, shift to zero
bxrts -= np.mean(bxrts[-30:])
bxrts /= np.max(bxrts)

# Initialize a plasma_state (The initial conditions are not relevant for
# all the parameter that we vary)

# Create the probing setups

# Set up the energies
_sim_bxrts_E, _bxrts_mask = simulation_energies(bE)

# sharding2 = jax.sharding.PositionalSharding(jax.devices())
# sim_bxrts_E = jax.device_put(_sim_bxrts_E, sharding2)
# sharding3 = jax.sharding.PositionalSharding(jax.devices())
# bxrts_mask = jax.device_put(_bxrts_mask, sharding3)

sim_bxrts_E = _sim_bxrts_E
bxrts_mask = _bxrts_mask

# The instrument function:
bxrts_inst_e, bxrts_inst = load_lineout(
    "if_voigt_gauss_template_bxrts.txt"
)

SF_bxrts_inst_e, SF_bxrts_inst = load_lineout(
    "SASE_260 X2PP 40 15_15_xrt.txt"
)
SF_bxrts_inst_e *= 1000
SF_bxrts_inst_e -= SF_bxrts_inst_e[np.argmax(SF_bxrts_inst)]


bxrts_inst_e -= bxrts_inst_e[np.argmax(bxrts_inst)]

bxrts_inst /= np.trapz(bxrts_inst, bxrts_inst_e)
bxrts_inst_e = jnp.array(bxrts_inst_e)
bxrts_inst = jnp.array(bxrts_inst)

conv_grid = bxrts_inst_e - jnpu.mean(bxrts_inst_e)


SF_bxrts_interp_I = jnp.interp(conv_grid, SF_bxrts_inst_e, SF_bxrts_inst)

full_SIF = jnp.convolve(
    bxrts_inst,
    SF_bxrts_interp_I,
    mode="same",
)

plt.plot(bxrts_inst_e + 8165, full_SIF)
# import matplotlib.pyplot as plt
# plt.plot(fxrts_inst_e, fxrts_inst)
# plt.show()
# exit()

doping = 0.15

ions = [jaxrts.Element("C"), jaxrts.Element("H"), jaxrts.Element("O")]
number_fraction = jnp.array([1 / 3, 8 / 15, 2 / 15])  # Sample composition: C5H8O2
mass_fraction = jaxrts.helpers.mass_from_number_fraction(number_fraction, ions)

if doping > 0:
    ions.append(jaxrts.Element("Co"))
    mass_fraction = (mass_fraction * (1 - doping)).tolist()
    mass_fraction.append(doping)
    mass_fraction = jnp.array(mass_fraction)

plasma_state = jaxrts.PlasmaState(
    ions=ions,
    Z_free=jnp.zeros_like(mass_fraction),
    mass_density=ureg("260mg/cc") * mass_fraction,
    T_e=jnp.array([80]) * ureg.electron_volt / ureg.k_B,
)


plasma_state["ee-lfc"] = jaxrts.models.ElectronicLFCStaticInterpolation()
# plasma_state["ipd"] = jaxrts.models.StewartPyattIPD()
plasma_state["ipd"] = jaxrts.models.Neglect()
# plasma_state["screening length"] = jaxrts.models.ArbitraryDegeneracyScreeningLength()
plasma_state["screening length"] = jaxrts.models.DebyeHueckelScreeningLength()
plasma_state["electron-ion Potential"] = jaxrts.hnc_potentials.CoulombPotential()
plasma_state["screening"] = jaxrts.models.FiniteWavelengthScreening()
plasma_state["ion-ion Potential"] = jaxrts.hnc_potentials.DebyeHueckelPotential()
plasma_state["ionic scattering"] = jaxrts.models.OnePotentialHNCIonFeat()
# plasma_state["BM S_ii"] = jaxrts.models.Sum_Sii()
plasma_state["BM S_ii"] = jaxrts.models.AverageAtom_Sii()
plasma_state["free-free scattering"] = jaxrts.models.RPA_DandreaFit()
# plasma_state["free-free scattering"] = jaxrts.models.BornMermin_Fit()
# plasma_state["bound-free scattering"] = jaxrts.models.SchumacherImpulse(r_k=2.0)
plasma_state["bound-free scattering"] = jaxrts.models.SchumacherImpulseFitRk()
plasma_state["free-bound scattering"] = jaxrts.models.DetailedBalance()

# fxrts_setup = jaxrts.setup.Setup(
#     ureg("17°"),
#     8165 * ureg.electron_volt,
#     sim_fxrts_E * ureg.electron_volt,
#     lambda x: inst_func_fxrts(x, 1.0),
# )

bxrts_setup = jaxrts.setup.Setup(
    ureg("170°"),
    8165 * ureg.electron_volt,
    extend_bE * ureg.electron_volt,
    lambda x: inst_func_bxrts(x, 1.0),
)

@jax.jit
def inst_func_fxrts(w, scale=1):
    w = w * ureg.hbar
    wmag = w.to(ureg.electron_volt).magnitude
    fxrts_inst_func = jnp.interp(
        wmag, fxrts_inst_e * scale, fxrts_inst, left=0, right=0
    )
    return fxrts_inst_func * ureg.second


@jax.jit
def inst_func_bxrts(w, scale=1):
    w = w * ureg.hbar
    wmag = w.to(ureg.electron_volt).magnitude
    bxrts_inst_func = jnp.interp(wmag, bxrts_inst_e * scale, full_SIF, left=0, right=0)
    return bxrts_inst_func * ureg.second


def set_plasma_state(state, T_e, mass_dens):
    # state.Z_free = jnp.array([Z_C, Z_H, Z_O, Z_Co])
    Z_free = list(calculate_mean_free_charge_saha(state))#[:-1] + [jnp.array(Z_Co)]
    print(list(Z_free))
    state.Z_free = jnp.array(Z_free)
    # print(state.Z_free)
    
    state.mass_density = mass_dens * ureg.gram / ureg.centimeter**3 * mass_fraction
    state.T_e = T_e * ureg.electron_volt / ureg.k_B
    state.T_i = jnp.array([T_e, T_e, T_e, T_e]) * ureg.electron_volt / ureg.k_B
    return state


# @jax.jit
def calculate_xrts(T_e, mass_dens, ampl):
    """
    Evalulate the bxrts. Using this function is required to work with
    tensorflow.

    See
    https://www.pymc.io/projects/docs/en/latest/learn/core_notebooks/pymc_overview.html#arbitrary-deterministics
    """
    # Update the Plasma State
    state = set_plasma_state(plasma_state, T_e, mass_dens)

    # FXRTS
    # E_shift_fxrts, ampl_fxrts = FXRTS_params
    # fxrts_setup = jaxrts.setup.Setup(
    #     ureg("17°"),
    #     8165 * ureg.electron_volt,
    #     (sim_fxrts_E - E_shift_fxrts) * ureg.electron_volt,
    #     lambda x: inst_func_fxrts(x, 1.0),
    # )

    # Expected value of outcome
    # mu_fxrts = state.probe(fxrts_setup).m_as(ureg.second)
    # mu_fxrts /= jnp.max(mu_fxrts)
    # mu_fxrts = mu_fxrts[fxrts_mask]

    # BXRTS
    # bxrts_setup = jaxrts.setup.Setup(
    #     ureg("160°"),
    #     8165 * ureg.electron_volt,
    #     (sim_bxrts_E - E_shift_bxrts) * ureg.electron_volt,
    #     lambda x: inst_func_bxrts(x, 1.0),
    # )

    # Expected value of outcome
    mu_bxrts = state.probe(bxrts_setup).m_as(ureg.second)[20:-20]

    return mu_bxrts * ampl


@jax.jit
def logLikeGausian_singular(model, observation, sigma):
    N = len(model)
    logl = -N / 2 * jnp.log(2 * jnp.pi * sigma**2) - jnp.sum(
        (model - observation) ** 2
    ) / (2 * sigma**2)

    # In case of numerical instabilities, treat the case as *VERY* unlikely : )
    logl = jnp.where(jnp.isnan(logl), -1e12, logl)
    return logl


# import time
# t0 = time.time()
# out1 = np.array(calculate_xrts(100.9, 2.9, 4.3, 1.0, 5.1, 6.0, 1.0))
# print(out1)
# t1 = time.time()
# out2 = np.array(calculate_xrts(102.9, 2.2, 2.3, 0.3, 3.1, 2.1, 1.1))
# print(out2)
# t2 = time.time()
# print(t1 - t0, t2 - t1)

# plt.plot(bE, out1)
# plt.plot(bE, out2)

# plt.savefig("out2.pdf")

import time

def distance(params, data):
    t0 = time.time()
    model = calculate_xrts(
        params["T_e"].value,
        params["rho"].value,
        # params["Z_C"].value,
        # params["Z_H"].value,
        # params["Z_O"].value,
        # params["Z_Co"].value,
        params["ampl_BXRTS"].value,
    )
    dist = data - model
    dist = np.where(np.isnan(dist), 100, dist)
    print(np.sum(dist), time.time()-t0, "[", params["T_e"].value, params["rho"].value, params["ampl_BXRTS"].value, "]")
    return dist


def main():
    params = lmfit.Parameters()
    params.add("T_e", value=20.0, min=0.0, max=100.0)
    params.add("rho", value=1.1, min=0.1, max=1.5)
    # params.add("Z_C", value=4.0, min=2, max=6)
    # params.add("Z_H", value=0.999, min=0, max=1, vary=False)
    # params.add("Z_O", value=4.0, min=2, max=8)
    # params.add("Z_Co", value=3.0, min=2, max=27)
    params.add("ampl_BXRTS", value=0.1, min=0.0, max=2.0, vary = True)

    out = lmfit.minimize(
            distance, 
            params, 
            kws={"data": bxrts},
            nan_policy="raise",
            # method="differential_evolution"
            # method='emcee'
        )

    print(lmfit.fit_report(out))

    # Show a nice sumary of the fit
    fit_time = time.time()
    with open(f"fit_report{fit_time}", "w") as f:
        f.writelines(lmfit.fit_report(out))

    plt.scatter(bE, bxrts, label="data", alpha=0.7)
    plt.plot(
        bE,
        calculate_xrts(
            out.params["T_e"].value,
            out.params["rho"].value,
            # out.params["Z_C"].value,
            # out.params["Z_H"].value,
            # out.params["Z_O"].value,
            # out.params["Z_Co"].value,
            out.params["ampl_BXRTS"].value,
        ),
        color="C1",
        label="fit result",
    )
    plt.savefig(f"lmfit_out{fit_time}.pdf")


if __name__ == "__main__":
    main()
