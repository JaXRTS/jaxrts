import pathlib
import sys

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.2)

sys.path.append(
    "C:/Users/Samuel/Desktop/PhD/Python_Projects/JAXRTS/jaxrts/src"
)

from jaxrts.ee_localfieldcorrections import eelfc_farid
import jaxrts
import jax.numpy as jnp
import jpu.numpy as jnpu
import numpy as onp

# jax.config.update("jax_disable_jit", True)

import matplotlib.pyplot as plt

from functools import partial

import time
import re

import os

# Allow jax to use 6 CPUs, see
# https://astralord.github.io/posts/exploring-parallel-strategies-with-jax/
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=6"

tstart = time.time()

ureg = jaxrts.ureg

file_dir = pathlib.Path(__file__).parent
mcss_file = (
    file_dir
    / "../tests/mcss_samples/without_rk/mcss_C[Z_f=4.0]_E=8978eV_theta=17_rho=4.5gcc_T=20.0eV_BM+STATINTERP.txt"
)


def load_data_from_mcss_file_name(name):
    elements_string = re.findall(r"[_]*[A-Za-z]*\[[A-Za-z0-9.=_]*\]", name)
    elements = []
    Zf = []
    number_frac = []
    for e_s in elements_string:
        element, Z = e_s[:-1].split("Z_f=")
        element = element[:-1]
        if "[frac=" in element:
            element, nf = element.split("[frac=")
            number_frac.append(float(nf))
        else:
            number_frac.append(1)
        Zf.append(float(Z))
        if element.startswith("_"):
            element = element[1:]
        elements.append(jaxrts.Element(element))
    E = re.findall(r"E=[0-9.]*", name)[0][2:]
    ang = re.findall(r"theta=[0-9.]*", name)[0][6:]
    rho = re.findall(r"rho=[0-9.]*", name)[0][4:]
    T = re.findall(r"T=[0-9.]*", name)[0][2:]
    return (
        elements,
        Zf,
        number_frac,
        float(E),
        float(ang),
        float(rho),
        float(T),
    )


name = mcss_file.stem
elements, Zf, number_frac, central_energy, theta, rho, T_e = (
    load_data_from_mcss_file_name(name)
)


E, S_el, S_bf, S_ff, S_tot = onp.genfromtxt(
    mcss_file,
    delimiter=",",
    unpack=True,
)
print(len(E))

state = jaxrts.PlasmaState(
    ions=elements,
    Z_free=Zf,
    mass_density=rho * ureg.gram / ureg.centimeter**3 * jnp.array(number_frac),
    T_e=T_e * ureg.electron_volt / ureg.k_B,
)

sharding = jax.sharding.PositionalSharding(jax.devices())
energy = (
    ureg(f"{central_energy} eV")
    - jnp.linspace(jnp.max(E), jnp.min(E), 2046) * ureg.electron_volt
)
sharded_energy = jax.device_put(energy, sharding)
# sharded_energy = energy

setup = jaxrts.setup.Setup(
    ureg(f"{theta}°"),
    ureg(f"{central_energy} eV"),
    sharded_energy,
    # ureg(f"{central_energy} eV")
    # + jnp.linspace(-700, 200, 2000) * ureg.electron_volt,
    partial(
        jaxrts.instrument_function.instrument_gaussian,
        sigma=ureg("10eV") / ureg.hbar / (2 * jnp.sqrt(2 * jnp.log(2))),
    ),
)

# state["chemical potential"] = jaxrts.models.ConstantChemPotential(
#     0 * ureg.electron_volt
# )

state["ee-lfc"] = jaxrts.models.ElectronicLFCStaticInterpolation()
state["ipd"] = jaxrts.models.StewartPyattIPD()
state["screening length"] = jaxrts.models.ArbitraryDegeneracyScreeningLength()
state["electron-ion Potential"] = jaxrts.hnc_potentials.CoulombPotential()
state["screening"] = jaxrts.models.FiniteWavelengthScreening()
state["ion-ion Potential"] = jaxrts.hnc_potentials.DebyeHuckelPotential()
state["ionic scattering"] = jaxrts.models.OnePotentialHNCIonFeat()
state["free-free scattering"] = jaxrts.models.BornMermin_Fit()
state["bound-free scattering"] = jaxrts.models.SchumacherImpulse(r_k=1)
state["free-bound scattering"] = jaxrts.models.Neglect()

# print(setup.full_k.to(1 / ureg.angstrom))
# print(
#     jaxrts.setup.dispersion_corrected_k(setup, state.n_e).to(1 / ureg.angstrom)
# )

I = state.probe(setup)
norm = jnpu.max(
    state.evaluate("free-free scattering", setup)
    # + state.evaluate("bound-free scattering", setup)
)
plt.plot(
    (setup.measured_energy).m_as(ureg.electron_volt),
    (I / norm).m_as(ureg.dimensionless),
    color="C0",
    label="BMA (LFC=StaticInterp, naive)",
)
plt.plot(
    (setup.measured_energy).m_as(ureg.electron_volt),
    (state.evaluate("bound-free scattering", setup) / norm).m_as(
        ureg.dimensionless
    ),
    color="C0",
    ls="dashed",
    alpha=0.7,
)
plt.plot(
    (setup.measured_energy).m_as(ureg.electron_volt),
    (state.evaluate("free-free scattering", setup) / norm).m_as(
        ureg.dimensionless
    ),
    color="C0",
    ls="dotted",
    alpha=0.7,
)
plt.plot(
    (setup.measured_energy).m_as(ureg.electron_volt),
    (state.evaluate("ionic scattering", setup) / norm).m_as(
        ureg.dimensionless
    ),
    color="C0",
    ls="dashdot",
    alpha=0.7,
)
state["free-free scattering"] = jaxrts.models.RPA_DandreaFit()

I = state.probe(setup)
norm = jnpu.max(
    state.evaluate("free-free scattering", setup)
    # + state.evaluate("bound-free scattering", setup)
)
plt.plot(
    (setup.measured_energy).m_as(ureg.electron_volt),
    (I / norm).m_as(ureg.dimensionless),
    color="C2",
    label="RPA",
)
plt.plot(
    (setup.measured_energy).m_as(ureg.electron_volt),
    (state.evaluate("free-free scattering", setup) / norm).m_as(
        ureg.dimensionless
    ),
    color="C2",
    ls="dotted",
    alpha=0.7,
)
MCSS_Norm = jnp.max(S_ff)
plt.plot(central_energy - E, S_tot / MCSS_Norm, color="C1", label="MCSS")
plt.plot(
    central_energy - E,
    S_bf / MCSS_Norm,
    color="C1",
    ls="dashed",
    alpha=0.7,
)
plt.plot(
    central_energy - E,
    S_ff / MCSS_Norm,
    color="C1",
    ls="dotted",
    alpha=0.7,
)

compton_shift = (
    setup.energy
    - jaxrts.plasma_physics.compton_energy(
        setup.energy, setup.scattering_angle
    )
).m_as(ureg.electron_volt)
plt.plot([compton_shift, compton_shift], [0.9, 1.1], color="black")
plt.plot([compton_shift - 20, compton_shift + 20], [1, 1], color="black")
plt.title(name)

plt.legend()

t0 = time.time()
state.probe(setup)
print(f"One sample takes {time.time()-t0}s.")
print(f"Full excecution took {time.time()-tstart}s.")
plt.show()
