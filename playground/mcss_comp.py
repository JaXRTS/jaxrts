import pathlib
import sys

import jax

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.2)

sys.path.append(
    "C:/Users/Samuel/Desktop/PhD/Python_Projects/JAXRTS/jaxrts/src"
)

import os
import re
import time
from functools import partial

import jax.numpy as jnp
import jpu.numpy as jnpu
import matplotlib.pyplot as plt
import numpy as onp

import jaxrts

# jax.config.update("jax_disable_jit", True)


# Allow jax to use 6 CPUs, see
# https://astralord.github.io/posts/exploring-parallel-strategies-with-jax/
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=6"

tstart = time.time()

ureg = jaxrts.ureg

file_dir = pathlib.Path(__file__).parent
mcss_file = (
    file_dir
    / "../tests/mcss_samples/without_rk/no_ipd/mcss_C[frac=0.5_Z_f=3.0]O[frac=0.5_Z_f=3.0]_E=8975eV_theta=120_rho=1.8gcc_T=10.0eV_RPA_NOLFC.txt"
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

mass_fraction = jaxrts.helpers.mass_from_number_fraction(number_frac, elements)

E, S_el, S_bf, S_ff, S_tot = onp.genfromtxt(
    mcss_file,
    delimiter=",",
    unpack=True,
)

state = jaxrts.PlasmaState(
    ions=elements,
    Z_free=Zf,
    mass_density=rho * ureg.gram / ureg.centimeter**3 * mass_fraction,
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
state["ee-lfc"] = jaxrts.models.ElectronicLFCConstant(1)
state["ipd"] = jaxrts.models.StewartPyattIPD()
state["ipd"] = jaxrts.models.Neglect()
state["screening length"] = jaxrts.models.ArbitraryDegeneracyScreeningLength()
# state["screening length"] = jaxrts.models.ConstantScreeningLength(ureg("4.38E-2 nm"))
state["electron-ion Potential"] = jaxrts.hnc_potentials.CoulombPotential()
state["screening"] = jaxrts.models.FiniteWavelengthScreening()
state["ion-ion Potential"] = jaxrts.hnc_potentials.DebyeHueckelPotential()
state["ionic scattering"] = jaxrts.models.OnePotentialHNCIonFeat()
state["BM S_ii"] = jaxrts.models.AverageAtom_Sii()
state["free-free scattering"] = jaxrts.models.BornMermin_Fortmann()
state["bound-free scattering"] = jaxrts.models.SchumacherImpulse(r_k=1)
state["free-bound scattering"] = jaxrts.models.Neglect()

print("W_R")
print(state["ionic scattering"].Rayleigh_weight(state, setup))
print("scattering length: ")
print(state.screening_length)
print("n_e:")
print(state.n_e.to(1 / ureg.centimeter**3))
print("chemPot")
print(state.evaluate("chemical potential", setup) / (1 * ureg.k_B * state.T_e))
# print(setup.full_k.to(1 / ureg.angstrom))
# print(
#     jaxrts.setup.dispersion_corrected_k(setup, state.n_e).to(1 / ureg.angstrom)
# )

I = state.probe(setup)
t0 = time.time()
state.probe(setup)
print(f"One sample takes {time.time()-t0}s.")
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
print(state["ionic scattering"].Rayleigh_weight(state, setup))

_I = state.probe(setup)
_norm = jnpu.max(
    state.evaluate("free-free scattering", setup)
    # + state.evaluate("bound-free scattering", setup)
)
print(jnpu.max(I) / norm / (jnpu.max(_I) / _norm))
I = _I
norm = _norm
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

print(f"Full excecution took {time.time()-tstart}s.")
plt.show()
