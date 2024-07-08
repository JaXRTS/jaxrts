import pathlib
import sys
sys.path.append(
    "C:/Users/Samuel/Desktop/PhD/Python_Projects/JAXRTS/jaxrts/src"
)

import jaxrts
import jax
import jax.numpy as jnp
import jpu.numpy as jnpu
import numpy as onp

import matplotlib.pyplot as plt

from functools import partial

import time
import re

ureg = jaxrts.ureg

file_dir = pathlib.Path(__file__).parent
mcss_file = (
    file_dir
    / "../tests/mcss_samples/without_rk/no_ipd/mcss_C[Z_f=3.0]_E=8978eV_theta=120_rho=3.0gcc_T=2.0eV_RPA_NOLFC.txt"
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

state = jaxrts.PlasmaState(
    ions=elements,
    Z_free=Zf,
    mass_density=rho * ureg.gram / ureg.centimeter**3 * jnp.array(number_frac),
    T_e=T_e * ureg.electron_volt / ureg.k_B,
)
setup = jaxrts.setup.Setup(
    ureg(f"{theta}°"),
    ureg(f"{central_energy} eV"),
    ureg(f"{central_energy} eV")
    + jnp.linspace(-700, 200, 2000) * ureg.electron_volt,
    partial(
        jaxrts.instrument_function.instrument_gaussian,
        sigma=ureg("10eV") / ureg.hbar / (2 * jnp.sqrt(2 * jnp.log(2))),
    ),
)

# state["chemical potential"] = jaxrts.models.ConstantChemPotential(
#     0 * ureg.electron_volt
# )
state["ee-lfc"] = jaxrts.models.ElectronicLFCStaticInterpolation()

state["ipd"] = jaxrts.models.ConstantIPD(0 * ureg.electron_volt)
state["screening length"] = jaxrts.models.ArbitraryDegeneracyScreeningLength()
state["electron-ion Potential"] = jaxrts.hnc_potentials.CoulombPotential()
state["screening"] = jaxrts.models.FiniteWavelengthScreening()
state["ion-ion Potential"] = jaxrts.hnc_potentials.DebyeHuckelPotential()
state["ionic scattering"] = jaxrts.models.OnePotentialHNCIonFeat()
state["free-free scattering"] = jaxrts.models.RPA_NoDamping()
state["bound-free scattering"] = jaxrts.models.SchumacherImpulse(r_k=1)
state["free-bound scattering"] = jaxrts.models.Neglect()

print(setup.k.to(1 / ureg.angstrom))
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
MCSS_Norm = jnp.max(S_ff)
plt.plot(
    central_energy - E,
    S_tot / MCSS_Norm,
    color="C1",
)
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

t0 = time.time()
state.probe(setup)
print(f"One sample takes {time.time()-t0}s.")
plt.show()
