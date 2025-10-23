"""
Compare our results to spectra generated with MCSS, written by D. Chapman
:cite:`Chapman.2016`.

.. note::

    We observed differences in the bound free feature, which we attribute to
    different binding energies implemented.

"""

import pathlib
import sys

import jax

# jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.2)

import os
import re
import time
from functools import partial

import jax.numpy as jnp
import jpu.numpy as jnpu
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as onp

import jaxrts

# jax.config.update("jax_disable_jit", True)


# Allow jax to use 6 CPUs, see
# https://astralord.github.io/posts/exploring-parallel-strategies-with-jax/
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=6"

tstart = time.time()

ureg = jaxrts.ureg

file_dir = pathlib.Path(__file__).parent
out_dir = file_dir / "mcss-comparison-output"
out_dir.mkdir(exist_ok=True)


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
    ff = re.findall(r"ff=[a-zA-Z_]*_lfc", name)[0][3:-4]
    lfc = re.findall(r"lfc=[A-Z_]*_r", name)[0][4:-2]
    rk = re.findall(r"rk=[0-9a-zA-Z]*", name)[0][3:]
    try:
        rk = float(rk)
    except ValueError:
        rk = None

    return (
        elements,
        Zf,
        number_frac,
        float(E),
        float(ang),
        float(rho),
        float(T),
        ff,
        lfc,
        rk,
    )


def plot_mcss_comparison(mcss_file):
    name = mcss_file.stem
    tcyclestart = time.time()

    elements, Zf, number_frac, central_energy, theta, rho, T_e, ff, lfc, rk = (
        load_data_from_mcss_file_name(name)
    )

    mass_fraction = jaxrts.helpers.mass_from_number_fraction(
        number_frac, elements
    )

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
    if jnp.any(state.Z_free - jnp.floor(state.Z_free) != 0):
        state = state.expand_integer_ionization_states()

    energy = (
        ureg(f"{central_energy} eV")
        - jnp.linspace(jnp.max(E), jnp.min(E), 2046) * ureg.electron_volt
    )

    setup = jaxrts.setup.Setup(
        ureg(f"{theta}°"),
        ureg(f"{central_energy} eV"),
        energy,
        # ureg(f"{central_energy} eV")
        # + jnp.linspace(-700, 200, 2000) * ureg.electron_volt,
        partial(
            jaxrts.instrument_function.instrument_gaussian,
            sigma=ureg("10eV") / ureg.hbar / (2 * jnp.sqrt(2 * jnp.log(2))),
        ),
        frc_exponent = 2.0,
    )

    state["chemical potential"] = jaxrts.models.IchimaruChemPotential()

    if lfc.lower() == "static_interp":
        state["ee-lfc"] = jaxrts.models.ElectronicLFCStaticInterpolation()
    state["ipd"] = jaxrts.models.Neglect()
    state["screening length"] = (
        jaxrts.models.ArbitraryDegeneracyScreeningLength()
    )
    state["electron-ion Potential"] = jaxrts.hnc_potentials.CoulombPotential()
    state["screening"] = jaxrts.models.FiniteWavelengthScreening()
    state["ion-ion Potential"] = jaxrts.hnc_potentials.DebyeHueckelPotential()
    state["ionic scattering"] = jaxrts.models.OnePotentialHNCIonFeat()
    if len(state.ions) > 1:
        state["BM S_ii"] = jaxrts.models.AverageAtom_Sii()
    else:
        state["BM S_ii"] = jaxrts.models.Sum_Sii()
    if ff.lower() == "dandrea_rpa_fit":
        state["free-free scattering"] = jaxrts.models.RPA_DandreaFit()
    elif ff.lower() == "born_mermin":
        state["free-free scattering"] = jaxrts.models.BornMermin(
            no_of_freq=40, KKT=False, RPA_rewrite=True
        )
        state["free-free scattering"].set_guessed_E_cutoffs(state, setup)
        print(
            state["free-free scattering"].E_cutoff_min.to(ureg.electron_volt)
        )
        print(
            state["free-free scattering"].E_cutoff_max.to(ureg.electron_volt)
        )
    state["bound-free scattering"] = jaxrts.models.SchumacherImpulseColdEdges(
        r_k=rk
    )
    state["free-bound scattering"] = jaxrts.models.Neglect()

    print("W_R")
    print(state["ionic scattering"].Rayleigh_weight(state, setup))
    print("scattering length: ")
    print(state.screening_length)
    print("n_e:")
    print(state.n_e.to(1 / ureg.centimeter**3))
    print("chemPot")
    print(
        state.evaluate("chemical potential", setup)
        / (1 * ureg.k_B * state.T_e)
    )

    I = state.probe(setup)
    t0 = time.time()
    I = state.probe(setup)
    jax.block_until_ready(I)
    print(f"One sample takes {time.time()-t0}s.")
    ff = state.evaluate("free-free scattering", setup)
    bf = state.evaluate("bound-free scattering", setup)
    el = state.evaluate("ionic scattering", setup)
    norm = jnpu.max(ff)

    fig, ax0 = plt.subplots()
    inset_ax = inset_axes(ax0, width="50%", height="50%", loc="upper left")

    for ax in [ax0, inset_ax]:
        ax.plot(
            (setup.measured_energy).m_as(ureg.electron_volt),
            ((bf + ff + el) / norm).m_as(ureg.dimensionless),
            color="black",
            label="JaXRTS",
        )
        ax.plot(
            (setup.measured_energy).m_as(ureg.electron_volt),
            (bf / norm).m_as(ureg.dimensionless),
            color="C0",
            ls="dashed",
            label="JaXRTS, bf",
            alpha=0.7,
        )
        ax.plot(
            (setup.measured_energy).m_as(ureg.electron_volt),
            (ff / norm).m_as(ureg.dimensionless),
            color="C0",
            ls="dotted",
            label="JaXRTS, ff",
            alpha=0.7,
        )
        ax.plot(
            (setup.measured_energy).m_as(ureg.electron_volt),
            (el / norm).m_as(ureg.dimensionless),
            color="C0",
            ls="dashdot",
            label="JaXRTS, elastic",
            alpha=0.7,
        )
        ax.plot(
            (setup.measured_energy).m_as(ureg.electron_volt),
            (I / norm).m_as(ureg.dimensionless),
            color="C0",
            label="JaXRTS",
        )
        MCSS_Norm = jnp.max(S_ff)
        ax.plot(
            central_energy - E, S_tot / MCSS_Norm, color="C1", label="MCSS"
        )
        ax.plot(
            central_energy - E,
            S_bf / MCSS_Norm,
            color="C1",
            ls="dashed",
            label="MCSS, bf",
            alpha=0.7,
        )
        ax.plot(
            central_energy - E,
            S_ff / MCSS_Norm,
            color="C1",
            ls="dotted",
            label="MCSS, ff",
            alpha=0.7,
        )

    inset_ax.set_xlim(8600, 9200)
    inset_ax.set_ylim(-0.1, 1.5)
    inset_ax.yaxis.set_label_position("right")
    inset_ax.yaxis.tick_right()
    inset_ax.set_xlabel("E [eV]")

    ax0.set_title(name, fontsize=7)

    ax0.set_xlabel("E [eV]")
    ax0.set_ylabel("Scattering intensity [a.u.]")
    ax0.legend()
    ax0.set_xlim(8600, 9200)

    print(f" One plot took {time.time()-tcyclestart}s.")
    plt.tight_layout()
    plt.savefig(out_dir / f"{mcss_file.stem}.pdf")
    plt.close()


for mcss_file in sorted(
    list((file_dir / "../tests/mcss_samples").glob("mcss*.txt"))
):
    print(mcss_file)
    plot_mcss_comparison(mcss_file)

print(f"Full excecution took {time.time()-tstart}s.")
