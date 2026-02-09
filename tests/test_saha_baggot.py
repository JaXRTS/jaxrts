from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt

import jaxrts
from jaxrts import ureg
from jaxrts.units import to_array
from jaxrts.plasma_physics import (
    fermi_energy,
    chem_pot_interpolationIchimaru,
    chem_pot_sommerfeld_fermi_interpolation,
    degeneracy_param,
)
from jaxrts.saha import (
    calculate_mean_free_charge_saha,
    calculate_charge_state_distribution,
)
from functools import partial
import jpu.numpy as jnpu
from collections import defaultdict
from tqdm import tqdm
import numpy as np

import pytest
import os


def plot_chemical_potentials():

    fig, ax = plt.subplots(figsize=(3.5, 3))

    T_e = 10 * ureg.eV / ureg.k_B

    densities = jnp.logspace(16, 27, 300) / (1 * ureg.cc)

    ax.plot(
        densities,
        fermi_energy(densities).m_as(ureg.eV),
        lw=3,
        alpha=0.8,
        ls="dashed",
        label=r"$E_F$",
    )
    ax.plot(
        densities,
        chem_pot_interpolationIchimaru(
            T_e * jnp.ones_like(densities), densities
        ).m_as(ureg.eV),
        label=r"Ichimaru",
        alpha=0.6,
        lw=3,
    )
    ax.plot(
        densities,
        chem_pot_sommerfeld_fermi_interpolation(
            T_e * jnp.ones_like(densities), densities
        ).m_as(ureg.eV),
        label=r"Sommerfeld",
        ls="dotted",
        alpha=0.8,
        lw=3,
    )
    ax.plot(
        densities,
        (
            1
            * ureg.boltzmann_constant
            * T_e
            * jnpu.log(
                degeneracy_param(densities, T_e * jnp.ones_like(densities)) / 2
            )
        ).m_as(ureg.eV),
        alpha=0.8,
        lw=3,
        label="Non-degenerate",
    )

    ax.legend(loc="upper left", fontsize=12)
    ax.set_ylabel(r"$\mu$", fontsize=12)
    ax.set_xlabel(r"$n^H (\text{cm}^{-3})$", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.show()


def test_compare_hydrogen_baggot():
    """
    Reproducing Fig. 3.6 from Baggot.2017.
    """

    TOLERANCE = 0.25
    DATA_DIR = "data/Baggot2017"

    if not os.path.exists(DATA_DIR):
        pytest.fail(f"Data directory '{DATA_DIR}' not found. Test cannot run.")

    # Setup Plasma State
    state = jaxrts.PlasmaState(
        ions=[jaxrts.Element("H")],
        Z_free=[1],
        mass_density=[ureg("1g/cc")],
        T_e=1 * ureg.electron_volt / ureg.k_B,
    )

    configs = [
        [
            jaxrts.models.Neglect(),
            jaxrts.models.NonDegenerateElectronChemPotential(),
            False,
        ],
        [
            jaxrts.models.DebyeHueckelIPD(),
            jaxrts.models.NonDegenerateElectronChemPotential(),
            False,
        ],
        [
            jaxrts.models.DebyeHueckelIPD(arb_deg=True),
            jaxrts.models.IchimaruChemPotential(),
            False,
        ],
        [
            jaxrts.models.DebyeHueckelIPD(),
            jaxrts.models.IchimaruChemPotential(),
            True,
        ],
    ]

    max_diffs = []

    for k, config in enumerate(configs):
        state["ipd"] = config[0]
        state["chemical potential"] = config[1]
        exclude_non_negative_energies = config[2]

        for T_e in [2, 5, 15]:
            filename = os.path.join(DATA_DIR, f"baggot{k+1}_{int(T_e)}eV.csv")

            if not os.path.exists(filename):
                pytest.fail(f"Reference file missing: {filename}")

            try:
                data = np.genfromtxt(
                    filename, delimiter=",", skip_header=1, unpack=True
                )
            except ValueError:
                pytest.fail(
                    f"File exists but is empty or malformed: {filename}"
                )

            ref_n, ref_Z = data

            calculated_Z_free = []

            for n in ref_n:
                state.T_e = T_e * ureg.eV / ureg.k_B
                state.T_i = [T_e] * ureg.eV / ureg.k_B
                state.mass_density = (
                    n * (1 / ureg.cc) * jaxrts.Element("H").atomic_mass
                )

                pop, Z_free = calculate_mean_free_charge_saha(
                    state,
                    use_ipd=True,
                    use_chem_pot=True,
                    use_distribution=True,
                    exclude_non_negative_energies=exclude_non_negative_energies,
                )

                if len(Z_free) == 1:
                    calculated_Z_free.append(Z_free[0])

            # Compare only data which is not in the WDM regime (Θ ~ 1), as there can be expected 
            # high differences!


            # Calculate Θ
            deg_param = degeneracy_param(
                        ref_n * (1 / ureg.cc), T_e * 1 * ureg.eV / ureg.k_B
                    ).m_as(ureg.dimensionless)

            diffs = np.abs(calculated_Z_free - ref_Z)[np.abs(deg_param) - 1 > 1]
            max_diff = np.max(
                diffs
            )

            max_diffs.append(max_diff)

    assert np.max(max_diffs) < TOLERANCE


def plot_hydrogen_baggot():
    """
    Reproducing Fig. 3.6 from Baggot.2017.
    """

    fig, axs = plt.subplots(figsize=(6.5, 6), ncols=2, nrows=2)

    state = jaxrts.PlasmaState(
        ions=[jaxrts.Element("H")],
        Z_free=[1],
        mass_density=[ureg("1g/cc")],
        T_e=1 * ureg.electron_volt / ureg.k_B,
    )

    configs = [
        [
            jaxrts.models.Neglect(),
            jaxrts.models.NonDegenerateElectronChemPotential(),
            False,
        ],
        [
            jaxrts.models.DebyeHueckelIPD(),
            jaxrts.models.NonDegenerateElectronChemPotential(),
            False,
        ],
        [
            jaxrts.models.DebyeHueckelIPD(arb_deg=True),
            jaxrts.models.IchimaruChemPotential(),
            False,
        ],
        [
            jaxrts.models.DebyeHueckelIPD(),
            jaxrts.models.IchimaruChemPotential(),
            True,
        ],
    ]

    for k, ax in enumerate(axs.flatten()):
        state["ipd"] = configs[k][0]
        state["chemical potential"] = configs[k][1]
        exclude_non_negative_energies = configs[k][2]

        densities = jnp.logspace(16, 26, 300)

        color = ["darkred", "darkblue", "darkgreen", "darkorange", "lime"]

        for i, T_e in enumerate([2, 5, 15]):

            Zfree = []

            for n in densities:

                state.T_e = (
                    T_e * 1 * ureg.electron_volt / ureg.boltzmann_constant
                )
                state.T_i = (
                    [T_e] * 1 * ureg.electron_volt / ureg.boltzmann_constant
                )
                state.mass_density = (
                    n * 1 / (1 * ureg.cc) * jaxrts.Element("H").atomic_mass
                )

                pop, Z_free = calculate_mean_free_charge_saha(
                    state,
                    use_ipd=True,
                    use_chem_pot=True,
                    use_distribution=True,
                    exclude_non_negative_energies=exclude_non_negative_energies,
                )
                Zfree.append(Z_free)

            ax.plot(
                densities,
                Zfree,
                ls="solid",
                lw=2,
                label=f"{T_e} eV",
                alpha=0.7,
                color=color[i],
            )

            try:
                n, Z = np.genfromtxt(
                    f"Baggot_data/baggot{k+1}_{int(T_e)}eV.csv",
                    delimiter=",",
                    skip_header=1,
                    unpack=True,
                )
                ax.scatter(n, Z, color=color[i])
            except:
                pass

            ax.plot(
                densities,
                jnp.clip(
                    fermi_energy(densities * 1 / (1 * ureg.cc)).m_as(ureg.eV)
                    / chem_pot_interpolationIchimaru(
                        T_e
                        * 1
                        * ureg.electron_volt
                        / ureg.boltzmann_constant
                        * jnp.ones_like(densities),
                        densities * 1 / (1 * ureg.cc),
                    ).m_as(ureg.eV),
                    0,
                    1,
                ),
                color=color[i],
                ls="dashed",
                lw=2,
                alpha=0.3,
            )

        ax.legend(loc="lower left", fontsize=12)
        ax.set_ylabel(r"$\langle Z^H\rangle$", fontsize=12)
        ax.set_xlabel(r"$n^H (\text{cm}^{-3})$", fontsize=12)
        ax.set_xscale("log")

    plt.tight_layout()
    plt.show()
