import pytest

from pathlib import Path

import numpy as np

import jaxrts

ureg = jaxrts.ureg


def test_impulse_approximation() -> None:
    """
    Test the result of the impuse approximation by re-creating fig. 4.5 from
    [Kraus.2012].

    [Kraus.2012] Kraus, Dominik, Characterization of phase transitions in warm
        dense matter with X-ray scattering: Charakterisierung von
        Phasenübergängen in warmer dichter Materie mit Röntgenstreuung. 2012.
    """
    incident_energy = ureg("4750eV")
    angle = ureg("135°")
    kraus_energy, kraus_intensity = np.genfromtxt(
        Path(__file__).parent / "data/Kraus2012/kraus.2012.fig4.5_a.csv",
        delimiter=",",
        unpack=True,
    )

    import matplotlib.pyplot as plt

    kraus_energy *= ureg.electron_volt
    for HR_c in [False, True]:
        s_be = jaxrts.bound_free.inelastic_structure_factor(
            kraus_energy,
            incident_energy,
            angle,
            6,
            {1: {0: 2}, 2: {0: 2, 1: 2}},
            {
                1: {0: 5.81 * ureg.dimensionless},
                2: {
                    0: 3.96 * ureg.dimensionless,
                    1: 3.09 * ureg.dimensionless,
                },
            },
            E_b={1: ureg("287eV"), 2: ureg("11eV")},
            J_approx="impulse",
            HR_Correction=HR_c,
        )
        plt.plot(
            kraus_energy, s_be / np.max(s_be), label=f"Calc., HR_C = {HR_c}"
        )
    plt.plot(kraus_energy, kraus_intensity, label="Literature")
    errors = kraus_intensity - (s_be / np.max(s_be).magnitude)
    plt.legend()

    plt.show()

    # Check that the most of the values are reasonable close
    assert np.quantile(errors**2, 0.9) < 0.1


def test_gregori_fig1() -> None:
    # 1s:
    E_b = ureg("286eV")
    rel_E = np.logspace(-1, 2, 500)
    E_c = E_b / rel_E

    # Convert this to an incident k
    k = np.sqrt((2 * ureg.m_e * E_c) / (ureg.hbar**2))

    # angle = ureg("180°")
    # assert np.sin(angle / 2) == 1.0
    # E_0 = (
    #     ureg.hbar
    #     * ureg.c
    #     / 2
    #     * np.sqrt((2 * ureg.m_e * E_c) / (ureg.hbar**2))
    # )

    # Use this for the further analysis


test_impulse_approximation()
