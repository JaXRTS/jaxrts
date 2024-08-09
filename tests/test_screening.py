import pathlib

import jpu.numpy as jnpu
import numpy as onp

import jaxrts

ureg = jaxrts.ureg


def test_screening_reproduces_Chapman2015():
    data_dir = pathlib.Path(__file__).parent / "data/Chapman2015b/Fig1c/"
    elements = [jaxrts.Element("C"), jaxrts.Element("H")]
    state = jaxrts.PlasmaState(
        elements,
        [4, 1],
        ureg("4.33g/cc")
        * jaxrts.helpers.mass_from_number_fraction([0.5, 0.5], elements),
        ureg("100eV") / ureg.k_B,
    )
    V_ei = jaxrts.hnc_potentials.CoulombPotential()
    V_ei.include_electrons = True

    # Check that the density is ok by comparing the fermi energy
    assert jnpu.absolute(
        ureg("36.5eV")
        - jaxrts.plasma_physics.fermi_energy(state.n_e).to(ureg.electron_volt)
    ) < ureg("0.1eV")

    for T in [10, 100, 1000]:
        k_lit, q_lit = onp.genfromtxt(
            data_dir / f"FWL_{T}.csv", unpack=True, delimiter=","
        )
        T *= ureg.electron_volt / ureg.k_B
        state.T_e = T
        V_eik = V_ei.full_k(state, k_lit / (1 * ureg.a_0))[-1, :-1]
        q = jaxrts.ion_feature.q_FiniteWLChapman2015(
            k_lit / (1 * ureg.a_0), V_eik, state.T_e, state.n_e
        )
        assert jnpu.max(jnpu.absolute(q[0, :] / 4 - q_lit)) < 0.02
        assert jnpu.max(jnpu.absolute(q[1, :] / 1 - q_lit)) < 0.02
