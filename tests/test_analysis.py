"""
These tests investigate that certain properties of a generated spectrum hold
true.
"""

import pytest

from functools import partial

import jax
from jax import numpy as jnp
from jpu import numpy as jnpu

import jaxrts

ureg = jaxrts.ureg


class TestITCFInstance:
    test_state = jaxrts.PlasmaState(
        ions=[jaxrts.Element("Be")],
        Z_free=jnp.array([2]),
        mass_density=jnp.array([3]) * ureg.gram / ureg.centimeter**3,
        T_e=60 * ureg.electron_volt / ureg.k_B,
    )
    test_setup = jaxrts.Setup(
        scattering_angle=ureg("30°"),
        energy=ureg("6900 eV"),
        measured_energy=ureg("6900 eV")
        + jnp.linspace(-120, 120, 1000) * ureg.electron_volt,
        instrument=partial(
            jaxrts.instrument_function.instrument_gaussian,
            sigma=ureg("1.0eV") / ureg.hbar / (2 * jnp.sqrt(2 * jnp.log(2))),
        ),
    )

    test_state["ionic scattering"] = jaxrts.models.OnePotentialHNCIonFeat()
    test_state["free-free scattering"] = jaxrts.models.RPA_DandreaFit()
    test_state["bound-free scattering"] = jaxrts.models.SchumacherImpulse()
    test_state["free-bound scattering"] = jaxrts.models.DetailedBalance()

    def get_T(self, raw):
        S_ee = self.test_state.probe(self.test_setup)
        T = jaxrts.analysis.ITCFT(
            S_ee,
            ureg("60/keV"),
            self.test_setup,
            ureg("100eV"),
            raw=raw,
        )
        return T

    def test_ITCFT_without_instument_function(self):
        self.test_setup.correct_k_dispersion = False
        assert jnpu.absolute(
            self.get_T(raw=True) - self.test_state.T_e
        ) < ureg("2e3K")

    def test_ITCFT_with_instument_function(self):
        self.test_setup.correct_k_dispersion = False
        assert jnpu.absolute(
            self.get_T(raw=False) - self.test_state.T_e
        ) < ureg("2e3K")

    @pytest.mark.xfail(
        reason="The k-dispersion should violate detailed balance"
    )
    def test_k_dispersion_conflicts_with_detailed_balance(self):
        self.test_setup.correct_k_dispersion = True
        assert jnpu.absolute(
            self.get_T(raw=True) - self.test_state.T_e
        ) < ureg("2e3K")


class TestSSFInstance:
    # We use a fully ionized state, so that bound-free is not contributing
    test_state = jaxrts.PlasmaState(
        ions=[jaxrts.Element("C")],
        Z_free=jnp.array([6]),
        mass_density=jnp.array([0.3]) * ureg.gram / ureg.centimeter**3,
        T_e=100 * ureg.electron_volt / ureg.k_B,
    )
    test_setup = jaxrts.Setup(
        scattering_angle=ureg("60°"),
        energy=ureg("70 keV"),
        measured_energy=ureg("70 keV")
        + jnp.linspace(-15, 15, 6000) * ureg.kiloelectron_volt,
        instrument=partial(
            jaxrts.instrument_function.instrument_gaussian,
            sigma=ureg("50eV") / ureg.hbar / (2 * jnp.sqrt(2 * jnp.log(2))),
        ),
    )
    test_setup.correct_k_dispersion = False

    test_state["ion-ion Potential"] = (
        jaxrts.hnc_potentials.DebyeHueckelPotential()
    )
    test_state["ionic scattering"] = jaxrts.models.OnePotentialHNCIonFeat()
    test_state["free-free scattering"] = jaxrts.models.RPA_DandreaFit()
    test_state["bound-free scattering"] = jaxrts.models.SchumacherImpulse()
    test_state["free-bound scattering"] = jaxrts.models.DetailedBalance()

    def test_sff_to_unity_fully_ionized(self):
        self.test_state.Z_free = self.test_state.Z_A
        S_ee = self.test_state.probe(self.test_setup)
        ssf_raw = jaxrts.analysis.ITCF_ssf(
            S_ee, self.test_setup, ureg("14.8keV"), raw=True
        )
        ssf = jaxrts.analysis.ITCF_ssf(
            S_ee, self.test_setup, ureg("14.8keV"), raw=False
        )
        assert jnpu.absolute(ssf_raw - 1) < 0.02
        assert jnpu.absolute(ssf - 1) < 0.02

    def test_sff_to_unity_with_bound_free_contrib(self):
        """
        Setting r_k to auto should be fine, here, because it goes to 1 for
        :math:`k\\rightarrow \\infty`. However, there is the binding energy and
        further unknowns.
        """
        self.test_state.Z_free = (
            self.test_state.Z_A - jnp.ones(len(self.test_state.ions)) * 2
        )
        S_ee = self.test_state.probe(self.test_setup)
        ssf_raw = jaxrts.analysis.ITCF_ssf(
            S_ee, self.test_setup, ureg("14.8keV"), raw=True
        )
        ssf = jaxrts.analysis.ITCF_ssf(
            S_ee, self.test_setup, ureg("14.8keV"), raw=False
        )
        assert jnpu.absolute(ssf_raw - 1) < 0.05
        assert jnpu.absolute(ssf - 1) < 0.05


class TestFsumRuleInstance:
    test_state = jaxrts.PlasmaState(
        ions=[jaxrts.Element("O")],
        Z_free=jnp.array([4]),
        mass_density=jnp.array([5]) * ureg.gram / ureg.centimeter**3,
        T_e=60 * ureg.electron_volt / ureg.k_B,
    )
    test_setup = jaxrts.Setup(
        scattering_angle=ureg("40°"),
        energy=ureg("7.5 keV"),
        measured_energy=ureg("7.5 keV")
        + jnp.linspace(-2, 2, 5000) * ureg.kiloelectron_volt,
        instrument=partial(
            jaxrts.instrument_function.instrument_gaussian,
            sigma=ureg("40eV") / ureg.hbar / (2 * jnp.sqrt(2 * jnp.log(2))),
        ),
    )

    test_state["ionic scattering"] = jaxrts.models.OnePotentialHNCIonFeat()
    test_state["free-free scattering"] = jaxrts.models.RPA_DandreaFit()
    test_state["bound-free scattering"] = jaxrts.models.SchumacherImpulse()
    test_state["free-bound scattering"] = jaxrts.models.DetailedBalance()

    def fsum(self, raw):
        S_ee = self.test_state.probe(self.test_setup)
        fsum = jaxrts.analysis.ITCF_fsum(
            S_ee,
            self.test_setup,
            ureg("1900eV"),
            raw=raw,
        )
        return fsum

    @property
    def fsum_value(self):
        return -((ureg.hbar * self.test_setup.k) ** 2) / (
            2 * ureg.electron_mass
        )

    def test_fsum_rule_fully_ionized(self):
        self.test_setup.correct_k_dispersion = False
        # Set to fully ionized
        self.test_state.Z_free = self.test_state.Z_A
        assert (
            jnpu.absolute(
                (self.fsum(raw=False) - self.fsum_value) / self.fsum_value
            )
            < 0.005
        )

    @pytest.mark.xfail(
        reason="The current implementation of bound-free scattering breaks the f-sum rule the fitting r_k is not provided"  # noqa: E501
    )
    def test_bound_free_breaks_fsum_rule(self):
        self.test_setup.correct_k_dispersion = False
        # Allow some bound-free contribution
        self.test_state.Z_free = self.test_state.Z_A - jnp.ones(
            len(self.test_state.ions)
        )
        assert (
            jnpu.absolute(
                (self.fsum(raw=False) - self.fsum_value) / self.fsum_value
            )
            < 0.005
        )
