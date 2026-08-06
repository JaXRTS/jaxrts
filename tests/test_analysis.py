"""
These tests investigate that certain properties of a generated spectrum hold
true.
"""

import pytest
from jax import numpy as jnp
from jpu import numpy as jnpu

import jaxrts

ureg = jaxrts.ureg


class ITCFInstance:
    def __init__(self):
        self.test_state = jaxrts.PlasmaState(
            ions=[jaxrts.Element("Be")],
            Z_free=jnp.array([2]),
            mass_density=jnp.array([3]) * ureg.gram / ureg.centimeter**3,
            T_e=60 * ureg.electron_volt / ureg.k_B,
        )
        self.test_setup = jaxrts.Setup(
            scattering_angle=ureg("30°"),
            energy=ureg("6900 eV"),
            measured_energy=ureg("6900 eV")
            + jnp.linspace(-170, 170, 1000) * ureg.electron_volt,
            instrument=jaxrts.instrument_function.TwinGaussian(
                fwhm1=ureg("2eV"), fwhm2=ureg("4eV")
            ),
        )

        self.test_setup.correct_k_dispersion = False
        self.test_state["ionic scattering"] = (
            jaxrts.models.OnePotentialHNCIonFeat()
        )
        self.test_state["free-free scattering"] = (
            jaxrts.models.RPA_DandreaFit()
        )
        self.test_state["bound-free scattering"] = (
            jaxrts.models.SchumacherImpulse()
        )
        self.test_state["free-bound scattering"] = (
            jaxrts.models.DetailedBalance()
        )

    def get_T(self, raw=False, x=ureg("150eV")):
        S_ee = self.test_state.probe(self.test_setup)
        T = jaxrts.analysis.ITCFT(
            S_ee,
            ureg("60/keV"),
            self.test_setup,
            x,
            raw=raw,
        )
        return T


@pytest.fixture
def itcf():
    return ITCFInstance()


def test_ITCFT_without_instument_function(itcf):
    """
    This only works with a very narrow and syummetric instrument function. This
    is expected.
    """
    itcf.test_setup.instrument = jaxrts.instrument_function.Gaussian(
        ureg("1eV")
    )
    assert jnpu.absolute(itcf.get_T(raw=True) - itcf.test_state.T_e) < ureg(
        "2e3K"
    )


def test_ITCFT_with_instument_function(itcf):
    assert jnpu.absolute(itcf.get_T(raw=False) - itcf.test_state.T_e) < ureg(
        "2e3K"
    )


def test_ITCFT_stable_against_moving_energy_grid(itcf):
    # We require small values for x. For big x, convergence should always
    # reach the correct value
    _x = jnp.array([10, 15, 20]) * ureg.electron_volt
    T1 = (
        jnp.array([itcf.get_T(x=x).m_as(ureg.kelvin) for x in _x])
        * ureg.kelvin
    )
    itcf.test_setup.measured_energy += 0.1 * ureg.electron_volt
    T2 = (
        jnp.array([itcf.get_T(x=x).m_as(ureg.kelvin) for x in _x])
        * ureg.kelvin
    )
    assert jnpu.max(jnpu.absolute(T1 - T2)) < ureg("1e3K")


@pytest.mark.xfail(reason="The k-dispersion should violate detailed balance")
def test_k_dispersion_conflicts_with_detailed_balance(itcf):
    itcf.test_setup.correct_k_dispersion = True
    assert jnpu.absolute(itcf.get_T(raw=True) - itcf.test_state.T_e) < ureg(
        "2e3K"
    )


class SSFInstance:
    def __init__(self):
        self.test_state = jaxrts.PlasmaState(
            ions=[jaxrts.Element("C")],
            Z_free=jnp.array([3.65]),
            mass_density=jnp.array([0.3]) * ureg.gram / ureg.centimeter**3,
            T_e=100 * ureg.electron_volt / ureg.k_B,
        )
        self.test_setup = jaxrts.Setup(
            scattering_angle=ureg("60°"),
            energy=ureg("70 keV"),
            measured_energy=ureg("70 keV")
            + jnp.linspace(-15, 15, 6000) * ureg.kiloelectron_volt,
            instrument=jaxrts.instrument_function.TwinGaussian(
                fwhm1=ureg("40eV"), fwhm2=ureg("60eV")
            ),
        )
        self.test_setup.correct_k_dispersion = False

        self.test_state["ion-ion Potential"] = (
            jaxrts.hnc_potentials.DebyeHueckelPotential()
        )
        self.test_state["ionic scattering"] = (
            jaxrts.models.OnePotentialHNCIonFeat()
        )
        self.test_state["free-free scattering"] = (
            jaxrts.models.RPA_DandreaFit()
        )
        self.test_state["bound-free scattering"] = (
            jaxrts.models.SchumacherImpulseFitRk()
        )
        self.test_state["free-bound scattering"] = (
            jaxrts.models.DetailedBalance()
        )


@pytest.fixture
def ssf():
    return SSFInstance()


def test_sff_to_unity_fully_ionized(ssf):
    ssf.test_state.Z_free = ssf.test_state.Z_A
    S_ee = ssf.test_state.probe(ssf.test_setup)
    ssf_raw = jaxrts.analysis.ITCF_ssf(
        S_ee, ssf.test_setup, ureg("14.8keV"), raw=True
    )
    ssf = jaxrts.analysis.ITCF_ssf(
        S_ee, ssf.test_setup, ureg("14.8keV"), raw=False
    )
    assert jnpu.absolute(ssf_raw - 1) < 0.02
    assert jnpu.absolute(ssf - 1) < 0.02


def test_sff_to_unity_with_bound_free_contrib(ssf):
    """
    The SchumacherRkFit is using the fsum rule. Check that if that is
    fulfilled, we reach the correct SFF limit.
    """
    ssf.test_state.Z_free = (
        ssf.test_state.Z_A - jnp.ones(len(ssf.test_state.ions)) * 2
    )
    S_ee = ssf.test_state.probe(ssf.test_setup)
    ssf_raw = jaxrts.analysis.ITCF_ssf(
        S_ee, ssf.test_setup, ureg("14.8keV"), raw=True
    )
    ssf = jaxrts.analysis.ITCF_ssf(
        S_ee, ssf.test_setup, ureg("14.8keV"), raw=False
    )
    assert jnpu.absolute(ssf_raw - 1) < 0.02
    assert jnpu.absolute(ssf - 1) < 0.02


class FsumRuleInstance:
    def __init__(self):
        self.test_state = jaxrts.PlasmaState(
            ions=[jaxrts.Element("O")],
            Z_free=jnp.array([4]),
            mass_density=jnp.array([5]) * ureg.gram / ureg.centimeter**3,
            T_e=60 * ureg.electron_volt / ureg.k_B,
        )
        self.test_setup = jaxrts.Setup(
            scattering_angle=ureg("40°"),
            energy=ureg("7.5 keV"),
            measured_energy=ureg("7.5 keV")
            + jnp.linspace(-2, 2, 5000) * ureg.kiloelectron_volt,
            instrument=jaxrts.instrument_function.TwinGaussian(
                fwhm1=ureg("30eV"), fwhm2=ureg("50eV")
            ),
        )
        self.test_setup.correct_k_dispersion = False

        self.test_state["ionic scattering"] = (
            jaxrts.models.OnePotentialHNCIonFeat()
        )
        self.test_state["free-free scattering"] = (
            jaxrts.models.RPA_DandreaFit()
        )
        self.test_state["bound-free scattering"] = (
            jaxrts.models.SchumacherImpulse()
        )
        self.test_state["free-bound scattering"] = (
            jaxrts.models.DetailedBalance()
        )

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


@pytest.fixture()
def fsum():
    return FsumRuleInstance()


def test_fsum_rule_fully_ionized(fsum):
    # Set to fully ionized
    fsum.test_state.Z_free = fsum.test_state.Z_A
    assert (
        jnpu.absolute(
            (fsum.fsum(raw=False) - fsum.fsum_value) / fsum.fsum_value
        )
        < 0.005
    )


@pytest.mark.xfail(
    reason="The current implementation of bound-free scattering breaks the f-sum rule the fitting r_k is not provided"  # noqa: E501
)
def test_bound_free_breaks_fsum_rule(fsum):
    # Allow some bound-free contribution
    fsum.test_state.Z_free = fsum.test_state.Z_A - jnp.ones(
        len(fsum.test_state.ions)
    )
    assert (
        jnpu.absolute(
            (fsum.fsum(raw=False) - fsum.fsum_value) / fsum.fsum_value
        )
        < 0.005
    )
