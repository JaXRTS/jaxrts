"""Bridge closures, the HNC limit and independent Yukawa benchmarks."""

import io
from pathlib import Path

import jax
import numpy as onp
import pytest
from jax import numpy as jnp

import jaxrts
import jaxrts.hnc_bridges as bridges
import jaxrts.hypernetted_chain as hnc
from jaxrts.units import ureg


def inputs(gamma=10.0, pot=11, rmax=40):
    state = make_state(1, rmax=rmax, pot=pot)
    model, potential = state["ionic scattering"], state["ion-ion Potential"]
    return (
        gamma * potential.short_r(state, model.r),
        gamma * potential.long_k(state, model.k),
        model.r,
        potential.T(state),
        state.n_i,
    )


def solve(gamma=10.0, **kwargs):
    return bridges.solve_vmhnc(
        *inputs(gamma), points_per_diameter=64, **kwargs
    )


def test_transform_normalization_and_roundtrip():
    # DST-I is used only for the PY reference, not for the HNC iteration.
    r = jnp.arange(1, 1024) * 20 / 1024
    k = jnp.arange(1, 1024) * jnp.pi / 20
    f = jnp.exp(-(r**2))
    transformed = bridges._forward(f, r, k)
    expected = jnp.pi**1.5 * jnp.exp(-(k**2) / 4)
    assert jnp.max(jnp.abs(transformed - expected)) < 2e-14
    assert jnp.allclose(
        bridges._inverse(transformed, r, k), f, rtol=1e-7, atol=2e-14
    )


@pytest.mark.parametrize("eta", [0.1, 0.3, 0.4, 0.48])
def test_hard_sphere_py(eta):
    x, bridge, g = bridges._hard_sphere_py(eta, 64)
    assert onp.all(g[x < 1] == 0)
    assert float(g[63]) == pytest.approx((1 + eta / 2) / (1 - eta) ** 2)
    assert onp.max(bridge) <= 1e-14
    assert abs(bridge[-1]) < 1e-12


def test_quantity_conversions_and_jit():
    vs, vl, r, temperature, density = inputs()
    converted = jax.jit(
        lambda v, w, x, t, n: bridges.solve_vmhnc(
            v, w, x, t, n, points_per_diameter=64
        )
    )(
        vs.to(ureg.joule),
        vl.to(ureg.joule * ureg.meter**3),
        r.to(ureg.meter),
        temperature.to(ureg.kelvin),
        density.to(1 / ureg.centimeter**3),
    )
    original = solve()
    assert converted.converged
    assert jnp.max(jnp.abs(converted.g_r - original.g_r)) < 1e-8
    assert converted.sigma.m_as(ureg.a_0) == pytest.approx(
        original.sigma.m_as(ureg.a_0), rel=1e-10
    )


def test_vmap_over_coupling():
    result = jax.jit(jax.vmap(lambda gamma: solve(gamma)))(
        jnp.array([1.0, 10.0])
    )
    assert onp.all(result.converged)
    assert jnp.max(jnp.abs(result.g_r[1] - solve().g_r)) < 1e-8


@pytest.mark.parametrize(
    "kwargs,status",
    [
        ({"eta_bounds": (0.35, 0.40)}, 3),
        ({"eta_max_iter": 1, "eta_tol": 1e-10}, 4),
        ({"mix": 1.0}, 1),
        ({"eta_tol": -1}, 1),
    ],
)
def test_failed_solves_are_explicit(kwargs, status):
    result = solve(**kwargs)
    assert not result.converged and result.status == status
    assert onp.all(onp.isnan(result.g_r)) and onp.all(onp.isnan(result.s_k))


def test_inner_failure_is_explicit():
    # An unbounded attraction produces overflow in the native HNC closure.
    result = solve(-1.0)
    assert result.status == 2 and not result.converged
    assert onp.all(onp.isnan(result.g_r)) and onp.all(onp.isnan(result.s_k))


@pytest.mark.parametrize("change", ["temperature", "density", "grid"])
def test_invalid_physical_inputs(change):
    args = list(inputs())
    if change == "temperature":
        args[3] *= -1
    elif change == "density":
        args[4] *= 0
    else:
        args[2] = args[2].magnitude.at[10].add(0.001) * ureg.a_0
    result = bridges.solve_vmhnc(*args, points_per_diameter=64)
    assert result.status == 1 and onp.all(onp.isnan(result.s_k))


def test_static_validation():
    with pytest.raises(ValueError, match="eta_bounds"):
        solve(eta_bounds=(0.3, 0.2))
    args = list(inputs())
    args[0] = jnp.broadcast_to(args[0].magnitude, (2, 2, 2048)) * ureg.eV
    with pytest.raises(ValueError, match="one-component"):
        bridges.solve_vmhnc(*args)
    with pytest.raises(ValueError, match="points_per_diameter"):
        bridges._hard_sphere_py(0.3, 16)
    args = list(inputs(pot=10))
    args[:3] = [arg[..., :-1] for arg in args[:3]]
    with pytest.raises(ValueError, match="even grid"):
        bridges.solve_vmhnc(*args)


def make_state(gamma=10, rmax=20, pot=10):
    ion = jaxrts.Element("H")
    a = 2 * ureg.a_0
    temperature = ureg.elementary_charge**2 / (
        4 * jnp.pi * ureg.epsilon_0 * a * gamma * ureg.k_B
    )
    state = jaxrts.PlasmaState(
        ions=[ion],
        Z_free=[1],
        mass_density=[3 / (4 * jnp.pi * a**3) * ion.atomic_mass],
        T_e=temperature,
    )
    state["screening length"] = jaxrts.models.ConstantScreeningLength(a)
    state["ionic scattering"] = jaxrts.models.OnePotentialVMHNCIonFeat(
        rmin=0.0001 * a,
        rmax=rmax * a,
        pot=pot,
        points_per_diameter=64,
    )
    return state


def test_model_and_serialization():
    state = make_state()
    model = state["ionic scattering"]
    result = model.pair_distribution_function(state)
    assert result.converged
    assert model.check_supports_S_ii_grid(state)
    k_grid, s_grid = model.S_ii_on_grid(state)
    assert jnp.allclose(k_grid.magnitude, model.k.magnitude, rtol=1e-7, atol=0)
    assert jnp.allclose(s_grid.magnitude, result.s_k, rtol=1e-7, atol=0)
    setup = jaxrts.Setup(
        90 * ureg.degree,
        5 * ureg.keV,
        jnp.linspace(4.9, 5.1, 20) * ureg.keV,
        jaxrts.instrument_function.Gaussian(sigma=1e14 / ureg.second),
    )
    sii = model.S_ii(state, setup)
    assert sii.shape == (1, 1)
    expected = jnp.interp(
        setup.k.m_as(1 / ureg.a_0),
        model.k.m_as(1 / ureg.a_0),
        result.s_k[0, 0],
    )
    assert jnp.allclose(
        sii.m_as(ureg.dimensionless), expected, rtol=1e-7, atol=0
    )
    assert onp.all(
        onp.isfinite(state.evaluate("ionic scattering", setup).magnitude)
    )
    buffer = io.StringIO()
    jaxrts.saving.dump(model, buffer)
    buffer.seek(0)
    restored = jaxrts.saving.load(buffer, ureg)
    assert jnp.allclose(
        restored.S_ii(state, setup).magnitude,
        sii.magnitude,
        rtol=1e-7,
        atol=0,
    )
    assert "Faussurier" in restored.citation("bibtex")
    # The model must match the original solver with the effective potential.
    pot = state["ion-ion Potential"]
    expected, _ = hnc.pair_distribution_function_HNC(
        pot.short_r(state, model.r)
        - ureg.k_B * pot.T(state) * result.bridge_r,
        pot.long_k(state, model.k),
        model.r,
        pot.T(state),
        state.n_i,
        model.mix,
        model.tmult,
    )
    assert jnp.max(jnp.abs(result.g_r - expected.magnitude)) < 1e-10


def test_model_rejects_mixtures():
    state = jaxrts.PlasmaState(
        ions=[jaxrts.Element("H"), jaxrts.Element("C")],
        Z_free=[1, 2],
        mass_density=jnp.array([1.0, 2.0]) * ureg.gram / ureg.cm**3,
        T_e=10 * ureg.eV / ureg.k_B,
    )
    with pytest.raises(ValueError, match="one ion species"):
        state["ionic scattering"] = jaxrts.models.OnePotentialVMHNCIonFeat()


@pytest.mark.parametrize("kwargs", [{"pot": 0}, {"rmax": -1 * ureg.a_0}])
def test_model_grid_validation(kwargs):
    with pytest.raises(ValueError):
        jaxrts.models.OnePotentialVMHNCIonFeat(**kwargs)


@pytest.mark.parametrize("species", [1, 2])
def test_pair_bridges_reuse_native_hnc(species):
    # The shared adapter accepts arbitrary pair bridges, also for mixtures;
    # only VMHNC's hard-sphere reference/variational search is one-component.
    r = jnp.linspace(0.001, 12, 256) * ureg.a_0
    pair = jnp.array([[0.2, 0.3], [0.3, 0.4]])[:species, :species]
    v = pair[..., None] * jnp.exp(-r.magnitude**2) * ureg.eV
    vl = jnp.zeros_like(v.magnitude) * ureg.eV * ureg.a_0**3
    n = jnp.array([0.03, 0.07])[:species] * (1 / ureg.a_0**3)
    temperature = 1 * ureg.eV / ureg.k_B
    bridge = -0.02 * jnp.exp(-r.magnitude**2) * jnp.ones_like(v.magnitude)
    for b in (jnp.zeros_like(bridge), bridge):
        g, iterations = bridges.pair_distribution_function_with_bridge(
            v, vl, r, temperature, n, b
        )
        expected, count = hnc.pair_distribution_function_HNC(
            v - ureg.k_B * temperature * b, vl, r, temperature, n
        )
        assert jnp.array_equal(g.magnitude, expected.magnitude)
        assert iterations == count < 2000
    with pytest.raises(ValueError, match="same shape"):
        bridges.pair_distribution_function_with_bridge(
            v, vl, r, temperature, n, bridge[0, 0]
        )


def test_weak_coupling_tends_to_native_hnc():
    differences, bridge_sizes = [], []
    for gamma in (0.01, 0.1, 1.0):
        result = solve(gamma)
        g, _ = hnc.pair_distribution_function_HNC(
            *inputs(gamma), mix=0.5, tmult=[10.0, 3.0]
        )
        assert result.converged
        differences.append(float(jnp.max(jnp.abs(result.g_r - g.magnitude))))
        bridge_sizes.append(float(jnp.max(jnp.abs(result.bridge_r))))
    assert bridge_sizes[0] == 0 < bridge_sizes[1] < bridge_sizes[2]
    assert differences[0] < differences[1] < differences[2]
    assert differences[0] < 1e-12 and differences[1] < 1e-5
    ideal = solve(0.0)
    assert ideal.converged and ideal.eta == 0
    assert jnp.max(jnp.abs(ideal.g_r - 1)) < 1e-12
    assert jnp.max(jnp.abs(ideal.s_k - 1)) < 1e-12


def test_hnc_limit_depends_on_bridge_accuracy():
    loose = solve(1.0, bridge_tol=0.1)
    fine = solve(1.0)
    assert loose.converged and fine.converged
    assert loose.eta == 0 < fine.eta
    bounded = solve(0.01, eta_bounds=(0.05, 0.1))
    assert bounded.status == 3 and not bounded.converged


@pytest.mark.parametrize("gamma", [1.0, 5.0, 10.0, 20.0, 50.0, 100.0])
def test_yukawa_against_md(gamma):
    args = inputs(gamma, pot=15, rmax=320)
    g, iterations = hnc.pair_distribution_function_HNC(
        *args, mix=0.5, tmult=[10.0, 3.0]
    )
    result = bridges.solve_vmhnc(*args, points_per_diameter=256)
    assert result.converged and iterations < 2000
    data = onp.genfromtxt(
        Path(__file__).parent / "data/vmhnc/yukawa_lammps.csv",
        delimiter=",",
        names=True,
    )
    md = data[
        (data["gamma"] == gamma)
        & (data["r_over_a"] >= 0.5)
        & (data["r_over_a"] <= 3)
    ]
    r = (args[2] / (2 * ureg.a_0)).m_as(ureg.dimensionless)
    gh = onp.interp(md["r_over_a"], r, g[0, 0].magnitude)
    gv = onp.interp(md["r_over_a"], r, result.g_r[0, 0])
    hnc_error = onp.sqrt(onp.mean((gh - md["g_r"]) ** 2))
    vmhnc_error = onp.sqrt(onp.mean((gv - md["g_r"]) ** 2))
    assert vmhnc_error < 0.035
    if gamma <= 5:
        assert vmhnc_error < 0.01
    if gamma >= 50:
        # First-shell structure, including the peak and following minimum.
        assert vmhnc_error < 0.5 * hnc_error
    k = make_state(1, rmax=320, pot=15)["ionic scattering"].k
    sh = hnc.S_ii_HNC(k, g, args[4], args[2]).magnitude[0, 0]
    k = (k * (2 * ureg.a_0)).m_as(ureg.dimensionless)
    data = onp.genfromtxt(
        Path(__file__).parent / "data/vmhnc/yukawa_lammps_sk.csv",
        delimiter=",",
        names=True,
    )
    md = data[data["gamma"] == gamma]
    sh = onp.interp(md["k_times_a"], k, sh)
    sv = onp.interp(md["k_times_a"], k, result.s_k[0, 0])
    hnc_error = onp.sqrt(onp.mean((sh - md["s_k"]) ** 2))
    vmhnc_error = onp.sqrt(onp.mean((sv - md["s_k"]) ** 2))
    assert vmhnc_error < 0.08
    if gamma >= 50:
        assert vmhnc_error < hnc_error
