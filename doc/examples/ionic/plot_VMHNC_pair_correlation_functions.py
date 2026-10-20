r"""
VMHNC: pair correlations and structure factors
==============================================

HNC and VMHNC use the closure

.. math::

    g(r) = \exp[-\beta V(r) + h(r) - c(r) + B(r)],

where :math:`\beta=1/(k_B T)`, :math:`h=g-1`, and :math:`c` is the direct
correlation function. HNC sets :math:`B(r)=0`; other closures use different
approximations for :math:`B(r)`.

Single-component VMHNC (variational modified HNC) uses a Percus--Yevick
hard-sphere bridge with a variationally determined packing fraction
:cite:`Faussurier.2004,LuccoCastello.2021`.

This example compares HNC, VMHNC and molecular dynamics (MD) results for
:math:`g_{ii}(r)` and :math:`S_{ii}(k)` of :math:`\mathrm{C}^{4+}` at
:math:`T_e=T_i=T`, using the Yukawa potential

.. math::

    V(r) = A\frac{\exp(-\kappa r/a)}{r/a}, \qquad
    A = \frac{(Ze)^2}{4\pi\epsilon_0 a}, \qquad
    \Gamma = \frac{A}{k_B T}.

The ion Wigner--Seitz radius is :math:`a=(3/(4\pi n_i))^{1/3}` and
:math:`\kappa=a/\lambda_{\mathrm{scr}}=1`.
MD circles show NVT means for 2048 particles; shading indicates two block
standard errors. MD structure factors are sampled from nonzero reciprocal
density modes.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from matplotlib.lines import Line2D

import jaxrts
from jaxrts import hypernetted_chain as hnc
from jaxrts import ureg

jax.config.update("jax_enable_x64", True)
plt.style.use("science")
try:
    current_folder = Path(__file__).parent
except NameError:
    current_folder = Path.cwd()
data_dir = current_folder / "../../../tests/data/vmhnc"
data = np.genfromtxt(data_dir / "yukawa_lammps.csv", delimiter=",", names=True)
sk_data = np.genfromtxt(
    data_dir / "yukawa_lammps_sk.csv", delimiter=",", names=True
)
a = 2 * ureg.a_0
ion = jaxrts.Element("C")
Z = 4
fig, ax = plt.subplots(1, 2, figsize=(10.5, 5))
for index, gamma in enumerate((1, 5, 10, 20, 50, 100)):
    temperature = (Z * ureg.elementary_charge) ** 2 / (
        4 * jnp.pi * ureg.epsilon_0 * a * gamma * ureg.k_B
    )
    state = jaxrts.PlasmaState(
        ions=[ion],
        Z_free=[Z],
        mass_density=[3 * ion.atomic_mass / (4 * jnp.pi * a**3)],
        T_e=temperature,
    )
    state["screening length"] = jaxrts.models.ConstantScreeningLength(a)
    state["ionic scattering"] = model = jaxrts.models.OnePotentialVMHNCIonFeat(
        rmin=0.0001 * a,
        rmax=320 * a,
        pot=15,
    )
    vmhnc = model.pair_distribution_function(state)
    potential = state["ion-ion Potential"]
    gh, iterations = hnc.pair_distribution_function_HNC(
        potential.short_r(state, model.r),
        potential.long_k(state, model.k),
        model.r,
        potential.T(state),
        state.n_i,
        model.mix,
        model.tmult,
    )
    if not bool(vmhnc.converged) or int(iterations) >= 2000:
        raise RuntimeError(f"Closure did not converge at Gamma={gamma}")
    sh = hnc.S_ii_HNC(model.k, gh, state.n_i, model.r)
    r = model.r.m_as(ureg.angstrom)
    k = model.k.m_as(1 / ureg.angstrom)
    md = data[data["gamma"] == gamma]
    sk_md = sk_data[sk_data["gamma"] == gamma]
    color = f"C{index}"
    for axis, x, y, error, stride in (
        (
            ax[0],
            md["r_over_a"] * a.m_as(ureg.angstrom),
            md["g_r"],
            md["g_sem"],
            3,
        ),
        (
            ax[1],
            sk_md["k_times_a"] / a.m_as(ureg.angstrom),
            sk_md["s_k"],
            sk_md["s_sem"],
            1,
        ),
    ):
        axis.fill_between(
            x, y - 2 * error, y + 2 * error, color=color, alpha=0.15
        )
        axis.plot(
            x,
            y,
            "o",
            color=color,
            ms=3,
            mfc="none",
            markevery=stride,
            alpha=0.9,
            zorder=4,
        )
    for axis, x, gv, gh_values in (
        (ax[0], r, vmhnc.g_r[0, 0], gh[0, 0].magnitude),
        (ax[1], k, vmhnc.s_k[0, 0], sh[0, 0].magnitude),
    ):
        axis.plot(
            x,
            gv,
            color=color,
            lw=1.5,
            alpha=0.7,
            label=rf"$T={(temperature * ureg.k_B).m_as(ureg.eV):.2f}"
            rf"\,\mathrm{{eV}},\ \Gamma={gamma:g}$",
        )
        axis.plot(
            x,
            gh_values,
            "--",
            color=color,
            lw=1.2,
            alpha=0.85,
            zorder=3,
        )
ax[0].set(
    xlabel=r"$r\ (\mathrm{\AA})$",
    ylabel=r"$g_{ii}(r)$",
    xlim=(0, 4.5 * a.m_as(ureg.angstrom)),
    ylim=(-0.04, 2.05),
)
ax[1].set(
    xlabel=r"$k\ (\mathrm{\AA}^{-1})$",
    ylabel=r"$S_{ii}(k)$",
    xlim=(0, 8),
    ylim=(-0.04, None),
)
fig.legend(
    handles=[
        Line2D([], [], color="k", ls="--", lw=1.2, alpha=0.85),
        Line2D([], [], color="k", lw=1.5, alpha=0.7),
        Line2D(
            [],
            [],
            color="k",
            ls="none",
            marker="o",
            mfc="none",
            ms=3,
            alpha=0.9,
        ),
    ],
    labels=["HNC", "VMHNC", "MD"],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.9),
    ncol=3,
    fontsize=9,
)
fig.legend(
    *ax[0].get_legend_handles_labels(),
    loc="upper center",
    bbox_to_anchor=(0.5, 0.84),
    ncol=3,
    fontsize=8,
)
fig.suptitle(
    r"Yukawa C$^{4+}$: $T_e=T_i=T$, $\kappa=1$, $N_{\mathrm{MD}}=2048$"
    "\n"
    r"$n_i=2.014\times10^{23}\,\mathrm{cm}^{-3}$, "
    rf"$\rho={state.mass_density[0].m_as(ureg.g / ureg.cm**3):.3f}"
    rf"\,\mathrm{{g\,cm^{{-3}}}}$, "
    rf"$\lambda_{{\mathrm{{scr}}}}={a.m_as(ureg.angstrom):.3f}"
    rf"\,\mathrm{{\AA}}$",
    fontsize=10,
)
for axis in ax:
    axis.grid(False, which="both")
fig.tight_layout(rect=(0, 0, 1, 0.86))
plt.show()
