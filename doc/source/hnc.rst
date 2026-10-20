Hypernetted Chain
=================

The calculation of the ion-ion static structure using hypernetted chain (HNC)
calculations can provide good agreement with more costly techniques if adequate
potentials are chosen :cite:`Fletcher.2015,Wunsch.2009`.

It allows for obtaining pair distribution functions :math:`g_{ab} = 1 + h_{ab}`
by iteratively solving the Ornstein Zernike equation of a classical liquid,
splitting it into a direct correlation function :math:`c_{\text{ab}}` and an
indirect term.

.. math::

   h_{\text{ab}}(r) = c_{\text{ab}}(r) + \sum_\text{j} n_\text{j} \int
   d\mathbf{r}' \, c_{\text{aj}}\left(|\mathbf{r} - \mathbf{r}'|\right)
   h_{\text{jb}}(|\mathbf{r}'|),

and closing it with the HNC approximation, where :math:`V_\text{ab}` are the
potentials between particles :math:`a` and :math:`b`.

.. math::

    g_{\text{ab}}(r) = \exp\left[-\beta V_{\text{ab}}(r) + h_{\text{ab}}(r) -
    c_{\text{ab}}(r)\right],

Structure factors can the be calculated via

.. math::

   S_{\text{ab}}(\mathbf{k}) = \delta_{\text{ab}} + \sqrt{n_a
   n_b}\int_V\mathrm{d}\mathbf{r}e^{-i\mathbf{k}\mathbf{r}}\left[g_{\text{ab}}(\mathbf{r})
   - 1\right].

Our implementation is based on the work of Kathrin Wünsch :cite:`Wunsch.2011`.
See especially the flowchart in Figure 4.1 therein, and also
:cite:`Schumacher.2025` and :cite:`Shaffer.2017`.

Within the HNC modules, all quantities have three axis with :math:`(n\times n
\times m)` entries, where :math:`n` is the number of ion species considered and
:math:`m` is the number of :math:`r` or :math:`k` points considered.

.. image:: images/ThreePotentialHNC.svg
   :width: 600

The HNC approach is capable of incorporating electrons, natively, by adding it
as an additional ion species. This is normally achieved by setting
:py:attr:`jaxrts.hnc_potentials.HNCPotential.include_electrons` to
``"SpinAverged"``.
This adds one additional entry to the first two dimensions (see figure above).

If a user requires to separate two kinds of electrons for their different
spins, set :py:attr:`jaxrts.hnc_potentials.HNCPotential.include_electrons` to
``"SpinSeparated"``. This will introduce two additional entries, instead, with
half of the electron density for each of them.

Bridge closures
---------------

:py:func:`jaxrts.hnc_bridges.pair_distribution_function_with_bridge` accepts a
dimensionless pair bridge :math:`B_{ab}(r)` with the same axes as the potential.
It solves HNC with
:math:`V_{s,ab}^{\mathrm{eff}}=V_{s,ab}-k_B T_{ab}B_{ab}` and returns
``(g, iterations)``. A zero bridge gives HNC. Pair bridges for mixtures use
the same interface.

For one ion species, :py:class:`jaxrts.models.OnePotentialVMHNCIonFeat` uses
a variational Percus--Yevick hard-sphere bridge,

.. math::

   g(r)=\exp[-\beta V(r)+h(r)-c(r)+B_{\mathrm{PY}}(r/\sigma;\eta)],
   \qquad \eta=\pi n_i\sigma^3/6.

The bridge-universality approximation follows :cite:`Rosenfeld.1979` and the
hard-sphere reference :cite:`Wertheim.1963`. The closure and variational
condition are given in :cite:`LuccoCastello.2021`, Eqs. (7)--(10); their
application to Yukawa fluids is described in :cite:`Faussurier.2004`.
The implementation is based on Otter :cite:`Otter.2026` and supports a
single ion species.

For example::

   state["ionic scattering"] = jaxrts.models.OnePotentialVMHNCIonFeat(
       rmin=0.001 * ureg.a_0, rmax=100 * ureg.a_0, pot=14,
   )
   result = state["ionic scattering"].pair_distribution_function(state)
   if not bool(result.converged):
       raise RuntimeError(f"VMHNC failed with status {int(result.status)}")

Calculations require JAX double precision.
:py:func:`jaxrts.hnc_bridges.solve_vmhnc` accepts split potentials on a uniform
radial grid and returns correlations, the packing fraction and convergence
information. ``eta_tol`` bounds the packing-fraction bracket width;
``bridge_tol`` sets the bridge amplitude below which HNC is recovered.
Failed correlations are NaN; check ``converged`` before use. JIT and vmap are
supported; reverse-mode differentiation through convergence loops is not.

The :doc:`example <gen_examples/ionic/plot_VMHNC_pair_correlation_functions>`
compares HNC and VMHNC with MD data for 2048 particles at Yukawa :math:`\kappa=1`,
:math:`\Gamma=1,5,10,20,50,100`.
``tests/data/vmhnc/yukawa_lammps.csv`` and ``yukawa_lammps_sk.csv`` contain
the means and standard errors from 20 NVT blocks for :math:`g_{ii}(r)` and
:math:`S_{ii}(k)`, respectively. MD structure factors are sampled directly
from nonzero reciprocal density modes. Tests and plots read the CSV data.

At weak coupling the bridge correction vanishes. At strong coupling the
benchmarks measure errors in the first coordination shell, where HNC
underestimates the correlation peak. These tests establish accuracy only for
the sampled states and observable. Increase the HNC box as well as radial
resolution when checking convergence of the variational packing fraction.
