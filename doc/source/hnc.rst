Hypernetted Chain
=================

The calculation of the ion-ion static structure using hypernetted chain (HNC)
calculations can provide good agreement with more costly techniques if adequate
potentials are chosen :cite:`Fletcher.2015,Wunsch.2009`.

If allows for obtaining pair distribution functions :math:`g_{ab} = 1 + h_{ab}`
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
See especially the flowchart in Figure 4.1, therein, and also
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
