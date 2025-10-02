SVT–OZ and notation
===================

Consider :math:`M` species (electron included, namely: M-1 ions plus electron) with number densities :math:`\{n_a\}`, masses
:math:`\{m_a\}`, and temperatures :math:`\{T_a\}` (:math:`a=1,\dots,M`).
In :math:`k`-space, the multi-component SVT–OZ relation can be obtained by extending the work of :cite:`Shaffer.2017` through summation over all species, yielding
.. math::

   \hat h_{ab}
   =\hat c_{ab}
   +\sum_{s=1}^M n_s\frac{m_{ab}\,T_{as}}{m_a\,T_{ab}}\;\hat c_{as}\hat h_{sb}
   +\sum_{s=1}^M n_s\frac{m_{ab}\,T_{sb}}{m_b\,T_{ab}}\;\hat h_{as}\hat c_{sb},
   \qquad a,b\in\{1,\dots,M\}.
   \label{eq:svt-oz}

The mass-weighted cross temperatures and reduced masses are

.. math::

   \begin{aligned}
   m_{ab}&=\frac{m_a m_b}{m_a+m_b}, &
   T_{ab}&=\frac{m_a T_b + m_b T_a}{m_a+m_b}, \label{eq:def-Tab-mab}\\
   T_{as}&=\frac{m_a T_s + m_s T_a}{m_a+m_s}, &
   T_{sb}&=\frac{m_s T_b + m_b T_s}{m_s+m_b}.
   \end{aligned}

For brevity define

.. math::

   \alpha_{abs}=n_s\,\frac{m_{ab}\,T_{as}}{m_a\,T_{ab}},
   \qquad
   \beta_{abs}=n_s\,\frac{m_{ab}\,T_{sb}}{m_b\,T_{ab}}.
   \label{eq:def-alpha-beta}

Moving all terms to the left, except :math:`\hat c_{ab}`.

.. math::

   \hat h_{ab}
   -\sum_{s=1}^M \alpha_{abs}\,\hat c_{as}\,\hat h_{sb}
   -\sum_{s=1}^M \beta_{abs}\,\hat h_{as}\,\hat c_{sb}
   =\hat c_{ab}.
   \label{eq:svt-oz-LHS}

In real space we use the HNC closure with the *cross* temperature
:math:`T_{ab}`,

.. math::

   \ln g_{ab}(r)=-\beta_{ab}V_{ab}(r)+N_{ab}(r),\qquad
   \beta_{ab}=\frac{1}{k_B T_{ab}},\quad
   h_{ab}=g_{ab}-1,\quad N_{ab}=h_{ab}-c_{ab}.
   \label{eq:hnc-closure}

For isotropic fields we use the 3D radial transforms

.. math::

   \begin{aligned}
   \hat f(k)&=\frac{4\pi}{k}\int_0^\infty r\,f(r)\sin(kr)\,dr,
   &
   f(r)&=\frac{1}{2\pi^2 r}\int_0^\infty k\,\hat f(k)\sin(kr)\,dk.
   \label{eq:radialFT}
   \end{aligned}

For two-component plasma :math:`M{=}2` system
---------------------------------------------

For :math:`M{=}2`, the four unknowns are

.. math::

   \mathrm{vec}(\hat H)=
   \begin{bmatrix}\hat h_{11}&\hat h_{12}&\hat h_{21}&\hat h_{22}\end{bmatrix}^{\!\top}\!,
   \qquad
   \mathrm{vec}(\hat C)=
   \begin{bmatrix}\hat c_{11}&\hat c_{12}&\hat c_{21}&\hat c_{22}\end{bmatrix}^{\!\top}.

Writing `[eq:svt-oz-LHS] <#eq:svt-oz-LHS>`__ for
:math:`(a,b)=(1,1),(1,2),(2,1),(2,2)` gives

.. math::

   \begin{aligned}
   \hat{h}_{11} - 
   \alpha_{111}\, \hat{c}_{11}\, \hat{h}_{11}  - 
   \alpha_{112}\, \hat{c}_{12}\, \hat{h}_{21} - 
   \beta_{111}\, \hat{h}_{11}\, \hat{c}_{11} - 
   \beta_{112}\, \hat{h}_{12}\, \hat{c}_{21} &= 
   \hat{c}_{11},\\
   \hat{h}_{12} - 
   \alpha_{121}\, \hat{c}_{11}\, \hat{h}_{12}  - 
   \alpha_{122}\, \hat{c}_{12}\, \hat{h}_{22} - 
   \beta_{121}\, \hat{h}_{11}\, \hat{c}_{12} - 
   \beta_{122}\, \hat{h}_{12}\, \hat{c}_{22} &= 
   \hat{c}_{12},\\
   \hat{h}_{21} - 
   \alpha_{211}\, \hat{c}_{21}\, \hat{h}_{11}  - 
   \alpha_{212}\, \hat{c}_{22}\, \hat{h}_{21} - 
   \beta_{211}\, \hat{h}_{21}\, \hat{c}_{11} - 
   \beta_{212}\, \hat{h}_{22}\, \hat{c}_{21} &= 
   \hat{c}_{21},\\
   \hat{h}_{22} - 
   \alpha_{221}\, \hat{c}_{21}\, \hat{h}_{12}  - 
   \alpha_{222}\, \hat{c}_{22}\, \hat{h}_{22} - 
   \beta_{221}\, \hat{h}_{21}\, \hat{c}_{12} - 
   \beta_{222}\, \hat{h}_{22}\, \hat{c}_{22} &= 
   \hat{c}_{22}.
   \end{aligned}

Collecting like terms yields the :math:`4\times4` linear system

.. math::

   \underbrace{
   \begin{pmatrix}
   1 - (\alpha_{111} + \beta_{111})\hat{c}_{11} & -\beta_{112}\hat{c}_{21} & -\alpha_{112}\hat{c}_{12} & 0 \\
   -\beta_{121}\hat{c}_{12} & 1 - \alpha_{121}\hat{c}_{11} - \beta_{122}\hat{c}_{22} & 0 & -\alpha_{122}\hat{c}_{12} \\
   -\alpha_{211}\hat{c}_{21} & 0 & 1 - \alpha_{212}\hat{c}_{22} - \beta_{211}\hat{c}_{11} & -\beta_{212}\hat{c}_{21} \\
   0 & -\alpha_{221}\hat{c}_{21} & -\beta_{221}\hat{c}_{12} & 1 - (\alpha_{222} + \beta_{222})\hat{c}_{22}
   \end{pmatrix}
   }_{\displaystyle A}
   \;
   \underbrace{
   \begin{pmatrix}
   \hat{h}_{11} \\
   \hat{h}_{12} \\
   \hat{h}_{21} \\
   \hat{h}_{22}
   \end{pmatrix}
   }_{\displaystyle \mathrm{vec}(\hat H)}
   =
   \underbrace{
   \begin{pmatrix}
   \hat{c}_{11} \\
   \hat{c}_{12} \\
   \hat{c}_{21} \\
   \hat{c}_{22}
   \end{pmatrix}
   }_{\displaystyle \mathrm{vec}(\hat C)}.
   \label{eq:M2-matrix}

This explicit :math:`M{=}2` pattern shows that each row couples only to
one column-block :math:`\{\hat h_{sb}\}_s` and one row-block
:math:`\{\hat h_{as}\}_s`.

From the :math:`M{=}2` pattern to the general :math:`M`: a selector-:math:`\delta` derivation
---------------------------------------------------------------------------------------------

We now derive the general matrix entries :math:`A[p,q]`

Flattening.
^^^^^^^^^^^

Fix a wave number :math:`k` and vectorize by lexicographic order. Using
0-based indexing,

.. math::

   p=\mathrm{idx}(a,b)=(a-1)M+(b-1),\qquad
   q=\mathrm{idx}(u,v)=(u-1)M+(v-1).
   \label{eq:idx}

Turn the single :math:`s`-sum into a matrix–vector product.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a fixed row :math:`(a,b)` (i.e. fixed :math:`p`), rewrite each term
of `[eq:svt-oz-LHS] <#eq:svt-oz-LHS>`__ as a sum over *all* column
indices :math:`(u,v)` using Kronecker deltas that *select* which columns
are hit:

.. math::

   \begin{aligned}
   \hat{h}_{ab} &= \sum_{u,v} \delta_{u,a} \delta_{v,b} \, \hat{h}_{uv}, \\
   \sum_{s} \alpha_{abs} \, \hat{c}_{as} \, \hat{h}_{sb} &= \sum_{u,v} \left( \sum_{s} \alpha_{abs} \, \hat{c}_{as} \, \delta_{u,s} \, \delta_{v,b} \right) \hat{h}_{uv} = \sum_{u,v} \left( \alpha_{ab,u} \, \hat{c}_{au} \, \delta_{v,b} \right) \hat{h}_{uv}, \\
   \sum_{s} \beta_{abs} \, \hat{h}_{as} \, \hat{c}_{sb} &= \sum_{u,v} \left( \sum_{s} \beta_{abs} \, \delta_{u,a} \, \delta_{v,s} \, \hat{c}_{sb} \right) \hat{h}_{uv} = \sum_{u,v} \left( \beta_{ab,v} \, \hat{c}_{vb} \, \delta_{u,a} \right) \hat{h}_{uv}.
   \end{aligned}

Therefore,

.. math::

   \sum_{u,v}\Big[\,
   \delta_{u,a}\delta_{v,b}
   -\alpha_{ab\,u}\,\hat c_{a u}\,\delta_{v,b}
   -\beta_{ab\,v}\,\hat c_{v b}\,\delta_{u,a}\,\Big]\hat h_{uv}
   =\hat c_{ab}.

Comparing with :math:`\sum_q A[p,q]\,H[q]=C[p]` and
:math:`H[q]=\hat h_{uv}`, :math:`C[p]=\hat c_{ab}`, we *define* the
entry formula

.. math::

   \boxed{%
   A\big[p,q]
   =A\big[(a{-}1)M+(b{-}1),\ (u{-}1)M+(v{-}1)\big]
   =\delta_{u,a}\delta_{v,b}
   -\alpha_{ab\,u}\,\hat c_{a u}\,\delta_{v,b}
   -\beta_{ab\,v}\,\hat c_{v b}\,\delta_{u,a}.}
   \label{eq:A-elem}
