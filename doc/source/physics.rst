Physics
=======

XRTS is describing the scattering of an X-ray photon on electrons. In the WDM
community, term summarizes Raman, Rayleigh and Thomson scattering processes
:cite:`Glenzer.2009`. 

In a typical setup, a light-source provides photons carying momentum
:math:`\hbar\vec{k_0}` and energy :math:`\hbar\omega_0`.
This light scatteres off a target, often a WDM or plasma state, defined by a
set of plasma paramters, such as density, temperature, and ionization.
A detector, located at an angle :math:`\theta` collects scattered photons, with
energy :math:`\hbar\omega_s`, and momentum :math:`\hbar\vec{k}_s`. The
transferred quantities shall be denoted with :math:`\vec{k}` (the **scattering vector**) and
:math:`\omega` (**photon frequency shift**).

Since the plasma is isotropic, we simplify that only the magnitude of the scattering vector :math:`\vec{k}` was relevant. 
This can be expressed as

.. math::

   k = \sqrt{k_s^2 + k_0^2 - 2 k_s k_0 \cos \theta}
       \approx 2 \frac{\omega_0}{c} \sin\left(\frac{\theta}{2}\right),

where the approximation holds when :math:`k` is small compared to :math:`k_0`. A sketch of the geometry of
a typical XRTS experiment can be seen in the figure.

.. image:: images/XRTS_geometry.svg
   :width: 600

An actual XRTS signal arises from several distinct
mechanisms that are interpreted according to the chemical picture introduced by Chihara :cite:`Chihara.2000`.  
In this model, and in jaxrts, the total electron–electron dynamic structure
factor is decomposed into a sum of contributions that correspond to
different physical origins—namely, elastic (**el**), free–free (**ff**), bound–free (**bf**), and 
free–bound (**fb**) interactions between electrons and ions.

.. math::

   S_{ee}^{\text{tot}}(k, \omega) =
       S_{ee}^{\text{el}}(k, \omega)
       + S_{ee}^{\text{ff}}(k, \omega)
       + S_{ee}^{\text{bf}}(k, \omega)
       + S_{ee}^{\text{fb}}(k, \omega)

Each of these four contributions have to be defined, in order to generate a spectrum.
See :doc:`first-spectrum` on how this is done.
A comprehensive list of available models can be found under :doc:`models`.

:math:`S_{ee}^{\text{tot}}` is related to the intensity :math:`I` measured in
an experiment via:

.. math::
   I(k, \omega) \propto \left(\frac{\omega + \omega_0}{\omega_0}\right)^\nu S_{\text{ee}}^\text{tot}(k, \omega) \circledast R\left(\omega\right)\quad,
   \label{eqn:signal}

Here, :math:`R` is the combined source instrument function
(:py:attr:`jaxrts.setup.Setup.instrument`), :math:`\circledast` is the
convolution, and the exponent :math:`\mu` accounts for the frequency
redistrubution correction (:py:attr:`jaxrts.setup.Setup.frc_exponent`, see
:cite:`Crowley.2013`.

