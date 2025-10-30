jaxrts
======

A Python package for X-ray Thomson scattering, using jax.

.. image:: images/jaxrts_logo.svg
   :width: 600

X-ray Thomson scattering (XRTS) is a diagnostics widely used in the warm dense
matter and dense plasma community.
The theory of XRTS has been discussed in many publications, e.g.
:cite:`Glenzer.2009, Gregori.2003, Gregori.2004, Wunsch.2011, Chapman.2015`,

Since the method of XRTS embarked, many models have been formulated to
calculate spectra. This code aims on collecting several of these efforts and
thereby assist experiment analysis and planning with easy forward calculations.
Splitting the full signal in different contributions, according to the Chihara
decomposition :cite:`Chihara.2000, Gregori.2003`, we allow users to combine
different implemented models, applicable to different plasma conditions and
involving varying computational cost and find a description according to their
needs.

This documentation aims on providing an overview about the capabilities of
jaxrts, and should help you taking your first steps. A discussion focussing on
more details about the physics implemented is currently in the works.
A detailed description of jaxrts module is available under
:doc:`module_overview`. If you prefer to learn by examples, we kindly refer you
to the :doc:`example gallery<gen_examples/index>`.

We would not have been able to write jaxrts without the seminal work by D.
Chapman, G. Gregori, K. Wünsch and D. O. Gericke, and many other authors. A
full bibliography can be found in the :doc:`bibliography`.

The code is currently maintained by Samuel Schumacher and Julian Lütgert in the
High Energy Density Group of Dominik Kraus at the University of Rostock.
`See here <https://github.com/JaXRTS/jaxrts/graphs/contributors>`__ for a list
of contributors to the software.

.. warning::

   This code is still under active development, so are it's dependencies. While
   we successfully reproduce data found in the literature and applied several
   models to experimental data, we are still implementing new features and
   correct bugs. Please kindly reach out to us if you note faulty behavior, and
   understand that -- while we attempt to maintain a stable user experience --
   some results might change when errors are fixed.
   See https://github.com/jaxrts/jaxrts for the most up-to-date version of
   jaxrts.

   We highly recommend installing the code in a virtual environemnt of it's
   own, to not interfere with other python packages in your system.

.. note::

   Currently, the code is only tested linux operating systems. While the
   project itself should run on other platforms, too, we do not actively test
   jaxrts against them. Please also note that currently, GPU acceleration is
   only available on linux,
   `as we rely on jax <https://docs.jax.dev/en/latest/installation.html#supported-platforms>`__ .
   CPU computation should be supported on Mac, Windows, and linux.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   installing.rst
   usage.rst
   gen_examples/index.rst
   development.rst
   bibliography
   module_overview


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
