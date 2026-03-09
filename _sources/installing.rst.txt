Installation
============

.. note::

   We highly recommend installing jaxrts in a (virtual) environment, separate
   from a system-wide python installation.

The easiest way to include this project into your workflow is to
``pip install`` the newest ``.whl`` file, that can be found on the
`release
page <https://github.com/jaxrts/jaxrts/releases>`__.


To install the current development version clone the git repository under
https://github.com/jaxrts/jaxrts:

.. code:: bash

   git clone https://github.com/jaxrts/jaxrts.git
   cd jaxrts

Create a new python environment and activate it. Then, install the project

.. code:: bash

   python -m pip install .

Or, just run

.. code:: bash

   python -m pip git+https://github.com/jaxrts/jaxrts

Installing additional features
------------------------------

Some optional features of jaxrts require specific, additional packages to be
installed on top of the the standard requirements. Currently these are the
tools to:

* use the Neural networkinterpolation for static structure factors (`NN`),
* and packages required to use some (experimental) graphical interfaces (`gui`)

To install them, just add the required keys in square brackets, e.g.

.. code:: bash

   python -m pip install ".[NN]"

