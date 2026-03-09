Development and Poetry
======================

For development, please clone the code first

.. code:: bash

   git clone https://github.com/jaxrts/jaxrts.git
   cd jaxrts


Afterwards, all dependencies can easily be installed using
`poetry <https://python-poetry.org/>`__ by calling

.. code:: bash

   poetry install

in the root of this repository.

This will create an virual environment for the project and install the
required dependencies. To execute commands within the virtual
environment, prefix them with ``poetry run``.

However, you can completely ignore poetry, if you need to. In that case, it
is recommended to install the package in edit mode ( ``pip install -e .`` ), in
a virtual environment.

To install optional dependencies, use the ``--extras`` flat.
I.e., use ``poetry install --extras=NN`` for installing the ``NN``
dependencies.


Building a new release version
------------------------------

A new release can be built using

.. code:: bash

   poetry build
