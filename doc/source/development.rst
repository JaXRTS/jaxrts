Development and Poetry
======================

The code can easily be installed using
`poetry <https://python-poetry.org/>`__ by calling

.. code:: bash

   poetry install

in the root of this repository.

This will create an virual environment for the project and install the
required dependencies. To execute commands within the virtual
environment, prefix them with ``poetry run``.

**However, you can completely ignore poetry, if you need to. Just make
sure you have the required packages installed**

A new release can be built using

.. code:: bash

   poetry build

Creating a new version
----------------------

To bump a new version

1. Change ``__version__`` in the ``src/logbook_4463/__init__.py`` file
2. Change the ``version`` in ``pyproject.toml``
3. Commit your changes
4. Create a new git tag via ``git tag 0.3.0``
5. Push the tag ``git push --tags``
