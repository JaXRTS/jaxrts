# JaXRTS

**Python code for x-ray Thomson scattering, made fast by using [jax](https://jax.readthedocs.io/en/latest/index.html).**

## Documentation

The [Documentation can be found online](http://aghed.hed.physik.uni-rostock.de/lue/jaxrts/). However, as we don't generate it automatically, yet, it might be outdated. If you want to generate it for yourself, just run

```bash
poetry make html
```

in the `doc` directory, after installing the module.

## Installing

We recommend to install the module using [poetry](https://python-poetry.org/) by calling

```bash
poetry install
```

in the root of this repository.

This will create an virtual environment for the project and install the required dependencies. To execute commands within the virtual environment, prefix them with ``poetry run``.

However, you should also be able to just pip install the module, after cloning it:

```bash
pip install -e .
```

This `-e` flag installs the module in 'edit' mode, i.e., changes you made are available without the need of reinstalling the package.
