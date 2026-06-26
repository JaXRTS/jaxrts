"""
A Python package for X-ray Thomson Scattering from dense plasmas, using jax.
"""

# These version placeholders will be replaced by poetry-dynamic-versioning
__version__ = "0.0.0"
__version_tuple__ = (0, 0, 0)

from . import (
    analysis,
    bound_free,
    collections,
    elements,
    form_factors,
    free_bound,
    free_free,
    helpers,
    hnc_potentials,
    instrument_function,
    ion_feature,
    math,
    models,
    plasma_physics,
    plasmastate,
    ionization,
    saving,
    setup,
    static_structure_factors,
    units,
)
from .elements import Element
from .plasmastate import PlasmaState
from .setup import Setup
from .units import ureg

from .collections import get_all_models, get_all_models_list

__all__ = [
    "Element",
    "PlasmaState",
    "Setup",
    "analysis",
    "bound_free",
    "collections",
    "elements",
    "form_factors",
    "free_bound",
    "free_free",
    "get_all_models",
    "get_all_models_list",
    "helpers",
    "hnc_potentials",
    "hypernetted_chain",
    "instrument_function",
    "ion_feature",
    "ionization",
    "math",
    "models",
    "plasma_physics",
    "plasmastate",
    "saving",
    "setup",
    "static_structure_factors",
    "units",
    "ureg",
]


# Register all models

import jax

_all_models = get_all_models_list()

for _model in _all_models:
    jax.tree_util.register_pytree_node(
        _model,
        _model._tree_flatten,
        _model._tree_unflatten,
    )
