"""
Convenience functions collecting all available
:py:class:`jaxrts.models.Models`, etc.

This is a submodule of its own to avoid circular imports.
"""

from collections import defaultdict
from typing import overload, Dict, Literal
import inspect
from . import hnc_potentials, models


@overload
def get_all_models(
    types: list[Literal["models", "hnc_potentials"]],
    names_only: Literal[False] = False,
) -> Dict[str, list[models.Model | hnc_potentials.HNCPotential]]: ...
@overload
def get_all_models(
    types: list[Literal["models", "hnc_potentials"]],
    names_only: Literal[True],
) -> Dict[str, list[str]]: ...
def get_all_models(
    types: list[Literal["models", "hnc_potentials"]] = [
        "models",
        "hnc_potentials",
    ],
    names_only: bool = False,
) -> Dict[str, list[models.Model | hnc_potentials.HNCPotential | str]]:
    """
    Get a dictionary of all :py:class:`jaxrts.models.Model` defined within
    jaxrts that are valid to attach to a
    :py:class:`jaxrts.plasma_state.PlasmaState`.
    The `allowed_keys` are the key of the returned dictionary.

    Parameters
    ----------
    types: List[Literal["models", "hnc_potentials"]]
        default ["models", "hnc_potentials"]. Which types of models should be
        included.

    names_only: bool, default False
        If ``true``, the values of the returned dict are a list of
        :py:class:`jaxrts.models.Model` object. If false, the values are rather
        list of strings with of the name of the models.

    Returns
    -------
    Dictionary of keys and lists of either models or model-names that are
    implemented in jaxrts.
    """
    all_models = defaultdict(list)
    considered_modules = []
    if "models" in types:
        considered_modules.append(models)
    if "hnc_potentials" in types:
        considered_modules.append(hnc_potentials)

    for module in considered_modules:
        for obj_name in dir(module):
            if "__class__" in dir(obj_name):
                obj = getattr(module, obj_name)
                # The model must be concrete (cannot be abstract), has to
                # advertise allowed keys, and cannot have a private name that
                # starts with ``_``.
                if (
                    not inspect.isabstract(obj)
                    and ("allowed_keys" in dir(obj))
                    and not obj_name.startswith("_")
                ):
                    keys = obj.allowed_keys
                    for k in keys:
                        if names_only:
                            all_models[k].append(obj_name)
                        else:
                            all_models[k].append(obj)
    return all_models
