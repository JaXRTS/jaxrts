"""
This module allows for saving and loading :py:class:`jaxrts.models.Model` and
:py:class:`jaxrts.plasmastate.PlasmaState`
"""

import dill as pickle
import base64
import json
from .plasmastate import PlasmaState
from .models import Model
import jaxrts
from .elements import Element
from .units import Quantity
from .hnc_potentials import HNCPotential
from jaxlib.xla_extension import ArrayImpl
import jax
import jpu.numpy as jnpu
import jax.numpy as jnp
import numpy as onp

import functools


def partialclass(cls, *args, **kwds):
    """
    See https://stackoverflow.com/a/38911383
    """

    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls


def _flatten_obj(obj):
    children, aux = obj._tree_flatten()
    if hasattr(obj, "_children_labels"):
        children = {l: c for (l, c) in zip(obj._children_labels, children)}
    if hasattr(obj, "_aux_labels"):
        aux = {l: a for (l, a) in zip(obj._aux_labels, aux)}
    return (children, aux)


def _parse_tree_save(obj, children, aux):
    """
    We do not unflatten, here, so allow in-place changes.
    """
    if hasattr(obj, "_children_labels"):
        children = tuple([children[key] for key in obj._children_labels])
    if hasattr(obj, "_aux_labels"):
        aux = tuple([aux[key] for key in obj._aux_labels])
    return (children, aux)


class JaXRTSEncoder(json.JSONEncoder):
    """
    Decode class, taking care of all classes that might be decoded.

    See https://gist.github.com/simonw/7000493
    """

    def default(self, obj):
        if isinstance(obj, PlasmaState):
            return {
                "_type": "PlasmaState",
                "value": _flatten_obj(obj),
            }
        if isinstance(obj, HNCPotential):
            out = _flatten_obj(obj)

            # Get _transform_r. If it exists, it is the first entry in children
            if hasattr(obj, "_transform_r"):
                if isinstance(out[0], dict):
                    _transform_r = out[0]["_transform_r"]
                    out[0]["_transform_r"] = (
                        {
                            "start": jnpu.min(_transform_r),
                            "stop": jnpu.max(_transform_r),
                            "num": len(_transform_r),
                        },
                    )
                else:
                    _transform_r = out[0][0]
                    out = (
                        (
                            {
                                "start": jnpu.min(_transform_r),
                                "stop": jnpu.max(_transform_r),
                                "num": len(_transform_r),
                            },
                            *out[0][1:],
                        ),
                        out[1],
                    )
            return {"_type": "HNCPotential", "value": (obj.__name__, out)}
        elif isinstance(obj, Model):
            return {
                "_type": "Model",
                "value": (obj.__name__, _flatten_obj(obj)),
            }
        elif isinstance(obj, Element):
            return {
                "_type": "Element",
                "value": obj.symbol,
            }
        elif isinstance(obj, Quantity):
            return {"_type": "Quantity", "value": obj.to_tuple()}
        elif isinstance(obj, ArrayImpl):
            try:
                return {"_type": "Array", "value": list(onp.array(obj))}
            except TypeError:
                return float(onp.array(obj))
        elif isinstance(obj, onp.ndarray):
            return {
                "_type": "ndArray",
                "value": list(obj),
            }
        elif isinstance(obj, onp.int32):
            return int(obj)
        elif isinstance(obj, onp.int64):
            return int(obj)
        elif isinstance(obj, jax.tree_util.Partial):
            return {
                "_type": "jaxPartial",
                "value": base64.b64encode(pickle.dumps(obj)).decode("utf-8"),
            }
        return super().default(obj)


class JaXRTSDecoder(json.JSONDecoder):
    """
    .. warning ::

       In the current implementation, you cannot resotre custom models. To
       restore them, you have to provide these in a custom mapping:

    """

    def __init__(self, ureg, additional_mappings={}, *args, **kwargs):
        self.ureg = ureg
        self.additional_mappings = additional_mappings
        json.JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs
        )

    @property
    def hnc_potentials(self) -> dict:
        pot_dict = {
            key: value
            for (key, value) in jaxrts.hnc_potentials.__dict__.items()
            if (value in jaxrts.hnc_potentials._all_hnc_potentals)
            and not key.startswith("_")
        }
        pot_dict.update(self.additional_mappings)
        return pot_dict

    @property
    def models(self) -> dict:
        model_dict = {
            key: value
            for (key, value) in jaxrts.models.__dict__.items()
            if (value in jaxrts.models._all_models) and not key.startswith("_")
        }
        model_dict.update(self.additional_mappings)
        return model_dict

    def object_hook(self, obj):
        if "_type" not in obj:
            return obj
        _type = obj["_type"]
        val = obj["value"]
        if _type == "jaxPartial":
            return pickle.loads(base64.b64decode(val))
        if _type == "ndArray":
            return onp.array(val)
        elif _type == "Quantity":
            return self.ureg.Quantity.from_tuple(val)
        elif _type == "Array":
            return jnp.array(val)
        elif _type == "Element":
            return Element(val)
        elif _type == "Model":
            name, tree = val

            model = self.models[name]
            new = object.__new__(model)

            children, aux_data = _parse_tree_save(new, *tree)
            new = new._tree_unflatten(aux_data, children)
            return new
        elif _type == "HNCPotential":
            name, tree = val

            pot = self.hnc_potentials[name]
            new = object.__new__(pot)
            children, aux_data = _parse_tree_save(new, *tree)

            new = new._tree_unflatten(aux_data, children)

            # Fix the transform_r
            # This uses that _transform_r will always be the first entry of the
            # children tuple.
            if hasattr(new, "_transform_r"):
                new._transform_r = jnpu.linspace(**children[0])
            return new
        elif _type == "PlasmaState":
            new = object.__new__(PlasmaState)
            children, aux_data = _parse_tree_save(new, *val)
            new = new._tree_unflatten(aux_data, children)
            return new
        return obj


def dump(obj, fp, *args, **kwargs):
    kwargs.update({"cls": JaXRTSEncoder})
    json.dump(obj, fp, *args, **kwargs)


def dumps(obj, *args, **kwargs):
    kwargs.update({"cls": JaXRTSEncoder})
    return json.dumps(obj, *args, **kwargs)


def load(fp, unit_reg, additional_mappings={}, *args, **kwargs):
    dec = partialclass(
        JaXRTSDecoder, ureg=unit_reg, additional_mappings=additional_mappings
    )
    kwargs.update({"cls": dec})
    return json.load(fp, *args, **kwargs)
