"""
This submodule contains high-level wrappers for the different Models
implemented.
"""

import abc

import jax.numpy as jnp

from .setup import Setup
from .plasmastate import PlasmaState
from .ion_feature import q

# This defines a Model, abstractly.
class Model(metaclass=abc.ABCMeta):
    def __init__(self, state: PlasmaState):
        """
        As not different prerequisites exist for different models, make sure to
        test that all relevant information is given in the PlasmaState, amend
        defaults if necessary in the ``__init__``, rather than the ``evaluate``
        method. Please log assumtpions, properly
        """
        self.plasma_state = state

    @abc.abstractmethod
    def evaluate(self, setup: Setup) -> jnp.ndarray: ...

# Here list of Models ...
# =======================

# Form Factor Models

class PaulingFormFactors(Model):
    def evaluate(self, setup: Setup) -> jnp.ndarray:
        pass

# The ion-feature
# -----------------

class ArphipovIonFeat(Model):
    def evaluate(self, setup: Setup) -> jnp.ndarray:
        pass
