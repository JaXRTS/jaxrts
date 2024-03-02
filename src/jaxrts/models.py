import abc

import jax.numpy as jnp

from .setup import Setup
from .plasmastate import PlasmaState


class Model(metaclass=abc.ABCMeta):
    def __init__(self, state: PlasmaState):
        self.plasma_state = state

    @abc.abstractmethod
    def evaluate(self, setup: Setup) -> jnp.ndarray: ...


# Here list of Models ...
