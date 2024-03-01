import jax.numpy as jnp
import jax

from .setup import *

class Model:
    
    def __init__(self, model_func):
        self.model_func = model_func
    
    @jax.jit
    def evaluate(self, setup : Setup) -> jnp.ndarray:
        return self.model_func(setup)
    
    
# Here list of Models ...

    
    
     