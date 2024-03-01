import jax.numpy as jnp

import jpu

from .units import ureg, Quantity

class Setup:
    
    def __init__(self, scattering_angle : Quantity, energy : Quantity, instrument : jnp.ndarray):
        
        self.scattering_angle = scattering_angle
        self.energy = energy
        self.lambda0 = ureg.planck_constant * ureg.c / energy
        self.instrument = instrument
    
    @property
    def k(self):
        return (4 * jnp.pi / self.lambda0) * jnp.sin(jnp.deg2rad(self.scattering_angle) / 2.0)
    
    
