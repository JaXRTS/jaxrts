from .units import ureg, Quantity
from typing import List
import numpy as np

class PlasmaState:
    
    def __init__(self,
                 ions : List,
                 Z_A : List | Quantity,
                 Z_free : List | Quantity,
                 atomic_masses : List | Quantity,
                 density_fractions : List | float,
                 mass_density : List | Quantity,
                 T_e : List | Quantity,
                 T_i : List | Quantity):

        assert (len(ions) == len(Z_A)) and (len(ions) == len(Z_free)) and (len(ions) == len(atomic_masses)) and (len(ions) == len(density_fractions)) and (len(ions) == len(mass_density)) and (len(ions) == len(T_e)) and (len(ions) == len(T_i)), "WARNING: Input parameters should be the same shape as <ions>!"
    
        self.ions = ions   
        
        # Define charge configuration 
        self.Z_A = Z_A
        self.Z_free = Z_free
        self.Z_core = Z_A - Z_free
        
        self.atomic_masses = atomic_masses
        self.density_fractions = density_fractions
        self.mass_density = mass_density
        
        self.T_e = T_e
        self.T_i = T_i if T_i else T_e
        
    def _jSii(k : Quantity, E : np.ndarray | List | Quantity):
        pass
    
    def Sii(k : Quantity, E : np.ndarray | List | Quantity):
        pass
    
    def probe(energy : Quantity, theta : float):
        
        lambda_0 = (energy.to(ureg.joule) / (1 * ureg.planck_constant)).to_base_units()
        
        