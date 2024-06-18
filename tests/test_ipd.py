import pytest

import sys

sys.path.append(
    "C:/Users/Samuel/Desktop/PhD/Python_Projects/JAXRTS/jaxrts/src"
)

from jax import numpy as jnp
import jaxrts
import copy

import os

ureg = jaxrts.ureg

def test_IPD():
    
    test_setup = jaxrts.setup.Setup(
    ureg("145°"),
    ureg("5keV"),
    jnp.linspace(4.5, 5.5) * ureg.kiloelectron_volts,
    lambda x: jaxrts.instrument_function.instrument_gaussian(
        x, 1 / ureg.second
    ),
    )
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
 
    nes = []
    
    ipds = {"dh" : [], "sp" : [], "is" : [], "pb" : []}
    keys = ["dh", "sp", "is", "pb"]

    md = jnp.linspace(0.01, 600, 500)
    for m in md:
        
        test_state = jaxrts.PlasmaState(
        ions=[jaxrts.Element("C")],
        Z_free=jnp.array([5]),
        mass_density=jnp.array([m]) * ureg.gram / ureg.centimeter**3,
        T_e=jnp.array([100]) * ureg.electron_volt / ureg.k_B,
        )

        nes.append(test_state.n_e.m_as(1 / ureg.cc))
        
        for i, IPDModel in enumerate([jaxrts.models.DebyeHueckelIPD(),
                        jaxrts.models.StewartPyattIPD(),
                        jaxrts.models.IonSphereIPD(),
                        jaxrts.models.PauliBlockingIPD()]):
        
            IPDModel.model_key = "ipd"
            shift = IPDModel.evaluate(plasma_state = test_state, setup = test_setup).m_as(ureg.electron_volt)
            
            # print(test_state.n_e.m_as(1 / ureg.cc), shift)
            ipds[keys[i]].append(-shift[0])
        
    
    ax.plot(nes, ipds["dh"], label = "Debye Hueckel Model")
    ax.plot(nes, ipds["sp"], label = "Stewart Pyatt Model")
    ax.plot(nes, ipds["is"], label = "Ion Sphere Model")
    ax.plot(nes, ipds["pb"], label = "Pauli Blocking Model")
    
    plt.xscale("log")
    # plt.yscale("symlog")
    plt.xlabel("$n_i$ [cm$^{-3}$]")
    plt.ylabel("IPD [eV]")
    plt.legend()
    plt.xlim(1E23, 1E26)
    # plt.ylim(0.5, 15)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    test_IPD()