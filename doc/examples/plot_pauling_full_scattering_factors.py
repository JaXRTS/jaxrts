"""
Full atomic Formfactors according to Pauling and Sherman
========================================================

This figure reproduces Fig.8 in :cite:`Pauling.1932` for elements up to
:math:`Z=36`.
"""

import numpy as onp
import matplotlib.pyplot as plt
import scienceplots

import jaxrts
from jpu import numpy as jnpu

ureg = jaxrts.ureg

plt.style.use("science")


k = onp.linspace(0, 1.5, 400) * (4 * onp.pi) / ureg.angstrom

for Z in range(1, 37):
    element = jaxrts.Element(Z)
    Zstar = jaxrts.form_factors.pauling_effective_charge(Z)
    F = jnpu.sum(
        jaxrts.form_factors.pauling_all_ff(k, Zstar)
        * element.electron_distribution[:, onp.newaxis],
        axis=0,
    )
    plt.plot(k.m_as(1 / ureg.angstrom), F, color="C0")

plt.xlabel("$k$ [1/$\\AA$]")
plt.ylabel("$F$")
plt.tight_layout()
plt.show()
