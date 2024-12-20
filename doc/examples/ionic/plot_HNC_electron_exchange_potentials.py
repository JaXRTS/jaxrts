"""
Comparison of classical map to derived analytical formulas
==========================================================
"""

import jaxrts
import jax
import jax.numpy as jnp


import matplotlib.pyplot as plt

ureg = jaxrts.ureg

state = jaxrts.PlasmaState(
    ions=[jaxrts.Element("H")],
    Z_free=jnp.array([1.0]),
    mass_density=[ureg(f"{40}g/cc")],
    T_e=ureg(f"{600e4}K"),
)


r = jnp.linspace(1e-3, 0.1, 3000) * ureg.angstrom

# Spin separated exchange
V_CM = jaxrts.hnc_potentials.PauliClassicalMap()
V_D = jaxrts.hnc_potentials.SpinSeparatedEEExchange()

for V in [V_CM, V_D]:
    V.include_electrons = "SpinSeparated"

    plt.plot(
        r.m_as(ureg.angstrom),
        V.full_r(state, r)[1, 1, :].m_as(ureg.hartree),
        label="Deutsch" if V == V_D else "CM (SpinSeparated)",
    )

V_CM = jaxrts.hnc_potentials.PauliClassicalMap()
V_D = jaxrts.hnc_potentials.SpinAveragedEEExchange()

for V in [V_CM, V_D]:
    V.include_electrons = "SpinAveraged"

    plt.plot(
        r.m_as(ureg.angstrom),
        V.full_r(state, r)[1, 1, :].m_as(ureg.hartree),
        label="Huang" if V == V_D else "CM (SpinAveraged)",
        ls="dashed",
    )


plt.plot(
    r.m_as(ureg.angstrom),
    jnp.zeros_like(r.m_as(ureg.angstrom)),
    color="black",
    ls="dashed",
)
plt.legend()
plt.show()
