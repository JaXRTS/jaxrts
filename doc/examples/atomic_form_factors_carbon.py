"""
Atomic form factors of C, by Pauling and Sherman
================================================

Calculate the atomic formfactors of carbon using a hydrogen-like atom, as
presented in :cite:`Pauling.1932`.

.. note::
    Instead of calling the functions :py:func:`jaxrts.form_factors.pauling_f10`
    and so on, directly, you could also use
    :py:func:`jaxrts.form_factors.pauling_atomic_ff`, instead.
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

import jaxrts
import jaxrts.form_factors as ff

ureg = jaxrts.units.ureg

k = np.linspace(0, 10, 100) / ureg.angstrom

CZ1s = 6 - 0.19
CZ2s = 6 - 2.04
CZ2p = 6 - 2.91

fC10 = ff.pauling_f10(k, CZ1s)
fC20 = ff.pauling_f20(k, CZ2s)
fC21 = ff.pauling_f21(k, CZ2s)

plt.style.use("science")
plt.plot(k, fC10 + fC20 + fC21, label="Full form factor")

plt.plot(
    k,
    fC10,
    label=r"$1s$ electrons",
    color="grey",
    linestyle=((0, (1, 1))),
)
plt.plot(
    k,
    fC20,
    label=r"$2s$ electrons",
    color="grey",
    linestyle=((0, (5, 1))),
)
plt.plot(
    k,
    fC21,
    label=r"$2p$ electrons",
    color="grey",
    linestyle=(0, (3, 1, 1, 1, 1, 1)),
)

plt.title(
    "Atomic form factors for Carbon\n(Pauling and Sherman)"
)

plt.xlabel(r"$q$ [1/$\AA$]")
plt.ylabel(r"$f_i$")

plt.legend()
plt.tight_layout()
plt.show()
