from jaxrts.saha import solve_saha
from jaxrts.elements import Element
from jaxrts.units import ureg, to_array

import numpy as np

def rho_to_ne(rho, m_a, Z=1):
    return (Z * rho / (m_a)).to(1 / (1 * ureg.cc))

if __name__ == "__main__":

    H = Element("H")
    He = Element("He")

    sol = solve_saha((H, He), 1.549e+9  / 11604.51812, to_array([rho_to_ne((1662424176 / (4.2955e18)) * ureg.gram / ureg.cc, H.atomic_mass, 1).m_as(1 / ureg.cc) * (1/0.5076e5)**3 * (4.2955e18), rho_to_ne((719087680)/(4.2955e18) * ureg.gram / ureg.cc, He.atomic_mass, 1).m_as(1 / ureg.cc) * (1/0.5076e5)**3 * (4.2955e18)]))

    ionization_state_charge = np.array([0, 1, 0, 1, 2])

    fractions = sol / np.array([np.sum(sol[:2]), np.sum(sol[:2]), np.sum(sol[2:]), np.sum(sol[2:]), np.sum(sol[2:])])

    Z_free = fractions * ionization_state_charge
    print("Z_Free_avg: H:", np.sum(Z_free[:2]), "He:", np.sum(Z_free[2:]))