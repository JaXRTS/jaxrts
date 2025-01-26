from .elements import Element, _absorption_edges
from typing import List
from functools import partial
from .units import Quantity, ureg
import numpy as onp
import jax

import jax.numpy as jnp
import jpu.numpy as jnpu

# fmt: off
_stat_weight = {
    "H": [2, 1], "He": [1, 2, 1], "Li": [2, 1, 2, 1], "Be": [1, 2, 1, 2, 1], "B": [2, 1, 2, 1, 2, 1],
    "C": [1, 2, 1, 2, 1, 2, 1], "N": [4, 1, 2, 1, 2, 1, 2, 1], "O": [5, 4, 1, 2, 1, 2, 1, 2, 1],
    "F": [4, 5, 4, 1, 2, 1, 2, 1, 2, 1], "Ne": [1, 4, 5, 4, 1, 2, 1, 2, 1, 2, 1],
    "Na": [2, 1, 4, 5, 4, 1, 2, 1, 2, 1, 2, 1], "Mg": [1, 2, 1, 4, 5, 4, 1, 2, 1, 2, 1, 2, 1],
    "Al": [2, 1, 2, 1, 4, 5, 4, 1, 2, 1, 2, 1, 2, 1], "Si": [1, 2, 1, 2, 1, 4, 5, 4, 1, 2, 1, 2, 1, 2, 1],
    "P": [4, 1, 2, 1, 2, 1, 4, 5, 4, 1, 2, 1, 2, 1, 2, 1], "S": [5, 4, 1, 2, 1, 2, 1, 4, 5, 4, 1, 2, 1, 2, 1, 2, 1],
    "Cl": [4, 5, 4, 1, 2, 1, 2, 1, 4, 5, 4, 1, 2, 1, 2, 1, 2, 1], "Ar": [1, 4, 5, 4, 1, 2, 1, 2, 1, 4, 5, 4, 1, 2, 1, 2, 1, 2, 1],
    "K": [4, 1, 4, 5, 4, 1, 2, 1, 2, 1, 4, 5, 4, 1, 2, 1, 2, 1, 2, 1], "Ca": [1, 4, 1, 4, 5, 4, 1, 2, 1, 2, 1, 4, 5, 4, 1, 2, 1, 2, 1, 2, 1],
    "Sc": [4, 5, 4, 1, 4, 5, 4, 1, 2, 1, 2, 1, 4, 5, 4, 1, 2, 1, 2, 1, 2, 1], "Ti": [5, 4, 5, 4, 1, 4, 5, 4, 1, 2, 1, 2, 1, 4, 5, 4, 1, 2, 1, 2, 1, 2, 1],
    "V": [4, 1, 4, 5, 4, 1, 4, 5, 4, 1, 2, 1, 2, 1, 4, 5, 4, 1, 2, 1, 2, 1, 2, 1], "Cr": [6, 1, 4, 5, 4, 1, 4, 5, 4, 1, 2, 1, 2, 1, 4, 5, 4, 1, 2, 1, 2, 1, 2, 1],
    "Mn": [9, 6, 1, 4, 5, 4, 1, 4, 5, 4, 1, 2, 1, 2, 1, 4, 5, 4, 1, 2, 1, 2, 1, 2, 1], "Fe": [9, 10, 9, 6, 1, 4, 5, 4, 1, 4, 5, 4, 1, 2, 1, 2, 1, 4, 5, 4, 1, 2, 1, 2, 1, 2, 1], "Co": [4, 9, 10, 9, 6, 1, 4, 5, 4, 1, 4, 5, 4, 1, 2, 1, 2, 1, 4, 5, 4, 1, 2, 1, 2, 1, 2, 1]
}

ionization_energies = {
    1: [13.598434599702], 2: [24.587389011, 54.4177655282], 3: [5.391714996, 75.640097, 122.45435913],
    4: [9.322699, 18.21115, 153.896205, 217.71858459], 5: [8.298019, 25.15483, 37.93059, 259.374379, 340.2260225],
    6: [11.260288, 24.383143, 47.88778, 64.49352, 392.09056, 489.99320779],
    7: [14.53413, 29.60125, 47.4453, 77.4735, 97.8901, 552.06741, 667.0461377],
    8: [13.618055, 35.12112, 54.93554, 77.4135, 113.899, 138.1189, 739.32697, 871.4099138],
    9: [17.42282, 34.97081, 62.70798, 87.175, 1103.1175302, 114.249, 157.16311, 185.1868, 953.8983],
    10: [21.564541, 40.96297, 63.4233, 97.19, 1195.8082, 126.247, 157.934, 207.271, 239.097, 1362.199256],
    11: [5.13907696, 47.28636, 71.62, 98.936, 299.856, 138.404, 172.23, 208.504, 264.192, 1465.0992, 1648.702285],
    12: [7.646236, 15.035271, 80.1436, 109.2654, 327.99, 141.33, 186.76, 225.02, 265.924, 367.489, 1761.8049, 1962.663889],
    13: [5.985769, 18.82855, 28.447642, 119.9924, 330.21, 153.8252, 190.49, 241.76, 284.64, 398.65, 442.005, 2085.97693, 2304.140359],
    14: [8.15168, 16.34585, 33.493, 45.14179, 351.28, 166.767, 205.279, 246.57, 303.59, 401.38, 476.273, 523.415, 2437.65805, 2673.177958],
    15: [10.486686, 19.76949, 30.20264, 51.44387, 372.31, 65.02511, 220.43, 263.57, 309.6, 424.4, 479.44, 560.62, 611.741, 2816.90868, 3069.842145],
    16: [10.36001, 23.33788, 34.86, 47.222, 379.84, 72.5945, 88.0529, 280.954, 328.794, 447.7, 504.55, 564.41, 651.96, 706.994, 3223.78057, 3494.188518],
    17: [12.967633, 23.81364, 39.8, 53.24, 400.851, 67.68, 96.94, 114.2013, 348.306, 456.7, 530.0, 591.58, 656.3, 750.23, 809.198, 3658.34366, 3946.29179],
    18: [15.7596119, 27.62967, 40.735, 59.58, 422.6, 74.84, 91.29, 124.41, 143.4567, 479.76, 540.4, 619.0, 685.5, 755.13, 855.5, 918.375, 4120.66559, 4426.22407],
    19: [4.34066373, 31.625, 45.8031, 60.917, 175.8174, 82.66, 99.44, 117.56, 154.87, 503.67, 565.6, 631.1, 714.7, 786.3, 4934.04979, 860.92, 967.7, 1034.542, 4610.80714],
    20: [6.11315547, 11.871719, 50.91316, 67.2732, 188.54, 84.34, 108.78, 127.21, 147.24, 211.275, 591.6, 658.2, 728.6, 817.2, 5128.8576, 894.0, 973.7, 1086.8, 1157.726, 5469.86358],
    21: [6.56149, 12.79977, 24.756839, 73.4894, 180.03, 91.95, 110.68, 137.99, 158.08, 225.18, 249.798, 687.36, 757.7, 833.2, 1287.957, 926.5, 1008.6, 1093.5, 1213.1, 5674.9036, 6033.75643],
    22: [6.82812, 13.5755, 27.49171, 43.26717, 192.1, 99.299, 119.533, 140.68, 170.5, 215.92, 265.07, 291.5, 787.67, 864.0, 1346.3, 944.5, 1042.5, 1130.2, 1220.3, 1425.257, 6249.0226, 6625.81023],
    23: [6.746187, 14.634, 29.3111, 46.709, 206.0, 65.28165, 128.125, 150.72, 173.55, 230.5, 254.8, 308.5, 336.274, 896.0, 1354.2, 977.2, 1062.9, 1165.2, 1258.9, 1486.7, 1569.656, 6851.3112, 7246.12624],
    24: [6.76651, 16.486305, 30.959, 49.16, 209.5, 69.46, 90.6349, 160.29, 184.76, 244.5, 270.8, 296.7, 354.7, 384.163, 1394.5, 1011.6, 1097.2, 1188.0, 1294.8, 1495.1, 1634.1, 1721.183, 7481.8624, 7894.80289],
    25: [7.434038, 15.63999, 33.668, 51.21, 221.89, 72.41, 95.604, 119.203, 195.5, 248.6, 286.1, 314.4, 343.6, 402.95, 1430.9, 435.172, 1133.7, 1224.1, 1320.3, 1537.2, 1643.2, 1788.7, 1879.873, 8140.7864, 8571.95438],
    26: [7.9024681, 16.19921, 30.651, 54.91, 233.6, 75.0, 98.985, 124.9671, 151.06, 262.1, 290.9, 330.8, 361.0, 392.2, 1460.0, 456.2, 489.312, 1262.7, 1357.8, 1575.6, 1687.0, 1798.4, 1950.4, 2045.759, 8828.1864, 9277.6886],
    27: [7.88101, 17.0844, 33.5, 51.27, 186.14, 79.5, 102.0, 128.9, 157.8, 275.4, 305.32, 336.1, 378.5, 410.0, 1504.5, 441.1, 511.96, 546.588, 1397.2, 1606.0, 1724.0, 1844.0, 1960.8, 2119.4, 2218.876, 9544.1817, 10012.1297],
    28: [7.639878, 18.168838, 35.187, 54.92, 193.2, 76.06, 108.0, 132.0, 162.0, 224.7, 319.5, 351.6, 384.5, 429.3, 1540.1, 462.8, 495.4, 571.07, 607.02, 1646.0, 1758.0, 1880.0, 2008.1, 2130.5, 2295.6, 2399.259, 10288.8848, 10775.3948],
    29: [7.72638, 20.29239, 36.841, 57.38, 198.0, 79.8, 103.0, 139.0, 166.0, 232.2, 265.33, 367.0, 401.0, 436.0, 670.608, 483.1, 518.7, 552.8, 632.5, 1690.5, 1800.0, 1918.0, 2044.0, 2179.4, 11567.6237, 2307.3, 2479.1, 2586.954, 11062.4309],
    30: [9.394197, 17.96439, 39.7233, 59.573, 203.0, 82.6, 108.0, 133.9, 173.9, 238.0, 274.4, 310.8, 417.6, 453.4, 697.5, 490.6, 540.0, 577.8, 613.3, 737.366, 1846.8, 1961.0, 2085.0, 2214.0, 11864.9401, 2358.0, 2491.5, 2669.9, 2781.996, 12388.9427],
    31: [5.999302, 20.51514, 30.72576, 63.241, 211.0, 86.01, 112.7, 140.8, 169.9, 244.0, 280.0, 319.0, 356.0, 471.2, 677.0, 508.8, 548.3, 599.8, 640.0, 765.7, 807.308, 2010.0, 2129.0, 2258.0, 2984.426, 2391.0, 2543.9, 2683.0, 2868.0, 12696]
}
# fmt: on

h = 1 * ureg.planck_constant
k_B = 1 * ureg.boltzmann_constant
m_e = 1 * ureg.electron_mass


@jax.jit
def bisection(func, a, b, tolerance=1e-4, max_iter=1e4):

    def condition(state):
        prev_x, next_x, count = state

        return (
            (count < max_iter)
            & (jnp.abs(func(next_x)) > tolerance)
            & (jnp.abs(prev_x - next_x) > tolerance)
        )

    def body(state):
        a, b, i = state
        c = (a + b) / 2  # middlepoint
        bound = jnp.where(jnp.sign(func(c)) == jnp.sign(func(a)), b, a)
        return bound, c, i + 1

    initial_state = (a, b, 0)

    _, final_state, iterations = jax.lax.while_loop(
        condition, body, initial_state
    )

    return final_state, iterations


@jax.jit
def saha_equation(gi: float, gj: float, T_e: Quantity, energy_diff: Quantity):

    return (
        2
        * (gj / gi)
        * ((2 * jnp.pi * 1 * m_e * k_B * T_e) ** 1.5 / (1 * h**3))
        * jnpu.exp(((-energy_diff) / (1 * k_B * T_e)))
    )


@partial(jax.jit, static_argnames=["element_list"])
def solve_saha(element_list, T_e: Quantity, ion_number_densities: Quantity):

    Z = [i.Z for i in element_list]
    M = jnp.zeros(
        (
            len(element_list) + onp.sum(Z) + 1,
            len(element_list) + onp.sum(Z) + 1,
        )
    )

    max_ne = jnpu.sum(
        jnp.array([elem.Z for elem in element_list]) * ion_number_densities
    )

    ne_scale = max_ne * 1e5

    skip = 0
    ionization_states = []
    for ion_dens, element in zip(ion_number_densities, element_list):

        # Flip the array, so that the weights start with no ionization and go to max
        stat_weight = jnp.array(_stat_weight[element.symbol])
        Eb = jnp.array(ionization_energies[element.Z]) * (
            1 * ureg.electron_volt
        )

        diag = jnp.diag(
            (
                (-1)
                * (
                    saha_equation(
                        stat_weight[:-1],
                        stat_weight[1:][::-1],
                        T_e,
                        Eb,
                    )
                    / ne_scale
                )
            ).m_as(ureg.dimensionless)
        )
        dens_row = jnp.ones((element.Z + 1))

        M = M.at[skip : skip + element.Z, skip : skip + element.Z].set(diag)
        M = M.at[skip : skip + element.Z + 1, skip + element.Z].set(dens_row)

        M = M.at[-1, skip + element.Z].set(
            (ion_dens / ne_scale).m_as(ureg.dimensionless)
        )

        skip += element.Z + 1
        ionization_states += list(jnp.arange(element.Z + 1))

    M = M.at[:-1, -1].set(jnp.array(ionization_states))

    def insert_ne(_M, ne):

        ne_line = jnp.ones(len(element_list) + onp.sum(Z)) * ne

        skip = -1
        for element in element_list:

            ne_line = ne_line.at[skip + element.Z + 1].set(0.0)
            skip += element.Z + 1

        _diag = jnp.diag(ne_line, -1)
        out = M + _diag
        out = out.at[-1, -1].set(ne)
        return out.T

    def det_M(ne):
        res = jnp.linalg.det(insert_ne(M, ne))
        return res

    # print("Max n_e:", max_ne)
    sol_ne, iterations = bisection(
        jax.tree_util.Partial(det_M),
        0.0,
        (max_ne / ne_scale).m_as(ureg.dimensionless),
        tolerance=1e-12,
        max_iter=1e5,
    )
    # jax.debug.print("Needed iterations for bisection: {x}", x=iterations)

    M = insert_ne(M, sol_ne)

    MM = jnp.array(M)
    M1 = MM[: (len(MM[0]) - 1), 0 : (len(MM[0]) - 1)]
    M2 = MM[: (len(MM[0]) - 1), (len(MM[0]) - 1)]

    # The solution in form of (nh0,nh1,nhe0,nhe1,nhe2,...)
    ionised_number_densities = jnp.linalg.solve(M1, M2)

    return ionised_number_densities * ne_scale


def calculate_mean_free_charge_saha(plasma_state):
    """
    Calculates the mean charge of each ion in a plasma using the Saha-Boltzmann equation.

    Parameters:
    plasma_state (PlasmaState): The plasma state object.

    Returns:
    jnp.ndarray: An array containing the mean charge of each ion in the plasma.
    """

    sol = solve_saha(
        tuple(plasma_state.ions),
        plasma_state.T_e,
        (plasma_state.mass_density / plasma_state.atomic_masses),
    ).m_as(1 / ureg.cc)

    indices = jnp.cumsum(
        jnp.array([0] + list([ion.Z + 1 for ion in plasma_state.ions]))
    )
    Z_total = []
    Z_free = []
    for i in range(len(indices) - 1):
        idx = jnp.arange(len(sol))
        relevant_part = jnp.where(
            (idx >= indices[i]) & (idx < indices[i + 1]), sol, 0
        )
        ionizations = jnp.where(
            (idx >= indices[i]) & (idx < indices[i + 1]),
            jnp.arange(len(sol)) - indices[i],
            0,
        )
        Z_total.append(jnp.sum(relevant_part))
        Z_free.append(jnp.sum(relevant_part / Z_total[i] * ionizations))
    Z_free = jnp.array(Z_free)

    return Z_free
