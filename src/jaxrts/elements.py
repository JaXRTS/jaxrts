"""
This submodule contains data for different chemical elements.
"""

from typing import Any, List

import jax
import jpu.numpy as jnpu
from jax import numpy as jnp

from .helpers import invert_dict, orbital_array
from .units import Quantity, ureg

_element_symbols = {
    1: "H",  # Hydrogen
    2: "He",  # Helium
    3: "Li",  # Lithium
    4: "Be",  # Beryllium
    5: "B",  # Boron
    6: "C",  # Carbon
    7: "N",  # Nitrogen
    8: "O",  # Oxygen
    9: "F",  # Fluorine
    10: "Ne",  # Neon
    11: "Na",  # Sodium
    12: "Mg",  # Magnesium
    13: "Al",  # Aluminum
    14: "Si",  # Silicon
    15: "P",  # Phosphorus
    16: "S",  # Sulfur
    17: "Cl",  # Chlorine
    18: "Ar",  # Argon
    19: "K",  # Potassium
    20: "Ca",  # Calcium
    21: "Sc",  # Scandium
    22: "Ti",  # Titanium
    23: "V",  # Vanadium
    24: "Cr",  # Chromium
    25: "Mn",  # Manganese
    26: "Fe",  # Iron
    27: "Co",  # Cobalt
    28: "Ni",  # Nickel
    29: "Cu",  # Copper
    30: "Zn",  # Zinc
    31: "Ga",  # Gallium
    32: "Ge",  # Germanium
    33: "As",  # Arsenic
    34: "Se",  # Selenium
    35: "Br",  # Bromine
    36: "Kr",  # Krypton
}
_element_names = {
    1: "Hydrogen",
    2: "Helium",
    3: "Lithium",
    4: "Beryllium",
    5: "Boron",
    6: "Carbon",
    7: "Nitrogen",
    8: "Oxygen",
    9: "Fluorine",
    10: "Neon",
    11: "Sodium",
    12: "Magnesium",
    13: "Aluminum",
    14: "Silicon",
    15: "Phosphorus",
    16: "Sulfur",
    17: "Chlorine",
    18: "Argon",
    19: "Potassium",
    20: "Calcium",
    21: "Scandium",
    22: "Titanium",
    23: "Vanadium",
    24: "Chromium",
    25: "Manganese",
    26: "Iron",
    27: "Cobalt",
    28: "Nickel",
    29: "Copper",
    30: "Zinc",
    31: "Gallium",
    32: "Germanium",
    33: "Arsenic",
    34: "Selenium",
    35: "Bromine",
    36: "Krypton",
}
_element_masses = {
    1: 1.008,
    2: 4.002602,
    3: 6.94,
    4: 9.0121831,
    5: 10.81,
    6: 12.011,
    7: 14.007,
    8: 15.999,
    9: 18.998403163,
    10: 20.1797,
    11: 22.98976928,
    12: 24.305,
    13: 26.9815385,
    14: 28.085,
    15: 30.973761998,
    16: 32.06,
    17: 35.45,
    18: 39.948,
    19: 39.0983,
    20: 40.078,
    21: 44.955908,
    22: 47.867,
    23: 50.9415,
    24: 51.9961,
    25: 54.938044,
    26: 55.845,
    27: 58.933194,
    28: 58.6934,
    29: 63.546,
    30: 65.38,
    31: 69.723,
    32: 72.63,
    33: 74.921595,
    34: 78.971,
    35: 79.904,
    36: 83.798,
}

# Taken from the python package mendeleev
# L. M. Mentel, mendeleev - A Python resource for properties of chemical
# elements, ions and isotopes. , 2014-- . Available at:
# https://github.com/lmmentel/mendeleev.
# which cites John C Slater. Atomic Radii in Crystals. The Journal of Chemical
# Physics, 41(10):3199, 1964. URL:
# http://scitation.aip.org/content/aip/journal/jcp/41/10/10.1063/1.1725697,
# doi:10.1063/1.1725697.
# For Z=36 krypton, where no empirical value is supplied, we use the calculated
# value of E. Clementi; D.L.Raimondi; W.P. Reinhardt (1967). "Atomic Screening
# Constants from SCF Functions. II. Atoms with 37 to 86 Electrons". The Journal
# of Chemical Physics. 47 (4): 1300–1307.
# See https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
# in pm
_atomic_radii_emp = {
    1: 25.0,
    2: 120.0,
    3: 145.0,
    4: 105.0,
    5: 85.0,
    6: 70.0,
    7: 65.0,
    8: 60.0,
    9: 50.0,
    10: 160.0,
    11: 180.0,
    12: 150.0,
    13: 125.0,
    14: 110.0,
    15: 100.0,
    16: 100.0,
    17: 100.0,
    18: 71.0,
    19: 220.0,
    20: 180.0,
    21: 160.0,
    22: 140.0,
    23: 135.0,
    24: 140.0,
    25: 140.0,
    26: 140.0,
    27: 135.0,
    28: 135.0,
    29: 135.0,
    30: 135.0,
    31: 130.0,
    32: 125.0,
    33: 115.0,
    34: 115.0,
    35: 115.0,
    36: 88.0,
}
# E. Clementi; D.L.Raimondi; W.P. Reinhardt (1967). "Atomic Screening
# Constants from SCF Functions. II. Atoms with 37 to 86 Electrons". The Journal
# of Chemical Physics. 47 (4): 1300–1307.
# See https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
# in pm
_atomic_radii_calc = {
    1: 53,
    2: 31,
    3: 167,
    4: 112,
    5: 87,
    6: 67,
    7: 56,
    8: 48,
    9: 42,
    10: 38,
    11: 190,
    12: 145,
    13: 118,
    14: 111,
    15: 98,
    16: 88,
    17: 79,
    18: 71,
    19: 243,
    20: 194,
    21: 184,
    22: 176,
    23: 171,
    24: 166,
    25: 161,
    26: 156,
    27: 152,
    28: 149,
    29: 145,
    30: 142,
    31: 136,
    32: 125,
    33: 114,
    34: 103,
    35: 94,
    36: 88,
}

# These Values are taken from Shannon, R. D., "Revised effective ionic radii
# and systematic studies of interatomic distances in halides and
# chalcogenides", Acta Crystallographica Section A, vol. 32, no 5, 1976. as
# they are presented by Baloch, Ahmer A. B. and Alqahtani, Saad M. and Mumtaz,
# Faisal and Muqaibel, Ali H. and Rashkeev, Sergey N. and Alharbi, Fahhad H.,
# "Extending Shannon's ionic radii database using machine learning", Physical
# Review Materials, vol. 5, no 4, 2021. at https://cmd-ml.github.io/.
#
# The entries are formatted as follows: (Oxidation State, Coordination Number,
# radius [Angstrom]
_ionic_radii = {
    1: [],
    2: [],
    3: [(1, 4, 0.59), (1, 6, 0.76), (1, 8, 0.92)],
    4: [(2, 3, 0.16), (2, 4, 0.27), (2, 6, 0.45)],
    5: [(3, 3, 0.01), (3, 4, 0.11), (3, 6, 0.27)],
    6: [(4, 4, 0.15), (4, 6, 0.16)],
    7: [(5, 6, 0.13), (3, 6, 0.16), (-3, 4, 1.46)],
    8: [
        (-2, 2, 1.35),
        (-2, 3, 1.36),
        (-2, 4, 1.38),
        (-2, 6, 1.4),
        (-2, 8, 1.42),
    ],
    9: [
        (7, 6, 0.08),
        (-1, 2, 1.285),
        (-1, 3, 1.3),
        (-1, 4, 1.31),
        (-1, 6, 1.33),
    ],
    10: [],
    11: [
        (1, 4, 0.99),
        (1, 5, 1),
        (1, 6, 1.02),
        (1, 7, 1.12),
        (1, 8, 1.18),
        (1, 9, 1.24),
        (1, 12, 1.39),
    ],
    12: [(2, 4, 0.57), (2, 5, 0.66), (2, 6, 0.72), (2, 8, 0.89)],
    13: [(3, 4, 0.39), (3, 5, 0.48), (3, 6, 0.535)],
    14: [(4, 4, 0.26), (4, 6, 0.4)],
    15: [(5, 4, 0.17), (5, 5, 0.29), (5, 6, 0.38), (3, 6, 0.44)],
    16: [(6, 4, 0.12), (6, 6, 0.29), (4, 6, 0.37), (-2, 6, 1.84)],
    17: [(7, 4, 0.08), (5, 3, 0.12), (7, 6, 0.27), (-1, 6, 1.81)],
    18: [],
    19: [
        (1, 4, 1.37),
        (1, 6, 1.38),
        (1, 7, 1.46),
        (1, 8, 1.51),
        (1, 9, 1.55),
        (1, 10, 1.59),
        (1, 12, 1.64),
    ],
    20: [
        (2, 6, 1),
        (2, 7, 1.06),
        (2, 8, 1.12),
        (2, 9, 1.18),
        (2, 10, 1.23),
        (2, 12, 1.34),
    ],
    21: [(3, 6, 0.745), (3, 8, 0.87)],
    22: [
        (4, 4, 0.42),
        (4, 5, 0.51),
        (4, 6, 0.605),
        (3, 6, 0.67),
        (4, 8, 0.74),
        (2, 6, 0.86),
    ],
    23: [
        (5, 4, 0.355),
        (5, 5, 0.46),
        (4, 5, 0.53),
        (5, 6, 0.54),
        (4, 6, 0.58),
        (3, 6, 0.64),
        (4, 8, 0.72),
        (2, 6, 0.79),
    ],
    24: [
        (6, 4, 0.26),
        (5, 4, 0.345),
        (4, 4, 0.41),
        (6, 6, 0.44),
        (5, 6, 0.49),
        (4, 6, 0.55),
        (5, 8, 0.57),
        (3, 6, 0.615),
        (2, 6, 0.73),
    ],
    25: [
        (7, 4, 0.25),
        (6, 4, 0.255),
        (5, 4, 0.33),
        (4, 4, 0.39),
        (7, 6, 0.46),
        (4, 6, 0.53),
        (3, 5, 0.58),
        (3, 6, 0.645),
        (2, 4, 0.66),
        (2, 5, 0.75),
        (2, 6, 0.83),
        (2, 7, 0.9),
        (2, 8, 0.96),
    ],
    26: [
        (6, 4, 0.25),
        (3, 4, 0.49),
        (3, 5, 0.58),
        (4, 6, 0.585),
        (2, 4, 0.64),
        (3, 6, 0.645),
        (2, 6, 0.78),
        (3, 8, 0.78),
        (2, 8, 0.92),
    ],
    27: [
        (4, 4, 0.4),
        (4, 6, 0.53),
        (2, 4, 0.58),
        (3, 6, 0.61),
        (2, 5, 0.67),
        (2, 6, 0.745),
        (2, 8, 0.9),
    ],
    28: [(4, 6, 0.48), (2, 4, 0.55), (3, 6, 0.6), (2, 5, 0.63), (2, 6, 0.69)],
    29: [
        (1, 2, 0.46),
        (3, 6, 0.54),
        (2, 4, 0.57),
        (1, 4, 0.6),
        (2, 5, 0.65),
        (2, 6, 0.73),
        (1, 6, 0.77),
    ],
    30: [(2, 4, 0.6), (2, 5, 0.68), (2, 6, 0.74), (2, 8, 0.9)],
    31: [(3, 4, 0.47), (3, 5, 0.55), (3, 6, 0.62)],
    32: [(4, 4, 0.39), (4, 6, 0.53), (2, 6, 0.73)],
    33: [(5, 4, 0.335), (5, 6, 0.46), (3, 6, 0.58)],
    34: [(6, 4, 0.28), (6, 6, 0.42), (4, 6, 0.5), (-2, 6, 1.98)],
    35: [
        (7, 4, 0.25),
        (5, 3, 0.31),
        (7, 6, 0.39),
        (3, 4, 0.59),
        (-1, 6, 1.96),
    ],
    36: [],
}

# This data is taken from hephaestus https://bruceravel.github.io/demeter/
_absorption_edges = {
    1: [13.6, 0, 0, 0, 0, 0, 0, 0, 0],
    2: [24.6, 0, 0, 0, 0, 0, 0, 0, 0],
    3: [54.8, 5.3, 0, 0, 0, 0, 0, 0, 0],
    4: [111.5, 8, 3, 0, 0, 0, 0, 0, 0],
    5: [188, 12.6, 4.7, 0.0, 0, 0, 0, 0, 0],
    6: [284.2, 18, 7.2, 0.0, 0, 0, 0, 0, 0],
    7: [409.9, 37.3, 17.5, 0.0, 0, 0, 0, 0, 0],
    8: [543.1, 41.6, 18.2, 0.0, 0, 0, 0, 0, 0],
    9: [696.7, 45, 19.9, 0.0, 0, 0, 0, 0, 0],
    10: [870.2, 48.5, 21.65, 0.0, 0, 0, 0, 0, 0],
    11: [1070.8, 63.5, 30.4, 0.0, 0, 0, 0, 0, 0],
    12: [1303, 88.6, 49.4, 2, 1, 0, 0, 0, 0],
    13: [1559, 117.8, 72.7, 4, 2, 0, 0, 0, 0],
    14: [1839, 149.7, 99.5, 8, 2, 0, 0, 0, 0],
    15: [2145.5, 189, 135.5, 12.0, 6.5, 0, 0, 0, 0],
    16: [2472.0, 230.9, 163.05, 14.0, 7.5, 0, 0, 0, 0],
    17: [2822.0, 270.0, 201.0, 18, 10, 0, 0, 0, 0],
    18: [3205.9, 326.3, 249.3, 29.3, 15.8, 0, 0, 0, 0, 0],
    19: [3608.4, 378.6, 294.95, 34.8, 18.3, 0, 0, 0, 0, 0],
    20: [4038.5, 438.4, 347.95, 44.3, 25.4, 0, 0, 0, 0, 0],
    21: [4492.0, 498.0, 401.15, 51.1, 28.3, 0, 0, 0, 0, 0],
    22: [4966.0, 560.9, 457.0, 58.7, 32.6, 2, 0, 0, 0, 0],
    23: [5465.0, 626.7, 515.95, 66.3, 37.2, 2, 0, 0, 0, 0],
    24: [5989.0, 696.0, (583.8 + 574.1) / 2, 74.1, 42.2, 2, 0, 0, 0, 0],
    25: [6539, 769.1, (649.9 + 638.7) / 2, 82.3, 47.2, 2, 0, 0, 0, 0],
    26: [7112, 844.6, (719.9 + 706.8) / 2, 91.3, 52.7, 2, 0, 0, 0, 0],
    27: [7709, 925.1, (778.1 + 793.2) / 2, 101, 58.9, 3, 0, 0, 0, 0],
    28: [
        8333,
        1008.6,
        (870.0 + 852.7) / 2,
        110.8,
        (68.0 + 66.2) / 2,
        4,
        0,
        0,
        0,
        0,
    ],
    29: [
        8979,
        1096.7,
        (952.3 + 932.7) / 2,
        122.5,
        (77.3 + 75.1) / 2,
        5,
        0,
        0,
        0,
        0,
    ],
    30: [
        9659.0,
        1196.2,
        (1044.9 + 1021.8) / 2,
        139.8,
        (91.4 + 88.6) / 2,
        10.15,
        0,
        1,
        0,
        0,
    ],
    31: [
        10367,
        1299,
        (1143.2 + 1116.4) / 2,
        159.51,
        (103.5 + 100.0) / 2,
        18.7,
        1,
        2,
        0,
        0,
    ],
    32: [
        11103,
        1414.6,
        (1248.1 + 1217) / 2,
        180.1,
        (124.9 + 120.8) / 2,
        (29.8 + 29.2) / 2,
        5,
        3,
        0,
        0,
    ],
    33: [
        11867,
        1527,
        (1359.1 + 1323.6) / 2,
        204.7,
        (146.2 + 141.2) / 2,
        41.7,
        8,
        3,
        0,
        0,
    ],
    34: [
        12658,
        1652,
        (1474.3 + 1433.9) / 2,
        229.6,
        (166.5 + 160.7) / 2,
        55.56,
        12,
        3,
        0,
        0,
    ],
    35: [
        13474,
        1782,
        (1596 + 1550) / 2,
        257,
        (189 + 182) / 2,
        (70 + 69) / 2,
        27,
        3,
        0,
        0,
    ],
    36: [
        14326,
        1921,
        (1730.9 + 1678.4) / 2,
        292.8,
        (222.2 + 214.4) / 2,
        (95.0 + 93.8) / 2,
        27.5,
        14.1,
        0,
        0,
    ],
}


def electron_distribution(atomic_number: int) -> jnp.ndarray:
    """
    Returns number of electrons for each orbital (defined by the quantum
    numbers n and l) for a neutral atom with the given ``atomic_number``.

    Parameter,
    ----------
    atomic_number: int
        The atomic number of the element.

    Returns
    -------
    jnp.ndarray
        The number of electrons per orbital as a flat array, starting with
        ``1s``, up to ``4f``.
        To find the corresponding indices for individual orbitals, one can use
        :py:data:`~.helpers.orbital_map`
    """
    electrons_remaining = jnp.array(atomic_number)
    occupancy = []

    while electrons_remaining > 0:
        for n in range(1, 5):  # Consider orbitals up to 4th shell (n=4)
            for l in range(n):  # noqa: E741
                max_electrons = 2 * (2 * l + 1)  # Max. electrons in an orbital

                if electrons_remaining >= max_electrons:
                    occupancy.append(max_electrons)
                    electrons_remaining = jnp.maximum(
                        0, electrons_remaining - max_electrons
                    )
                else:
                    occupancy.append(electrons_remaining)
                    electrons_remaining = 0

                if electrons_remaining == 0:
                    break
            if electrons_remaining == 0:
                break
        if electrons_remaining == 0:
            break

    return orbital_array(*occupancy)


@jax.jit
def electron_distribution_ionized_state(Z_core: float) -> jnp.ndarray:
    """
    Interpolate between electron populations if the number of core electrons is
    not an integer.

    Assume the population of electrons be behave like a neutral atom with
    reduced number of electrons. I.e., a 1.5 times ionized carbon is like
    Beryllium (and half a step to Boron).

    Parameters
    ------------
    Z_core: float
        Number of electrons still bound to the core

    Returns
    -------
    jnp.ndarray
        An array of populations.
    """
    S1s = jnpu.interp(Z_core, jnp.array([1, 2]), jnp.array([1, 2]))
    S2s = jnpu.interp(Z_core, jnp.array([2, 4]), jnp.array([0, 2]))
    S2p = jnpu.interp(Z_core, jnp.array([4, 10]), jnp.array([0, 6]))
    S3s = jnpu.interp(Z_core, jnp.array([10, 12]), jnp.array([0, 2]))
    S3p = jnpu.interp(Z_core, jnp.array([12, 18]), jnp.array([0, 6]))
    S3d = jnpu.interp(Z_core, jnp.array([18, 28]), jnp.array([0, 10]))
    S4s = jnpu.interp(Z_core, jnp.array([28, 30]), jnp.array([0, 2]))
    S4p = jnpu.interp(Z_core, jnp.array([30, 36]), jnp.array([0, 6]))
    # Note: the lines below are wrong, but we are only interested in Z<37
    S4d = jnpu.interp(Z_core, jnp.array([0, 99]), jnp.array([0, 0]))
    S4f = jnpu.interp(Z_core, jnp.array([0, 99]), jnp.array([0, 0]))
    return jnp.array(
        [
            S1s,
            S2s,
            S2p,
            S3s,
            S3p,
            S3d,
            S4s,
            S4p,
            S4d,
            S4f,
        ]
    )


class Element:
    def __init__(self, identifier: str | int) -> None:
        if isinstance(identifier, str):
            #: The atomic number of the element
            self.Z = invert_dict(_element_symbols)[identifier]
            #: The abbreviated symbol no the element
            self.symbol = identifier
        else:
            self.Z = int(identifier)
            self.symbol = _element_symbols[self.Z]

        #: The name of the element
        self.name: str = _element_names[self.Z]
        #: The electron distribution, returned as a flat array
        self.electron_distribution = electron_distribution_ionized_state(
            self.Z
        )

        #: The atomic mass of this element
        self.atomic_mass: Quantity = _element_masses[self.Z] * (
            1 * ureg.atomic_mass_constant
        )

        Eb = _absorption_edges[self.Z]
        Eb1s = Eb[0]
        Eb2s = Eb[1]
        Eb2p = 1 / 2 * (Eb[2] + Eb[3])
        Eb3s = Eb[4]
        Eb3p = 1 / 2 * (Eb[5] + Eb[6])
        Eb3d = 1 / 2 * (Eb[7] + Eb[8])
        Eb4s = 0
        Eb4p = 0
        Eb4d = 0
        Eb4f = 0

        self.binding_energies = jnp.array(
            [Eb1s, Eb2s, Eb2p, Eb3s, Eb3p, Eb3d, Eb4s, Eb4p, Eb4d, Eb4f]
        ) * (1 * ureg.electron_volt)

        self.atomic_radius_calc = _atomic_radii_calc[self.Z] * ureg.picometer

    def __eq__(self, other: Any) -> bool:
        """
        If this function returns ``True``, the two instances self and other are
        considered identical.
        """
        if not isinstance(other, Element):
            # don't attempt to compare against unrelated types
            raise NotImplementedError(
                "Cannot compare {} to an object of type {}".format(
                    type(self), type(other)
                )
            )
        if isinstance(other, MixElement):
            return False

        return self.Z == other.Z

    def __repr__(self) -> str:
        return f"Element {self.name} ({self.symbol}) Z={self.Z}"

    def __hash__(self) -> int:
        return self.Z

class MixElement(Element):
    """
    This helper class is used to define Average Atoms to perform calculations
    on.

    This class is not intended to be used everywhere, where a real
    :py:class:`~.Element` can be used. Rather, we see it as a convenience tool.
    """

    def __init__(self, Zmix, avgMass, name=""):
        self.Z: float = Zmix
        self.electron_distribution = electron_distribution_ionized_state(
            self.Z
        )
        self.atomic_mass: Quantity = avgMass
        self.symbol = "MIX"
        self.name = name

        # These are entries that do not make semse for an Average Atom
        self.binding_energies = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * (
            1 * ureg.electron_volt
        )

        self.atomic_radius_calc = 0 * ureg.picometer
