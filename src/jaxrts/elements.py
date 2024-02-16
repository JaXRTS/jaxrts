"""
This submodule contains data for different chemical elements.
"""

from typing import Any

from jax import numpy as jnp

from .helpers import orbital_array, invert_dict

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


def electron_distribution(atomic_number: int) -> jnp.ndarray:
    """
    Returns number of electrons for each orbital (defined by the quantum
    numbers n and l) for a neutral atom with the given ``atomic_number``.

    Parameters
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
    electrons_remaining = atomic_number
    occupancy = []

    while electrons_remaining > 0:
        for n in range(1, 5):  # Consider orbitals up to 4th shell (n=4)
            for l in range(n):  # noqa: E741
                max_electrons = 2 * (2 * l + 1)  # Max. electrons in an orbital

                if electrons_remaining >= max_electrons:
                    occupancy.append(max_electrons)
                    electrons_remaining -= max_electrons
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
        self.name = _element_names[self.Z]
        #: The electron distribution, retuned as a flat array
        self.electron_distribution = electron_distribution(self.Z)

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

        return self.Z == other.Z

    def __repr__(self) -> str:
        return f"Element {self.name} ({self.symbol}) Z={self.Z}"
