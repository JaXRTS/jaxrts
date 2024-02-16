import pytest

from jaxrts import elements
import numpy as onp


def test_element_creatation() -> None:
    assert elements.Element("C") == elements.Element(6)


def test_electron_distribution() -> None:
    for Z in elements._element_names.keys():
        electron_occupation = elements.electron_distribution(Z)
        # Check that the number of electrons is correct
        assert onp.sum(electron_occupation) == Z
        # Check that no orbital is overfilled
        assert electron_occupation[0] <= 2
        assert electron_occupation[1] <= 2
        assert electron_occupation[2] <= 6
        assert electron_occupation[3] <= 2
        assert electron_occupation[4] <= 6
        assert electron_occupation[5] <= 10
        assert electron_occupation[6] <= 2
        assert electron_occupation[7] <= 6
        assert electron_occupation[8] <= 10
        assert electron_occupation[9] <= 14
