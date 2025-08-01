import numpy as onp
import jpu.numpy as jnpu

from jaxrts import elements, ureg


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


def test_cold_ionization_energies_match_zero_ionization() -> None:
    """
    Test that the cold ionization energies match the zero
    ionization energies.
    """
    acceptable_deviation = ureg("10eV")
    for z in range(35):
        element = elements.Element(z + 1)
        if element == elements.Element("N") or element == elements.Element(
            "O"
        ):
            # The L edge caused by the s orbital is quite off for
            # nitrogen and oxygen.
            assert (
                element.cold_binding_energies[0]
                - element.get_binding_energies(0)[0]
            ) < acceptable_deviation
            assert (
                element.cold_binding_energies[1]
                - element.get_binding_energies(0)[1]
            ) < 2 * acceptable_deviation

            assert (
                element.cold_binding_energies[2]
                - element.get_binding_energies(0)[2]
            ) < acceptable_deviation
        else:
            assert (
                jnpu.max(
                    element.cold_binding_energies
                    - element.get_binding_energies(0)
                )
                < acceptable_deviation
            )
