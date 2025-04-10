import mendeleev
from jaxrts.helpers import orbital_map, orbital_array

def parse_econf(config):
    """
    Parse the electron configuration from mendeleev into an jaxrts orbital map
    style array.
    """
    out = orbital_array()
    for key in config.keys():
        value = config[key]
        orbital_name = f"{key[0]}{key[1]}"
        out = out.at[orbital_map[orbital_name]].set(value)
    return out


def get_J(config):
    """
    Calculate the combined angular momentum J = L + S from an electronic
    configuration.
    """
    J = 0.0
    for key in config.keys():
        value = config[key]
        match (key, value):
            case ((_, "s"), 0):
                J += 0
            case ((_, "s"), 1):
                J += 0.5
            case ((_, "s"), 2):
                J += 0
            case ((_, "p"), 0):
                J += 0
            case ((_, "p"), 1):
                J += 0.5
            case ((_, "p"), 2):
                J += 0
            case ((_, "p"), 3):
                J += 1.5
            case ((_, "p"), 4):
                J += 2
            case ((_, "p"), 5):
                J += 1.5
            case ((_, "p"), 6):
                J += 0
            case ((_, "d"), 0):
                J += 0
            case ((_, "d"), 1):
                J += 1.5
            case ((_, "d"), 2):
                J += 2
            case ((_, "d"), 3):
                J += 1.5
            case ((_, "d"), 4):
                J += 0
            case ((_, "d"), 5):
                J += 2.5
            case ((_, "d"), 6):
                J += 4
            case ((_, "d"), 7):
                J += 4.5
            case ((_, "d"), 8):
                J += 4
            case ((_, "d"), 9):
                J += 2.5
            case ((_, "d"), 10):
                J += 0
    return J


elements = mendeleev.get_all_elements()

cutoffZ = 36

ionization_econf = {
    element.atomic_number: [
        parse_econf(element.ec.ionize(z).conf) for z in range(element.atomic_number + 1)
    ]
    for element in elements
    if element.atomic_number <= cutoffZ
}
ionization_g = {
    element.atomic_number: [
        1 + 2 * get_J(element.ec.ionize(z).conf)
        for z in range(element.atomic_number + 1)
    ]
    for element in elements
    if element.atomic_number <= cutoffZ
}
ionization_energies = {
    element.atomic_number: [ie.energy for ie in element._ionization_energies]
    for element in elements
    if element.atomic_number <= cutoffZ
}
print(ionization_energies)
print(ionization_econf)
print(ionization_g)
