from jaxrts.saha import solve_saha
from jaxrts.elements import Element
from jaxrts.units import ureg, to_array

if __name__ == "__main__":

    print(solve_saha((Element("H"), Element("He")), 4200 * ureg.kelvin, to_array([1E19 * 1/ureg.cc, 1E19 * 1/ureg.cc])))