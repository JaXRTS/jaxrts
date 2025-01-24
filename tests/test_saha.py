from jaxrts.saha import solve_saha
from jaxrts.elements import Element
from jaxrts.units import ureg, to_array
from tqdm import tqdm
import jpu.numpy as jnpu

import matplotlib.pyplot as plt

import numpy as np

import csv

def load_csv_data(filenames):
    """
    Load CSV data from a list of filenames.
    
    Parameters:
        filenames (list of str): List of CSV file paths.
        
    Returns:
        tuple: A tuple containing two lists:
            - all_x_values: List of lists, where each inner list contains x values from a file.
            - all_y_values: List of lists, where each inner list contains y values from a file.
    """
    all_x_values = []
    all_y_values = []
    
    for filename in filenames:
        x_values = []
        y_values = []
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:  # Ensure at least two columns exist
                    try:
                        x = float(row[0])
                        y = float(row[1])
                        x_values.append(x)
                        y_values.append(y)
                    except ValueError:
                        # Skip rows with non-numeric data
                        continue
        all_x_values.append(x_values)
        all_y_values.append(y_values)
    
    return all_x_values, all_y_values


def rho_to_ne(rho, m_a, Z=1):
    return (Z * rho / (m_a)).to(1 / (1 * ureg.cc))

if __name__ == "__main__":

    H = Element("H")
    He = Element("He")

    xdata, ydata = load_csv_data([f"plot-data({i}).csv" for i in range(4)])

    T = jnpu.linspace(0.1 * ureg.electron_volt / ureg.boltzmann_constant, 2 * ureg.electron_volt / ureg.boltzmann_constant, 200)
    
    fig, ax = plt.subplots()
    i=0
    for ne in [1E16 / ureg.meter**3, 1E19 / ureg.meter**3, 1E22 / ureg.meter**3, 1E25 / ureg.meter**3]:
        ionizations = []
        for Tv in tqdm(T):
            sol = solve_saha(tuple([H]), Tv.to(ureg.electron_volt / ureg.boltzmann_constant), to_array([ne]))

            ionization_state_charge = np.array([0, 1])

            sol = sol.m_as(1/ureg.cc)
            fractions = sol / np.array([np.sum(sol[:2]), np.sum(sol[:2])])
            #, #np.sum(sol[2:]), np.sum(sol[2:]), np.sum(sol[2:])])

            Z_free = jnpu.sum(fractions * ionization_state_charge)
        
            ionizations.append(Z_free)

        ax.plot(T.m_as(ureg.electron_volt / ureg.boltzmann_constant), ionizations, label = f"n_e = {ne}", color=f"C{i}")
        i += 1

    i=0
    for xv, yv in zip(xdata, ydata):
        ax.scatter(xv, np.array(yv) / 100.0, color=f"C{i}", s=15)
        i+=1

    ax.set_title("Hydrogen Plasma")
    ax.set_ylabel("Mean Free Charge")
    ax.set_xlabel("Temperature [eV]")
    plt.show()
    # print("Z_Free_avg: H:", np.sum(Z_free[:2]))
    #, "He:", np.sum(Z_free[2:]))