from jaxrts.saha import solve_saha, calculate_mean_free_charge_saha
from jaxrts.elements import Element
from jaxrts.units import ureg, to_array
from tqdm import tqdm
import jpu.numpy as jnpu

import jaxrts

import time

import jax.numpy as jnp

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


    ions = [jaxrts.Element("C"), jaxrts.Element("H"), jaxrts.Element("O"), jaxrts.Element("Co")]
    number_fraction = jnp.array([1 / 3, 1/ 3, 1 / 3-0.06, 0.06])  # Sample composition: C5H8O2
    mass_fraction = jaxrts.helpers.mass_from_number_fraction(number_fraction, ions)

    plasma_state = jaxrts.PlasmaState(
        ions=ions,
        Z_free=jnp.ones_like(mass_fraction),
        mass_density=ureg("260mg/cc") * mass_fraction,
        T_e=jnp.array([40]) * ureg.electron_volt / ureg.k_B,
    )

    # for k in range(1):
    #     t0 = time.time()
    #     Zf = calculate_mean_free_charge_saha(plasma_state)
    #     print(f"Time needed: {time.time()-t0}")


    # exit()

    Z_free = [[],[],[],[]]
    Tes = jnp.logspace(-1, 3, 1000)* ureg.electron_volt / ureg.k_B
    for Te in tqdm(Tes):
        plasma_state.T_e = Te
        res = calculate_mean_free_charge_saha(plasma_state)
        Z_free[0].append(res[0])
        Z_free[1].append(res[1])
        Z_free[2].append(res[2])
        Z_free[3].append(res[3])

    fig, ax = plt.subplots()

    ax.plot(Tes.m_as(ureg.electron_volt / ureg.k_B), Z_free[0], label = r"$Z_C$")
    ax.plot(Tes.m_as(ureg.electron_volt / ureg.k_B), Z_free[1], label = r"$Z_H$")
    ax.plot(Tes.m_as(ureg.electron_volt / ureg.k_B), Z_free[2], label = r"$Z_O$")
    ax.plot(Tes.m_as(ureg.electron_volt / ureg.k_B), Z_free[3], label = r"$Z_{Co}$")
    ax.set_xscale("log")
    ax.set_xlabel("T [eV]")
    ax.set_ylabel("Ionization")
    plt.show()
    exit()
    Al = Element("Al")

    xdata, ydata = load_csv_data(["dash.csv", "solid.csv", "dashdot.csv", "longdash.csv"])

    T = jnp.logspace(
        -3,
        1,
        1000,
    ) * ureg.kiloelectron_volt / ureg.boltzmann_constant

    fig, ax = plt.subplots()
    i = 0

    for ne in [
        rho_to_ne(0.027 * ureg.gram / ureg.cc, Al.atomic_mass),
        rho_to_ne(0.00027 * ureg.gram / ureg.cc, Al.atomic_mass),
        rho_to_ne(270.0 * ureg.gram / ureg.cc, Al.atomic_mass),
        rho_to_ne(2.7 * ureg.gram / ureg.cc, Al.atomic_mass)
    ]:
        ionizations = []
        for Tv in tqdm(T):
            sol = solve_saha(
                tuple([Al]),
                Tv.to(ureg.electron_volt / ureg.boltzmann_constant),
                to_array([ne]),
            )

            ionization_state_charge = np.array(range(14))

            sol = sol.m_as(1 / ureg.cc)
            fractions = sol / np.sum(sol)
            # , #np.sum(sol[2:]), np.sum(sol[2:]), np.sum(sol[2:])])

            Z_free = np.sum(fractions * ionization_state_charge)

            ionizations.append(Z_free)

        ionizations = np.array(ionizations)

        ax.plot(
            T.m_as(ureg.kiloelectron_volt / ureg.boltzmann_constant),
            ionizations,
            label=f"n_e = {ne}",
            color=f"C{i}",
        )
        i += 1

    i = 0
    for xv, yv in zip(xdata, ydata):
        ax.scatter(xv, np.array(yv), color=f"C{i}", s=15)
        i += 1

    ax.set_title("Aluminium Plasma")
    ax.set_ylabel("Mean Free Charge")
    ax.set_xlabel("Temperature [keV]")
    ax.set_xscale("log")
    plt.show()
