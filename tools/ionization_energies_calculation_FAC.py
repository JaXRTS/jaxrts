# ======================================================================================
# SCRIPT: ionization_energies_calculation_FAC.py
#
# HOW TO USE:
#   python ionization_energies_calculation_FAC.py
#   this script was tested on ubuntu 24.04 with Python 3.12
#
# DESCRIPTION:
#   This self-contained script calculates ionization energies for all elements from Z=1
#   to Z=36. It uses the 'multiprocessing' module to ensure each FAC calculation
#   runs in an isolated process, preventing state-related errors from the fac library.
#
# DEPENDENCIES:
#   - fac: The Flexible Atomic Code (FAC) Python wrapper.
#     https://github.com/flexible-atomic-code/fac
#
# CHANGE LOG (v4.1 - Multiprocessing Fix):
#   - BUG FIX: Re-introduced process isolation using the 'multiprocessing' module
#     to fix state-leakage errors from the underlying fac library during
#     sequential calculations (e.g., "AddConfigToList" error).
#   - ROBUSTNESS: Each call to the FAC library now runs in a clean, separate
#     process, mimicking the stability of the original subprocess-based script.
#   - Time: 31.07.2025
# ======================================================================================

import os
import re
import pprint
from datetime import datetime
import multiprocessing

# --- Configuration Data ---
NEUTRAL_CONFIGS = {
    1: ("H", "1s1"),
    2: ("He", "1s2"),
    3: ("Li", "1s2 2s1"),
    4: ("Be", "1s2 2s2"),
    5: ("B", "1s2 2s2 2p1"),
    6: ("C", "1s2 2s2 2p2"),
    7: ("N", "1s2 2s2 2p3"),
    8: ("O", "1s2 2s2 2p4"),
    9: ("F", "1s2 2s2 2p5"),
    10: ("Ne", "1s2 2s2 2p6"),
    11: ("Na", "1s2 2s2 2p6 3s1"),
    12: ("Mg", "1s2 2s2 2p6 3s2"),
    13: ("Al", "1s2 2s2 2p6 3s2 3p1"),
    14: ("Si", "1s2 2s2 2p6 3s2 3p2"),
    15: ("P", "1s2 2s2 2p6 3s2 3p3"),
    16: ("S", "1s2 2s2 2p6 3s2 3p4"),
    17: ("Cl", "1s2 2s2 2p6 3s2 3p5"),
    18: ("Ar", "1s2 2s2 2p6 3s2 3p6"),
    19: ("K", "1s2 2s2 2p6 3s2 3p6 4s1"),
    20: ("Ca", "1s2 2s2 2p6 3s2 3p6 4s2"),
    21: ("Sc", "1s2 2s2 2p6 3s2 3p6 3d1 4s2"),
    22: ("Ti", "1s2 2s2 2p6 3s2 3p6 3d2 4s2"),
    23: ("V", "1s2 2s2 2p6 3s2 3p6 3d3 4s2"),
    24: ("Cr", "1s2 2s2 2p6 3s2 3p6 3d5 4s1"),
    25: ("Mn", "1s2 2s2 2p6 3s2 3p6 3d5 4s2"),
    26: ("Fe", "1s2 2s2 2p6 3s2 3p6 3d6 4s2"),
    27: ("Co", "1s2 2s2 2p6 3s2 3p6 3d7 4s2"),
    28: ("Ni", "1s2 2s2 2p6 3s2 3p6 3d8 4s2"),
    29: ("Cu", "1s2 2s2 2p6 3s2 3p6 3d10 4s1"),
    30: ("Zn", "1s2 2s2 2p6 3s2 3p6 3d10 4s2"),
    31: ("Ga", "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p1"),
    32: ("Ge", "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p2"),
    33: ("As", "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p3"),
    34: ("Se", "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p4"),
    35: ("Br", "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p5"),
    36: ("Kr", "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6"),
}
ORBITAL_L_VALUES = {"s": 0, "p": 1, "d": 2, "f": 3}

# --- Helper Functions for Configuration Management ---


def parse_orbital(part):
    match = re.match(r"(\d+)([spdfgh])(\d+)", part)
    if not match:
        return None
    n, l_char, occupancy = match.groups()
    return int(n), ORBITAL_L_VALUES[l_char], int(occupancy)


def create_ion_config(neutral_config, charge):
    if charge == 0:
        return neutral_config
    if charge < 0:
        raise ValueError("Anions not supported.")
    parts = neutral_config.split()
    orbitals = [(*parse_orbital(p), p) for p in parts if parse_orbital(p)]
    for _ in range(charge):
        if not orbitals:
            break
        orbitals.sort(key=lambda x: (x[0], x[1]), reverse=True)
        n, l, occupancy, original_part = orbitals[0]
        if occupancy > 1:
            orbitals[0] = (
                n,
                l,
                occupancy - 1,
                f"{n}{list(ORBITAL_L_VALUES.keys())[l]}{occupancy - 1}",
            )
        else:
            orbitals.pop(0)
    final_parts = [
        orb[3] for orb in sorted(orbitals, key=lambda x: (x[0], x[1]))
    ]
    return " ".join(final_parts)


def get_occupied_orbitals(config):
    return [
        re.match(r"(\d+[spdfgh])", part).group(1) for part in config.split()
    ]


# --- Integrated Calculation Logic (from facionize.py) ---


def _run_single_pfac_config(element_z, config_string):
    """
    Internal worker to calculate total energy for a single configuration.
    IMPORTANT: This function should only be run inside a separate process
    to ensure the pfac state is clean.
    """
    try:
        import pfac.fac as fac
    except ImportError:
        raise ImportError(
            "The 'pfac' library is required but not installed. Please run 'pip install pfac'."
        )

    # Use PID for unique filenames, which is essential in multiprocessing
    pid = os.getpid()
    config_label = f"worker_config_{pid}"
    binary_file = f"{config_label}.b"
    text_file = f"{config_label}.lev"

    try:
        fac.SetAtom(int(element_z))
        fac.Config(config_label, config_string)
        fac.OptimizeRadial([config_label])
        fac.Structure(binary_file, [config_label])
        fac.PrintTable(binary_file, text_file, 1)

        energy = 0.0
        found = False
        with open(text_file, "r") as f:
            for line in f:
                if line.strip().startswith("E0"):
                    parts = line.split(",")
                    energy = float(parts[1])
                    found = True
                    break
        if not found:
            raise ValueError(
                f"Could not find 'E0' energy in FAC output for config '{config_string}'."
            )
        return energy
    finally:
        # Ensure cleanup of temporary files
        if os.path.exists(binary_file):
            os.remove(binary_file)
        if os.path.exists(text_file):
            os.remove(text_file)
        if "fac" in locals() and hasattr(fac, "CloseSFAC"):
            fac.CloseSFAC()


def create_final_config(initial_config, orbital_to_remove):
    """Generates the final state configuration string after removing one electron."""
    parts = initial_config.split()
    new_parts = []
    found_orbital = False
    pattern = re.compile(r"(\d+[spdfgh])(\d+)")

    for part in parts:
        match = pattern.match(part)
        if not match:
            new_parts.append(part)
            continue

        orbital, occupancy_str = match.groups()
        occupancy = int(occupancy_str)

        if orbital == orbital_to_remove:
            found_orbital = True
            if occupancy > 1:
                new_parts.append(f"{orbital}{occupancy - 1}")
        else:
            new_parts.append(part)

    if not found_orbital:
        raise ValueError(
            f"Orbital '{orbital_to_remove}' not found in config '{initial_config}'."
        )

    return " ".join(new_parts)


# <-- 2. create a wrapper function to run the calculation in an isolated subprocess -->
def _worker_target(queue, func, *args):
    """A target function for the worker process to run."""
    try:
        result = func(*args)
        queue.put(result)
    except Exception as e:
        queue.put(e)  # Put the exception object in the queue on failure


def run_in_worker(func, *args):
    """Runs a function in a separate process and returns its result."""
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_worker_target, args=(queue, func, *args)
    )
    p.start()
    result = queue.get()  # Get result (or exception) from the queue
    p.join()

    if isinstance(result, Exception):
        raise result  # Re-raise the exception in the main process
    return result


def calculate_ionization_energy(element_z, initial_config, orbital_to_remove):
    """
    Calculates ionization energy by finding the energy difference between
    the initial and final states, with each calculation in a separate process.
    """
    final_config = create_final_config(initial_config, orbital_to_remove)

    e_initial = run_in_worker(
        _run_single_pfac_config, element_z, initial_config
    )

    if not final_config.strip():
        e_final = 0.0
    else:
        e_final = run_in_worker(
            _run_single_pfac_config, element_z, final_config
        )

    return e_final - e_initial


# --- Main Execution Block ---


def main():
    """Main function to run the batch calculations and save results."""
    output_basename = "ionization_energies"
    element_z_range = range(1, 37)
    all_results_list = []

    start_time = datetime.now()
    print(
        f"Batch calculation started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"Output files will be named '{output_basename}_data.py'")

    for z in element_z_range:
        symbol, neutral_config = NEUTRAL_CONFIGS[z]
        print(f"\n--- Processing Element: {symbol} (Z={z}) ---")

        element_data = {"element": symbol, "Z": z, "ions": {}}

        for charge in range(z):
            try:
                ion_config = create_ion_config(neutral_config, charge)
                if not ion_config:
                    print(
                        f"  Charge = +{charge}: Skipped (no electrons left)."
                    )
                    continue

                print(f"  Charge = +{charge}, Config = '{ion_config}'")
                occupied_orbitals = get_occupied_orbitals(ion_config)

                ion_results = {
                    "config": ion_config,
                    "ionization_energies_eV": {},
                }
                element_data["ions"][charge] = ion_results

            except Exception as e:
                element_data["ions"][charge] = {
                    "config": "ERROR",
                    "ionization_energies_eV": {"error": str(e)},
                }
                continue

            for orbital in occupied_orbitals:
                print(
                    f"    -> Calculating removal from {orbital}...",
                    end="",
                    flush=True,
                )
                try:
                    energy_val = calculate_ionization_energy(
                        z, ion_config, orbital
                    )
                    energy = round(energy_val, 4)
                    print(f" Done ({energy} eV)")
                    ion_results["ionization_energies_eV"][orbital] = energy

                except Exception as e:
                    error_message = f"ERROR: {e}"
                    print(f" Failed ({e})")
                    ion_results["ionization_energies_eV"][
                        orbital
                    ] = error_message

        all_results_list.append(element_data)

    print("\n--- All calculations complete. Saving results... ---")

    try:
        py_filename = f'{output_basename.replace("-", "_")}_data.py'
        with open(py_filename, "w", encoding="utf-8") as f:
            f.write("#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n\n")
            f.write(
                "# Ionization Energy Data Generated by FAC-code: https://github.com/flexible-atomic-code/fac\n"
            )
            f.write(
                "# This data was generated by the script 'ionization_energies_calculation_FAC.py' script.\n\n"
            )
            f.write("ENERGY_DATA = ")
            f.write(pprint.pformat(all_results_list, indent=2, width=100))
            f.write("\n")
        print(f"Successfully saved results to {py_filename}")
    except Exception as e:
        print(f"Error saving to Python file: {e}")

    end_time = datetime.now()
    print(
        f"\nBatch calculation finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"Total duration: {end_time - start_time}")


if __name__ == "__main__":
    # This is important for multiprocessing on some platforms (Windows, macOS)
    multiprocessing.freeze_support()
    main()
