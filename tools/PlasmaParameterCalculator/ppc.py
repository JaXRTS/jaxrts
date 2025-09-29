import tkinter as tk
from tkinter import ttk, messagebox
import math
import jaxrts

ureg = jaxrts.ureg
import jpu

import numpy as np

import jaxrts.plasma_physics as pp

eps0 = 1 * ureg.epsilon_0
e_charge = 1 * ureg.elementary_charge
k_B = 1 * ureg.boltzmann_constant
m_e = 1 * ureg.electron_mass
h = 1 * ureg.planck_constant
m_u = 1 * ureg.u

ELEMENTS_Z = {jaxrts.elements._element_symbols[k]: k for k in range(1, 37)}

ELEMENTS = {
    jaxrts.elements._element_symbols[k]: jaxrts.elements._element_masses[k]
    * 1
    * ureg.u
    for k in range(1, 37)
}


def compute_plasma_parameters(rho, element_symbol, T_e, Z_avg):
    if element_symbol not in ELEMENTS:
        raise ValueError("Unknown-Element.")
    if rho <= 0 or T_e <= 0 or Z_avg <= 0:
        raise ValueError("All entries must be > 0.")
    if Z_avg > ELEMENTS_Z[element_symbol]:
        raise ValueError("Ionization greater than ion charge!")

    mass_ion = ELEMENTS[element_symbol]

    kT = 1 * ureg.boltzmann_constant * T_e

    n_i = (rho / mass_ion).to("1/cm^3")
    n_e = (Z_avg * n_i).to("1/cm^3")

    lambda_D = pp.Debye_Hueckel_screening_length(n_e, T_e, Z_avg)

    N_D = (4 / 3) * math.pi * n_e * lambda_D**3

    a_ws_e = (3 / (4 * math.pi * n_e)) ** (1 / 3)
    a_ws_i = (3 / (4 * math.pi * (n_e / Z_avg))) ** (1 / 3)
    r_dB_e = pp.therm_de_broglie_wl(T_e)

    Gamma_ee = (e_charge**2) / (4 * math.pi * eps0 * a_ws_e * kT)
    Gamma_ii = ((Z_avg * e_charge) ** 2) / (4 * math.pi * eps0 * a_ws_i * kT)
    Gamma_ei = (Z_avg * e_charge**2) / (4 * math.pi * eps0 * a_ws_i * kT)

    E_Fe = pp.fermi_energy(n_e)
    E_Fi = pp.fermi_energy(n_i)

    Theta_e = kT / E_Fe
    Theta_i = kT / E_Fi

    omega_pe = pp.plasma_frequency(n_e)
    f_pe = omega_pe / (2 * math.pi)

    return {
        "rho": rho,
        "n_i": n_i,
        "n_e": n_e,
        "lambda_D": lambda_D.to("nm"),
        "N_D": N_D.to(ureg.dimensionless),
        "a_ws_e": a_ws_e.to("nm"),
        "a_ws_i": a_ws_i.to("nm"),
        "r_dB_e": r_dB_e.to("nm"),
        "Gamma_ee": Gamma_ee.to(ureg.dimensionless),
        "Gamma_ii": Gamma_ii.to(ureg.dimensionless),
        "Gamma_ei": Gamma_ei.to(ureg.dimensionless),
        "E_Fe": E_Fe.to("eV"),
        "E_Fi": E_Fi.to("eV"),
        "Theta_e": Theta_e.to(ureg.dimensionless),
        "Theta_i": Theta_i.to(ureg.dimensionless),
        "omega_pe": omega_pe.to("rad/s"),
        "f_pe": f_pe.to("Hz"),
    }


class PlasmaGUI:
    def __init__(self, root):
        self.root = root
        root.title("Plasma Parameters Calculator")
        root.geometry("900x500")
        root.resizable(False, False)

        left_frame = ttk.Frame(root, padding=(10, 10))
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        right_frame = ttk.Frame(root, padding=(10, 10))
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(
            left_frame,
            text="Plasma conditions:",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, columnspan=2, pady=(0, 8))

        row = 1
        ttk.Label(left_frame, text="Mass density ρ [g/cm³]").grid(
            row=row, column=0, sticky=tk.W, pady=4
        )
        self.rho_entry = ttk.Entry(left_frame, width=20)
        self.rho_entry.grid(row=row, column=1, pady=4)
        self.rho_entry.insert(0, "1.0")

        row += 1
        ttk.Label(left_frame, text="Element").grid(
            row=row, column=0, sticky=tk.W, pady=4
        )
        self.element_combo = ttk.Combobox(
            left_frame,
            values=list(ELEMENTS.keys()),
            width=17,
            state="readonly",
        )
        self.element_combo.grid(row=row, column=1, pady=4)
        self.element_combo.set("H")

        row += 1
        ttk.Label(left_frame, text="Temperature T [eV]").grid(
            row=row, column=0, sticky=tk.W, pady=4
        )
        self.T_entry = ttk.Entry(left_frame, width=20)
        self.T_entry.grid(row=row, column=1, pady=4)
        self.T_entry.insert(0, "10")

        row += 1
        ttk.Label(left_frame, text="Ionization Z").grid(
            row=row, column=0, sticky=tk.W, pady=4
        )
        self.Z_entry = ttk.Entry(left_frame, width=20)
        self.Z_entry.grid(row=row, column=1, pady=4)
        self.Z_entry.insert(0, "1")

        row += 1
        ttk.Button(
            left_frame, text="Calculate", command=self.on_calculate
        ).grid(row=row, column=0, columnspan=2, pady=(12, 0), ipadx=10)

        ttk.Label(
            right_frame, text="Results:", font=("Segoe UI", 11, "bold")
        ).pack(anchor=tk.NW)
        self.output_label = tk.Label(
            right_frame,
            text="",
            justify=tk.LEFT,
            anchor="nw",
            bg="#dddddd",
            fg="black",
            font=("Courier New", 10),
            bd=2,
            relief=tk.SUNKEN,
        )
        self.output_label.pack(
            fill=tk.BOTH, expand=True, padx=(0, 10), pady=(6, 10)
        )

    def on_calculate(self):
        try:
            rho = float(self.rho_entry.get())
            element = self.element_combo.get()
            T_e = float(self.T_entry.get())
            Z = float(self.Z_entry.get())
            results = compute_plasma_parameters(
                rho * 1 * ureg.gram / ureg.cc,
                element,
                T_e * ureg.electron_volt / ureg.boltzmann_constant,
                Z,
            )
            self.display_results(results, element, T_e, Z)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_results(self, r, element, T_e, Z):
        lines = []
        lines.append(
            f"Element: {element}    T = {T_e:.3g} eV    Z_free = {Z:.3g}"
        )
        lines.append(
            f"Mass density ρ = {r['rho']:.3gP}".replace(
                "gram/cubic_centimeter", "g/cm³"
            )
        )
        lines.append("-" * 60)
        lines.append(
            f"Ion number density n_i = {r['n_i']:.3eP}".replace(
                "1/centimeter", "1/cm"
            )
        )
        lines.append(
            f"Electron number density n_e = {r['n_e']:.3eP}".replace(
                "1/centimeter", "1/cm"
            )
        )
        lines.append("")
        lines.append(f"Debye length λ_D = {r['lambda_D']:.3eP}")
        lines.append(
            f"Particles in a Debye sphere N_D = {r['N_D']:.3eP} (≈ {np.round(r['N_D'])})".replace(
                "dimensionless", ""
            )
        )
        lines.append("")
        lines.append(
            f"Wigner-Seitz radius (electrons) a_ws_e = {r['a_ws_e']:.3eP}".replace(
                "nanometer", "nm"
            )
        )
        lines.append(
            f"Wigner-Seitz radius (ions) a_ws_i = {r['a_ws_i']:.3eP}".replace(
                "nanometer", "nm"
            )
        )
        lines.append(
            f"Thermal de Broglie wavelength (e) r_dB = {r['r_dB_e']:.3eP}".replace(
                "nanometer", "nm"
            )
        )
        lines.append("")
        lines.append(
            f"electron-electron coupling parameter Γ_ee = {r['Gamma_ee']:.3eP}".replace(
                "dimensionless", ""
            )
        )
        lines.append(
            f"ion-ion coupling parameter Γ_ii = {r['Gamma_ii']:.3eP}".replace(
                "dimensionless", ""
            )
        )
        lines.append(
            f"electron-ion coupling parameter Γ_ei = {r['Gamma_ei']:.3eP}".replace(
                "dimensionless", ""
            )
        )
        lines.append("")
        lines.append(
            f"Fermi energy E_F (electrons) = {r['E_Fe']:.3eP}".replace(
                "electron_volt", "eV"
            )
        )
        lines.append(
            f"Fermi energy E_F (ions) = {r['E_Fi']:.3eP}".replace(
                "electron_volt", "eV"
            )
        )
        lines.append(
            f"Degeneracy parameter Θ (electrons)= {r['Theta_e']:.3fP}".replace(
                "dimensionless", ""
            )
        )
        lines.append(
            f"Degeneracy parameter Θ (ions) = {r['Theta_i']:.3fP}".replace(
                "dimensionless", ""
            )
        )
        lines.append("")
        lines.append(
            f"Plasma frequency ω_pe = {r['omega_pe']:.3eP}".replace(
                "radian/second", "rad/s"
            )
        )
        lines.append(
            f"Plasma frequency f_pe = {r['f_pe']:.3eP}".replace("hertz", "Hz")
        )
        lines.append("-" * 60)

        def determine_level(x):

            upper = 1.5
            lower = 0.5

            if x > upper:
                return "strongly"
            if lower <= x <= upper:
                return "moderately"
            if x < lower:
                return "weakly"

        level = {1: "weakly", 2: "moderately", 3: "strongly"}
        coupling_i = determine_level(r["Gamma_ii"])
        coupling_e = determine_level(r["Gamma_ee"])
        degen_i = determine_level(1 / r["Theta_i"])
        degen_e = determine_level(1 / r["Theta_e"])

        lines.append(
            f">>> Ions are {coupling_i} coupled and {degen_i} degenerate."
        )
        lines.append(
            f">>> Electron are {coupling_e} coupled and {degen_i} degenerate."
        )

        self.output_label.config(text="\n".join(lines))


if __name__ == "__main__":
    root = tk.Tk()

    style = ttk.Style()
    style.theme_use("default")  # try "default", "clam", "alt", "classic"

    app = PlasmaGUI(root)
    root.mainloop()
