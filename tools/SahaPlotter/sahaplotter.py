import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFrame,
    QComboBox,
    QCheckBox,
    QSpacerItem,
    QMessageBox,
    QRadioButton,
    QSizePolicy,
    QButtonGroup,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
import matplotlib.pyplot as plt
import numpy as np

from jaxrts.units import ureg
from jaxrts.saha import calculate_mean_free_charge_saha
from jaxrts.plasmastate import PlasmaState
import jaxrts.elements

import jax.numpy as jnp
import jpu.numpy as jnpu
import jpu

import jaxrts

import re

ipd_options = {
    "None": jaxrts.models.Neglect(),
    "Stewart-Pyatt": jaxrts.models.StewartPyattIPD(),
    "Debye-Huckel": jaxrts.models.DebyeHueckelIPD(),
    "Ion-Sphere": jaxrts.models.IonSphereIPD(),
    "Ecker-Kröll": jaxrts.models.EckerKroellIPD(),
    "Pauli-Blocking": jaxrts.models.PauliBlockingIPD(),
}


def create_latex_string(elements):
    tex_string = r"$"

    for element, count in elements.items():
        if count == 1:
            tex_string += f"{element}"
        else:
            tex_string += f"{element}_{{{count}}}"

    tex_string += r"$"

    if tex_string == r"$$":
        return ""
    return tex_string


def chemical_to_list(chem_str):
    elem_list = {}
    res = re.split(r"(\d+)", chem_str)
    try:
        element_names = res.copy()[::2]
        element_names.remove("")
    except:
        element_names = res.copy()[::2]

    try:
        stochio = res.copy()[1::2]
        stochio.remove("")
    except:
        stochio = res.copy()[1::2].copy()

    stochio = [int(st) for st in stochio]

    return dict(zip(element_names, stochio))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Saha-Boltzmann Plotter")
        self.setGeometry(100, 100, 1000, 500)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(5)

        input_layout = QVBoxLayout()
        input_layout.setSpacing(3)

        form_layout = QHBoxLayout()
        label_column = QVBoxLayout()
        input_column = QVBoxLayout()

        self.plotted = False
        self.calc_Tspace = True
        self.logscale = False

        EFLabel = QLabel("Element formula:")
        EFLabel.setStyleSheet("font-size: 14px;")
        label_column.addWidget(EFLabel)
        self.element_input = QLineEdit()
        self.element_input.setPlaceholderText("Enter formula, e.g. C1H1")
        self.element_input.setStyleSheet("font-size: 14px;")
        input_column.addWidget(self.element_input)

        temp_layout = QHBoxLayout()
        self.labelT1 = QLabel("T<sub>1</sub>:")
        self.labelT1.setStyleSheet("font-size: 14px;")
        temp_layout.addWidget(self.labelT1)

        self.T1_input = QLineEdit()
        self.T1_input.setPlaceholderText("e.g. 1eV")
        self.T1_input.setStyleSheet("font-size: 14px;")
        temp_layout.addWidget(self.T1_input)

        self.labelT2 = QLabel("T<sub>2</sub>:")
        self.labelT2.setStyleSheet("font-size: 14px;")
        temp_layout.addWidget(self.labelT2)
        self.T2_input = QLineEdit()
        self.T2_input.setPlaceholderText("e.g. 1000eV")
        self.T2_input.setStyleSheet("font-size: 14px;")
        temp_layout.addWidget(self.T2_input)
        input_column.addLayout(temp_layout)

        self.TRLabel = QLabel("Temperature Range:")
        self.TRLabel.setStyleSheet("font-size: 14px;")
        label_column.addWidget(self.TRLabel)

        self.MDLabel = QLabel("Mass Density:")
        self.MDLabel.setStyleSheet("font-size: 14px;")
        label_column.addWidget(self.MDLabel)
        self.mass_density_input = QLineEdit()
        self.mass_density_input.setPlaceholderText("Density e.g. 1g/cc")
        self.mass_density_input.setStyleSheet("font-size: 14px;")
        input_column.addWidget(self.mass_density_input)

        self.dependency_swap_button = QPushButton("T ↔ ρ")
        self.dependency_swap_button.setFixedSize(50, 30)
        self.dependency_swap_button.clicked.connect(self.set_dependency)
        self.dependency_swap_button.setStyleSheet("font-size: 10px;")

        form_layout.addLayout(label_column)
        form_layout.addLayout(input_column)

        input_layout.addLayout(form_layout)
        left_layout.addLayout(input_layout)

        # Add a separator line
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        left_layout.addWidget(separator1)

        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.setFixedSize(400, 60)
        self.calculate_button.setStyleSheet("font-size: 16px;")
        self.calculate_button.clicked.connect(self.calculate)
        left_layout.addWidget(self.calculate_button)

        # Add a separator line
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        left_layout.addWidget(separator2)

        optionsLabel = QLabel("Options:")
        optionsLabel.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.keep_plots = QCheckBox("Keep plots")
        self.log_scale = QCheckBox("Log scale T-axis")
        self.log_scale.clicked.connect(self.set_scale)

        checkboxes = QHBoxLayout()
        checkboxes.addWidget(optionsLabel)
        checkboxes.addWidget(self.keep_plots)
        checkboxes.addWidget(self.log_scale)
        checkboxes.addWidget(self.dependency_swap_button)
        left_layout.addLayout(checkboxes)

        main_layout.addLayout(left_layout, stretch=2)

        right_layout = QVBoxLayout()
        self.canvas = FigureCanvas(plt.figure())
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        main_layout.addLayout(right_layout, stretch=3)

        self.option_label = QLabel("IPD:")
        self.option_label.setStyleSheet("font-size: 14px;")
        label_column.addWidget(self.option_label)

        self.option_dropdown = QComboBox()
        self.option_dropdown.addItems(
            [
                "None",
                "Stewart-Pyatt",
                "Debye-Huckel",
                "Ion-Sphere",
                "Ecker-Kröll",
                "Pauli-Blocking",
            ]
        )
        self.option_dropdown.currentIndexChanged.connect(
            self.update_selected_ipd
        )
        input_column.addWidget(self.option_dropdown)

        self.selected_ipd = "None"  # Default selection

        form_layout.addLayout(label_column)
        form_layout.addLayout(input_column)

        input_layout.addLayout(form_layout)
        left_layout.addLayout(input_layout)
        main_layout.addLayout(left_layout, stretch=2)

    def update_selected_ipd(self, index):
        self.selected_ipd = self.option_dropdown.itemText(index)

    def set_dependency(self):

        reply = QMessageBox.question(
            self,
            "Warning",
            "This will reset the current plot. Are you sure?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.ax.cla()
            self.canvas.draw()
        else:
            return -1

        self.calc_Tspace = not self.calc_Tspace

        if not self.calc_Tspace:
            self.mass_density_input.setPlaceholderText("Temperature e.g. 1eV")
            self.mass_density_input.setText("")
            self.T1_input.setPlaceholderText("e.g. 1g/cc")
            self.T1_input.setText("")
            self.T2_input.setPlaceholderText("e.g. 100g/cc")
            self.T2_input.setText("")
            self.TRLabel.setText("Density Range:")
            self.MDLabel.setText("Temperature:")
            self.labelT1.setText("ρ<sub>1</sub>")
            self.labelT2.setText("ρ<sub>2</sub>")
            self.log_scale.setText("Log scale ρ-axis")
        else:
            self.mass_density_input.setPlaceholderText(
                "Mass Density e.g. 1g/cc"
            )
            self.mass_density_input.setText("")
            self.T1_input.setPlaceholderText("e.g. 1eV")
            self.T1_input.setText("")
            self.T2_input.setPlaceholderText("e.g. 1000eV")
            self.T2_input.setText("")
            self.MDLabel.setText("Mass Density:")
            self.TRLabel.setText("Temperature Range:")
            self.labelT1.setText("T<sub>1</sub>")
            self.labelT2.setText("T<sub>2</sub>")
            self.log_scale.setText("Log scale T-axis")

    def set_scale(self):

        self.logscale = not self.logscale

        try:
            self.ax.set_xscale("log" if self.logscale else "linear")
            self.canvas.draw()
        except AttributeError as err:
            pass

        if self.logscale:
            self.log_scale.setChecked(True)
            self.log_scale.update()

        if not self.logscale:
            self.log_scale.setChecked(False)
            self.log_scale.update()

    def update_plot(self, data=None, labels=None, title=None):

        if not self.keep_plots.isChecked():
            self.ax.cla()

        xdata = (
            data[0].m_as(ureg.electron_volt / ureg.boltzmann_constant)
            if self.calc_Tspace
            else data[0].m_as(ureg.gram / ureg.cc)
        )

        for ion in range(len(data[1][0])):

            self.ax.plot(
                xdata,
                np.array(data[1])[:, ion],
                label=labels[ion] if labels else "",
            )

        if self.calc_Tspace:
            self.ax.set_xlabel("Temperature [eV]")
        else:
            self.ax.set_xlabel("ρ [g/cc]")

        self.ax.set_ylabel("Mean Charge State")

        self.ax.grid(True, alpha=0.5, color="gray")
        self.ax.set_ylim(0)
        self.ax.set_xlim(np.min(xdata), np.max(xdata))

        self.ax.legend(fontsize=9)
        self.ax.set_title(title, fontsize=11)
        self.canvas.draw()

    def calculate(self):

        if self.calc_Tspace:
            if not self.plotted:
                self.ax = self.canvas.figure.add_subplot(111)
                self.plotted = True

            element_string = self.element_input.text()
            T1 = self.T1_input.text()
            T2 = self.T2_input.text()
            mass_density = self.mass_density_input.text()

            if (
                ureg(T1).to_base_units().units
                == ureg("1eV").to_base_units().units
            ):
                T1 += "/k_B"

            if (
                ureg(T2).to_base_units().units
                == ureg("1eV").to_base_units().units
            ):
                T2 += "/k_B"

            elements = chemical_to_list(element_string)

            try:
                T1 = ureg(T1)
                T2 = ureg(T2)
                mass_density = ureg(mass_density)
                if (
                    (
                        T1.to_base_units().units
                        != ureg("1K").to_base_units().units
                    )
                    or (
                        T2.to_base_units().units
                        != ureg("1K").to_base_units().units
                    )
                    or (
                        mass_density.to_base_units().units
                        != ureg("1g/cc").to_base_units().units
                    )
                ):
                    print("Invalid units for temperature or mass density.")
                    return -1
            except ValueError:
                print(
                    "Invalid input for temperature or mass density. Please enter numeric values."
                )
                return -1

            ions = [jaxrts.Element(el) for el in elements.keys()]
            number_fraction = jnp.array(list(elements.values()))
            number_fraction /= np.sum(number_fraction)

            mass_fraction = jaxrts.helpers.mass_from_number_fraction(
                number_fraction, ions
            )

            plasma_state = jaxrts.PlasmaState(
                ions=ions,
                Z_free=jnp.zeros_like(mass_fraction),
                mass_density=mass_density * mass_fraction,
                T_e=T1,
            )

            plasma_state["ipd"] = ipd_options[self.selected_ipd]

            T_e_plot = jnpu.linspace(T1, T2, 500)
            Zfree = []

            for T_e in T_e_plot:
                plasma_state.T_e = T_e
                Zfree.append(calculate_mean_free_charge_saha(plasma_state))

            labels = list(
                zip(
                    list(elements.keys()),
                    [
                        " ("
                        + str(mass_density.m_as(ureg.gram / ureg.cc))
                        + "g/cc)"
                    ]
                    * len(list(elements.keys())),
                )
            )
            labels = ["".join(str(xs) for xs in x) for x in labels]
            self.update_plot(
                [T_e_plot, Zfree],
                labels,
                title=create_latex_string(chemical_to_list(element_string)),
            )

        else:
            if not self.plotted:
                self.ax = self.canvas.figure.add_subplot(111)
                self.plotted = True

            element_string = self.element_input.text()
            rho1 = self.T1_input.text()
            rho2 = self.T2_input.text()
            temperature = self.mass_density_input.text()

            if (
                ureg(temperature).to_base_units().units
                == ureg("1eV").to_base_units().units
            ):
                temperature += "/k_B"

            elements = chemical_to_list(element_string)

            try:
                rho1 = ureg(rho1)
                rho2 = ureg(rho2)
                temperature = ureg(temperature)
                if (
                    (
                        rho1.to_base_units().units
                        != ureg("1g/cc").to_base_units().units
                    )
                    or (
                        rho2.to_base_units().units
                        != ureg("1g/cc").to_base_units().units
                    )
                    or (
                        temperature.to_base_units().units
                        != ureg("1K").to_base_units().units
                    )
                ):
                    print("Invalid units for temperature or mass density.")
                    return -1
            except ValueError:
                print(
                    "Invalid input for temperature or mass density. Please enter numeric values."
                )
                return -1

            ions = [jaxrts.Element(el) for el in elements.keys()]
            number_fraction = jnp.array(list(elements.values()))
            number_fraction /= np.sum(number_fraction)

            mass_fraction = jaxrts.helpers.mass_from_number_fraction(
                number_fraction, ions
            )

            plasma_state = jaxrts.PlasmaState(
                ions=ions,
                Z_free=jnp.zeros_like(mass_fraction),
                mass_density=rho1 * mass_fraction,
                T_e=temperature,
            )

            plasma_state["ipd"] = ipd_options[self.selected_ipd]

            rho_plot = jnpu.linspace(rho1, rho2, 500)
            Zfree = []

            for rho in rho_plot:
                plasma_state.mass_density = rho * mass_fraction
                Zfree.append(calculate_mean_free_charge_saha(plasma_state))

            labels = list(
                zip(
                    list(elements.keys()),
                    [
                        " ("
                        + str(
                            temperature.m_as(
                                ureg.electron_volt / ureg.boltzmann_constant
                            )
                        )
                        + "eV)"
                    ]
                    * len(list(elements.keys())),
                )
            )
            labels = ["".join(str(xs) for xs in x) for x in labels]

            self.update_plot(
                [rho_plot, Zfree],
                labels,
                title=create_latex_string(chemical_to_list(element_string)),
            )


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
