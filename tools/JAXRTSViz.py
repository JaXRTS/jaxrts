import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1] / "src"))

import inspect

import jaxrts

import sys
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QWidget,
    QLabel,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QCheckBox,
    QSizePolicy,
    QLineEdit,
    QPushButton,
    QMainWindow,
)
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure

elements_counter = 1


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class LabelDropdownApp(QMainWindow):

    def __init__(self):
        super().__init__()

        global elements_counter
        self.setWindowTitle("JAXRTSViz")
        self.setGeometry(100, 100, 1000, 600)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QHBoxLayout(central_widget)

        # Store textboxes for later retrieval
        self.textboxes = []

        ##########################################################PLASMA STATE################################################################
        # Create left side layout
        self.left_layout = QVBoxLayout()
        main_layout.addLayout(
            self.left_layout, 1
        )  # Adjusted proportion for self.left_layout

        # Header for Section 1
        header = QLabel("Plasma State Settings", self)
        header.setStyleSheet("font-weight: bold;")
        self.left_layout.addWidget(header)

        # Additional label-textbox pairs under each other
        additional_layout1 = QHBoxLayout()
        label3 = QLabel(r"Mass density ρ=")
        text_box3 = QLineEdit()
        text_box3.setMaximumWidth(
            label3.sizeHint().width()
        )  # Match width to label
        additional_layout1.addWidget(label3, alignment=QtCore.Qt.AlignLeft)
        additional_layout1.addWidget(text_box3, alignment=QtCore.Qt.AlignLeft)
        label4 = QLabel(r"g/cm³")
        additional_layout1.addWidget(label4, alignment=QtCore.Qt.AlignLeft)
        self.left_layout.addLayout(additional_layout1)
        self.textboxes.append(text_box3)

        additional_layout2 = QHBoxLayout()
        label4 = QLabel(r"Temperature T=")
        text_box4 = QLineEdit()
        text_box4.setMaximumWidth(
            label4.sizeHint().width()
        )  # Match width to label
        additional_layout2.addWidget(label4, alignment=QtCore.Qt.AlignLeft)
        additional_layout2.addWidget(text_box4, alignment=QtCore.Qt.AlignLeft)
        label5 = QLabel(r"eV")
        additional_layout2.addWidget(label5, alignment=QtCore.Qt.AlignLeft)
        self.left_layout.addLayout(additional_layout2)
        self.textboxes.append(text_box4)

        # Labels and textboxes above the initial dropdown-textbox-textbox
        grid_layout = QGridLayout()
        self.left_layout.addLayout(grid_layout)

        # Dropdown Label
        label_dropdown = QLabel("Element")
        combo_box = QComboBox()
        combo_box.setMaximumWidth(70)
        combo_box.addItems(list(jaxrts.elements._element_symbols.values()))
        grid_layout.addWidget(label_dropdown, 0, 0, 1, 1)  # Row 0, Col 0
        grid_layout.addWidget(
            combo_box, 1, 0, 1, 1, alignment=QtCore.Qt.AlignLeft
        )  # Row 1, Col 0
        elements_counter += 1
        # Textbox 1 Label
        label_textbox1 = QLabel(r"Ionization Z")
        text_box1 = QLineEdit()
        text_box1.setMaximumWidth(70)
        grid_layout.addWidget(
            label_textbox1, 0, 1, 1, 1, alignment=QtCore.Qt.AlignLeft
        )  # Row 0, Col 1
        grid_layout.addWidget(
            text_box1, 1, 1, 1, 1, alignment=QtCore.Qt.AlignLeft
        )  # Row 1, Col 1
        self.textboxes.append(text_box1)

        # Textbox 2 Label
        label_textbox2 = QLabel()
        label_textbox2.setText(r"Fraction f")
        text_box2 = QLineEdit()
        text_box2.setMaximumWidth(70)
        grid_layout.addWidget(
            label_textbox2, 0, 2, 1, 1, alignment=QtCore.Qt.AlignLeft
        )  # Row 0, Col 2
        grid_layout.addWidget(
            text_box2, 1, 2, 1, 1, alignment=QtCore.Qt.AlignLeft
        )  # Row 1, Col 2
        self.textboxes.append(text_box2)

        label_blank = QLabel()
        label_blank.setText(r"")
        label_blank.setMaximumWidth(20)
        grid_layout.addWidget(
            text_box2, 0, 3, 1, 1, alignment=QtCore.Qt.AlignLeft
        )  # Row 1, Col 2

        # Layout for initial dropdown and textboxes
        self.dropdown_layouts = []
        self.initial_row_layout = QHBoxLayout()
        self.initial_row_layout.addWidget(
            combo_box, alignment=QtCore.Qt.AlignLeft
        )

        label_blank = QLabel()
        label_blank.setText(r"")
        label_blank.setMaximumWidth(50)
        self.initial_row_layout.addWidget(
            label_blank, alignment=QtCore.Qt.AlignLeft
        )

        self.initial_row_layout.addWidget(
            text_box1, alignment=QtCore.Qt.AlignLeft
        )
        self.initial_row_layout.addWidget(
            text_box2, alignment=QtCore.Qt.AlignLeft
        )

        self.left_layout.addLayout(self.initial_row_layout)

        # Button layout for adding new rows
        self.button_layout = QHBoxLayout()

        def add_new_row():
            global elements_counter

            combo_box = QComboBox()
            combo_box.setMaximumWidth(70)
            combo_box.addItems(list(jaxrts.elements._element_symbols.values()))
            text_box1 = QLineEdit()
            text_box1.setMaximumWidth(70)
            text_box2 = QLineEdit()
            text_box2.setMaximumWidth(70)
            delete_row_button = QPushButton("-")
            delete_row_button.setFixedSize(20, 20)

            counter = elements_counter
            delete_row_button.clicked.connect(lambda x: remove_row(x, counter))
            self.button_layout.addWidget(delete_row_button)
            row_layout = QHBoxLayout()
            row_layout.addWidget(combo_box, alignment=QtCore.Qt.AlignLeft)
            row_layout.addWidget(text_box1, alignment=QtCore.Qt.AlignLeft)
            row_layout.addWidget(text_box2, alignment=QtCore.Qt.AlignLeft)
            row_layout.addWidget(
                delete_row_button, alignment=QtCore.Qt.AlignLeft
            )

            row_layout.setObjectName("Element" + str(elements_counter))

            self.left_layout.insertLayout(
                self.left_layout.count() - 24, row_layout
            )  # Insert before the button layout
            self.dropdown_layouts.append(row_layout)

            self.textboxes.append(text_box1)
            self.textboxes.append(text_box2)

            elements_counter += 1

        def remove_row(x, k):
            global elements_counter
            for layout in self.dropdown_layouts:
                if layout.objectName() == "Element" + str(k):
                    self.dropdown_layouts.remove(layout)
                    while layout.count():
                        item = layout.takeAt(0)
                        widget = item.widget()
                        if widget:
                            widget.deleteLater()
                        else:
                            sublayout = item.layout()
                            if sublayout:
                                self.remove_row(sublayout)

            elements_counter -= 1
            elements_counter = max(1, elements_counter)

        # Initial row + button for adding new rows
        add_row_button = QPushButton("+")
        add_row_button.setFixedSize(40, 40)
        add_row_button.clicked.connect(add_new_row)
        self.button_layout.addWidget(add_row_button)
        self.left_layout.addLayout(self.button_layout)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: gray;")
        self.left_layout.addWidget(line)

        # Spacer to push sections to the top
        ##########################################################PLASMA STATE################################################################
        ############################################################MODELS####################################################################

        # Header
        header = QLabel("Models Settings", self)
        header.setStyleSheet("font-weight: bold;")
        self.left_layout.addWidget(header)

        # Model layout box
        # row_layout2 = QHBoxLayout()

        def add_new_model():
            pass

        models = {}

        # for name in list(models.keys()):
        for obj_name in dir(jaxrts.models):
            if "__class__" in dir(obj_name):
                attributes = getattr(jaxrts.models, obj_name)
                if "allowed_keys" in dir(attributes):
                    key = getattr(attributes, "allowed_keys")
                    if ("Model" not in obj_name) & ("model" not in obj_name):
                        for k in key:
                            try:
                                models[k].append(obj_name)
                            except:
                                models[k] = [obj_name]

        base_models = [
            "ionic scattering",
            "free-free scattering",
            "bound-free scattering",
            "free-bound scattering",
        ]

        for mod in list(models.keys()):
            row_layout2 = QHBoxLayout()
            if mod in base_models:
                label = QLabel()
                combo_box = QComboBox()
                combo_box.setMaximumWidth(200)
                combo_box.addItems(list(models[mod]))
                label.setText(mod + str(":"))
                row_layout2.addWidget(label)
                row_layout2.addWidget(combo_box)

            self.left_layout.addLayout(row_layout2)

        # Initial row + button for adding new rows
        self.button_layout2 = QHBoxLayout()
        add_row_button = QPushButton("+")
        add_row_button.setFixedSize(40, 40)
        add_row_button.clicked.connect(add_new_model)
        self.button_layout2.addWidget(add_row_button)
        self.left_layout.addLayout(self.button_layout2)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: gray;")
        self.left_layout.addWidget(line)

        ############################################################MODELS#########################################################
        ############################################################SETUP##########################################################

        header = QLabel("Setup Settings", self)
        header.setStyleSheet("font-weight: bold;")
        self.left_layout.addWidget(header)

        setup_layout1 = QHBoxLayout()
        label3 = QLabel(r"Probe Energy E=")
        text_box3 = QLineEdit()
        text_box3.setMaximumWidth(
            label3.sizeHint().width()
        )  # Match width to label
        setup_layout1.addWidget(label3, alignment=QtCore.Qt.AlignLeft)
        setup_layout1.addWidget(text_box3, alignment=QtCore.Qt.AlignLeft)
        label4 = QLabel(r"eV")
        setup_layout1.addWidget(label4, alignment=QtCore.Qt.AlignLeft)
        self.left_layout.addLayout(setup_layout1)
        self.textboxes.append(text_box3)

        setup_layout2 = QHBoxLayout()
        label4 = QLabel(r"Scattering Angle θ=")
        text_box4 = QLineEdit()
        text_box4.setMaximumWidth(
            label4.sizeHint().width()
        )  # Match width to label
        setup_layout2.addWidget(label4, alignment=QtCore.Qt.AlignLeft)
        setup_layout2.addWidget(text_box4, alignment=QtCore.Qt.AlignLeft)
        label5 = QLabel(r"°")
        setup_layout2.addWidget(label5, alignment=QtCore.Qt.AlignLeft)
        self.left_layout.addLayout(setup_layout2)
        self.textboxes.append(text_box4)

        setup_layout3 = QHBoxLayout()
        label4 = QLabel(r"E1=")
        text_box4 = QLineEdit()
        text_box4.setMaximumWidth(50)  # Match width to label
        setup_layout3.addWidget(label4, alignment=QtCore.Qt.AlignLeft)
        setup_layout3.addWidget(text_box4, alignment=QtCore.Qt.AlignLeft)
        label5 = QLabel(r"°")
        setup_layout3.addWidget(label5, alignment=QtCore.Qt.AlignLeft)
        self.textboxes.append(text_box4)

        label4 = QLabel(r"E2=")
        text_box4 = QLineEdit()
        text_box4.setMaximumWidth(50)  # Match width to label
        setup_layout3.addWidget(label4, alignment=QtCore.Qt.AlignLeft)
        setup_layout3.addWidget(text_box4, alignment=QtCore.Qt.AlignLeft)
        self.textboxes.append(text_box4)

        label4 = QLabel(r"N=")
        text_box4 = QLineEdit()
        text_box4.setMaximumWidth(50)  # Match width to label
        setup_layout3.addWidget(label4, alignment=QtCore.Qt.AlignLeft)
        setup_layout3.addWidget(text_box4, alignment=QtCore.Qt.AlignLeft)
        self.textboxes.append(text_box4)

        self.left_layout.addLayout(setup_layout3)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: gray;")
        self.left_layout.addWidget(line)

        self.probe_button_layout = QHBoxLayout()
        self.probe_button = QPushButton("Probe")
        self.probe_button.setFixedSize(70, 60)
        self.probe_button.clicked.connect(self.print_textbox_values)

        self.toggle_probe_button = QCheckBox()
        self.toggle_probe_button.stateChanged.connect(lambda state, button=self.probe_button: self.on_toggle_probe(state, button))
        self.toggle_probe_button.setText("Toggle Probe")

        self.probe_button_layout.addWidget(self.probe_button, alignment=QtCore.Qt.AlignCenter)
        self.probe_button_layout.addWidget(self.toggle_probe_button, alignment=QtCore.Qt.AlignRight)
        self.left_layout.addLayout(self.probe_button_layout)
        self.left_layout.addStretch(1)

        ############################################################SETUP##########################################################
        # Create right side layout for canvas
        right_layout = QVBoxLayout()
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        self.canvas = MplCanvas(self, width=5, height=6, dpi=100)
        right_layout.addWidget(self.canvas)

        main_layout.addWidget(
            right_widget, 3
        )  # Adjusted proportion for right_widget

        self.plot_initial_graph()

    def on_toggle_probe(self, state, button):
        button.setEnabled(
            not self.toggle_probe_button.isChecked()
        )

    def toggle_probe(self):
        pass

    def plot_initial_graph(self):
        self.canvas.axes.plot([0, 1, 2, 3], [10, 1, 20, 3])
        self.canvas.axes.set_title("Initial Plot")
        self.canvas.draw()

    def print_textbox_values(self):
        for textbox in self.textboxes:
            print(textbox.text())


def main():
    app = QApplication(sys.argv)
    window = LabelDropdownApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
