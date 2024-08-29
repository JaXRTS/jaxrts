import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parents[2] / "src"))

import sys
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QObject, QRect, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import jaxrts
from jaxrts.units import to_array, ureg

# Get the base colors and their names
base_colors = mcolors.BASE_COLORS


def save_state():
    pass


def load_state():
    pass


class ConstantValueInputDialog(QDialog):
    def __init__(self, typ, unit_v):
        super().__init__()

        self.initUI(typ, unit_v)

    def initUI(self, typ, unit_v):
        self.setWindowTitle("Please insert value:")
        self.setFixedSize(400, 100)
        layout = QHBoxLayout()

        self.label = QLabel(typ, self)
        self.label.setMinimumWidth(150)
        self.line_edit = QLineEdit(self)
        self.set_button = QPushButton("Set", self)
        self.set_button.setMaximumWidth(50)
        self.set_button.clicked.connect(self.return_value)
        self.label_u = QLabel(str(unit_v), self)

        layout.addWidget(self.label)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.set_button)
        layout.addWidget(self.label_u)

        self.setLayout(layout)

    def return_value(self):
        self.accept()  # Close the dialog and indicate that it was accepted

    def get_value(self):
        return self.line_edit.text()


class CustomTabWidget(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.opentabs = 1
        self.setTabsClosable(True)
        self.tabCloseRequested.connect(self.close_tab)
        self.tabBarClicked.connect(self.handle_tab_click)

        # Add initial tab and the "+" tab
        self.add_new_tab()
        self.add_plus_tab()

    def add_new_tab(self):
        # Create a new JAXRTSViz widget
        new_tab = JAXRTSViz()
        self.opentabs += 1
        # Insert the new tab before the "+" tab
        index = self.count() - 1
        self.insertTab(index, new_tab, f"Board {self.opentabs - 1}")

        # Set the new tab as the current tab
        self.setCurrentIndex(index)

    def add_plus_tab(self):
        # Add a "+" tab at the end
        plus_tab = QWidget()
        index = self.addTab(plus_tab, "+")
        self.tabBar().setTabButton(index, QTabBar.RightSide, None)
        self.tabBar().setTabButton(index, QTabBar.LeftSide, None)

    def close_tab(self, index):
        if self.tabText(index) != "+":
            self.removeTab(index)

    def handle_tab_click(self, index):
        if self.tabText(index) == "+":
            self.add_new_tab()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JAXRTSViz")
        self.setGeometry(20, 20, 1200, 850)
        # Create a CustomTabWidget
        self.tabs = CustomTabWidget()
        self.setCentralWidget(self.tabs)


class StatusSwitch(QPushButton):
    def __init__(self, parent=None, is_compiled=False):
        super().__init__(parent)
        self.is_compiled = is_compiled
        self.setCheckable(True)
        self.setMinimumWidth(66)
        self.setMinimumHeight(22)

    def paintEvent(self, event):

        label = "Compiled" if self.is_compiled else "Uncompiled"
        bg_color = Qt.green if self.is_compiled else Qt.red

        radius = 10
        width = 32
        center = self.rect().center()

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.translate(center)
        painter.setBrush(QtGui.QColor(0, 0, 0))

        pen = QtGui.QPen(Qt.black)
        pen.setWidth(2)
        painter.setPen(pen)

        painter.drawRoundedRect(
            QRect(-width, -radius, 2 * width, 2 * radius), radius, radius
        )
        painter.setBrush(QtGui.QBrush(bg_color))
        sw_rect = QRect(-radius, -radius, width + radius, 2 * radius)
        if not self.is_compiled:
            sw_rect.moveLeft(-width)
        painter.drawRoundedRect(sw_rect, radius, radius)
        painter.setFont(QFont("Times", 5))
        painter.drawText(sw_rect, Qt.AlignCenter, label)


class EmittingStream(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        if text.strip():  # Avoid emitting empty lines
            self.text_written.emit(">> " + str(text))

    def flush(self):
        pass


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=6, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)


class ConsoleOutputWorker(QThread):
    # Define a signal to send console output updates
    update_console = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._is_running = True

    def run(self):
        pass
        # while self._is_running:
        #     time.sleep(1)
        #     # Simulate console output
        #     self.update_console.emit(f"Console output message {counter}")
        #     counter += 1

    def stop(self):
        self._is_running = False


class CustomToolbar(NavigationToolbar):
    def __init__(self, canvas, parent=None, is_compiled=False):
        super(CustomToolbar, self).__init__(canvas, parent)

        self.is_compiled = is_compiled

        self.compile_status = StatusSwitch(is_compiled=self.is_compiled)
        self.compile_status.setFixedSize(80, 30)
        # self.compile_status.setCheckable(False)

        self.is_compiled = False
        self.addWidget(self.compile_status)
        self.addData = QPushButton("Load Spectrum")

        self.addData.setFixedSize(90, 40)
        self.addWidget(self.addData)

        self.data_norm_to_elastic = QRadioButton("Norm to elastic peak")
        self.data_norm_to_elastic.setFont(QFont(None, 8))
        self.data_norm_to_inelastic = QRadioButton("Norm to inelastic peak")
        self.data_norm_to_inelastic.setFont(QFont(None, 8))

        # Create a button group to ensure only one radio button can be selected at a time
        button_group = QButtonGroup(self)
        button_group.addButton(self.data_norm_to_elastic)
        button_group.addButton(self.data_norm_to_inelastic)

        # Create a layout to hold the radio buttons
        radiobutton_layout = QVBoxLayout()
        self.comboBoxData = QComboBox()
        radiobutton_layout.addWidget(self.comboBoxData)
        radiobutton_layout.addWidget(self.data_norm_to_elastic)
        radiobutton_layout.addWidget(self.data_norm_to_inelastic)

        # Create a widget to contain the layout
        radiobutton_widget = QWidget()
        radiobutton_widget.setLayout(radiobutton_layout)

        # Add the widget with the layout to the toolbar
        self.addWidget(radiobutton_widget)

        self.models_norm_to_elastic = QRadioButton("Norm to elastic peak")
        self.models_norm_to_elastic.setFont(QFont(None, 8))
        self.models_norm_to_inelastic = QRadioButton("Norm to inelastic peak")
        self.models_norm_to_inelastic.setFont(QFont(None, 8))

        # Create a button group to ensure only one radio button can be selected at a time
        button_group = QButtonGroup(self)
        button_group.addButton(self.models_norm_to_elastic)
        button_group.addButton(self.models_norm_to_inelastic)

        # Create a layout to hold the radio buttons
        radiobutton_layout = QVBoxLayout()
        self.comboBoxModels = QComboBox()
        radiobutton_layout.addWidget(self.comboBoxModels)
        radiobutton_layout.addWidget(self.models_norm_to_elastic)
        radiobutton_layout.addWidget(self.models_norm_to_inelastic)

        # Create a widget to contain the layout
        radiobutton_widget = QWidget()
        radiobutton_widget.setLayout(radiobutton_layout)

        # Add the widget with the layout to the toolbar
        self.addWidget(radiobutton_widget)

    def update_combobox_entries(self, dmdict):
        self.comboBoxData.clear()
        self.comboBoxData.addItems(dmdict["Data"])
        self.comboBoxModels.clear()
        self.comboBoxModels.addItems(dmdict["Models"])


class JAXRTSViz(QWidget):

    def __init__(self):
        super().__init__()

        self.base_models = [
            "ionic scattering",
            "free-free scattering",
            "bound-free scattering",
            "free-bound scattering",
            "BM S_ii",
        ]
        self.current_models = []
        self.current_fwhm = 0.0

        self.instrument_function_data = []
        self.spectra_data = []
        self.model_data = []

        self.spectra_model_names = {"Data": [], "Models": []}

        self.current_state = None
        self.current_setup = None

        self.is_compiled = False

        self.offset_elements = 27
        self.elements_counter = 0

        self.atomic_number = {
            v: k for k, v in jaxrts.elements._element_symbols.items()
        }

        # self.setWindowTitle("JAXRTSViz")
        # self.setGeometry(100, 100, 1000, 600)

        # Central widget
        central_widget = QWidget()
        # self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout1 = QHBoxLayout()
        main_layout2 = QHBoxLayout()

        # Store textboxes for later retrieval
        self.textboxes = []
        self.comboBoxesList = []
        self.activeModels = {}
        self.model_dropdown_menu_actions = []

        ##########################################################PLASMA STATE################################################################
        # Create left side layout
        self.left_layout = QVBoxLayout()

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
        text_box3.setObjectName("rho")
        label4 = QLabel(r"g/cm³")
        additional_layout1.addWidget(label4, alignment=QtCore.Qt.AlignLeft)
        self.left_layout.addLayout(additional_layout1)
        self.textboxes.append(text_box3)

        additional_layout2 = QHBoxLayout()
        label4 = QLabel(r"Temperature T=")
        text_box4 = QLineEdit()
        text_box4.setObjectName("T")
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
        self.grid_layout = QGridLayout()
        self.left_layout.addLayout(self.grid_layout)

        # Dropdown Label
        label_dropdown = QLabel("Element")
        label_textbox1 = QLabel(r"Ionization Z")
        label_textbox2 = QLabel(r"Fraction f")
        label_blank = QLabel("")
        label_blank.setMinimumWidth(40)

        # Layout for initial dropdown and textboxes
        self.dropdown_layouts = []
        self.initial_row_layout = QHBoxLayout()

        self.initial_row_layout.addWidget(label_dropdown)
        self.initial_row_layout.addWidget(label_textbox1)
        self.initial_row_layout.addWidget(label_textbox2)
        self.initial_row_layout.addWidget(label_blank)

        self.left_layout.addLayout(self.initial_row_layout)

        combo_box = QComboBox()
        counter = self.elements_counter
        combo_box.setObjectName("Element" + str(counter))
        combo_box.setMaximumWidth(70)
        combo_box.addItems(list(jaxrts.elements._element_symbols.values()))

        text_box1 = QLineEdit()
        text_box1.setObjectName("Zf_Element" + str(self.elements_counter))
        text_box1.setMaximumWidth(70)
        text_box2 = QLineEdit()
        text_box2.setObjectName("f_Element" + str(self.elements_counter))
        text_box2.setMaximumWidth(70)

        self.textboxes.append(text_box1)
        self.textboxes.append(text_box2)
        self.comboBoxesList.append(combo_box)

        label_blank = QLabel("")
        label_blank.setMinimumWidth(20)

        row_layout = QHBoxLayout()
        row_layout.addWidget(combo_box, alignment=QtCore.Qt.AlignLeft)
        row_layout.addWidget(text_box1, alignment=QtCore.Qt.AlignLeft)
        row_layout.addWidget(text_box2, alignment=QtCore.Qt.AlignLeft)
        row_layout.addWidget(label_blank, alignment=QtCore.Qt.AlignLeft)

        row_layout.setObjectName("Element" + str(self.elements_counter))

        self.left_layout.addLayout(row_layout)

        # Button layout for adding new rows
        self.button_layout = QHBoxLayout()

        # Initial row + button for adding new rows
        add_row_button = QPushButton("+")
        add_row_button.setFixedSize(40, 40)
        add_row_button.clicked.connect(self.add_new_row)
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

        self.Allmodels = {}

        # for name in list(models.keys()):
        for obj_name in dir(jaxrts.models):
            if "__class__" in dir(obj_name):
                attributes = getattr(jaxrts.models, obj_name)
                if "allowed_keys" in dir(attributes):
                    key = getattr(attributes, "allowed_keys")
                    if ("Model" not in obj_name) & ("model" not in obj_name):
                        for k in key:
                            try:
                                self.Allmodels[k].append(obj_name)
                            except:
                                self.Allmodels[k] = [obj_name]

        for mod in list(self.Allmodels.keys()):
            row_layout2 = QHBoxLayout()
            if mod in self.base_models:
                label = QLabel()
                combo_box = QComboBox()
                combo_box.setObjectName("Model-" + mod)
                combo_box.setMaximumWidth(200)
                combo_box.addItems(list(self.Allmodels[mod]))
                combo_box.currentTextChanged.connect(self.set_compile_off)
                label.setText(mod + str(":"))
                row_layout2.addWidget(label)
                row_layout2.addWidget(combo_box)
                self.comboBoxesList.append(combo_box)

            self.left_layout.addLayout(row_layout2)

        # Create a dropdown menu
        self.model_dropdown_menu = QMenu(self)

        for mod in list(self.Allmodels.keys()):
            if mod not in self.base_models:
                self.add_model_menu_element(mod)
        # Add actions to the dropdown menu

        # Initial row + button for adding new rows
        self.button_layout2 = QHBoxLayout()
        self.model_add_button = QPushButton("+")
        self.model_add_button.setFixedSize(40, 40)
        self.model_add_button.clicked.connect(self.add_new_model)
        self.button_layout2.addWidget(self.model_add_button)
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
        text_box3.setObjectName("Energy")
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
        text_box4.setObjectName("Theta")
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
        text_box4.setObjectName("E1")
        text_box4.setMaximumWidth(35)  # Match width to label
        setup_layout3.addWidget(label4, alignment=QtCore.Qt.AlignLeft)
        setup_layout3.addWidget(text_box4, alignment=QtCore.Qt.AlignLeft)
        labelE1 = QLabel("eV")
        setup_layout3.addWidget(labelE1, alignment=QtCore.Qt.AlignLeft)
        self.textboxes.append(text_box4)

        label4 = QLabel(r"E2=")
        text_box4 = QLineEdit()
        text_box4.setObjectName("E2")
        text_box4.setMaximumWidth(35)  # Match width to label
        setup_layout3.addWidget(label4, alignment=QtCore.Qt.AlignLeft)
        setup_layout3.addWidget(text_box4, alignment=QtCore.Qt.AlignLeft)
        labelE2 = QLabel("eV")
        setup_layout3.addWidget(labelE2, alignment=QtCore.Qt.AlignLeft)
        self.textboxes.append(text_box4)

        label4 = QLabel(r"N=")
        text_box4 = QLineEdit()
        text_box4.setObjectName("npoints")
        text_box4.setMaximumWidth(50)  # Match width to label
        setup_layout3.addWidget(label4, alignment=QtCore.Qt.AlignLeft)
        setup_layout3.addWidget(text_box4, alignment=QtCore.Qt.AlignLeft)
        self.textboxes.append(text_box4)
        self.left_layout.addLayout(setup_layout3)

        setup_Ins = QHBoxLayout()
        headerIns = QLabel("Instrument Function:", self)
        headerIns.setStyleSheet("font-weight: bold;")
        setup_Ins.addWidget(headerIns, alignment=QtCore.Qt.AlignLeft)
        self.left_layout.addLayout(setup_Ins)

        setup_layout2 = QHBoxLayout()
        label4 = QLabel(r"fwhm=")
        text_box4 = QLineEdit()
        text_box4.setObjectName("fwhm")
        text_box4.textChanged.connect(self.set_compile_off)
        text_box4.setMaximumWidth(30)
        setup_layout2.addWidget(label4, alignment=QtCore.Qt.AlignLeft)
        setup_layout2.addWidget(text_box4, alignment=QtCore.Qt.AlignLeft)
        label5 = QLabel(r"eV")
        setup_layout2.addWidget(label5, alignment=QtCore.Qt.AlignLeft)
        self.left_layout.addLayout(setup_layout2)
        self.textboxes.append(text_box4)
        self.loadIns = QPushButton("Load")
        self.loadIns.setFont(QFont(None, 10))
        self.loadIns.setFixedSize(40, 30)
        self.loadIns.clicked.connect(self.load_instrument)
        setup_layout2.addWidget(self.loadIns, alignment=QtCore.Qt.AlignLeft)

        self.labelIns = QLabel("")
        self.labelIns.setObjectName("Instrument")
        self.labelIns.setMinimumWidth(50)
        setup_layout2.addWidget(self.labelIns, alignment=QtCore.Qt.AlignLeft)

        self.delIns = QPushButton("x")
        self.delIns.setFixedSize(20, 20)
        self.delIns.setVisible(False)
        self.delIns.clicked.connect(self.del_instrument)
        setup_layout2.addWidget(self.delIns, alignment=QtCore.Qt.AlignLeft)

        setup_layout2.addWidget(self.labelIns, alignment=QtCore.Qt.AlignLeft)
        self.left_layout.addLayout(setup_layout2)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: gray;")
        self.left_layout.addWidget(line)

        self.probe_button_layout = QHBoxLayout()
        self.probe_button = QPushButton("Probe")
        self.probe_button.setFixedSize(120, 60)
        self.probe_button.setFont(QFont(None, 12))
        self.probe_button.clicked.connect(self.toggle_probe)
        
        self.save_button = QPushButton("Save")
        self.save_button.setFixedSize(60, 30)
        self.save_button.clicked.connect(self.save_state)

        self.load_button = QPushButton("Load")
        self.load_button.setFixedSize(60, 30)
        self.load_button.clicked.connect(self.load_state)

        self.toggle_probe_button = QCheckBox()
        self.toggle_probe_button.stateChanged.connect(
            lambda state, button=self.probe_button: self.on_toggle_probe(
                state, button
            )
        )
        self.toggle_probe_button.setText("Toggle Probe")

        self.probe_button_layout.addWidget(
            self.probe_button, alignment=QtCore.Qt.AlignCenter
        )
        self.probe_button_layout.addWidget(self.save_button, alignment=QtCore.Qt.AlignRight
        )
        self.probe_button_layout.addWidget(
            self.load_button, alignment=QtCore.Qt.AlignRight
        )
        self.probe_button_layout.addWidget(
            self.toggle_probe_button, alignment=QtCore.Qt.AlignRight
        )
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

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas.ax.patch.set_facecolor("#f0f0f0")
        self.canvas.fig.set_facecolor("#f0f0f0")
        self.toolbar = CustomToolbar(
            self.canvas, self, is_compiled=self.is_compiled
        )
        self.toolbar.addData.clicked.connect(self.add_spectra)
        # self.figure, self.ax = plt.subplots(figsize=(6, 6))  # Adjust the width and height as needed

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

        main_layout1.addLayout(
            self.left_layout, 1
        )  # Adjusted proportion for self.left_layout

        main_layout1.addWidget(
            right_widget, 3
        )  # Adjusted proportion for right_widget

        # Plasma parameter output console
        self.console_output_pp = QPlainTextEdit()
        self.console_output_pp.setFixedHeight(
            150
        )  # Fixed height for the console area
        self.console_output_pp.setPlaceholderText("Plasma parameters")
        self.console_output_pp.setReadOnly(True)  # Make it read-only
        self.console_output_pp.setStyleSheet(
            """
            QPlainTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                font-size: 8pt; /* Adjust font size if needed */
                line-height: 3px; /* Adjust line height */
                padding: 3px;
            }
        """
        )
        main_layout2.addWidget(self.console_output_pp, 1)

        # Info output console
        self.console_output = QPlainTextEdit()
        self.console_output.setFixedHeight(
            150
        )  # Fixed height for the console area
        self.console_output.setPlaceholderText("Output")
        self.console_output.setReadOnly(True)  # Make it read-only
        self.console_output.setStyleSheet(
            """
            QPlainTextEdit {
                background-color: #282823;
                border: 2px solid black;
                color: white;
                font-size: 8pt; /* Adjust font size if needed */
                line-height: 5px; /* Adjust line height */
                padding: 3px;
            }
        """
        )

        # Create and start the console output worker thread
        self.console_worker = ConsoleOutputWorker()
        self.console_worker.update_console.connect(self.update_console_output)
        self.console_worker.start()

        # Redirect stdout and stderr
        # sys.stdout = EmittingStream(text_written=self.update_console_output)
        # sys.stderr = EmittingStream(text_written=self.update_console_output)

        main_layout2.addWidget(self.console_output, 2)

        main_layout.addLayout(main_layout1)
        main_layout.addLayout(main_layout2)
        self.setLayout(main_layout)

    def add_model_menu_element(self, action_name):
        action = QAction(action_name, self)
        action.setObjectName(action_name)
        self.model_dropdown_menu_actions.append(action)
        action.triggered.connect(
            lambda checked, name=action_name: self.model_menu_element_triggered(
                name
            )
        )
        self.model_dropdown_menu.addAction(action)
        
    def save_state(self):
        state = {}
        for textbox in self.textboxes:
            state[textbox.objectName()] = textbox.text()
        for combobox in self.comboBoxesList:
            state[combobox.objectName()] = combobox.currentText()
        state['toggle_probe'] = self.toggle_probe_button.isChecked()
        
        file_name, _ = QFileDialog.getSaveFileName(self, "Save State", "", "JSON Files (*.json)")
        if file_name:
            with open(file_name, 'w') as f:
                json.dump(state, f)
            print(f"State saved to {file_name}")

    def load_state(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load State", "", "JSON Files (*.json)")
        if file_name:
            with open(file_name, 'r') as f:
                state = json.load(f)
            
            for textbox in self.textboxes:
                if textbox.objectName() in state:
                    textbox.setText(state[textbox.objectName()])
            
            for combobox in self.comboBoxesList:
                if combobox.objectName() in state:
                    index = combobox.findText(state[combobox.objectName()])
                    if index >= 0:
                        combobox.setCurrentIndex(index)
            
            if 'toggle_probe' in state:
                self.toggle_probe_button.setChecked(state['toggle_probe'])
            
            print(f"State loaded from {file_name}")
            self.is_compiled = False
            self.toolbar.compile_status.repaint()

    def add_spectra(self):
        fdialog = QFileDialog.Options()
        fdialog |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            "",
            "All Files (*);;Text Files (*.txt)",
            options=fdialog,
        )
        if file_name:
            try:
                energy, intensity = np.genfromtxt(
                    file_name, delimiter=",", unpack=True, skip_header=1
                )
            except:
                print("Please select a .csv file!")
                return

            self.spectra_model_names["Data"].append(
                file_name.split("/")[-1].split(".txt")[0]
            )
            global base_colors
            self.toolbar.update_combobox_entries(self.spectra_model_names)
            self.toolbar.repaint()
            self.spectra_data.append([energy, intensity])
            self.canvas.ax.scatter(
                energy,
                intensity,
                label=file_name.split("/")[-1].split(".txt")[0],
                s=3,
                color=list(base_colors.values())[len(self.spectra_data)],
            )
            self.canvas.ax.legend()
            self.canvas.draw()

    def model_menu_element_triggered(self, action_name):

        self.is_compiled = False
        self.base_models.append(action_name)

        for mod in list(self.Allmodels.keys()):
            row_layout2 = QHBoxLayout()
            if mod == action_name:
                label = QLabel()
                combo_box = QComboBox()
                combo_box.setObjectName("Model-" + mod)
                combo_box.setMaximumWidth(200)
                combo_box.addItems(list(self.Allmodels[mod]))
                combo_box.currentTextChanged.connect(self.set_compile_off)
                label.setText(mod + str(":"))

                row_layout2.addWidget(label)
                row_layout2.addWidget(combo_box)
                self.comboBoxesList.append(combo_box)

            self.left_layout.insertLayout(
                self.left_layout.count() - 12, row_layout2
            )
            for act in self.model_dropdown_menu_actions:
                if act.objectName() == action_name:
                    self.model_dropdown_menu.removeAction(act)

        self.offset_elements += 13

    def add_new_model(self):
        # Get the position of the button
        button_position = self.model_add_button.mapToGlobal(
            self.model_add_button.rect().bottomLeft()
        )

        # Show the dropdown menu at the button's position
        self.model_dropdown_menu.exec_(button_position)

    def set_compile_off(self):

        self.models_changed = True
        self.fwhm_changed = True

        for cb in self.comboBoxesList:
            try:
                if "Model" in cb.objectName():
                    self.models_changed = (
                        cb.currentText() not in self.current_models
                    ) and (len(self.current_models) > 0)
                    if self.models_changed:
                        break
            except:
                pass

        for textb in self.textboxes:
            try:
                if textb.objectName() == "fwhm":
                    if textb.text() != "":
                        try:
                            self.fwhm_changed = float(textb.text()) != float(
                                self.current_fwhm
                            )
                            if self.fwhm_changed:
                                break
                        except:
                            self.fwhm_changed = True

            except:
                pass

        if (not self.fwhm_changed) and (not self.models_changed):
            self.is_compiled = True
        else:
            self.is_compiled = False

        self.toolbar.compile_status.repaint()

    def del_instrument(self):
        for textb in self.textboxes:
            try:
                if textb.objectName() == "fwhm":
                    textb.setEnabled(True)
            except:
                pass
        self.instrument_function_data = []
        self.labelIns.setText("")
        self.delIns.setVisible(False)
        self.delIns.repaint()

    def load_instrument(self):
        fdialog = QFileDialog.Options()
        fdialog |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            "",
            "All Files (*);;Text Files (*.txt)",
            options=fdialog,
        )
        if file_name:
            try:
                energy, intensity = np.genfromtxt(
                    file_name, delimiter=",", unpack=True, skip_header=1
                )
            except:
                print("Please select a .csv file!")
                return
            self.instrument_function_data = [energy, intensity]
            self.delIns.setVisible(True)
            self.delIns.repaint()
            self.labelIns.setFont(QFont("Times", 6))
            self.labelIns.setText(file_name.split("/")[-1][:5] + "...csv")

        for textb in self.textboxes:
            try:
                if textb.objectName() == "fwhm":
                    textb.setEnabled(False)
            except:
                pass

    def add_new_row(self):

        self.elements_counter += 1

        for cb in self.comboBoxesList:
            try:
                if "ionic scattering" in cb.objectName():
                    cb.clear()
                    cb.addItems(
                        [
                            m
                            for m in list(self.Allmodels["ionic scattering"])
                            if "HNC" in m
                        ]
                    )

                # Currently BM does not work with MC ...
                if "free-free scattering" in cb.objectName():
                    cb.clear()
                    cb.addItems(
                        [
                            m
                            for m in list(
                                self.Allmodels["free-free scattering"]
                            )
                            # if "BornMermin" not in m
                        ]
                    )
            except:
                pass
        counter = self.elements_counter
        combo_box = QComboBox()
        combo_box.setObjectName("Element" + str(counter))
        combo_box.setMaximumWidth(70)
        combo_box.addItems(list(jaxrts.elements._element_symbols.values()))

        text_box1 = QLineEdit()
        text_box1.setObjectName("Zf_Element" + str(self.elements_counter))
        text_box1.setMaximumWidth(70)
        text_box2 = QLineEdit()
        text_box2.setObjectName("f_Element" + str(self.elements_counter))
        text_box2.setMaximumWidth(70)
        delete_row_button = QPushButton("-")
        delete_row_button.setFixedSize(20, 20)
        delete_row_button.clicked.connect(
            lambda x: self.remove_row(x, counter)
        )
        self.button_layout.addWidget(delete_row_button)
        row_layout = QHBoxLayout()
        row_layout.addWidget(combo_box, alignment=QtCore.Qt.AlignLeft)
        row_layout.addWidget(text_box1, alignment=QtCore.Qt.AlignLeft)
        row_layout.addWidget(text_box2, alignment=QtCore.Qt.AlignLeft)
        row_layout.addWidget(delete_row_button, alignment=QtCore.Qt.AlignLeft)

        row_layout.setObjectName("Element" + str(self.elements_counter))

        self.left_layout.insertLayout(
            self.left_layout.count() - self.offset_elements, row_layout
        )  # Insert before the button layout
        self.dropdown_layouts.append(row_layout)

        self.textboxes.append(text_box1)
        self.textboxes.append(text_box2)
        self.comboBoxesList.append(combo_box)

        self.elements_counter += 1
        self.is_compiled = False
        self.toolbar.compile_status.repaint()

    def remove_row(self, x, k):

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

        self.elements_counter -= 1

        if self.elements_counter == 1:
            for cb in self.comboBoxesList:
                try:
                    if "ionic scattering" in cb.objectName():
                        cb.clear()
                        cb.addItems(
                            [
                                m
                                for m in list(
                                    self.Allmodels["ionic scattering"]
                                )
                            ]
                        )
                    if "free-free scattering" in cb.objectName():
                        cb.clear()
                        cb.addItems(
                            [
                                m
                                for m in list(
                                    self.Allmodels["free-free scattering"]
                                )
                            ]
                        )
                except:
                    pass

        self.elements_counter = max(1, self.elements_counter)

    def on_toggle_probe(self, state, button):
        button.setEnabled(not self.toggle_probe_button.isChecked())

    def update_console_output(self, message):
        self.console_output.appendPlainText(message)

    def closeEvent(self, event):
        # Restore stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        super().closeEvent(event)

    def open_value_input_dialog(self, typ, unit_v):
        dialog = ConstantValueInputDialog(typ, unit_v)
        if dialog.exec_() == QDialog.Accepted:
            value = dialog.get_value()
            return value
        else:
            print("A value is required.")
            return None

    def toggle_probe(self):

        probing_values_and_models = {}
        elements = []
        Z_free = []
        n_frac = []

        for textb in self.textboxes:
            try:
                probing_values_and_models[textb.objectName()] = float(
                    textb.text()
                )
            except Exception as err:
                if err == ValueError:
                    print("Please check entries!")
                    return

        for cb in self.comboBoxesList:
            try:
                if "Model" in cb.objectName():
                    probing_values_and_models[cb.objectName()[6:]] = (
                        cb.currentText()
                    )
                else:
                    probing_values_and_models[cb.objectName()] = (
                        cb.currentText()
                    )
            except Exception as err:
                if err == ValueError:
                    print("Please check entries!")
                    return

        for key in list(probing_values_and_models.keys()):
            if ("Element" in key) & (not ("_" in key)):
                try:
                    elements.append(probing_values_and_models[key])
                    Z_free.append(
                        float(probing_values_and_models["Zf_" + key])
                    )
                    n_frac.append(float(probing_values_and_models["f_" + key]))
                except ValueError:
                    print("Please check entries!")
                    return
                except KeyError:
                    print("Please check entries!")
                    return
        if jnp.abs(jnp.sum(jnp.array(n_frac)) - 1.0) >= 0.001:
            print("Please check that the density fractions add up to 1!")
            return

        try:
            self.current_fwhm = probing_values_and_models["fwhm"]
        except KeyError:
            self.current_fwhm = 0.0

        self.probe_button.setEnabled(False)

        self.probe_button.repaint()

        def instrument_data(E):
            return (
                jnp.interp(
                    x=E.m_as(1 / (1 * ureg.second)),
                    xp=jnp.array(self.instrument_function_data[0]),
                    fp=jnp.array(self.instrument_function_data[1]),
                    left=None,
                    right=None,
                    period=None,
                )
                * 1
                * ureg.second
            )

        instrument = (
            partial(
                jaxrts.instrument_function.instrument_gaussian,
                sigma=float(probing_values_and_models["fwhm"])
                * 1
                * ureg.electron_volt
                / ureg.hbar
                / (2 * jnp.sqrt(2 * jnp.log(2))),
            )
            if len(self.instrument_function_data) == 0
            else instrument_data
        )

        if self.is_compiled:

            energy = (
                jnp.linspace(
                    probing_values_and_models["E1"],
                    probing_values_and_models["E2"],
                    int(probing_values_and_models["npoints"]),
                )
                * ureg.electron_volt
            )

            self.current_state.Z_free = to_array(Z_free)
            self.current_state.mass_density = (
                jnp.array(
                    [probing_values_and_models["rho"] * f for f in n_frac]
                )
                * 1
                * ureg.gram
                / ureg.centimeter**3
            )

            self.current_state.T_e = (
                probing_values_and_models["T"]
                * len(n_frac)
                * 1
                * ureg.electron_volt
                / ureg.k_B
            )

            self.current_setup.energy = (
                probing_values_and_models["Energy"] * 1 * ureg.electron_volt
            )
            self.current_setup.scattering_angle = (
                probing_values_and_models["Theta"] * 1 * ureg.degrees
            )
            self.current_setup.measured_energy = energy
            self.current_setup.instrument = jax.tree_util.Partial(instrument)

            try:
                I = self.current_state.probe(self.current_setup)
            except AttributeError:
                self.probe_button.setEnabled(True)
                print("This didn't work.")
                return

            self.probe_button.setEnabled(True)

            # self.canvas.ax.clear()
            self.model_data.append(
                [
                    (self.current_setup.measured_energy).m_as(
                        ureg.electron_volt
                    ),
                    I.m_as(ureg.second),
                ]
            )
            self.spectra_model_names["Models"].append(
                "Model " + str(len(self.model_data))
            )
            global base_colors
            self.toolbar.update_combobox_entries(self.spectra_model_names)
            self.toolbar.repaint()
            self.canvas.ax.plot(
                (self.current_setup.measured_energy).m_as(ureg.electron_volt),
                I.m_as(ureg.second) / jnp.max(I.m_as(ureg.second)),
                label="Model " + str(len(self.model_data)),
                color=list(base_colors.keys())[len(self.model_data)],
            )
            self.canvas.ax.set_xlabel("E [eV]")
            self.canvas.ax.set_ylabel("I [1/s]")
            self.canvas.ax.legend()
            self.canvas.draw()

        else:

            print("Compiling ...")
            self.console_output.repaint()

            self.current_state = jaxrts.PlasmaState(
                ions=[jaxrts.elements.Element(e) for e in elements],
                Z_free=Z_free,
                mass_density=jnp.array(
                    [probing_values_and_models["rho"] * f for f in n_frac]
                )
                * 1
                * ureg.gram
                / ureg.centimeter**3,
                T_e=probing_values_and_models["T"]
                * 1
                * ureg.electron_volt
                / ureg.k_B,
            )

            # sharding = jax.sharding.PositionalSharding(jax.devices())
            energy = (
                jnp.linspace(
                    probing_values_and_models["E1"],
                    probing_values_and_models["E2"],
                    int(probing_values_and_models["npoints"]),
                )
                * ureg.electron_volt
            )
            # sharded_energy = jax.device_put(energy, sharding)
            #         # sharded_energy = energy

            self.current_setup = jaxrts.setup.Setup(
                probing_values_and_models["Theta"] * 1 * ureg.degrees,
                probing_values_and_models["Energy"] * 1 * ureg.electron_volt,
                energy,
                # ureg(f"{central_energy} eV")
                # + jnp.linspace(-700, 200, 2000) * ureg.electron_volt,
                instrument,
            )
            self.current_models = []
            for typ in self.base_models:
                if "Constant" in probing_values_and_models[typ]:

                    if "length" in typ:
                        unit_value = ureg.angstrom
                    elif ("ipd" in typ) or ("potential" in typ):
                        unit_value = ureg.electron_volt
                    elif "lfc" in typ:
                        unit_value = ureg.dimensionless

                    value = self.open_value_input_dialog(
                        typ="Value for " + str(typ) + ":", unit_v=unit_value
                    )
                    if value == None:
                        print(f"A value for {typ} is required!")
                        return
                    if float(value):
                        self.current_state[typ] = eval(
                            "jaxrts.models." + probing_values_and_models[typ]
                        )(float(value) * unit_value)
                    else:
                        print("Please check entry!")
                        return
                else:
                    if typ=="bound-free scattering":
                        self.current_state[typ] = eval(
                            "jaxrts.models." + probing_values_and_models[typ]
                        )(r_k = 1.0)
                    else:
                        self.current_state[typ] = eval(
                            "jaxrts.models." + probing_values_and_models[typ]
                        )()  
                self.current_models.append(probing_values_and_models[typ])

            try:
                I = self.current_state.probe(self.current_setup)
            except AttributeError:
                print("This didn't work.")
                return

            self.is_compiled = True
            self.model_data.append(
                [
                    (self.current_setup.measured_energy).m_as(
                        ureg.electron_volt
                    ),
                    I.m_as(ureg.second),
                ]
            )
            self.spectra_model_names["Models"].append(
                "Model " + str(len(self.model_data))
            )
            self.toolbar.update_combobox_entries(self.spectra_model_names)
            self.toolbar.repaint()
            self.canvas.ax.plot(
                (self.current_setup.measured_energy).m_as(ureg.electron_volt),
                I.m_as(ureg.second) / jnp.max(I.m_as(ureg.second)),
                label="Model " + str(len(self.model_data)),
                color=list(base_colors.keys())[len(self.model_data)],
            )
            self.canvas.ax.set_xlabel("E [eV]")
            self.canvas.ax.set_ylabel("I [1/s]")
            self.canvas.ax.legend()
            self.canvas.draw()

            # Update plasma parameters

            n_e = (
                self.current_state.n_e.m_as(1 / ureg.centimeter**3)
                if self.current_state is not None
                else np.nan
            )
            try:
                kappa_sc = (
                    self.current_state.screening_length.m_as(1 / ureg.angstrom)
                    if self.current_state is not None
                    else np.nan
                )
            except:
                kappa_sc = np.nan
            w_p = (
                jaxrts.plasma_physics.plasma_frequency(
                    self.current_state.n_e
                ).m_as(1 / ureg.second)
                if self.current_state is not None
                else np.nan
            )
            theta_e = (
                jaxrts.plasma_physics.degeneracy_param(
                    self.current_state.n_e, self.current_state.T_e
                ).m_as(ureg.dimensionless)
                if self.current_state is not None
                else np.nan
            )
            gamma_ee = (
                jaxrts.plasma_physics.coupling_param(
                    -1, -1, self.current_state.n_e, self.current_state.T_e
                )
                if self.current_state is not None
                else np.nan
            )
            compton = (
                jaxrts.plasma_physics.compton_energy(
                    probing_values_and_models["Energy"]
                    * 1
                    * ureg.electron_volt,
                    probing_values_and_models["Theta"] * 1 * ureg.degrees,
                ).m_as(ureg.electron_volt)
                if self.current_state is not None
                else np.nan
            )

            self.console_output_pp.setPlainText(
                "Plasma parameters:\n\n"
                + "➢  Free electron density: "
                + "{:0.3E}".format(n_e)
                + " g/cm³\n"
                + "➢  screening length: "
                + "{:0.3E}".format(kappa_sc, 3)
                + " 1/Å\n"
                + "➢  plasma frequency: "
                + "{:0.3E}".format(w_p, 3)
                + " 1/s\n"
                + "➢  electron degeneracy parameter: "
                + "{:0.3f}".format(theta_e, 3)
                + "\n"
                + "➢  electron coupling parameter: "
                + "{:0.3f}".format(gamma_ee, 3)
                + " \n"
                + "➢  compton energy: "
                + "{:0.3E}".format(compton, 3)
                + " eV\n"
            )

            self.canvas.ax.patch.set_facecolor("#f0f0f0")
            self.canvas.fig.set_facecolor("#f0f0f0")

            self.probe_button.setEnabled(True)
            self.is_compiled = True
            self.toolbar.compile_status.repaint()


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
