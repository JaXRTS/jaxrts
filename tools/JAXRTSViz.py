import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1] / "src"))

import inspect
from functools import partial

import numpy as np

import jax.numpy as jnp
import jaxrts
import jax
import sys
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QWidget,
    QLabel,
    QPlainTextEdit,
    QComboBox,
    QMenu,
    QAction,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QCheckBox,
    QSizePolicy,
    QLineEdit,
    QPushButton,
    QMainWindow,
)
from PyQt5.QtGui import QTextCursor, QTextBlockFormat, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QRect
from PyQt5 import QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import time

from jaxrts.units import ureg, Quantity, to_array


base_models = [
    "ionic scattering",
    "free-free scattering",
    "bound-free scattering",
    "free-bound scattering",
    "BM S_ii"
]
current_models = []
current_fwhm = 0.0

current_state = None
current_setup = None

is_compiled = False

offset_elements = 27
elements_counter = 1

atomic_number = {v: k for k, v in jaxrts.elements._element_symbols.items()}

class StatusSwitch(QPushButton):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setMinimumWidth(66)
        self.setMinimumHeight(22)

    def paintEvent(self, event):
        global is_compiled
        label = "Compiled" if is_compiled else "Uncompiled"
        bg_color = Qt.green if is_compiled else Qt.red

        radius = 10
        width = 32
        center = self.rect().center()

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.translate(center)
        painter.setBrush(QtGui.QColor(0,0,0))

        pen = QtGui.QPen(Qt.black)
        pen.setWidth(2)
        painter.setPen(pen)

        painter.drawRoundedRect(QRect(-width, -radius, 2*width, 2*radius), radius, radius)
        painter.setBrush(QtGui.QBrush(bg_color))
        sw_rect = QRect(-radius, -radius, width + radius, 2*radius)
        if not is_compiled:
            sw_rect.moveLeft(-width)
        painter.drawRoundedRect(sw_rect, radius, radius)
        painter.setFont(QFont('Times', 5))
        painter.drawText(sw_rect, Qt.AlignCenter, label)

class EmittingStream(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        if text.strip():  # Avoid emitting empty lines
            self.text_written.emit(str(text))

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
        counter = 0
        # while self._is_running:
        #     time.sleep(1)
        #     # Simulate console output
        #     self.update_console.emit(f"Console output message {counter}")
        #     counter += 1

    def stop(self):
        self._is_running = False
        
class CustomToolbar(NavigationToolbar):
    def __init__(self, canvas, parent=None):
        super(CustomToolbar, self).__init__(canvas, parent)

        self.compile_status = StatusSwitch()
        self.compile_status.setFixedSize(80, 30)
        # self.compile_status.setCheckable(False)
        global is_compiled
        is_compiled = False
        self.addWidget(self.compile_status)
        self.addData = QPushButton("Load Spectrum")
        self.addData.setFixedSize(90, 40)
        self.addWidget(self.addData)
class JAXRTSViz(QMainWindow):

    def __init__(self):
        super().__init__()

        global elements_counter
        global base_models
        global offset_elements
        self.setWindowTitle("JAXRTSViz")
        self.setGeometry(100, 100, 1000, 600)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

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
        grid_layout = QGridLayout()
        self.left_layout.addLayout(grid_layout)

        # Dropdown Label
        label_dropdown = QLabel("Element")
        combo_box1 = QComboBox()
        combo_box1.setObjectName("Element" + str(elements_counter))
        combo_box1.setMaximumWidth(70)
        combo_box1.addItems(list(jaxrts.elements._element_symbols.values()))
        grid_layout.addWidget(label_dropdown, 0, 0, 1, 1)  # Row 0, Col 0
        grid_layout.addWidget(
            combo_box1, 1, 0, 1, 1, alignment=QtCore.Qt.AlignLeft
        )  # Row 1, Col 0
        self.comboBoxesList.append(combo_box1)
        # Textbox 1 Label
        label_textbox1 = QLabel(r"Ionization Z")
        text_box1 = QLineEdit()
        text_box1.setObjectName("Zf_Element" + str(elements_counter))
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
        text_box2.setObjectName("f_Element" + str(elements_counter))
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
            text_box2, 0, 2, 1, 1, alignment=QtCore.Qt.AlignLeft
        )  # Row 1, Col 2

        # Layout for initial dropdown and textboxes
        self.dropdown_layouts = []
        self.initial_row_layout = QHBoxLayout()
        self.initial_row_layout.addWidget(
            combo_box1, alignment=QtCore.Qt.AlignLeft
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
            if mod in base_models:
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
            if mod not in base_models:
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
        text_box4.setMaximumWidth(50)  # Match width to label
        setup_layout3.addWidget(label4, alignment=QtCore.Qt.AlignLeft)
        setup_layout3.addWidget(text_box4, alignment=QtCore.Qt.AlignLeft)
        self.textboxes.append(text_box4)

        label4 = QLabel(r"E2=")
        text_box4 = QLineEdit()
        text_box4.setObjectName("E2")
        text_box4.setMaximumWidth(50)  # Match width to label
        setup_layout3.addWidget(label4, alignment=QtCore.Qt.AlignLeft)
        setup_layout3.addWidget(text_box4, alignment=QtCore.Qt.AlignLeft)
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
        setup_Ins .addWidget(headerIns, alignment=QtCore.Qt.AlignLeft)
        self.left_layout.addLayout(setup_Ins)
        
        setup_layout2 = QHBoxLayout()
        label4 = QLabel(r"fwhm=")
        text_box4 = QLineEdit()
        text_box4.setObjectName("fwhm")
        text_box4.textChanged.connect(self.set_compile_off)
        text_box4.setMaximumWidth(50)
        setup_layout2.addWidget(label4, alignment=QtCore.Qt.AlignLeft)
        setup_layout2.addWidget(text_box4, alignment=QtCore.Qt.AlignLeft)
        label5 = QLabel(r"eV")
        setup_layout2.addWidget(label5, alignment=QtCore.Qt.AlignLeft)
        self.left_layout.addLayout(setup_layout2)
        self.textboxes.append(text_box4)
        self.loadIns = QPushButton("Load")
        self.loadIns.setFixedSize(50, 30)
        self.loadIns.clicked.connect(self.load_instrument)
        setup_layout2.addWidget(self.loadIns, alignment=QtCore.Qt.AlignLeft)
    
        labelIns = QLabel("")
        labelIns.setObjectName("Instrument")
        setup_layout2.addWidget(labelIns, alignment=QtCore.Qt.AlignLeft)
        
        self.delIns = QPushButton("-")
        self.delIns.setFixedSize(20, 20)
        self.delIns.clicked.connect(self.del_instrument)
        setup_layout2.addWidget(self.delIns, alignment=QtCore.Qt.AlignLeft)
        
        setup_layout2.addWidget(labelIns, alignment=QtCore.Qt.AlignLeft)
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

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas.ax.patch.set_facecolor('#f0f0f0')
        self.canvas.fig.set_facecolor('#f0f0f0')
        self.toolbar = CustomToolbar(self.canvas, self)
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
        self.console_output_pp.setFixedHeight(150)  # Fixed height for the console area
        self.console_output_pp.setPlaceholderText("Plasma parameters")
        self.console_output_pp.setReadOnly(True)  # Make it read-only
        self.console_output_pp.setStyleSheet("""
            QPlainTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                font-size: 8pt; /* Adjust font size if needed */
                line-height: 3px; /* Adjust line height */
                padding: 3px;
            }
        """)
        main_layout2.addWidget(self.console_output_pp, 1)
        
        # Info output console
        self.console_output = QPlainTextEdit()
        self.console_output.setFixedHeight(150)  # Fixed height for the console area
        self.console_output.setPlaceholderText("Output")
        self.console_output.setReadOnly(True)  # Make it read-only
        self.console_output.setStyleSheet("""
            QPlainTextEdit {
                background-color: #282823;
                border: 2px solid black;
                color: white;
                font-size: 8pt; /* Adjust font size if needed */
                line-height: 5px; /* Adjust line height */
                padding: 3px;
            }
        """)
        
        # Create and start the console output worker thread
        self.console_worker = ConsoleOutputWorker()
        self.console_worker.update_console.connect(self.update_console_output)
        self.console_worker.start()

        # Redirect stdout and stderr
        sys.stdout = EmittingStream(text_written=self.update_console_output)
        sys.stderr = EmittingStream(text_written=self.update_console_output)
        
        main_layout2.addWidget(self.console_output, 2)
        
        main_layout.addLayout(main_layout1)
        main_layout.addLayout(main_layout2)
        
    def add_model_menu_element(self, action_name):
        action = QAction(action_name, self)
        action.setObjectName(action_name)
        self.model_dropdown_menu_actions.append(action)
        action.triggered.connect(lambda checked, name=action_name: self.model_menu_element_triggered(name))
        self.model_dropdown_menu.addAction(action)

    def model_menu_element_triggered(self, action_name):

        global is_compiled
        global base_models
        is_compiled = False
        base_models.append(action_name)
        
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
    
            self.left_layout.insertLayout(self.left_layout.count() - 12, row_layout2)
            for act in self.model_dropdown_menu_actions:
                if act.objectName() == action_name:
                    self.model_dropdown_menu.removeAction(act)
                    
        global offset_elements
        offset_elements += 13
        
    def add_new_model(self):
        # Get the position of the button
        button_position = self.model_add_button.mapToGlobal(self.model_add_button.rect().bottomLeft())

        # Show the dropdown menu at the button's position
        self.model_dropdown_menu.exec_(button_position)
        
    def set_compile_off(self):
        global is_compiled
        global current_fwhm
        global current_models
        
        models_changed = True
        fwhm_changed = True

        for cb in self.comboBoxesList:
            try:
                if "Model" in cb.objectName(): 
                    models_changed = (cb.currentText() not in current_models) and (len(current_models) > 0)
                    if(models_changed):
                        break
            except:
                pass
        
        for textb in self.textboxes:
            try:
                if textb.objectName() == "fwhm":
                    if(textb.text() != ""):
                        try:
                            fwhm_changed = float(textb.text()) != float(current_fwhm)
                            if(fwhm_changed):
                                break
                        except:
                            fwhm_changed = True
                    
            except:
                pass

        if ((not fwhm_changed) and (not models_changed)):
            is_compiled = True
        else:
            is_compiled = False

        self.toolbar.compile_status.repaint()
         
    def del_instrument(self):
        for textb in self.textboxes:
            try:
                if textb.objectName() == "fwhm":
                    textb.setEnabled(
                True
                    )
            except:
                pass
        
    def load_instrument(self):
        for textb in self.textboxes:
            try:
                if textb.objectName() == "fwhm":
                    textb.setEnabled(
                False
                    )
            except:
                pass
            

    def add_new_row(self):
        global elements_counter
        global is_compiled
        
        elements_counter += 1
        counter = elements_counter
        combo_box = QComboBox()
        combo_box.setObjectName("Element" + str(counter))
        combo_box.setMaximumWidth(70)
        combo_box.addItems(list(jaxrts.elements._element_symbols.values()))
        
        text_box1 = QLineEdit()
        text_box1.setObjectName("Zf_Element" + str(elements_counter))
        text_box1.setMaximumWidth(70)
        text_box2 = QLineEdit()
        text_box2.setObjectName("f_Element" + str(elements_counter))
        text_box2.setMaximumWidth(70)
        delete_row_button = QPushButton("-")
        delete_row_button.setFixedSize(20, 20)
        delete_row_button.clicked.connect(lambda x: self.remove_row(x, counter))
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
            self.left_layout.count() - offset_elements, row_layout
        )  # Insert before the button layout
        self.dropdown_layouts.append(row_layout)

        self.textboxes.append(text_box1)
        self.textboxes.append(text_box2)
        self.comboBoxesList.append(combo_box)

        elements_counter += 1
        is_compiled = False
        self.toolbar.compile_status.repaint()

    def remove_row(self, x, k):
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

    def on_toggle_probe(self, state, button):
        button.setEnabled(
            not self.toggle_probe_button.isChecked()
        )
    
        
    def update_console_output(self, message):
        self.console_output.appendPlainText(message)

    def closeEvent(self, event):
        # Restore stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        super().closeEvent(event)

    def toggle_probe(self):
        
        global current_state
        global current_setup
        global is_compiled
        global base_models
        global current_models
        
        probing_values_and_models = {}
        elements = []
        Z_free = []
        n_frac = []
        
        for textb in self.textboxes:
            try:
                probing_values_and_models[textb.objectName()] = float(textb.text())
            except ValueError as err:
                print("Please check entries!")
                return
            
        for cb in self.comboBoxesList:
            try:
                if "Model" in cb.objectName():
                    probing_values_and_models[cb.objectName()[6:]] = cb.currentText()
                else:
                    probing_values_and_models[cb.objectName()] = cb.currentText()
            except ValueError as err:
                print("Please check entries!")
                return
            
        for key in list(probing_values_and_models.keys()):
            if ("Element" in key) & (not ("_" in key)):
                try:
                    elements.append(probing_values_and_models[key])
                    Z_free.append(float(probing_values_and_models["Zf_"+ key]))
                    n_frac.append(float(probing_values_and_models["f_"+ key]))
                except ValueError as err:
                    print("Please check entries!")
                    return
        if(jnp.abs(jnp.sum(jnp.array(n_frac)) - 1.0) >= 0.001):
            print("Please check that the density fractions add up to 1!")
            return  
        
        global current_fwhm
        current_fwhm = probing_values_and_models["fwhm"]
        
        self.probe_button.setEnabled(
            False
        )
        
        self.probe_button.repaint()
        
        if is_compiled:
            
            energy = (
                jnp.linspace(probing_values_and_models["E1"], probing_values_and_models["E2"], int(probing_values_and_models["npoints"])) * ureg.electron_volt
            )
            
            current_state.Z_free = to_array(Z_free)
            current_state.mass_density = to_array(probing_values_and_models["rho"] * ureg.gram / ureg.centimeter**3 * jnp.array(n_frac))
            current_state.T_e=to_array(probing_values_and_models["T"] * ureg.electron_volt / ureg.k_B)
            current_state.ions = [jaxrts.elements.Element(e) for e in elements]
            
            current_setup.energy = probing_values_and_models["Energy"] * 1 * ureg.electron_volt
            current_setup.scattering_angle = probing_values_and_models["Theta"] * 1 * ureg.degrees
            current_setup.measured_energy = energy
            
            try:
                I = current_state.probe(current_setup)
            except AttributeError as err:
                self.probe_button.setEnabled(
                    True
                )
                print("This didn't work.")
                return
            
            self.probe_button.setEnabled(
                True
            )
            
            self.canvas.ax.clear()
            self.canvas.ax.plot((current_setup.measured_energy).m_as(ureg.electron_volt), I.m_as(ureg.second))
            self.canvas.ax.set_xlabel("E [eV]")
            self.canvas.ax.set_ylabel("I [1/s]")
            self.canvas.draw()
        
        else:
            current_state = jaxrts.PlasmaState(
                ions=[jaxrts.elements.Element(e) for e in elements],
                Z_free=Z_free,
                mass_density=probing_values_and_models["rho"] * ureg.gram / ureg.centimeter**3 * jnp.array(n_frac),
                T_e=probing_values_and_models["T"] * ureg.electron_volt / ureg.k_B,
            )

            # sharding = jax.sharding.PositionalSharding(jax.devices())
            energy = (
                jnp.linspace(probing_values_and_models["E1"], probing_values_and_models["E2"], int(probing_values_and_models["npoints"])) * ureg.electron_volt
            )
            # sharded_energy = jax.device_put(energy, sharding)
    #         # sharded_energy = energy
            current_setup = jaxrts.setup.Setup(
                probing_values_and_models["Theta"] * 1 * ureg.degrees,
                probing_values_and_models["Energy"] * 1 * ureg.electron_volt,
                energy,
                # ureg(f"{central_energy} eV")
                # + jnp.linspace(-700, 200, 2000) * ureg.electron_volt,
                partial(
                    jaxrts.instrument_function.instrument_gaussian,
                    sigma=float(probing_values_and_models["fwhm"]) * 1 * ureg.electron_volt / ureg.hbar / (2 * jnp.sqrt(2 * jnp.log(2))),
                ),
    )
            current_models = []
            for typ in base_models:
                current_state[typ] = eval("jaxrts.models." + probing_values_and_models[typ])()
                current_models.append(probing_values_and_models[typ])
            
            try:
                I = current_state.probe(current_setup)
            except AttributeError as err:
                print("This didn't work.")
                return
            
            is_compiled= True
            
            self.canvas.ax.plot((current_setup.measured_energy).m_as(ureg.electron_volt), I.m_as(ureg.second))
            self.canvas.ax.set_xlabel("E [eV]")
            self.canvas.ax.set_ylabel("I [1/s]")
            self.canvas.draw()
            
            # Update plasma parameters
            
            n_e = current_state.n_e.m_as(1 / ureg.centimeter**3) if current_state is not None else np.nan
            try:
                kappa_sc = current_state.screening_length.m_as(1 / ureg.angstrom) if current_state is not None else np.nan
            except:
                kappa_sc = np.nan
            w_p = jaxrts.plasma_physics.plasma_frequency(current_state.n_e).m_as(1 / ureg.second) if current_state is not None else np.nan
            theta_e = jaxrts.plasma_physics.degeneracy_param(current_state.n_e, current_state.T_e).m_as(ureg.dimensionless) if current_state is not None else np.nan
            gamma_ee = jaxrts.plasma_physics.coupling_param(-1, -1, current_state.n_e, current_state.T_e) if current_state is not None else np.nan
            compton = jaxrts.plasma_physics.compton_energy(probing_values_and_models["Energy"] * 1 * ureg.electron_volt, probing_values_and_models["Theta"] * 1 * ureg.degrees).m_as(ureg.electron_volt) if current_state is not None else np.nan
            
            self.console_output_pp.setPlainText(
                "Plasma parameters:\n\n"
                + "➢  Free electron density: " + "{:0.3E}".format(n_e) + " g/cm³\n"
                + "➢  screening length: " + "{:0.3E}".format(kappa_sc,3) + " 1/Å\n"
                + "➢  plasma frequency: " + "{:0.3E}".format(w_p, 3) + " 1/s\n"
                + "➢  electron degeneracy parameter: " + "{:0.3f}".format(theta_e, 3) + "\n"
                + "➢  electron coupling parameter: " + "{:0.3f}".format(gamma_ee, 3) + " \n"
                + "➢  compton energy: " + "{:0.3E}".format(compton, 3) + " eV\n"
            )
            
            self.canvas.ax.patch.set_facecolor('#f0f0f0')
            self.canvas.fig.set_facecolor('#f0f0f0')
            
            self.probe_button.setEnabled(
                True
            )
            is_compiled = True
            self.toolbar.compile_status.repaint()
def main():
    app = QApplication(sys.argv)
    window = JAXRTSViz()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
