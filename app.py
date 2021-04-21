import sys
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg

from ui.main_window import Ui_Form

class Main(qtw.QWidget, Ui_Form):
	"""
	handles user interaction, loads data and updates GUI
	"""
	def __init__(self):
		"""
		initializes and sets up GUI widgets and its connections
		"""
		super().__init__()
		self.setupUi(self)
		self.setWindowTitle("Image to LaTeX App")

		# state values
		self.latex_text = ""
		self.file_path = ""

		# attach button to function
		self.browseImageButton.clicked.connect(self.browse_image)
		self.loadImageButton.clicked.connect(self.load_image)
		self.translateButton.clicked.connect(self.translate_to_latex)

		# auto-complete feauture
		self.filePathEdit.setText(self.file_path)

	def browse_image(self):
		"""
		opens dialog box to browse image, get its path and update the formula box
		"""
		self.file_path = qtw.QFileDialog.getOpenFileName(self, 'Open file', '.',
		                                                 "Image files (*.jpg *.png)")[0]

		# self.file_path = "./data/formula_images/0000ca7c3d3830b_basic.png" # for testing
		# check if path is loaded properly, then update and load image
		if not self.file_path: return
		self.filePathEdit.setText(self.file_path)
		print(self.file_path)
		self.load_image()

	def load_image(self):
		"""
		loads image if valid
		"""
		self.clear_layout(self.inputImageLayout)

		# check if path has leads to image, if not exit early
		self.file_path = self.filePathEdit.text()
		if not Path(self.file_path).is_file():
			self.latexFormulaEdit.setText("File not found!")
			print("File not found!")
			return

		# load image, set as label, align center and show label
		self.image = qtg.QPixmap(self.file_path)
		self.image_label = qtw.QLabel()
		self.image_label.setPixmap(self.image)
		self.inputImageLayout.addWidget(self.image_label)
		self.inputImageLayout.setAlignment(qtc.Qt.AlignHCenter)
		self.show()

	def translate_to_latex(self):
		"""
		given a latext_text, update the formula box
		"""
		self.latex_text = "\\int {a^{2}+b^{2}} = \frac{c^{2}}{d^{2}}"
		self.latexFormulaEdit.setText(self.latex_text)
		print(self.latex_text)

	def center_window(self):
		"""
		centers the fixed main window size according to user screen size
		"""
		screen = qtw.QDesktopWidget().screenGeometry()
		main_window = self.geometry()
		x = (screen.width() - main_window.width()) / 2

		# pulls the window up slightly (arbitrary)
		y = (screen.height() - main_window.height()) / 2 - 50
		self.setFixedSize(main_window.width(), main_window.height())
		self.move(x, y)

	def clear_layout(self, layout):
		"""
		clear all widget within a layout
		"""
		while layout.count() > 0:
			exist = layout.takeAt(0)
			if not exist: continue
			else: exist.widget().deleteLater()

if __name__ == "__main__":
	app = qtw.QApplication([])
	main = Main()
	main.center_window()
	main.show()
	sys.exit(app.exec_())
