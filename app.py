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
		browse image, get its path and update the formula box
		"""
		self.file_path = "../data/formula_images/0000ca7c3d3830b_basic.png"
		self.filePathEdit.setText(self.file_path)
		print(self.file_path)

	def load_image(self):
		"""
		loads stock data .csv from inputted filepath string on the GUI
		as StockData object, also autocompletes all inputs
		using information provided by the csv.
		Error handling
			invalid filepath :
				empty filepath or file could not be found.
			invalid .csv :
				.csv file is empty, missing date column, etc.
		"""
		self.file_path = Path(self.filePathEdit.text())
		image_label = qtw.QLabel(self)
		image = qtg.QPixmap(self.file_path)

	def translate_to_latex(self):
		"""
		given a latext_text, update the formula box
		"""
		self.latex_text = "\\int {a^{2}+b^{2}} = \frac{c^{2}}{d^{2}}"
		self.latexFormulaEdit.setText(self.latex_text)
		print(self.latex_text)

	def center(self):
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

if __name__ == "__main__":
	app = qtw.QApplication([])
	main = Main()
	main.center()
	main.show()
	sys.exit(app.exec_())
