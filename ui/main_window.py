# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\ui\main_window.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1120, 552)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        self.appTitleLabel = QtWidgets.QLabel(Form)
        self.appTitleLabel.setGeometry(QtCore.QRect(4, 10, 1111, 71))
        font = QtGui.QFont()
        font.setPointSize(31)
        self.appTitleLabel.setFont(font)
        self.appTitleLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.appTitleLabel.setObjectName("appTitleLabel")
        self.browseImageButton = QtWidgets.QPushButton(Form)
        self.browseImageButton.setGeometry(QtCore.QRect(10, 90, 151, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.browseImageButton.setFont(font)
        self.browseImageButton.setObjectName("browseImageButton")
        self.filePathEdit = QtWidgets.QLineEdit(Form)
        self.filePathEdit.setGeometry(QtCore.QRect(170, 90, 781, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.filePathEdit.setFont(font)
        self.filePathEdit.setObjectName("filePathEdit")
        self.verticalLayoutWidget = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 380, 1101, 161))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.outputImageLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.outputImageLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.outputImageLayout.setContentsMargins(0, 0, 0, 0)
        self.outputImageLayout.setObjectName("outputImageLayout")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(10, 150, 1101, 161))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.inputImageLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.inputImageLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.inputImageLayout.setContentsMargins(0, 0, 0, 0)
        self.inputImageLayout.setObjectName("inputImageLayout")
        self.latexFormulaEdit = QtWidgets.QLineEdit(Form)
        self.latexFormulaEdit.setGeometry(QtCore.QRect(170, 320, 781, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.latexFormulaEdit.setFont(font)
        self.latexFormulaEdit.setObjectName("latexFormulaEdit")
        self.translateButton = QtWidgets.QPushButton(Form)
        self.translateButton.setGeometry(QtCore.QRect(10, 320, 151, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.translateButton.setFont(font)
        self.translateButton.setObjectName("translateButton")
        self.loadImageButton = QtWidgets.QPushButton(Form)
        self.loadImageButton.setGeometry(QtCore.QRect(960, 90, 151, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.loadImageButton.setFont(font)
        self.loadImageButton.setObjectName("loadImageButton")
        self.translateButton_2 = QtWidgets.QPushButton(Form)
        self.translateButton_2.setGeometry(QtCore.QRect(960, 320, 151, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.translateButton_2.setFont(font)
        self.translateButton_2.setObjectName("translateButton_2")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.appTitleLabel.setText(_translate("Form", "Image to Latex App"))
        self.browseImageButton.setText(_translate("Form", "Browse Image"))
        self.translateButton.setText(_translate("Form", "Translate"))
        self.loadImageButton.setText(_translate("Form", "Load Image"))
        self.translateButton_2.setText(_translate("Form", "Render"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
