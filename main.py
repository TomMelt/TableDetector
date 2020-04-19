from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
from table_detect import threshold_img, modify_img

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("table detector")
        MainWindow.resize(856, 679)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)

        self.hslider_dilate = QtWidgets.QSlider(self.centralwidget)
        self.hslider_dilate.setOrientation(QtCore.Qt.Horizontal)
        self.hslider_dilate.setObjectName("hslider_dilate")
        self.hslider_dilate.setMinimum(0)
        self.hslider_dilate.setMaximum(10)
        self.hslider_dilate.setSingleStep(1)
        self.hslider_dilate.setTickPosition(QtWidgets.QSlider.TicksBothSides)


        self.hslider_dilate.valueChanged.connect(self.changed)

        self.horizontalLayout.addWidget(self.hslider_dilate)
        self.checkBox_livePreview = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_livePreview.setObjectName("checkBox_livePreview")
        self.horizontalLayout.addWidget(self.checkBox_livePreview)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.button_drawTable = QtWidgets.QPushButton(self.centralwidget)
        self.button_drawTable.setObjectName("button_drawTable")

        self.verticalLayout.addWidget(self.button_drawTable)
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 822, 634))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_preprocess = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_preprocess.setText("")

        in_file = './data/pre.png.bak'

        self.img = cv2.imread(in_file)
        self.preprocessed_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.preprocessed_img = threshold_img(self.preprocessed_img, method='adaptive')
        cv2.imwrite('./data/pre.png', self.preprocessed_img)

        self.label_preprocess.setPixmap(QtGui.QPixmap('./data/pre.png'))
        self.label_preprocess.setObjectName("label_preprocess")
        self.verticalLayout_2.addWidget(self.label_preprocess)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout.addWidget(self.scrollArea)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 856, 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("table detector", "table detector"))
        self.label_2.setText(_translate("table detector", "Dilate Image"))
        self.checkBox_livePreview.setText(_translate("table detector", "Live Preview"))
        self.button_drawTable.setText(_translate("table detector", "Draw Table"))

    def changed(self):
        size = self.hslider_dilate.value()
        print(size)
        temp = modify_img(self.preprocessed_img, method='dilate', iterations=self.hslider_dilate.value())
        cv2.imwrite('./data/pre.png', temp)
        self.label_preprocess.setPixmap(QtGui.QPixmap("./data/pre.png"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

