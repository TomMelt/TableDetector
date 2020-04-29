from gui.gui import Ui_MainWindow
from PyQt5 import QtWidgets
from image import Image, TableDetector, CharsDetector
import sys


class UiCustomMainWindow(Ui_MainWindow):
    def __init__(self):
        super().__init__()

    from _menu import init_menu, quit_program, open_file
    from _tab_table import (
            init_table_tab,
            init_table_button,
            init_table_cutoffs,
            init_table_hslider,
            init_table_preview,
            table_button_click,
            table_cutoff_changed,
            table_hslider_changed,
            table_update_preview,
            )
    from _tab_chars import (
            init_chars_tab,
            init_chars_button,
            init_chars_hslider,
            init_chars_preview,
            chars_button_click,
            chars_hslider_changed,
            chars_update_preview,
            )

    def setup_ui(self, mainwindow):
        super().setupUi(mainwindow)

        self.infile = './inputs/pre.png.bak'
        self.preview_file = './data/preview.png'
        self.image = Image()
        self.table_detector = TableDetector()
        self.chars_detector = CharsDetector()
        self.image.load(self.infile)
        self.init_menu()
        self.init_table_tab()
        self.init_chars_tab()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = QtWidgets.QMainWindow()
    ui = UiCustomMainWindow()
    ui.setup_ui(mainwindow)
    mainwindow.show()
    sys.exit(app.exec_())
