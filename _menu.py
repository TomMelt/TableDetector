from PyQt5.QtWidgets import QFileDialog
import sys


def init_menu(self):
    self.actionExit.triggered.connect(self.quit_program)
    self.actionExit.setShortcut('Ctrl+W')
    self.actionOpen.triggered.connect(self.open_file)
    self.actionOpen.setShortcut('Ctrl+O')


def quit_program(self):
    sys.exit()


def open_file(self):
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    infile, _ = QFileDialog.getOpenFileName(
            None,
            'QFileDialog.getOpenFileName()',
            '',
            'Images (*.png);;All (*)',
            options=options,
            )
    if infile:
        self.infile = infile
        self.init_table_preview()
#        self.init_char_preview()
