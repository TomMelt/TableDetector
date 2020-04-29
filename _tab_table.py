from image import Image
from PyQt5 import QtGui


def init_table_tab(self):
    self.init_table_preview()
    self.init_table_button()
    self.init_table_hslider()
    self.init_table_cutoffs()


def init_table_button(self):
    self.button_drawTable.clicked.connect(self.table_button_click)


def init_table_hslider(self):
    self.hslider_dilate.valueChanged.connect(self.table_hslider_changed)


def init_table_cutoffs(self):
    self.doubleSpinBox_col_cutoff.valueChanged.connect(self.table_cutoff_changed)
    self.doubleSpinBox_row_cutoff.valueChanged.connect(self.table_cutoff_changed)
    self.spinBox_col_plateau.valueChanged.connect(self.table_cutoff_changed)
    self.spinBox_row_plateau.valueChanged.connect(self.table_cutoff_changed)


def init_table_preview(self):
    self.image = Image()
    img = self.image
    img.load(path=self.infile)
    img.threshold()
    img.save(path=self.preview_file)
    self.label_preprocess.setPixmap(QtGui.QPixmap(self.preview_file))


def table_button_click(self):
    self.table_update_preview()


def table_cutoff_changed(self):
    if self.checkBox_livePreview.isChecked():
        self.table_update_preview()


def table_hslider_changed(self):
    size = self.hslider_dilate.value()
    self.image.dilate(size)
    if self.checkBox_livePreview.isChecked():
        self.table_update_preview()


def table_update_preview(self):
    img = self.image
    relative_heights = (
            self.doubleSpinBox_row_cutoff.value(),
            self.doubleSpinBox_col_cutoff.value(),
            )
    plateau_sizes = (
            self.spinBox_row_plateau.value(),
            self.spinBox_col_plateau.value(),
            )
    self.table_detector.detect_from_image(
            img=img.dilated.copy(),
            relative_heights=relative_heights,
            plateau_sizes=plateau_sizes,
            )
    img.add_table(self.table_detector.table)
    img.save(self.preview_file)
    self.label_preprocess.setPixmap(QtGui.QPixmap(self.preview_file))
    self.label_preprocess_2.setPixmap(QtGui.QPixmap(self.preview_file))
