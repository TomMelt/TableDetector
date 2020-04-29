from image import Image
from PyQt5 import QtGui


def init_chars_tab(self):
    self.init_chars_preview()
    self.init_chars_button()
    self.init_chars_hslider()


def init_chars_button(self):
    self.button_detect_chars.clicked.connect(self.chars_button_click)


def init_chars_hslider(self):
    self.hslider_dilate_2.valueChanged.connect(self.chars_hslider_changed)


def init_chars_preview(self):
    self.label_preprocess_2.setPixmap(QtGui.QPixmap(self.preview_file))


def chars_button_click(self):
    self.chars_update_preview()


def chars_hslider_changed(self):
    size = self.hslider_dilate_2.value()
    self.image.dilate(size)
    if self.checkBox_livePreview_2.isChecked():
        self.chars_update_preview()


def chars_update_preview(self):
    img = self.image
    img.save(self.preview_file)
    self.chars_detector.detect_from_image(
            img = img.dilated.copy(),
            cells = self.table_detector.cells,
            )
    img.add_chars(self.chars_detector.boundboxes)
    img.save(self.preview_file)
    self.label_preprocess_2.setPixmap(QtGui.QPixmap(self.preview_file))
