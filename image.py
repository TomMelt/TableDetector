import cv2
from table_detect import (
        threshold_img,
        modify_img,
        get_table,
        draw_table,
        cell_positions,
        get_bboxs,
        get_char_from_bbox,
        normalise_chars,
        )
import matplotlib.pyplot as plt


class Image():
    def __init__(self):
        pass

    def load(self, path):
        img = cv2.imread(path)
        self.original = img
        self.current = img

    def threshold(self):
        img = self.original.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = threshold_img(
                img,
                method='adaptive',
                )
        self.dilated = img
        self.thresholded = img
        self.current = img

    def save(self, path):
        img = self.current
        cv2.imwrite(path, img)

    def display(self):
        plt.imshow(self.current)
        plt.show()

    def dilate(self, size):
        img = self.thresholded.copy()
        if size > 0:
            img = modify_img(img, method='dilate', iterations=size)
        if size < 0:
            img = modify_img(img, method='erode', iterations=-size)
        self.dilated = img
        self.current = img

    def add_table(self, table):
        img = self.dilated.copy()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = draw_table(table, img=img)
        self.current = img

    def add_chars(self, chars):
        img = self.dilated.copy()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for char in chars:
            x, y, x2, y2 = char
            img = cv2.rectangle(img, (x, y), (x2, y2), color=(0,255,0))
        self.current = img



class TableDetector():
    def __init__(self):
        self.cells = None

    def detect_from_image(self, img, relative_heights, plateau_sizes):
        self.table = get_table(
                img=img,
                relative_heights=relative_heights,
                plateau_sizes=plateau_sizes,
                debug=False,
                )
        self.cells = cell_positions(self.table)


class CharsDetector():
    def __init__(self):
        pass

    def detect_from_image(self, img, cells, method='histogram'):
        n, m, _ = cells.shape
        boundboxes = []

        for i in range(n):
            for j in range(m):
                bboxs = get_bboxs(
                        cells[i][j],
                        method=method,
                        img=img
                        )
                if bboxs is not None:
                    boundboxes.append(bboxs)

        self.boundboxes = [b for row in boundboxes for b in row]
