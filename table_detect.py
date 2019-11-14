import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
import sys
from operator import xor
from scipy.signal import find_peaks

# https://stackoverflow.com/questions/50829874/how-to-find-table-like-structure-in-image

# This only works if there's only one table on a page
# Important parameters:
#  - morph_size
#  - min_text_height_limit
#  - max_text_height_limit
#  - cell_threshold
#  - min_columns

# image pixels go from top-left corner

def pre_process_image(img, save_in_file, morph_size=(2, 2)):

    # get rid of the color
    pre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu threshold
    pre = cv2.threshold(pre, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # dilate the text to make it solid spot
    cpy = pre.copy()
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
    cpy = cv2.erode(~cpy, struct, anchor=(-1, -1), iterations=1)
    cpy = cv2.erode(~cpy, struct, anchor=(-1, -1), iterations=2)
    pre = cpy

    if save_in_file is not None:
        cv2.imwrite(save_in_file, pre)
    return pre


def find_row_positions(img, rel_height=0.3):

    inverted_img = (255 - img)//255
    pixel_hist = np.sum(inverted_img, axis=1)
    pixel_hist = -1.0*pixel_hist

    cutoff = -1.*rel_height*max(abs(pixel_hist))

    positions, _ = find_peaks(pixel_hist, height=cutoff, distance=10)
    max_h = len(pixel_hist) - 1
    positions = np.sort(np.append(positions, [0, max_h]))
#    plt.plot(pixel_hist)
#    plt.plot(positions,pixel_hist[positions], 'x')
#    plt.plot([0, max_h], [cutoff, cutoff])
#    plt.show()

    return positions


def find_column_postions(img, rel_height=0.2):

    inverted_img = (255 - img)//255
    pixel_hist = np.sum(inverted_img, axis=0)
    pixel_hist = -1.0*pixel_hist

    cutoff = -1.*rel_height*max(abs(pixel_hist))

    positions, _ = find_peaks(pixel_hist, height=-10, plateau_size=10)
    max_w = len(pixel_hist) - 1
    positions = np.sort(np.append(positions, [0, max_w-1]))
#    plt.plot(pixel_hist)
#    plt.plot(positions,pixel_hist[positions], 'x')
#    plt.plot([0, max_w], [cutoff, cutoff])
#    plt.show()

    return positions


def build_lines(col_pos, row_pos):

    hor_lines, ver_lines = [], []

    min_y, max_y = min(row_pos), max(row_pos)
    min_x, max_x = min(col_pos), max(col_pos)


    for pos in col_pos:
        ver_lines.append((pos, min_y, pos, max_y))

    for pos in row_pos:
        hor_lines.append((min_x, pos, max_x, pos))

    return hor_lines, ver_lines


if __name__ == "__main__":
    in_file = str(sys.argv[1])
    pre_file = './data/pre.png'
    out_file = './data/out.png'

    img = cv2.imread(in_file)

    pre_processed = pre_process_image(img, pre_file)

    col_pos = find_column_postions(pre_processed)
    row_pos = find_row_positions(pre_processed)

    hor_lines, ver_lines = build_lines(col_pos, row_pos)

    # Visualize the result
    vis = img.copy()


    for line in hor_lines:
        [x1, y1, x2, y2] = line
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

    for line in ver_lines:
        [x1, y1, x2, y2] = line
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

    print(pytesseract.image_to_string(img))

    cv2.imwrite(out_file, vis)
