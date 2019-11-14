import cv2
import pytesseract
from pytesseract import Output
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


def cell_positions(col_pos, row_pos):

    n = len(row_pos)-1
    m = len(col_pos)-1
    cells = np.zeros((n,m,4), dtype=int)

    for r in range(n):
        for c in range(m):
            x1, x2 = col_pos[c], col_pos[c+1]
            y1, y2 = row_pos[r], row_pos[r+1]
            cells[r][c] = np.array([x1, y1, x2, y2])

    return cells


def cell_text(cell_img):

    if np.sum(255 - cell_img) == 0:
        # this means the image is blank space
        return 0, ''

    config = '--oem 3 -c tessedit_char_whitelist=.0123456789'
    data = pytesseract.image_to_data(cell_img, config=config, output_type=Output.DICT)
    conf, text = data['conf'][-1], data['text'][-1]

    if text == '':
        config = '--psm 10 --oem 3 -c tessedit_char_whitelist=.0123456789'
        data = pytesseract.image_to_data(cell_img, config=config, output_type=Output.DICT)
        conf, text = data['conf'][-1], data['text'][-1]

    if text == '':
        text = 'still not found'
        conf = 0

    return conf, text



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
    vis = vis*0. + 255

    cells = cell_positions(col_pos, row_pos)


    for line in hor_lines:
        [x1, y1, x2, y2] = line
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

    for line in ver_lines:
        [x1, y1, x2, y2] = line
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

    n, m, _ = cells.shape
    output = np.empty((n,m), dtype=object)

    for i in range(n):
        for j in range(m):
            [x1, y1, x2, y2] = cells[i][j]
            w = x2 - x1
            h = y2 - y1
            cell_img = img[y1:y2,x1:x2]
            conf, text = cell_text(cell_img)
            color = (0, 255, 0) if conf > 90 else (0, 0, 255)
            cv2.putText(vis, text, (x1,y2), cv2.FONT_HERSHEY_PLAIN, 1.0, color)
            output[i][j] = text

    np.savetxt('table.csv', output, fmt='%s', delimiter='\t')
    cv2.imwrite(out_file, np.vstack((img, vis)))
