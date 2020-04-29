#from pytesseract import Output
from scipy.signal import find_peaks
from sklearn.cluster import KMeans, SpectralClustering
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
#import pytesseract
import sys

# TODO
# =======
#  - blur image for finding contours
#  - use unblurred for finding contours
#  - create algorithm to compare sets of characters and reduce to given threshold


def threshold_img(img, method):

    method = method.lower()

    if method == 'otsu':
        img = cv2.GaussianBlur(img, (5,5), 0)
        img = cv2.threshold(
                img,
                0,
                255,
                cv2.THRESH_BINARY+cv2.THRESH_OTSU,
                )[1]
    elif method == 'otsu2':
        img = cv2.threshold(
                img,
                250,
                255,
                cv2.THRESH_OTSU,
                )[1]
    elif method == 'adaptive':
        img = cv2.adaptiveThreshold(
                img,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
                )
    else:
        msg = 'Method {0} not implemented. Please use "otsu" or "adaptive".'
        raise NotImplementedError(msg.format(method))

    return img


def modify_img(img, method, iterations, morph_size=(2, 2)):

    method = method.lower()

    struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)

    if method == 'dilate':
        img = ~img
    elif method == 'erode':
        pass
    else:
        msg = 'Method {0} is not implemented. Please use "erode" or "dilate".'
        raise NotImplementedError(msg.format(method))

    img = cv2.erode(~img, struct, anchor=(-1, -1), iterations=iterations)

    if method == 'erode':
        img = ~img

    return img


def get_table(img, relative_heights=(0.2, 0.2), plateau_sizes=(10,10), debug=False):

    inverted_img = (255 - img)//255
    table = {}

    row_pos, row_cutoff = find_row_positions(
            inverted_img,
            relative_heights[0],
            plateau_sizes[0],
            )

    col_pos, col_cutoff = find_column_positions(
            inverted_img,
            relative_heights[1],
            plateau_sizes[1],
            )

    table['row_pos'] = row_pos
    table['col_pos'] = col_pos

    if debug == True:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        fig = plt.figure(figsize=(7, 7))
        gs = fig.add_gridspec(3, 3)
        ax1 = fig.add_subplot(gs[1:, :-1])
        ax2 = fig.add_subplot(gs[0, :-1])
        ax3 = fig.add_subplot(gs[1:, -1])

        img = draw_table(table, img)

        ax1.imshow(img, aspect='auto')
        ax1.set_xlim([0, img.shape[1]])

        # columns
        y = -1.*np.sum(inverted_img, axis=0)
        x = [i for i in range(len(y))]
        ax2.plot(x, y)
        ax2.plot(col_pos, y[col_pos], 'x')
        ax2.plot([0, len(x)], [col_cutoff, col_cutoff])
        ax2.set_xlim([0, img.shape[1]])
        ax2.set_xticklabels('')

        # rows
        y = -1.*np.sum(inverted_img, axis=1)
        x = [i for i in range(len(y))]
        ax3.plot(y, x)
        ax3.plot(y[row_pos], row_pos, 'x')
        ax3.plot([row_cutoff, row_cutoff], [0, len(x)])
        ax3.set_ylim([0, img.shape[0]])
        ax3.set_yticklabels('')

        plt.show()

    return table


def find_row_positions(img, rel_height, plateau_size):

    pixel_hist = np.sum(img, axis=1)
    pixel_hist = -1.0*pixel_hist

    cutoff = -1.*rel_height*max(abs(pixel_hist))

    positions, _ = find_peaks(
            pixel_hist,
            height=cutoff,
            plateau_size=plateau_size,
            )
    max_h = len(pixel_hist) - 1
    positions = np.sort(np.append(positions, [0, max_h]))

    return positions, cutoff


def find_column_positions(img, rel_height, plateau_size):

    pixel_hist = np.sum(img, axis=0)
    pixel_hist = -1.0*pixel_hist

    cutoff = -1.*rel_height*max(abs(pixel_hist))

    positions, _ = find_peaks(
            pixel_hist,
            height=cutoff,
            plateau_size=plateau_size,
            )
    max_w = len(pixel_hist) - 1
    positions = np.sort(np.append(positions, [0, max_w]))

    return positions, cutoff


def draw_table(table, img):

    row_pos = table['row_pos']
    col_pos = table['col_pos']

    hor_lines, ver_lines = [], []

    min_y, max_y = min(row_pos), max(row_pos)
    min_x, max_x = min(col_pos), max(col_pos)

    for pos in col_pos:
        ver_lines.append((pos, min_y, pos, max_y))

    for pos in row_pos:
        hor_lines.append((min_x, pos, max_x, pos))

    for line in hor_lines:
        [x1, y1, x2, y2] = line
        img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for line in ver_lines:
        [x1, y1, x2, y2] = line
        img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return img


def trim_whitespace(img, threshold=0.01):
    inverted_img = (255 - img)//255
    y = np.sum(inverted_img, axis=0)
    xa, xb = trim_limits(y, threshold)
    y = np.sum(inverted_img, axis=1)
    ya, yb = trim_limits(y, threshold)
    return img[ya:yb+1, xa:xb+1], xa, ya


def trim_limits(y, threshold):
    y = np.cumsum(y)
    # lower limit
    a = np.min(np.argwhere(y>threshold))
    # upper limit
    b = np.max(np.argwhere(y[-1]-y>threshold))
    if b < len(y):
        b = b+1
    return a, b


def cell_positions(table):

    row_pos = table['row_pos']
    col_pos = table['col_pos']

    n = len(row_pos)-1
    m = len(col_pos)-1
    cells = np.zeros((n,m,4), dtype=int)

    for r in range(n):
        for c in range(m):
            x1, x2 = col_pos[c], col_pos[c+1]
            y1, y2 = row_pos[r], row_pos[r+1]
            cells[r][c] = np.array([x1, y1, x2, y2])

    return cells


#def cell_text(cell_img):
#
#    if np.sum(255 - cell_img) == 0:
#        # this means the image is blank space
#        return 0, ''
#
#    config = '--oem 3 --psm 6 -c tessedit_char_whitelist=-.0123456789'
#    data = pytesseract.image_to_string(cell_img, config=config)
#
#
##    if text == '':
##        config = '--psm 10 --oem 3 -c tessedit_char_whitelist=.0123456789'
##        data = pytesseract.image_to_data(cell_img, config=config, output_type=Output.DICT)
##        conf, text = data['conf'][-1], data['text'][-1]
#
##    if text == '':
##        text = 'still not found'
##        conf = 0
#
#    return data


def remove_overlapping_bbox(bboxs, cutoff):

    n = len(bboxs)

    duplicates = []
    unique_bboxs = []

    for i in range(n):
        for j in range(n):
            if (i != j) and (j > i):
                overlap, box = check_overlap_bbox(bboxs[i], bboxs[j])
                if overlap > cutoff:
                    duplicates.append(i)
                    bboxs[i] = box
                    bboxs[j] = box

    for i in range(n):
        if i not in duplicates:
            unique_bboxs.append(bboxs[i])

    return unique_bboxs


def get_cell_img(cell, img):

    [x1, y1, x2, y2] = cell

    cell_img = img[y1:y2, x1:x2]

    return cell_img


def get_bboxs(cell, method, img):
    method = method.lower()
    boundboxes = []
    # store absolute location of cell w.r.t. whole image
    abs_x1, abs_y1, abs_x2, abs_y2 = cell
    # get image for given cell location
    cell_img = img[abs_y1:abs_y2,abs_x1:abs_x2]

    inverted_img = (255 - cell_img)//255

    if np.sum(inverted_img) < 5:
        return(None)

    if method == 'contours':
        # pad whitespace around cell
        pad = 2
        cell_img = cv2.copyMakeBorder(cell_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, None, 255)
        # find contours
        contours, hierarchy = cv2.findContours(
                cell_img,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
                )[-2:]
        contours, hierarchy = sort_contours(contours, hierarchy)
        for k, cnt in enumerate(contours):
            if hierarchy[k][-1] == 0:
                x, y, w, h = cv2.boundingRect(cnt)
                x = x + abs_x1 - pad
                y = y + abs_y1 - pad
                boundboxes.append([x, y, x+w, y+h])
        boundboxes = remove_overlapping_bbox(boundboxes, cutoff=0.6)

    elif method == 'histogram':
        cell_img, xt, yt = trim_whitespace(cell_img)
        table = get_table(
                cell_img,
                relative_heights=(0.1,0.2),
                plateau_sizes=(0,0),
                debug=False,
                )
        boundboxes = cell_positions(table)[0]
        for i in range(len(boundboxes)):
            bbox = boundboxes[i]
            bbox = bbox + np.array([abs_x1+xt, abs_y1+yt, abs_x1+xt, abs_y1+yt])
            boundboxes[i] = bbox

    return boundboxes


def get_char_from_bbox(img, bbox):

    [abs_x1, abs_y1, abs_x2, abs_y2] = bbox
    char = img[abs_y1:abs_y2,abs_x1:abs_x2]

    return char


def normalise_chars(chars):

    norm_chars = []
    max_w = 0
    max_h = 0

    for char in chars:
        h, w = char.shape
        max_w = max(w, max_w)
        max_h = max(h, max_h)

    for char in chars:
        char = cv2.copyMakeBorder(
                char,
                0,
                max_h-char.shape[0],
                0,
                max_w-char.shape[1],
                cv2.BORDER_CONSTANT,
                None,
                255
                )
        norm_chars.append(char)

    return norm_chars


def check_overlap_bbox(boxA, boxB):

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    overlap = interArea / float(min(boxAArea, boxBArea))

    # determine the (x, y)-coordinates of the union rectangle
    xA = min(boxA[0], boxB[0])
    yA = min(boxA[1], boxB[1])
    xB = max(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])

    union = [xA, yA, xB, yB]

    return overlap, union


def sort_contours(contours, hierarchy, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    index = [i for i in range(len(boundingBoxes))]
    (index, boundingBoxes) = zip(*sorted(zip(index, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))

    contours = [contours[i] for i in index]
    hierarchy = [hierarchy[0][i] for i in index]

    return contours, hierarchy


if __name__ == "__main__":
    im = 0
    in_file = None
#    try:
#        in_file = str(sys.argv[1])
#    except:
#        os.system('gnome-screenshot -a -f ./data/screenshot.png')
#        pass
    if in_file is None:
        in_file = './inputs/pre.png.bak'
    out_file = './data/out.png'

    img = cv2.imread(in_file)
    preprocessed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    preprocessed_img = threshold_img(preprocessed_img, method='adaptive')

#    for m in ['dilate', 'erode']:
#        preprocessed_img = modify_img(img, method=m, iterations=2)
#        plt.imshow(np.hstack([img, preprocessed_img]), 'gray')
#        plt.title('orig                        '+m)
#        plt.show()

    preprocessed_img = modify_img(preprocessed_img, method='dilate', iterations=1)

    plt.imshow(preprocessed_img)
    plt.show()

    table = get_table(
            preprocessed_img,
            relative_heights=(0.1,0.1),
            plateau_sizes=(10,10),
            debug=True,
            )

    # Visualize the result
    vis = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
    vis = draw_table(table, img=vis)

    plt.imshow(vis)
    plt.show()

    cells = cell_positions(table)
    n, m, _ = cells.shape
    output = np.empty((n,m), dtype=object)

#    ---- use pixel algorithm to get bounding boxes (needs work) -----

    boundboxes = []

    for i in range(n):
        for j in range(m):
            bboxs = get_bboxs(
                    cells[i][j],
                    method='contours',
                    img=preprocessed_img,
                    )
            if bboxs is not None:
                boundboxes.append(bboxs)

    boundboxes = [b for row in boundboxes for b in row]

    for bbox in boundboxes:
        x, y, x2, y2 = bbox
        vis = cv2.rectangle(vis, (x, y), (x2, y2), color=(0,255,0))

    plt.imshow(vis)
    plt.show()

    chars = [get_char_from_bbox(preprocessed_img, bbox) for bbox in boundboxes]

    chars = normalise_chars(chars)

    n = len(chars)
#    print('n = ', n)
#    n = 50
    chars = chars[:n]
    sim_mat = np.zeros((n,n), dtype=np.float)

    for i in range(n):
        for j in range(n):
            if (i != j) and (j > i):
                value = ssim(chars[i], chars[j], data_range=255)
                if value > 0.5:
                    sim_mat[i,j] = value*value
            if i == j:
                sim_mat[i,i] = 0.0

    sim_mat = sim_mat + sim_mat.T

    n_clusters = 11
    clustering = SpectralClustering(
            assign_labels='kmeans',
#            assign_labels='discretize',
            affinity='precomputed',
            n_clusters=n_clusters
            ).fit(sim_mat)

    clusters = clustering.labels_
    order = np.argsort(clusters)
    clusters = clusters[order]
#    print('clusters', clusters)
#    print('order   ', order)

    chars = np.array(chars)
    sim_mat = sim_mat[order]
    sim_mat = sim_mat[:,order]
    chars = chars[order]
    chars = list(chars)

#    col_head = np.vstack(chars)//255
#    row_head = np.hstack([chars[0]*0.]+chars)//255
#    max_h, max_w = chars[0].shape
#    sim_mat = resize(sim_mat, (max_h*n, max_w*n), preserve_range=True, mode='edge', order=0)
#
#    sim_mat = np.hstack((col_head, sim_mat))
#    sim_mat = np.vstack((row_head, sim_mat))
#    plt.imshow(sim_mat)
#    plt.colorbar()
#    plt.show()

    matches = {}

    for i in range(n_clusters):
        matches[i] = []

    for i in range(n):
        matches[clusters[i]].append(chars[i])

    max_w = 0

    for k, v in matches.items():
        match_img = np.hstack(v)
        max_w = max(max_w, match_img.shape[1])
        matches[k] = match_img

    for k, v in matches.items():
        match_img = cv2.copyMakeBorder(v, 0, 0, 0, max_w - v.shape[1], cv2.BORDER_CONSTANT, None, 255)
        matches[k] = match_img

    matches_img = np.vstack(list(matches.values()))
    plt.imshow(matches_img)
    plt.show()

# --------------------- find character positions ------------------------------
#    col_pos = find_column_postions(cell_img, plateau_size=3)
#    row_pos = find_row_positions(cell_img)
#    hor_lines, ver_lines = build_lines(col_pos, row_pos)
#
#    for line in ver_lines:
#        line = np.array(line) + np.array([abs_x1, abs_y1, abs_x1, abs_y1,])
#        [x1, y1, x2, y2] = line
#        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imwrite(out_file, vis)
