import math

import cv2
import numpy as np
import painless_panes
import scipy.spatial.distance

image = cv2.imread("example1.jpg")
height, width, channels = image.shape

width = int(width * 800 / height)
height = int(height * 800 / height)

image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
windows = painless_panes.model.get_detections(image, conf=0.25)
window = painless_panes.cv.select_central_detection(image, windows)
bbox = window["bounding_box"]

window_corners = painless_panes.cv.find_window_corners(image, bbox, annotate=True)


def get_perspective_transform_matrix(corners):
    """Get the perspective transform from the corners of a rectangular object

    source: https://stackoverflow.com/q/38285229

    :param corners: _description_
    :type corners: _type_
    """
    # image center
    u0 = width / 2.0
    v0 = height / 2.0

    # detected corners on the original image
    # TODO: Apply sorting to put the corners in zig-zag order
    p = []
    p.append(corners[0])
    p.append(corners[1])
    p.append(corners[3])
    p.append(corners[2])

    # widths and heights of the projected image
    w1 = scipy.spatial.distance.euclidean(p[0], p[1])
    w2 = scipy.spatial.distance.euclidean(p[2], p[3])

    h1 = scipy.spatial.distance.euclidean(p[0], p[2])
    h2 = scipy.spatial.distance.euclidean(p[1], p[3])

    w = max(w1, w2)
    h = max(h1, h2)

    # visible aspect ratio
    ar_vis = float(w) / float(h)

    # make numpy arrays and append 1 for linear algebra
    m1 = np.array((p[0][0], p[0][1], 1)).astype("float32")
    m2 = np.array((p[1][0], p[1][1], 1)).astype("float32")
    m3 = np.array((p[2][0], p[2][1], 1)).astype("float32")
    m4 = np.array((p[3][0], p[3][1], 1)).astype("float32")

    # calculate the focal disrance
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    n21 = n2[0]
    n22 = n2[1]
    n23 = n2[2]

    n31 = n3[0]
    n32 = n3[1]
    n33 = n3[2]

    f = math.sqrt(
        np.abs(
            (1.0 / (n23 * n33))
            * (
                (n21 * n31 - (n21 * n33 + n23 * n31) * u0 + n23 * n33 * u0 * u0)
                + (n22 * n32 - (n22 * n33 + n23 * n32) * v0 + n23 * n33 * v0 * v0)
            )
        )
    )

    A = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]]).astype("float32")

    At = np.transpose(A)
    Ati = np.linalg.inv(At)
    Ai = np.linalg.inv(A)

    # calculate the real aspect ratio
    ar_real = math.sqrt(
        np.dot(np.dot(np.dot(n2, Ati), Ai), n2)
        / np.dot(np.dot(np.dot(n3, Ati), Ai), n3)
    )

    if ar_real < ar_vis:
        W = int(w)
        H = int(W / ar_real)
    else:
        H = int(h)
        W = int(ar_real * H)

    pts1 = np.array(p).astype("float32")
    pts2 = np.float32([[0, 0], [W, 0], [0, H], [W, H]])
    mat = cv2.getPerspectiveTransform(pts1, pts2)
    return mat


def get_transformed_image(mat, img):
    """_summary_

    source: https://stackoverflow.com/a/64608248

    :param mat: _description_
    :type mat: _type_
    :param img: _description_
    :type img: _type_
    :return: _description_
    :rtype: _type_
    """
    height, width, _ = img.shape

    # calculate the tranformation

    # new source: image corners
    corners = np.array([[0, height], [0, 0], [width, 0], [width, height]])

    # Transform the corners of the image
    corners_tranformed = cv2.perspectiveTransform(
        np.array([corners.astype("float32")]), mat
    )

    # These tranformed corners seems completely wrong/inverted x-axis
    print(corners_tranformed)

    x_mn = math.ceil(min(corners_tranformed[0].T[0]))
    y_mn = math.ceil(min(corners_tranformed[0].T[1]))

    x_mx = math.ceil(max(corners_tranformed[0].T[0]))
    y_mx = math.ceil(max(corners_tranformed[0].T[1]))

    width = x_mx - x_mn
    height = y_mx - y_mn

    analogy = height / 1000
    n_height = height / analogy
    n_width = width / analogy

    dst2 = corners_tranformed
    dst2 -= np.array([x_mn, y_mn])
    dst2 = dst2 / analogy

    mat2 = cv2.getPerspectiveTransform(
        corners.astype("float32"), dst2.astype("float32")
    )

    img_warp = cv2.warpPerspective(np.array(image), mat2, (int(n_width), int(n_height)))
    return img_warp


# project the image with the new w/h
mat = get_perspective_transform_matrix(window_corners)
image_warp = get_transformed_image(mat, image)
# height, width, channels = image_warp.shape

# width = int(width * 800 / height)
# height = int(height * 800 / height)

# image_warp = cv2.resize(image_warp, (width, height), interpolation=cv2.INTER_LINEAR)
# M = cv2.getPerspectiveTransform(pts1, pts2)

# dst = cv2.warpPerspective(image, M, (W, H))


cv2.imshow("img", image)
cv2.imshow("dst", image_warp)

cv2.waitKey(0)
