import math

import cv2
import numpy
import painless_panes

image = cv2.imread("example1.jpg")
height, width, channels = image.shape

width = int(width * 800 / height)
height = int(height * 800 / height)

image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
windows = painless_panes.model.get_detections(image, conf=0.25)
window = painless_panes.cv.select_central_detection(image, windows)
bbox = window["bounding_box"]

window_corners = painless_panes.cv.find_window_corners(image, bbox, annotate=True)

# project the image with the new w/h
trans, width, height = painless_panes.cv.align_image_perspective_on_rectangle(
    image, window_corners
)

# 12. Apply the final cropped transformation
image_final = cv2.warpPerspective(image, trans, (width, height))
# image_warp = get_transformed_image(mat, image)
# height, width, channels = image_warp.shape

# width = int(width * 800 / height)
# height = int(height * 800 / height)

# image_warp = cv2.resize(image_warp, (width, height), interpolation=cv2.INTER_LINEAR)
# M = cv2.getPerspectiveTransform(pts1, pts2)

# dst = cv2.warpPerspective(image, M, (W, H))


cv2.imshow("img", image)
cv2.imshow("dst", image_final)

cv2.waitKey(0)
