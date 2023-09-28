import cv2
import numpy
import painless_panes

image = cv2.imread("example1.jpg")
height, width, channels = image.shape

width = int(width * 800 / height)
height = int(height * 800 / height)

image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
windows = painless_panes.model.get_detections(image)
window = painless_panes.cv.select_central_detection(image, windows)
bbox = window["bounding_box"]

image = painless_panes.cv.find_window_corners(image, bbox)

# # Find the contours of the edges
# contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Approximate each contour to a polygon
# for contour in contours:
#     perimeter = cv2.arcLength(contour, True)
#     polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
#     if len(polygon) == 4:
#       cv2.drawContours(image, [polygon], -1, (0, 255, 0), 2)

cv2.imshow("1", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
