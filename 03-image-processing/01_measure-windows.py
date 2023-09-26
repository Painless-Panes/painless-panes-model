import cv2
import numpy
import painless_panes

image = cv2.imread("example1.jpg")
height, width, channels = image.shape

width = int(width * 640 / height)
height = int(height * 640 / height)

image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
windows = painless_panes.model.get_detections(image)
window = painless_panes.cv.select_central_detection(image, windows)
x0, y0, x1, y1 = window["bounding_box"]
# width, height, message = painless_panes.cv.measure_window(image, annotate=True)

# print(width, height, message)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image, (5, 5), 0)
# image = cv2.bilateralFilter(image, 9, 75, 75)

all_edges = cv2.Canny(image, 30, 150)
edges = numpy.zeros_like(all_edges)
edges[y0:y1, x0:x1] = all_edges[y0:y1, x0:x1]

fld = cv2.ximgproc.createFastLineDetector()
# Get line vectors from the image
lines = fld.detect(edges)
# vlines, hlines = painless_panes.cv.partition_lines_by_orientation(lines)
# Draw lines on the image
print(repr(lines))
print(lines.shape)
image = fld.drawSegments(image, lines, linecolor=(0, 255, 0))
# image = fld.drawSegments(image, hlines, linecolor=(255, 0, 0))


# # Find the contours of the edges
# contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Approximate each contour to a polygon
# for contour in contours:
#     perimeter = cv2.arcLength(contour, True)
#     polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
#     if len(polygon) == 4:
#       cv2.drawContours(image, [polygon], -1, (0, 255, 0), 2)

cv2.imshow("Blurred", image)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
