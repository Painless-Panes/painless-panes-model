import cv2
import numpy
import painless_panes

image = cv2.imread("example1.jpg")
height, width, _ = image.shape

width = int(width * 800 / height)
height = 800
image = cv2.resize(image, (width, height))

wwidth, wheight, image_out, message = painless_panes.cv.measure_window(image, annotate=True)

cv2.imshow("1", image_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
