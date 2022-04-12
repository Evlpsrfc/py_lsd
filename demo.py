import cv2 as cv

import lsd_ext


image = cv.imread("lsd_1.6/chairs.pgm")
output = lsd_ext.lsd(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
lines = output[:, 3::-1]
lines = lines.reshape(-1, 2, 2).astype(int)
for p, q in lines:
    p = tuple(p)
    q = tuple(q)
    cv.line(image, p, q, (0, 255, 0), 2)
cv.imwrite("chairs_plot.png", image)