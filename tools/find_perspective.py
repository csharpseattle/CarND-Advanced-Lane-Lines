#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import sys
import math

sys.path.append("..")
import camera
import image_utils


if len(sys.argv) < 2:
    print("Please provide an image filename on the command line")
    sys.exit()
else:
    filename = sys.argv[1]

img = mpimg.imread(filename, 1)
h, w = img.shape[:2]

f, axes = plt.subplots(1, 2, figsize=(12, 4.5))
f.tight_layout()
plt.subplots_adjust(left=0.03, right=.97, top=0.99, bottom=0.)

def p1xChanged(x):
    global p1x
    global w
    p1x = float(x / w)
    redrawFiles()

def p1yChanged(x):
    global p1y
    global p4y
    global h
    p1y = float(x / h)
    p4y = p1y
    redrawFiles()

def p2xChanged(x):
    global p2x
    global w
    p2x = float(x / w)
    redrawFiles()

def p2yChanged(x):
    global p2y
    global p3y
    global h
    p2y = float(x / h)
    p3y = float(x / h)
    redrawFiles()

def p3xChanged(x):
    global p3x
    global w
    p3x = float(x / w)
    redrawFiles()

def p3yChanged(x):
    global p3y
    global p2y
    global h
    p3y = float(x / h)
    p2y = float(x / h)
    redrawFiles()

def p4xChanged(x):
    global p4x
    global w
    p4x = float(x / w)
    redrawFiles()

def p4yChanged(x):
    global p4y
    global p1y
    global h
    p4y = float(x / h)
    p1y = p4y
    redrawFiles()



def redrawFiles():
    global mtx
    global dist
    global p1x
    global p2x
    global p3x
    global p4x
    global p1y
    global p2y
    global p3y
    global p4y

    print("--------------------------")
    print("p1x = " + str(p1x))
    print("p1y = " + str(p1y))
    print("p2x = " + str(p2x))
    print("p2y = " + str(p2y))
    print("p3x = " + str(p3x))
    print("p3y = " + str(p3y))
    print("p4x = " + str(p4x))
    print("p4y = " + str(p4y))
    print("--------------------------")
    print("TRANSFORM_SRC_POINTS = np.float32([(" + str(round(p1x, 5)) + ", " + str(round(p1y, 5)) + "), (" + str(round(p2x, 5)) + ", " + str(round(p2y, 5)) + "), (" + str(round(p3x, 5)) + ", " + str(round(p3y, 5)) + "), (" + str(round(p4x, 5)) + ", " + str(round(p4y, 5)) + ")])")


    img = mpimg.imread(filename, 1)

    undist = cv2.undistort(img, mtx, dist, None, mtx)
    height, width = undist.shape[:2]

    s1 = (int(p1x * width), int(p1y * height))
    s2 = (int(p2x * width), int(p2y * height))
    s3 = (int(p3x * width), int(p3y * height))
    s4 = (int(p4x * width), int(p4y * height))

    cv2.line(undist, s1, s2, [255, 0, 0], 2)
    cv2.line(undist, s2, s3, [0, 255, 0], 2)
    cv2.line(undist, s3, s4, [0, 0, 255], 2)

    src = np.float32([s1, s2, s3, s4])

    d1 = (int(p1x * width), height)
    d2 = (int(p1x * width), 0)
    d3 = (int(p4x * width), 0)
    d4 = (int(p4x * width), height)
    dst = np.float32([d1, d2, d3, d4])

    s = np.copy(src)
    d = np.copy(dst)

    M = cv2.getPerspectiveTransform(src, dst)
    transformed = cv2.warpPerspective(undist, M, (width, height), flags=cv2.INTER_LINEAR)

    axes[0].clear()
    axes[0].imshow(undist)
    axes[1].clear()
    axes[1].imshow(transformed)
    plt.draw()





#######################################################################################

TRANSFORM_SRC_POINTS = [(0.0, 0.91528), (0.21875, 0.73472), (0.78281, 0.73472), (1.0, 0.91528)]

#[(0.23828, 0.98611), (0.46563, 0.6375), (0.56406, 0.6375), (0.95312, 0.98611)]

(p1x, p1y), (p2x, p2y), (p3x, p3y), (p4x, p4y) = TRANSFORM_SRC_POINTS

# create trackbars if in interactive mode
shapefinder = cv2.imread(filename, 1)
height, width, _ = shapefinder.shape
maxYTrackbar = int(height)
maxXTrackbar = int(width)

#
# create a single window with all the sliders
#
cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Trackbars', 600, 100)

cv2.createTrackbar("p1-x", 'Trackbars', int(p1x * width), maxXTrackbar, p1xChanged)
cv2.createTrackbar("p1-y", 'Trackbars', int(p1y * height), maxYTrackbar, p1yChanged)
cv2.createTrackbar("p2-x", 'Trackbars', int(p2x * width), maxXTrackbar, p2xChanged)
cv2.createTrackbar("p2-y", 'Trackbars', int(p2y * height), maxYTrackbar, p2yChanged)
cv2.createTrackbar("p3-x", 'Trackbars', int(p3x * width), maxXTrackbar, p3xChanged)
cv2.createTrackbar("p3-y", 'Trackbars', int(p3y * height), maxYTrackbar, p3yChanged)
cv2.createTrackbar("p4-x", 'Trackbars', int(p4x * width), maxXTrackbar, p4xChanged)
cv2.createTrackbar("p4-y", 'Trackbars', int(p4y * height), maxYTrackbar, p4yChanged)

print("Calibrating camera....")
mtx, dist = camera.calibrate()
print("Done.")

redrawFiles()
plt.show()

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
