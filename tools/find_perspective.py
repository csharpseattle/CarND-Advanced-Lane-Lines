#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import sys
import math

#reading in an image
if len(sys.argv) < 2:
    filename = 'frame1.jpg'
    if not os.path.isfile(filename):
        print("Please provide an image filename on the command line")
        sys.exit()
else:
    filename = sys.argv[1]

img = mpimg.imread(filename, 1)
h, w, _ = img.shape

f, axes = plt.subplots(1, 2, figsize=(24, 9))
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


def calibrate_camera():
    num_x_corners = 9
    num_y_corners = 6

    obj_points = []
    img_points = []

    obj_p = np.zeros((num_x_corners * num_y_corners,  3), np.float32)
    obj_p[:, :2] = np.mgrid[0:num_x_corners, 0:num_y_corners].T.reshape(-1, 2)


    images = os.listdir("camera_cal")
    for imagefile in images:
        #
        # Read in the image file and convert to grayscale
        #
        img = mpimg.imread('camera_cal/' + imagefile)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #
        # use cv2 findChessboardCorners to get list of image corners.
        # If found, append these corners to the img_points array.
        # Note that the obj_points are the same for each new imagefile.
        #
        ret, corners = cv2.findChessboardCorners(gray, (num_x_corners, num_y_corners), None)

        if (ret == True):
            img_points.append(corners)
            obj_points.append(obj_p)

    #
    # use the img_points to pass to  opencv calibrateCamera()
    # and get the distortion coefficients and
    # camera calibration matrix to translate 2D image points.
    #
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    return mtx, dist



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
    height, width, _ = undist.shape

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

# p1x = 0.18671875
# p1y = 0.91667
# p2x = 0.30703125
# p2y = 0.7875
# p3x = 0.771875
# p3y = 0.7875
# p4x = 0.94453125
# p4y = 0.91667

# p1x = 0.18671875
# p1y = 0.91667
# p2x = 0.3515625
# p2y = 0.7361111111111112
# p3x = 0.70390625
# p3y = 0.7361111111111112
# p4x = 0.94453125
# p4y = 0.91667

# p1x = 0.18671875
# p1y = 0.91667
# p2x = 0.38125
# p2y = 0.7361111111111112
# p3x = 0.70390625
# p3y = 0.7361111111111112
# p4x = 0.94453125
# p4y = 0.91667


TRANSFORM_SRC_POINTS = np.float32([(0.0, 0.91528), (0.21875, 0.73472), (0.78281, 0.73472), (1.0, 0.91528)])

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

mtx, dist = calibrate_camera()
redrawFiles()
plt.show()

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
