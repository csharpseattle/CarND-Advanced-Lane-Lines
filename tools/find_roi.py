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

image = mpimg.imread(filename)

def p1xChanged(x):
    global p1x
    p1x = x
    redrawFiles()

def lower_y_changed(x):
    global p1y
    global p4y
    p1y = x
    p4y = x
    redrawFiles()

def p2xChanged(x):
    global p2x
    p2x = x
    redrawFiles()

def upper_y_changed(x):
    global p2y
    global p3y
    p2y = x
    p3y = x
    redrawFiles()

def p3xChanged(x):
    global p3x
    global p3x
    p3x = x
    redrawFiles()


def p4xChanged(x):
    global p4x
    p4x = x
    redrawFiles()


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image




def redrawFiles():
    global p1x
    global p2x
    global p3x
    global p4x
    global p1y
    global p2y
    global p3y
    global p4y
    global roi_points


    img = mpimg.imread(filename, 1)
    h, w, _ = img.shape

    points = np.array([(p1x, p1y), (p2x, p2y), (p3x, p3y), (p4x, p4y)])
    normed = points / (w, h)

    print("--------------------------")
    print("ROI_POINTS = np.array([(" + str(round(normed[0][0], 5)) + ", " + str(round(normed[0][1], 5)) + "), ("+ str(round(normed[1][0], 5)) + ", " + str(round(normed[1][1], 5)) + "), (" + str(round(normed[2][0], 5)) + ", " + str(round(normed[2][1], 5)) + "), ("+ str(round(normed[3][0], 5)) + ", " + str(round(normed[3][1], 5)) + ")])")
    print("--------------------------")

    verts = np.array([points], dtype=np.int32)
    masked = region_of_interest(img, verts)

    #out_image = weighted_img(lines, img)
    #out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Original", masked)




#######################################################################################

ROI_POINTS = np.array([(0.02969, 0.93472), (0.19688, 0.60278), (0.92109, 0.60278), (1.0, 0.93472)])

# create trackbars if in interactive mode
shapefinder = cv2.imread(filename, 1)
height, width, _ = shapefinder.shape
maxYTrackbar = int(height)
maxXTrackbar = int(width)
points = ROI_POINTS * (width, height)

p1x = int(points[0][0])
p1y = int(points[0][1])
p2x = int(points[1][0])
p2y = int(points[1][1])
p3x = int(points[2][0])
p3y = int(points[2][1])
p4x = int(points[3][0])
p4y = int(points[3][1])


#
# Create a window for all the stacked originals
#
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)


#
# create a single window with all the sliders
#
cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Trackbars', 600, 100)

cv2.createTrackbar("p1-x", 'Trackbars', p1x, maxXTrackbar, p1xChanged)
cv2.createTrackbar("lower-y", 'Trackbars', p1y, maxYTrackbar, lower_y_changed)
cv2.createTrackbar("p2-x", 'Trackbars', p2x, maxXTrackbar, p2xChanged)
cv2.createTrackbar("p3-x", 'Trackbars', p3x, maxXTrackbar, p3xChanged)
cv2.createTrackbar("upper-y", 'Trackbars', p3y, maxYTrackbar, upper_y_changed)
cv2.createTrackbar("p4-x", 'Trackbars', p4x, maxXTrackbar, p4xChanged)


redrawFiles()

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
