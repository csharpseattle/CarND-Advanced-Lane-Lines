#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math
import sys

#reading in an image
if len(sys.argv) < 2:
    filename = 'frame1.jpg'
    if not os.path.isfile(filename):
        print("Please provide an image filename on the command line")
        sys.exit()
else:
    filename = sys.argv[1]

f, axes = plt.subplots(2, 2, figsize=(12, 4.5))
f.tight_layout()
plt.subplots_adjust(left=0.03, right=.97, top=0.99, bottom=0.)


def l_threshLowChanged(x):
    global L_THRESHOLD_LOW
    L_THRESHOLD_LOW = x
    redrawFiles()

def l_threshHighChanged(x):
    global L_THRESHOLD_HIGH
    L_THRESHOLD_HIGH = x
    redrawFiles()

def b_threshLowChanged(x):
    global B_THRESHOLD_LOW
    B_THRESHOLD_LOW = x
    redrawFiles()

def b_threshHighChanged(x):
    global B_THRESHOLD_HIGH
    B_THRESHOLD_HIGH = x
    redrawFiles()


def redrawFiles():
    print("--------------------------")
    print("L_THRESHOLD_LOW  = " + str(L_THRESHOLD_LOW))
    print("L_THRESHOLD_HIGH = " + str(L_THRESHOLD_HIGH))
    print("B_THRESHOLD_LOW  = " + str(B_THRESHOLD_LOW))
    print("B_THRESHOLD_HIGH = " + str(B_THRESHOLD_HIGH))


     #
    # Read in the image file
    #
    img = mpimg.imread(filename, 1)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    #
    #
    # convert to LUV color space and threshold the l values.
    #
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l = luv[:, :, 0]
    sobell = cv2.Sobel(l, cv2.CV_64F, 1, 0)
    abs_sobell = np.absolute(sobell)
    scaled_sobell = np.uint8(255 * abs_sobell / np.max(abs_sobell))
    lbinary = np.zeros_like(scaled_sobell)
    lbinary[(l >= L_THRESHOLD_LOW) & (l <= L_THRESHOLD_HIGH)] = 1

    #
    # convert to LAB color space and threshold the b values.
    #
    # lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # b = lab[:, :, 2]
    # bbinary = np.zeros_like(b)
    # bbinary[(b >= B_THRESHOLD_LOW) & (b <= B_THRESHOLD_HIGH)] = 1

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    b = lab[:, :, 2]
    sobel_b = cv2.Sobel(b, cv2.CV_64F, 1, 0)
    abs_sobel_b = np.absolute(sobel_b)
    scaled_sobel_b = np.uint8(255 * abs_sobel_b / np.max(abs_sobel_b))
    bbinary = np.zeros_like(scaled_sobel_b)
    bbinary[(b >= B_THRESHOLD_LOW) & (b <= B_THRESHOLD_HIGH)] = 1

    #
    # Combine the three and return
    #
    combined = np.zeros_like(b)
    combined[(bbinary == 1) | (lbinary == 1)] = 1


    axes[0, 0, ].clear()
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 1, ].clear()
    axes[0, 1].imshow(lbinary, cmap='gray')
    axes[1, 0, ].clear()
    axes[1, 0].imshow(bbinary, cmap='gray')
    axes[1, 1, ].clear()
    axes[1, 1].imshow(combined, cmap='gray')

    plt.draw()





#######################################################################################
L_THRESHOLD_LOW  = 118
L_THRESHOLD_HIGH = 255
B_THRESHOLD_LOW  = 159
B_THRESHOLD_HIGH = 255

L_THRESHOLD_LOW  = 199
L_THRESHOLD_HIGH = 255
B_THRESHOLD_LOW  = 159
B_THRESHOLD_HIGH = 255

# This could work for low contrast with Sobel on l and b 48
# L_THRESHOLD_LOW  = 235
# L_THRESHOLD_HIGH = 255
# B_THRESHOLD_LOW  = 142
# B_THRESHOLD_HIGH = 146

# HIGH Contrast 147
# L_THRESHOLD_LOW  = 250
# L_THRESHOLD_HIGH = 255
# B_THRESHOLD_LOW  = 139
# B_THRESHOLD_HIGH = 255

L_THRESHOLD_LOW  = 224
L_THRESHOLD_HIGH = 255
B_THRESHOLD_LOW  = 172
B_THRESHOLD_HIGH = 255


#
# create a single window with all the sliders
#
cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Trackbars', 600, 100)

cv2.createTrackbar("L  Low", 'Trackbars', L_THRESHOLD_LOW, 255, l_threshLowChanged)
cv2.createTrackbar("L  HI", 'Trackbars', L_THRESHOLD_HIGH, 255, l_threshHighChanged)
cv2.createTrackbar("B  Low", 'Trackbars', B_THRESHOLD_LOW, 255, b_threshLowChanged)
cv2.createTrackbar("B  HI", 'Trackbars', B_THRESHOLD_HIGH, 255, b_threshHighChanged)

redrawFiles()
plt.show()

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
