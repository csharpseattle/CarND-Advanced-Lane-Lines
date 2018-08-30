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


BRIGHTNESS_THRESHOLD = 73
L_THRESHOLD_LC  = (179, 255)
L_THRESHOLD  = (203, 255)
B_THRESHOLD_LC  = (139, 255)
B_THRESHOLD  = (137, 255)

BRIGHTNESS_THRESHOLD = 100
L_THRESHOLD_LC = (125, 255)
B_THRESHOLD_LC = (138, 255)
L_THRESHOLD    = (224, 255)
B_THRESHOLD    = (172, 255)


f, axes = plt.subplots(1, 2, figsize=(12, 4.5))
f.tight_layout()
plt.subplots_adjust(left=0.03, right=.97, top=0.99, bottom=0.)

def Contrast_Changed(x):
    global BRIGHTNESS_THRESHOLD
    BRIGHTNESS_THRESHOLD = x
    redrawFiles()

def l_threshLowLCChanged(x):
    global L_THRESHOLD_LC
    L_THRESHOLD_LC = (x, L_THRESHOLD_LC[1])
    redrawFiles()

def l_threshHighLCChanged(x):
    global L_THRESHOLD_LC
    L_THRESHOLD_LC = (L_THRESHOLD_LC[0], x)
    redrawFiles()

def b_threshLowLCChanged(x):
    global B_THRESHOLD_LC
    B_THRESHOLD_LC = (x, B_THRESHOLD_LC[1])
    redrawFiles()

def b_threshHighLCChanged(x):
    global B_THRESHOLD_LC
    B_THRESHOLD_LC = (B_THRESHOLD_LC[0], x)
    redrawFiles()

def l_threshLowChanged(x):
    global L_THRESHOLD
    L_THRESHOLD = (x, L_THRESHOLD[1])
    redrawFiles()

def l_threshHighChanged(x):
    global L_THRESHOLD
    L_THRESHOLD = (L_THRESHOLD[0], x)
    redrawFiles()

def b_threshLowChanged(x):
    global B_THRESHOLD
    B_THRESHOLD = (x, B_THRESHOLD[1])
    redrawFiles()

def b_threshHighChanged(x):
    global B_THRESHOLD
    B_THRESHOLD = (B_THRESHOLD[0], x)
    redrawFiles()


def threshold_image(img):
    splits_y = 9
    splits_x = 80
    h, w, c = img.shape
    out_img = np.zeros(img[:, :, 0].shape)

    blur = cv2.GaussianBlur(img, (7, 7), 0)

    for i in range(splits_y):
        for j in range(splits_x):
            partial = blur[(h//splits_y) * i:(h//splits_y) * (i+1), (w//splits_x) * j: (w//splits_x) * (j+1)]
            gray_partial = cv2.cvtColor(partial, cv2.COLOR_RGB2GRAY)
            mean = np.mean(gray_partial)

            l_thresh = L_THRESHOLD
            b_thresh = B_THRESHOLD
            if (mean < BRIGHTNESS_THRESHOLD):
                l_thresh = L_THRESHOLD_LC
                b_thresh = B_THRESHOLD_LC

            #
            #
            # convert to LUV color space and threshold the l values.
            #
            luv = cv2.cvtColor(partial, cv2.COLOR_RGB2LUV)
            l = partial[:, :, 0]
            lbinary = np.zeros_like(l)
            lbinary[(l >= l_thresh[0]) & (l <= l_thresh[1])] = 1


            #
            # convert to LAB color space and threshold the b values.
            #
            lab = cv2.cvtColor(partial, cv2.COLOR_RGB2LAB)
            b = lab[:, :, 2]
            bbinary = np.zeros_like(b)
            bbinary[(b >= b_thresh[0]) & (b <= b_thresh[1])] = 1

            #
            # Combine the binaries and return
            #
            combined = np.zeros_like(b)
            combined[(bbinary == 1) | (lbinary == 1)] = 1

            out_img[(h//splits_y) * i:(h//splits_y) * (i+1), (w//splits_x) * j: (w//splits_x) * (j+1)] = combined

    return out_img



def redrawFiles():
    print("--------------------------")
    print("BRIGHTNESS_THRESHOLD = " + str(BRIGHTNESS_THRESHOLD))
    print("L_THRESHOLD_LC  = " + str(L_THRESHOLD_LC))
    print("L_THRESHOLD  = " + str(L_THRESHOLD))
    print("B_THRESHOLD_LC  = " + str(B_THRESHOLD_LC))
    print("B_THRESHOLD  = " + str(B_THRESHOLD))

    #
    # Read in the image file
    #
    img = mpimg.imread(filename, 1)

    out_img = threshold_image(img)



    axes[0].clear()
    axes[0].imshow(img, cmap='gray')
    axes[1].clear()
    axes[1].imshow(out_img, cmap='gray')

    plt.draw()





#######################################################################################
cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Trackbars', 600, 100)

cv2.createTrackbar("L_LC  Low", 'Trackbars', L_THRESHOLD_LC[0], 255, l_threshLowLCChanged)
cv2.createTrackbar("L_LC  HI", 'Trackbars', L_THRESHOLD_LC[1], 255, l_threshHighLCChanged)
cv2.createTrackbar("B_LC  Low", 'Trackbars', B_THRESHOLD_LC[0], 255, b_threshLowLCChanged)
cv2.createTrackbar("B_LC  HI", 'Trackbars', B_THRESHOLD_LC[1], 255, b_threshHighLCChanged)
cv2.createTrackbar("L  Low", 'Trackbars', L_THRESHOLD[0], 255, l_threshLowChanged)
cv2.createTrackbar("L  HI", 'Trackbars', L_THRESHOLD[1], 255, l_threshHighChanged)
cv2.createTrackbar("B  Low", 'Trackbars', B_THRESHOLD[0], 255, b_threshLowChanged)
cv2.createTrackbar("B  HI", 'Trackbars', B_THRESHOLD[1], 255, b_threshHighChanged)
cv2.createTrackbar("BRIGHTNESS", 'Trackbars', BRIGHTNESS_THRESHOLD, 255, Contrast_Changed)

redrawFiles()
plt.show()

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
