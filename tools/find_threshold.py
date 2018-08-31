#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math
import sys

sys.path.append("..")
import image_utils

#reading in an image
if len(sys.argv) < 2:
    filename = 'frame1.jpg'
    if not os.path.isfile(filename):
        print("Please provide an image filename on the command line")
        sys.exit()
else:
    filename = sys.argv[1]

BRIGHTNESS_THRESHOLD = 100
L_THRESHOLD_LC = (125, 255)
B_THRESHOLD_LC = (138, 255)
L_THRESHOLD    = (224, 255)
B_THRESHOLD    = (172, 255)
R_THRESHOLD_LC = (224, 255)
R_THRESHOLD    = (172, 255)
G_THRESHOLD_LC = (224, 255)
G_THRESHOLD    = (172, 255)


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

def r_threshLowLCChanged(x):
    global R_THRESHOLD_LC
    R_THRESHOLD_LC = (x, R_THRESHOLD_LC[1])
    redrawFiles()

def r_threshHighLCChanged(x):
    global R_THRESHOLD_LC
    R_THRESHOLD_LC = (R_THRESHOLD_LC[0], x)
    redrawFiles()

def g_threshLowLCChanged(x):
    global G_THRESHOLD_LC
    G_THRESHOLD_LC = (x, G_THRESHOLD_LC[1])
    redrawFiles()

def g_threshHighLCChanged(x):
    global G_THRESHOLD_LC
    G_THRESHOLD_LC = (G_THRESHOLD_LC[0], x)
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

def r_threshLowChanged(x):
    global R_THRESHOLD
    R_THRESHOLD = (x, R_THRESHOLD[1])
    redrawFiles()

def r_threshHighChanged(x):
    global R_THRESHOLD
    R_THRESHOLD = (R_THRESHOLD[0], x)
    redrawFiles()

def g_threshLowChanged(x):
    global G_THRESHOLD
    G_THRESHOLD = (x, G_THRESHOLD[1])
    redrawFiles()

def g_threshHighChanged(x):
    global G_THRESHOLD
    G_THRESHOLD = (G_THRESHOLD[0], x)
    redrawFiles()



def threshold_image(img, l_threshold, b_threshold, r_threshold, g_threshold, brightness_threshold):
    splits_y = 9
    splits_x = 20
    h, w = img.shape[:2]
    out_img = np.zeros(img[:, :, 0].shape)

    blur = cv2.GaussianBlur(img, (7, 7), 0)

    for i in range(splits_y):
        for j in range(splits_x):
            partial = blur[(h//splits_y) * i:(h//splits_y) * (i+1), (w//splits_x) * j: (w//splits_x) * (j+1)]
            gray_partial = cv2.cvtColor(partial, cv2.COLOR_RGB2GRAY)
            mean = np.mean(gray_partial)

            l_thresh = l_threshold[0]
            b_thresh = b_threshold[0]
            r_thresh = r_threshold[0]
            g_thresh = g_threshold[0]
            if (mean < brightness_threshold):
                l_thresh = l_threshold[1]
                b_thresh = b_threshold[1]
                r_thresh = r_threshold[1]
                g_thresh = g_threshold[1]


            r = partial[:, :, 0]
            rbinary = np.zeros_like(r)
            rbinary[(r >= r_thresh[0]) & (r <= r_thresh[1])] = 1

            g = partial[:, :, 1]
            gbinary = np.zeros_like(g)
            gbinary[(g >= g_thresh[0]) & (g <= g_thresh[1])] = 1
            #
            #
            # convert to LUV color space and threshold the l values.
            #
            luv = cv2.cvtColor(partial, cv2.COLOR_RGB2LUV)
            l = luv[:, :, 0]
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
            combined[(rbinary == 1) | (gbinary == 1) | (bbinary == 1) | (lbinary == 1)] = 1

            out_img[(h//splits_y) * i:(h//splits_y) * (i+1), (w//splits_x) * j: (w//splits_x) * (j+1)] = combined

    return out_img




def redrawFiles():
    print("--------------------------")
    print("BRIGHTNESS_THRESHOLD = " + str(BRIGHTNESS_THRESHOLD))
    print("L_THRESHOLD_LC  = " + str(L_THRESHOLD_LC))
    print("L_THRESHOLD  = " + str(L_THRESHOLD))
    print("B_THRESHOLD_LC  = " + str(B_THRESHOLD_LC))
    print("B_THRESHOLD  = " + str(B_THRESHOLD))
    print("R_THRESHOLD_LC  = " + str(R_THRESHOLD_LC))
    print("R_THRESHOLD  = " + str(R_THRESHOLD))
    print("G_THRESHOLD_LC  = " + str(G_THRESHOLD_LC))
    print("G_THRESHOLD  = " + str(G_THRESHOLD))

    #
    # Read in the image file
    #
    img = mpimg.imread(filename, 1)

    l = np.array([L_THRESHOLD, L_THRESHOLD_LC])
    b = np.array([B_THRESHOLD, B_THRESHOLD_LC])
    r = np.array([R_THRESHOLD, R_THRESHOLD_LC])
    g = np.array([G_THRESHOLD, G_THRESHOLD_LC])
    out_img = threshold_image(img, l, b, r, g,  BRIGHTNESS_THRESHOLD)

    #
    # To output file:
    #
    stack = np.dstack((out_img, out_img, out_img)) * 255
    cv2.imwrite('filename.jpg', stack)

    axes[0].clear()
    axes[0].imshow(img, cmap='gray')
    axes[1].clear()
    axes[1].imshow(out_img, cmap='gray')

    plt.draw()





#######################################################################################
cv2.namedWindow('R', cv2.WINDOW_NORMAL)
cv2.resizeWindow('R', 600, 100)

cv2.namedWindow('L', cv2.WINDOW_NORMAL)
cv2.resizeWindow('L', 600, 100)

cv2.namedWindow('B', cv2.WINDOW_NORMAL)
cv2.resizeWindow('B', 600, 100)

cv2.namedWindow('G', cv2.WINDOW_NORMAL)
cv2.resizeWindow('G', 600, 100)

cv2.createTrackbar("L_LC  Low", 'L', L_THRESHOLD_LC[0], 255, l_threshLowLCChanged)
cv2.createTrackbar("L_LC  HI", 'L', L_THRESHOLD_LC[1], 255, l_threshHighLCChanged)
cv2.createTrackbar("L  Low", 'L', L_THRESHOLD[0], 255, l_threshLowChanged)
cv2.createTrackbar("L  HI", 'L', L_THRESHOLD[1], 255, l_threshHighChanged)

cv2.createTrackbar("B  Low", 'B', B_THRESHOLD[0], 255, b_threshLowChanged)
cv2.createTrackbar("B  HI", 'B', B_THRESHOLD[1], 255, b_threshHighChanged)
cv2.createTrackbar("B_LC  Low", 'B', B_THRESHOLD_LC[0], 255, b_threshLowLCChanged)
cv2.createTrackbar("B_LC  HI", 'B', B_THRESHOLD_LC[1], 255, b_threshHighLCChanged)

cv2.createTrackbar("R_LC  Low", 'R', R_THRESHOLD_LC[0], 255, r_threshLowLCChanged)
cv2.createTrackbar("R_LC  HI", 'R', R_THRESHOLD_LC[1], 255, r_threshHighLCChanged)
cv2.createTrackbar("R  Low", 'R', R_THRESHOLD[0], 255, r_threshLowChanged)
cv2.createTrackbar("R  HI", 'R', R_THRESHOLD[1], 255, r_threshHighChanged)

cv2.createTrackbar("G  Low", 'G', G_THRESHOLD[0], 255, g_threshLowChanged)
cv2.createTrackbar("G  HI", 'G', G_THRESHOLD[1], 255, g_threshHighChanged)
cv2.createTrackbar("G_LC  Low", 'G', G_THRESHOLD_LC[0], 255, g_threshLowLCChanged)
cv2.createTrackbar("G_LC  HI", 'G', G_THRESHOLD_LC[1], 255, g_threshHighLCChanged)

cv2.createTrackbar("BRIGHTNESS", 'R', BRIGHTNESS_THRESHOLD, 255, Contrast_Changed)

redrawFiles()
plt.show()

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
