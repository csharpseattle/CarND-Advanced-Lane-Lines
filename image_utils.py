import numpy as np
import cv2


def threshold_image(img, l_threshold, b_threshold, brightness_threshold):
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
            if (mean < brightness_threshold):
                l_thresh = l_threshold[1]
                b_thresh = b_threshold[1]

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
            combined[(bbinary == 1) | (lbinary == 1)] = 1

            out_img[(h//splits_y) * i:(h//splits_y) * (i+1), (w//splits_x) * j: (w//splits_x) * (j+1)] = combined

    return out_img



'''
warps and image from perspective to bird's eye or
the inverse, bird's eye back to perspective.
'''
def warp_image(img, src_points, dst_points):
    h, w = img.shape[:2]

    #
    # get the transform matrix
    #
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    #
    # use cv2.warpPerspective() to warp the image.
    #
    warped = cv2.warpPerspective(img, M, (w, h), cv2.INTER_LINEAR)

    return warped



"""
Applies an image mask.

Only keeps the region of the image defined by the polygon
formed from `vertices`. The rest of the image is set to black.
`vertices` should be a numpy array of integer points.
"""
def region_of_interest(img, vertices):

    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    verts = np.array([vertices], dtype=np.int32)
    cv2.fillPoly(mask, verts, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
