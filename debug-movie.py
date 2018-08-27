import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

roi_points = np.array([(0.15625, 0.92917), (0.51016, 0.63194), (0.57344, 0.63194), (0.92266, 0.92917)])


R_THRESHOLD_LOW  = 15
R_THRESHOLD_HIGH = 255
L_THRESHOLD_LOW  = 208
L_THRESHOLD_HIGH = 255
B_THRESHOLD_LOW  = 144
B_THRESHOLD_HIGH = 255
frame_number = 0

left_fit = []
right_fit = []

'''
The goals / steps of this project are the following:

- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
'''

class Line():
    def __init__(self):
        self.detected = False


def threshold_image(img):
    #
    # Apply Sobel and Threshold the R values.
    #
    r = img[:, :, 0]
    sobelr = cv2.Sobel(r, cv2.CV_64F, 1, 0)
    abs_sobelr = np.absolute(sobelr)
    scaled_sobelr = np.uint8(255 * abs_sobelr / np.max(abs_sobelr))
    rbinary = np.zeros_like(scaled_sobelr)
    rbinary[(scaled_sobelr >= R_THRESHOLD_LOW) & (scaled_sobelr <= R_THRESHOLD_HIGH)] = 1

    #
    # convert to LUV color space and threshold the l values.
    #
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l = luv[:, :, 0]
    lbinary = np.zeros_like(l)
    lbinary[(l >= L_THRESHOLD_LOW) & (l <= L_THRESHOLD_HIGH)] = 1

    #
    # convert to LAB color space and threshold the b values.
    #
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    b = lab[:, :, 2]
    bbinary = np.zeros_like(b)
    bbinary[(b >= B_THRESHOLD_LOW) & (b <= B_THRESHOLD_HIGH)] = 1

    #
    # Combine the three and return
    #
    combined = np.zeros_like(b)
    #combined[(rbinary == 1) | (bbinary == 1) | (lbinary == 1)] = 1
    combined[(bbinary == 1) | (lbinary == 1)] = 1

    return combined


def calibrate_camera():
    num_x_corners = 9
    num_y_corners = 6

    obj_points = []
    img_points = []

    #
    # Prepare the object points.  These are the same for each calibration image.
    # the reshaped mgrid will give us an array like:
    # array([[ 0.,  0.,  0.],
    #        [ 1.,  0.,  0.],
    #        [ 2.,  0.,  0.],
    #        [ 3.,  0.,  0.],
    #        [ 4.,  0.,  0.],
    #        [ 5.,  0.,  0.],
    #        [ 6.,  0.,  0.],
    #        [ 7.,  0.,  0.],
    #        [ 0.,  1.,  0.],
    #        [ 1.,  1.,  0.],
    #         ...
    #        [ num_x_corners-1, num_y_corners-1, 0]])
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

def measure_curvature(ploty):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 15/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = ((1 + ((2 * left_fit[0] * y_eval) + left_fit[1]) ** 2) ** float(3/2)) / abs(2 * left_fit[0])
    right_curverad = ((1 + ((2 * right_fit[0] * y_eval) + right_fit[1]) ** 2) ** float(3/2)) / abs(2 * right_fit[0])

    return left_curverad, right_curverad



def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low), (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low), (win_xright_high,win_y_high),(0,255,0), 2)

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if (len(good_left_inds) >= minpix):
            avg_index = np.int(np.mean(nonzerox[good_left_inds]))
            leftx_current = avg_index

        if (len(good_right_inds) >= minpix):
            avg_index = np.int(np.mean(nonzerox[good_right_inds]))
            rightx_current = avg_index

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    global frame_number
    global left_fit
    global right_fit

    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    #Fit a  polynomial to each lane using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    right_fitx = np.clip(right_fitx, 0, 1279)
    out_img[ploty.astype(int), left_fitx.astype(int)] = [0, 255, 0]
    out_img[ploty.astype(int), right_fitx.astype(int)] = [0, 255, 0]

    return out_img


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    global left_fit
    global right_fit

    #
    # Fit polynomial to each line with np.polyfit()
    #
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    #
    # Generate x and y values for plotting
    #
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])

    #
    # Calc both polynomials using ploty, left_fit and right_fit ###
    #
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    return left_fitx, right_fitx, ploty


def search_around_poly(binary_warped):
    global frame_number
    global left_fit
    global right_fit

    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    right_fitx = np.clip(right_fitx, 0, 1279)
    result[ploty.astype(int), left_fitx.astype(int)] = [0, 255, 0]
    result[ploty.astype(int), right_fitx.astype(int)] = [0, 255, 0]

    #
    # Measure Curvature
    #
    leftrad, rightrad = measure_curvature(ploty)
    avg = (leftrad + rightrad) // 2
    avgText = "Curvature - " + str(avg)

    cv2.putText(result,avgText, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return result



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
    verts = np.array([vertices], dtype=np.int32)
    cv2.fillPoly(mask, verts, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def process_frame(frame):
    global frame_number
    global mtx
    global dist
    global roi_points

    frame_number += 1

    #
    # Undistort the image using the camera matrix and
    # distortion coefficients.
    #
    undist = cv2.undistort(frame, mtx, dist, None, mtx)
    h, w, _ = undist.shape

    #
    # Mask the region of interest
    #
    points = roi_points * (w, h)
    roi = region_of_interest(undist, points)


    #thresholded = threshold_image(undist)

    #
    # Source and destination points are normalized coordinates.  Translate
    # to screen coordinates using the shape of the image.
    #
    src_points = np.float32([(0.18671875, 0.91667), (0.30703125, 0.7875), (0.771875, 0.7875), (0.94453125, 0.91667)])

    # src_points for project_video
    src_points = np.float32([(0.18671875, 0.91667), (0.44375, 0.6222222222222222), (0.5739583333333333, 0.6222222222222222), (0.94453125, 0.91667)])

    # src_points for challenge video:
    src_points = np.float32([(0.18671875, 0.91667), (0.38125, 0.7361111111111112), (0.70390625, 0.7361111111111112), (0.94453125, 0.91667)])

    dst_points = np.float32([(src_points[0][0], 1.0), (src_points[0][0], 0.0), (src_points[3][0], 0.0), (src_points[3][0], 1.0)])

    # src_points[:, 0] *= w
    # src_points[:, 1] *= h
    # dst_points[:, 0] *= w
    # dst_points[:, 1] *= h
    src_points *= (w, h)
    dst_points *= (w, h)

    #
    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    #
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    #
    # use cv2.warpPerspective() to warp your image to a top-down view
    #
    warped = cv2.warpPerspective(roi, M, (w, h), cv2.INTER_LINEAR)

    thresholded = threshold_image(warped)


    if len(left_fit) == 0 or len(right_fit) == 0:
        out_img = fit_polynomial(thresholded)
    else:
        out_img = search_around_poly(thresholded)

    topstack = np.hstack((undist, out_img))
    bottomstack = np.hstack((warped, np.dstack((thresholded, thresholded, thresholded))*255))
    final = np.vstack((topstack, bottomstack))

    return final


################################################################################
################################################################################
################################################################################
mtx, dist = calibrate_camera()

video_output = 'myvideo.mp4'
#video_output = 'dude.mp4'
#clip2 = VideoFileClip('project_video.mp4').subclip(0, 10)
clip2 = VideoFileClip('project_video.mp4')
#clip2 = VideoFileClip('challenge_video.mp4').subclip(0,0.125)
video_clip = clip2.fl_image(process_frame)
video_clip.write_videofile(video_output, audio=False)
