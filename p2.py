import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

np.set_printoptions(suppress=True)

R_THRESHOLD_LOW  = 15
R_THRESHOLD_HIGH = 255
L_THRESHOLD_LOW  = 175
L_THRESHOLD_HIGH = 255
B_THRESHOLD_LOW  = 134
B_THRESHOLD_HIGH = 255


METERS_PER_PIXEL_X = 3.7/700
METERS_PER_PIXEL_Y = 30/720

frame_number = -1

average_curvature = []
average_curvature_over_x_frames = 10

average_left_coefficients = []
average_right_coefficients = []
average_coefficients_over_x_frames = 5

#
# Perspective transform matrix and inverse
#
M = None
M_inv = None

'''
The goals / steps of this project are the following:

- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
vvv- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
'''
class Frame():
    def __init__(self, number, w, h):
        self.frame_number = number
        self.left_line = None
        self.right_line = None
        self.frame_width = w
        self.frame_height = h

    def getLanePoints(self):
        left_pts = np.array([self.left_line.getLinePoints()])
        right_pts = np.array([np.flipud(self.right_line.getLinePoints())])
        return np.hstack((left_pts, right_pts))

    def measure_offset(self, img = None):
        rightmax = self.right_line.getLowestLinePoint()
        leftmax = self.left_line.getLowestLinePoint()

        lane_center = (rightmax + leftmax) / 2
        center_screen = (self.frame_width / 2)

        if img is not None:
            cv2.line(img, (int(lane_center), 719), (int(lane_center), 600), [255, 0, 0], 2)
            cv2.line(img, (int(rightmax), 719), (int(rightmax), 520), [255, 0, 0], 2)
            cv2.line(img, (int(leftmax), 719), (int(leftmax), 520), [255, 255, 0], 2)
            cv2.line(img, (int(center_screen), 719), (int(center_screen), 620), [0, 0, 255], 2)

        offset = round((center_screen - lane_center) * METERS_PER_PIXEL_X, 2)
        return offset



class Line:
    def __init__(self, x_pixels, y_pixels, height):
        self.fitx = []
        self.fity = []
        self.lane_pixels_x = x_pixels
        self.lane_pixels_y = y_pixels

        #
        # fit a polynomial using numpy polyfit
        #
        self.coeff = np.polyfit(self.lane_pixels_y, self.lane_pixels_x, 2)

        #
        # Generate values for plotting
        #
        self.fity = np.linspace(0, height-1, height)
        self.fitx = (self.coeff[0] * self.fity ** 2) + (self.coeff[1] * self.fity) + self.coeff[2]

    def getLineCoefficients(self):
            return self.coeff

    def getLinePoints(self):
        return np.transpose(np.vstack([self.fitx, self.fity]))


    def measure_curvature(self):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        #
        # convert our line coordinates from pixel space to world space.
        #
        fity = self.fity * METERS_PER_PIXEL_Y
        fitx = self.fitx * METERS_PER_PIXEL_X

        #
        # Get the maximum y value to calculate the radius of curvature.
        #
        maxy = np.max(fity)

        #
        # fit to the new world space coordinates and calculate curvature
        #
        world_poly = np.polyfit(fity, fitx, 2)
        curvature = ((1 + ((2 * world_poly[0] * maxy) + world_poly[1]) ** 2) ** float(3/2)) / abs(2 * world_poly[0])
        return curvature


    def getLowestLinePoint(self):
        #
        # Used to get the lowest point of the lane line, closest to the car,
        # when measuring the curvature of the lane.
        # I use the maximum y value (bottom of the screen) and plug into the
        # polynomial to calculate the x value
        #
        y = np.max(self.fity)
        return (self.coeff[0] * y ** 2) + (self.coeff[1] * y) + self.coeff[2]



def calibrate_camera(image_path, x_corners, y_corners):
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
    #        [ x_corners-1, y_corners-1, 0]])
    obj_p = np.zeros((x_corners * y_corners,  3), np.float32)
    obj_p[:, :2] = np.mgrid[0:x_corners, 0:y_corners].T.reshape(-1, 2)


    images = os.listdir(image_path)
    for imagefile in images:
        #
        # Read in the image file and convert to grayscale
        #
        img = mpimg.imread(image_path + '/' + imagefile)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #
        # use cv2 findChessboardCorners to get list of image corners.
        # If found, append these corners to the img_points array.
        # Note that the obj_points are the same for each new imagefile.
        #
        ret, corners = cv2.findChessboardCorners(gray, (x_corners, y_corners), None)

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



def getPerspectiveTransform(w, h):
    global M
    global M_inv

    if M is None:
        #
        # Source and destination points are normalized coordinates.
        # Translate to screen coordinates using the shape of the image.
        #

        #src_points = np.float32([(0.18671875, 0.91667), (0.30703125, 0.7875), (0.771875, 0.7875), (0.94453125, 0.91667)])
        src_points = np.float32([(0.18671875, 0.91667), (0.44375, 0.6222222222222222), (0.5739583333333333, 0.6222222222222222), (0.94453125, 0.91667)])
        dst_points = np.float32([(src_points[0][0], 1.0), (src_points[0][0], 0.0), (src_points[3][0], 0.0), (src_points[3][0], 1.0)])

        src_points[:, 0] *= w
        src_points[:, 1] *= h
        dst_points[:, 0] *= w
        dst_points[:, 1] *= h

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        M_inv = cv2.getPerspectiveTransform(dst_points, src_points)

    return M, M_inv



def fillLane(avg_left, avg_right, Minv, undist, warped):
    #
    # find our pixels to shade
    width, height, _ = undist.shape
    fity = np.linspace(0, height-1, height)
    left_fitx = (avg_left[0] * fity ** 2) + (avg_left[1] * fity) + avg_left[2]
    right_fitx = (avg_right[0] * fity ** 2) + (avg_right[1] * fity) + avg_right[2]

    #
    # transpose and prepare for fillpoly
    #
    left_transpose = np.transpose(np.vstack([left_fitx, fity]))
    right_transpose = np.transpose(np.vstack([right_fitx, fity]))

    left_pts = np.array([left_transpose])
    right_pts = np.array([np.flipud(right_transpose)])
    points =  np.hstack((left_pts, right_pts))

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    #
    # Draw the lane onto the warped blank image
    #
    cv2.fillPoly(color_warp, np.int_([points]), (0, 255, 0))

    #
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    #
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]), flags=cv2.INTER_LINEAR)

    #
    # Combine the result with the original image
    #
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result



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
    combined[(rbinary == 1) | (bbinary == 1) | (lbinary == 1)] = 1

    return combined


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Create an output image to draw on and visualize the result
    #out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # the number of sliding windows
    nwindows = 9
    # the width of the windows +/- margin
    margin = 100
    # minimum number of pixels found to recenter window
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

        #
        # Identify the nonzero pixels in x and y within the window
        #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if (len(good_left_inds) >= minpix):
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

        if (len(good_right_inds) >= minpix):
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def fit_polynomial(binary_warped):
    global frame_number

    # Find our lane pixels first
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)

    left_line = Line(leftx, lefty, binary_warped.shape[0])
    right_line = Line(rightx, righty, binary_warped.shape[0])

    return left_line, right_line


def search_around_poly(binary_warped, left_fit, right_fit):
    #
    # the width of the margin around the previous polynomial to search
    #
    margin = 100

    #
    # Grab activated pixels
    #
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    #
    # Set the area of search based on activated x-values
    # within the +/- margin of our polynomial function
    #
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))

    #
    # Again, extract left and right line pixel positions
    #
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_line = Line(leftx, lefty, binary_warped.shape[0])
    right_line = Line(rightx, righty, binary_warped.shape[0])

    return left_line, right_line


def process_frame(frame):
    global M
    global M_inv
    global average_curvature
    global frame_number
    global frames
    global average_left_coefficients
    global average_right_coefficients

    frame_number += 1

    # bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('frame' + str(frame_number) + '.jpg', bgr)

    undist = cv2.undistort(frame, mtx, dist, None, mtx)
    h, w, _ = undist.shape

    thresholded = threshold_image(undist)

    #
    # get the transform matrix for warping the image
    # into a bird's eye view
    #
    M, M_inv = getPerspectiveTransform(w, h)

    #
    # use cv2.warpPerspective() to warp the image to a top-down view
    #
    warped = cv2.warpPerspective(thresholded, M, (w, h), flags=cv2.INTER_LINEAR)


    #
    # fit the lines in the current frame
    #
    frame = Frame(frame_number, w, h)
    if (frame_number == 0) or len(frames[frame_number-1].left_line.fitx) == 0 or len(frames[frame_number-1].right_line.fitx) == 0:
        frame.left_line, frame.right_line = fit_polynomial(warped)
    else:
        prev_left_fitx = frames[frame_number-1].left_line.getLineCoefficients()
        prev_right_fitx = frames[frame_number-1].right_line.getLineCoefficients()
        frame.left_line, frame.right_line = search_around_poly(warped, prev_left_fitx, prev_right_fitx)

    frames.append(frame)

    #
    # Calculate the average left and right lines
    #
    current_left = frame.left_line.getLineCoefficients()
    if len(average_left_coefficients) < average_coefficients_over_x_frames:
        average_left_coefficients.append(current_left)
    else:
        if (len(average_left_coefficients) >= average_coefficients_over_x_frames):
            average_left_coefficients.pop(0)

        left_std = np.std(average_left_coefficients, axis = 0)
        left_mean = np.mean(average_left_coefficients, axis = 0)

        if np.all(current_left < left_mean + (0.05 * left_std)) and np.all(current_left > left_mean - (0.05 * left_std)):
            average_left_coefficients.append(current_left)

    current_right = frame.right_line.getLineCoefficients()
    if len(average_right_coefficients) < average_coefficients_over_x_frames:
        average_right_coefficients.append(current_right)
    else:
        if (len(average_right_coefficients) >= average_coefficients_over_x_frames):
            average_right_coefficients.pop(0)

        right_std = np.std(average_right_coefficients, axis = 0)
        right_mean = np.mean(average_right_coefficients, axis = 0)

        if np.all(current_right < right_mean + (0.05 * right_std)) and np.all(current_right > right_mean - (0.05 * right_std)):
            average_right_coefficients.append(current_right)


    avg_left = np.mean(average_left_coefficients, axis = 0)
    avg_right = np.mean(average_right_coefficients, axis = 0)


    out_img = fillLane(avg_left, avg_right, M_inv, undist, warped)

    #
    # Measure Curvature
    #
    leftrad = frame.left_line.measure_curvature()
    rightrad = frame.right_line.measure_curvature()
    avg = (leftrad + rightrad) // 2

    average_curvature.append(avg)

    if (len(average_curvature) > average_curvature_over_x_frames):
        average_curvature.pop(0)


    #
    # Don't update the text so often.  It gets hard to read.
    #
    avgText = "Curvature: " + str(int(np.mean(average_curvature))) + "m"
    cv2.putText(out_img, avgText, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    offsetText = "Center of Lane: " + str(frame.measure_offset(out_img)) + "m"
    cv2.putText(out_img, offsetText, (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    frameText = "Frame Number: " + str(frame_number)
    cv2.putText(out_img, frameText, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return out_img



################################################################################
################################################################################
################################################################################

#
# Setup a list to hold the frame data
#
frames = []

#
# Calibrate the camera. Get camera calibration matrix and
# distortion coefficients
#
mtx, dist = calibrate_camera("camera_cal", 9, 6)

video_output = 'dude.mp4'
#clip2 = VideoFileClip('project_video.mp4').subclip(0, 2)
#clip2 = VideoFileClip('project_video.mp4')
clip2 = VideoFileClip('challenge_video.mp4').subclip(0,4)
video_clip = clip2.fl_image(process_frame)
video_clip.write_videofile(video_output, audio=False)
