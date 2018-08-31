import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from enum import Enum
import project_data
import camera
import image_utils

video_input  = 'project_video.mp4'
if len(sys.argv) > 1:
    video_input = sys.argv[1]
    if not os.path.isfile(video_input):
        print("Could not find file: " + video_input)
        sys.exit()


#
# The Frames list and a number
# for keeping a frame count.
#
FRAMES = []
frame_number = -1

#
# Meters per pixel are used when converting from
# pixel to world space when measuring curvature and
# offset from the center of the lane.
#
METERS_PER_PIXEL_X = 3.7/700
METERS_PER_PIXEL_Y = 30/720

#
# Lane line polynomial coefficients are averaged
# over X frames
#
AVERAGE_OVER_X_FRAMES = 7

#
# Lines with less pixel will be invalidated
#
MIN_PIXELS_FOR_VALID_LINE = 3000

#
# A change of slope of this amount over the polynomial
# will invalidate the line
#
SLOPE_TOLERANCE = 1.2

#
# Perspective transform matrix and inverse
#
M = None
M_inv = None


'''
 Side enum.
 used for specifying a side of
 the lane to a line.
'''
class Side(Enum):
    LEFT = 0
    RIGHT = 1


'''
 the Frame class.

   Packages the frame number of the video along with both of the
 detected lane lines.
'''
class Frame():
    def __init__(self, number):
        self.frame_number = number
        self.left_line = None
        self.right_line = None



'''
 the Line class

  Houses the polynomial and pixel values associated with a lane line
  and provides operations on a lane line.
'''
class Line:
    def __init__(self, x_pixels, y_pixels, height, detected = False):
        self.fitx = []
        self.fity = []
        self.lane_pixels_x = x_pixels
        self.lane_pixels_y = y_pixels
        self.detected = detected

        if len(self.lane_pixels_y) > 0 and len(self.lane_pixels_x) > 0:
            #
            # fit a polynomial using numpy polyfit
            #
            self.line_coefficients = np.polyfit(self.lane_pixels_y, self.lane_pixels_x, 2)

            #
            # Generate values for plotting
            #
            self.fity = np.linspace(0, height-1, height)
            self.fitx = (self.line_coefficients[0] * self.fity ** 2) + (self.line_coefficients[1] * self.fity) + self.line_coefficients[2]

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
            self.curvature = ((1 + ((2 * world_poly[0] * maxy) + world_poly[1]) ** 2) ** float(3/2)) / abs(2 * world_poly[0])

            self.upper_slope = (2 * self.line_coefficients[0] * self.fity[0]) + self.line_coefficients[1]
            self.mid_slope = (2 * self.line_coefficients[0] * self.fity[height//2]) + self.line_coefficients[1]
            self.lower_slope = (2 * self.line_coefficients[0] * self.fity[height-1]) + self.line_coefficients[1]
        else:
            self.detected = False

    def isValid(self):
        return (self.detected == True and
                (len(self.line_coefficients) == 3) and
                (len(self.lane_pixels_x) > MIN_PIXELS_FOR_VALID_LINE) and
                (abs(self.curvature > 200)) and
                (abs(self.upper_slope - self.lower_slope) < 1.2) and
                (abs(self.upper_slope < SLOPE_TOLERANCE)) and
                (abs(self.mid_slope < SLOPE_TOLERANCE)) and
                (abs(self.lower_slope < SLOPE_TOLERANCE)))


    '''
    # Used to get the lowest point of the lane line, closest to the car,
    # when measuring the offset of the center of the car to the center of
    # the lane.
    # The maximum y value is used in the
    # polynomial to calculate the x value at the bottom of the screen
    '''
    def getLowestLinePoint(self):
        y = np.max(self.fity)
        return (self.line_coefficients[0] * y ** 2) + (self.line_coefficients[1] * y) + self.line_coefficients[2]


'''
 Measures the offset of the vehicle from the
 center of the lane.
'''
def measure_offset(left_line, right_line, width):
    #
    # Get the lowest x values (bottom of screen)
    #
    rightmax = right_line.getLowestLinePoint()
    leftmax = left_line.getLowestLinePoint()

    #
    # the center of the lane is the mid point
    # between the two detected lines.
    #
    lane_center = (rightmax + leftmax) / 2
    center_screen = (width / 2)

    #
    # Convert from pixel space to world space.
    #
    offset = round((center_screen - lane_center) * METERS_PER_PIXEL_X, 2)
    return offset



def getLastValidLines(side, count = 1):
    global frame_number
    global FRAMES

    original_count = count

    valid_lines = []

    frame_num = frame_number
    while frame_num >= 0 and count > 0:
        prev_frame = FRAMES[frame_num]
        if side is Side.LEFT:
            if prev_frame.left_line.isValid():
                valid_lines.append(prev_frame.left_line)
                count -= 1
        else:
            if prev_frame.right_line.isValid():
                valid_lines.append(prev_frame.right_line)
                count -= 1
        frame_num -= 1

    return valid_lines





'''
 Searches a thresholded and warped image for lane lines.
 Returns a left lane line and a right lane line if found.
'''
def find_lane_lines(binary_warped, left_line, right_line):
    height, width = binary_warped.shape

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    nwindows      = 9   # Choose the number of sliding windows
    window_margin = 50  # the width of the windows +/- margin
    poly_margin   = 50  # the margin when searching around poly
    minpix        = 200  # minimum number of pixels found to recenter window

    do_Sliding_Window = True
    new_left_line = None
    new_right_line = None
    if (left_line is not None and left_line.isValid()) and (right_line is not None and right_line.isValid()):
        #
        # We have valid lines from a previous frame.  We can search around the
        # polynomial for line pixels.
        #
        left_poly = left_line.line_coefficients
        right_poly = right_line.line_coefficients
        left_lane_inds = ((nonzerox > (left_poly[0]*(nonzeroy**2) + left_poly[1]*nonzeroy + left_poly[2] - poly_margin)) &
                          (nonzerox < (left_poly[0]*(nonzeroy**2) + left_poly[1]*nonzeroy + left_poly[2] + poly_margin)))
        right_lane_inds = ((nonzerox > (right_poly[0]*(nonzeroy**2) + right_poly[1]*nonzeroy + right_poly[2] - poly_margin)) &
                           (nonzerox < (right_poly[0]*(nonzeroy**2) + right_poly[1]*nonzeroy + right_poly[2] + poly_margin)))

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Our new lines.
        new_right_line = Line(rightx, righty, height, True)
        new_left_line = Line(leftx, lefty, height, True)

        if new_right_line.isValid() and new_right_line.isValid():
            do_Sliding_Window = False


    if (do_Sliding_Window):
        left_lane_inds = []
        right_lane_inds = []

        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[height//2:,:], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint       = np.int(histogram.shape[0]//2)
        leftx_base     = np.argmax(histogram[:midpoint])
        rightx_base    = np.argmax(histogram[midpoint:]) + midpoint
        leftx_current  = leftx_base
        rightx_current = rightx_base

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(height // nwindows)

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = height - (window+1) * window_height
            win_y_high = height - window * window_height

            ### Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - window_margin
            win_xleft_high = leftx_current + window_margin
            win_xright_low = rightx_current - window_margin
            win_xright_high = rightx_current + window_margin

            ###  Identify the nonzero pixels in x and y within the window ###
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
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Our new lines.
        new_left_line = Line(leftx, lefty, height, True)
        new_right_line = Line(rightx, righty, height, True)
        #
        # end of Sliding Window
        #

    left_fitx = new_left_line.fitx
    right_fitx = new_right_line.fitx
    ploty = np.linspace(0, height-1, height)

    #
    # validate lines.  Make sure they don't cross each other.
    #
    if new_left_line.isValid() and new_right_line.isValid() and np.min(np.absolute(right_fitx - left_fitx)) < 300:
        new_left_line.detected = False
        new_right_line.detected = False

    return new_left_line, new_right_line




def fillLane(avg_left, avg_right, src_points, dst_points,  undist, warped):

    #
    # find our pixels to shade
    #
    height, width = undist.shape[:2]
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
    newwarp = image_utils.warp_image(color_warp, dst_points, src_points)

    #
    # Combine the result with the original image
    #
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result


'''
Callback function to process a frame of video.
The given frame is first undistored and warped to a bird's-eye view.
Then lane lines are detected and the area between the lines if filled
with a translucent green.
'''
def process_frame(frame):
    global video_data
    global mtx
    global dist
    global M
    global M_inv
    global frame_number
    global FRAMES

    frame_number += 1

    #
    # Undistort the image using the camera matrix and
    # distortion coefficients.
    #
    undist = cv2.undistort(frame, mtx, dist, None, mtx)
    h, w, _ = undist.shape

    #
    # threshold color values.
    #
    l_threshold = (video_data['L_THRESHOLD'], video_data['L_THRESHOLD_LC'])
    b_threshold = (video_data['B_THRESHOLD'], video_data['B_THRESHOLD_LC'])
    brightness_threshold = video_data['BRIGHTNESS_THRESHOLD']
    thresholded = image_utils.threshold_image(undist, l_threshold, b_threshold, brightness_threshold)

    #
    # Mask the region of interest
    #
    points = np.array(video_data['ROI_POINTS']) * (w, h)
    roi = image_utils.region_of_interest(thresholded, points)

    #
    # Warp the image to a bird's eye view
    #
    # Source and destination points are normalized coordinates.
    # Translate to screen coordinates using the shape of the image.
    #
    src_points = np.float32(video_data['TRANSFORM_SRC_POINTS'])
    dst_points = np.float32([(src_points[0][0], 1.0), (src_points[0][0], 0.0), (src_points[3][0], 0.0), (src_points[3][0], 1.0)])
    src_points *= (w, h)
    dst_points *= (w, h)

    warped = image_utils.warp_image(roi, src_points, dst_points)

    #
    # Create the Frame object to hold the data for the lane lines
    #
    frame = Frame(frame_number)
    prev_left_line = None
    prev_right_line = None

    if (frame_number > 0):
        prev_left_line = FRAMES[frame_number-1].left_line
        prev_right_line = FRAMES[frame_number-1].right_line

    frame.left_line, frame.right_line = find_lane_lines(warped, prev_left_line, prev_right_line)
    FRAMES.append(frame)

    #
    # If lanes were not found or not valid we use the previous lines.
    #
    current_left_line = frame.left_line
    current_right_line = frame.right_line
    if not current_left_line.isValid():
        current_left_line = getLastValidLines(Side.LEFT, 1)[0]

    if not current_right_line.isValid():
        current_right_line = getLastValidLines(Side.RIGHT, 1)[0]


    #
    # Calculate the average left and right lines
    #
    valid_left_lines  = getLastValidLines(Side.LEFT,  AVERAGE_OVER_X_FRAMES)
    valid_right_lines = getLastValidLines(Side.RIGHT, AVERAGE_OVER_X_FRAMES)

    avg_left_coeffs = []
    [avg_left_coeffs.append(line.line_coefficients) for line in valid_left_lines]
    avg_left = np.mean(avg_left_coeffs, axis = 0)

    avg_right_coeffs = []
    [avg_right_coeffs.append(line.line_coefficients) for line in valid_right_lines]
    avg_right = np.mean(avg_right_coeffs, axis = 0)

    #
    # Use the average of the lines to fill the lane with
    # a translucent green.
    #
    out_img = fillLane(avg_left, avg_right, src_points, dst_points,  undist, warped)

    #
    # Measure Curvature.  Take the average of the left and right lines
    #
    avg_curve = []
    [avg_curve.append(line.curvature) for line in valid_left_lines]
    [avg_curve.append(line.curvature) for line in valid_right_lines]
    average_line_curvature = int(np.mean(avg_curve))

    #
    # S C R E E N   T E X T
    #
    avgText = "Curvature: " + str(average_line_curvature) + "m"
    cv2.putText(out_img, avgText, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    offset = measure_offset(current_left_line, current_right_line, w)
    offsetText = str(abs(offset)) + ("m left", "m right")[offset > 0] + " of center"
    cv2.putText(out_img, offsetText, (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return out_img



print("Calibrating Camera....")
mtx, dist = camera.calibrate()
print("Done.")

#video_input  = 'challenge_video.mp4'
#video_input  = 'harder_challenge_video.mp4'
video_output = 'myvideo.mp4'
video_data = project_data.getVideoData(video_input)

clip2 = VideoFileClip(video_input) #.subclip(36, 45)
video_clip = clip2.fl_image(process_frame)
video_clip.write_videofile(video_output, audio=False)
