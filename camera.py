import os
import numpy as np
import cv2
import matplotlib.image as mpimg



'''
 Determines and returns the camera calibration matrix and distortion
 coefficients for use in undistorting images taken by a particular camera
'''
def calibrate():
    camera_calibration_images_path = "camera_cal/"

    num_x_corners = 9
    num_y_corners = 6

    obj_points = []
    img_points = []

    #
    # Prepare the object points.  These are the same for each calibration image.
    # the reshaped mgrid will give us an array like:
    # array([[ 0.,  0.,  0.],
    #        [ 1.,  0.,  0.],
    #         ...
    #        [ num_x_corners-1, num_y_corners-1, 0]])
    obj_p = np.zeros((num_x_corners * num_y_corners,  3), np.float32)
    obj_p[:, :2] = np.mgrid[0:num_x_corners, 0:num_y_corners].T.reshape(-1, 2)


    images = os.listdir(camera_calibration_images_path)
    for imagefile in images:
        #
        # Read in the image file and convert to grayscale
        #
        img = mpimg.imread(camera_calibration_images_path + imagefile)
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
