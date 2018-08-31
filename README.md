## Advanced Lane Lines

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_images/undistorted.png "Undistorted example"
[image2]: ./writeup_images/thresholded.png "Thresholded example"
[image3]: ./writeup_images/warped.png "Warped example"
[image4]: ./writeup_images/equation.gif "Equation"
[image5]: ./writeup_images/visualization.png "Visualization"
[image6]: ./writeup_images/green_lane.jpg "Output"
[video1]: ./video_outputs/project_video.mp4 "Project Video"

---
### 1. Camera Calibration

Determining the camera calibration matrix and distortion coefficients is the first step toward correcting distortion in images.  To obtain these we use a set of chessboard images and OpenCV's `calibrateCamera()` and `findChessboardCorners()` functions.   Chessboard corners are detected in each image using `findChessboardCorners()` and are stored in the `img_points` array.  For each set of `img_points` found a corresponding set of 3D points representing the position of the corner in the world coordinate space is kept in `obj_points.`   These `obj_points` and `img_points` are then fed to `calibrateCamera()` to obtain our calibration matrix and distortion coefficients.  Given an image, the matrix, and the distortion coefficients, OpenCV's `undistort()` can then be used to correct for distortion on any image taken with that camera and lens.


#### Distortion-corrected images:

![Undistorted Images][image1]

Camera calibration code can be found in `camera.py`.  The second code cell in `P2.ipynp` uses this code to calibrate the camera and apply distortion correction to two test images.

---
### 2. Thresholding

The next step after distortion correction is thresholding.  Finding the correct thresholding to handle all different colors, shadows, reflections, and other noise was a tricky problem. To reduce noise in the image I initially perform a gaussian blur.  The color spaces I eventually settled on are the L Channel of the LUV color space for picking up white lines and the B Channel of the Lab color space for picking up yellow lines.  I used different threshold values for each video but I found that for the `project_video.mp4` L values between 199 and 255 and B values between 159 and 255 worked fairly well in the general case.  Shadows in the challenge videos were a particularly difficult problem and I attempted to do an adaptive threshold where sections of the image with lower brightness could be thresholded with different values but I only saw modest gains.  I finally settled on breaking the image up into 9 sections in the Y dimension and 20 sections in the X dimension. Lower brightness values for `project_video.mp4` were between 118 and 255 for the L channel and 159 and 255 for the B Channel.   All project thresholding values can be found in `project_data.py`

Below is an example of thresholding an undistored test image:

![Thresholded Image][image2]

Thresholding code can be found in `./image_utils.py`.  See the function `threshold_image()` at the top of the file.   To make deterimining these numbers easier I wrote a tool so that I could see the changes in threshold interactively.  The tool can be found at `./tools/find_threshold.py`

---
### 3. Transforming an image

The next step is to warp the image into a bird's-eye view.  The perspective matrix for performing this transformation is obtained by a call to the openCV function `getPerspectiveTransform()`  To calculate this `getPerspectiveTransform()` needs points from the source image mapped to locations in the destination image.  In an attempt to handle differing image sizes I kept source points normalized between 0 and 1.  Four source points were mapped to destination points in the following manner:


- (bottom_left_x, bottom_left_y) in the source mapped to (bottom_right_x, height_of_image) in the destination
- (top_left_x, top_left_y) in the source mapped to (bottom_left_x, 0) in the destination
- (top_right_x, top_right_y) in the source mapped to (bottom_right_x, 0) in the destination
- (bottom_right_x, bottom_right_y) in the source mapped to (bottom_right_x, height_of_image) in the destination

Once the proper source and destintion points are mapped all it takes is a call to `cv2.warpPerspective()` to take the image to/from a top-down view.  `warp_image()` in the file `image_utils.py` (lines 58-71) takes care of both obtaining the perspective matrix and warping the image.

The example below shows the original and the warped image.

![Warped image][image3]


---
### 4. Recognizing Lanes

The bulk of the work to recognize the lane lines is done in the function `find_lane_lines().`
This function takes one of two approaches to finding the lane lines:

1. If this is the first detection or the previous line was not been successfully detected:
- Take a histogram of the bottom half of the warped and thresholded binary image.
- Take the histogram peak on each half of the image as the starting point of each line.
- Using a 'sliding window approach' move upwards in the image adjusting the window position as necessary to follow the area with the largest number of pixels.
- use Numpy's `polyfit()` to fit the pixels to a polynomial.

OR:
2. If a previous line has been successfully detected and mapped to a polynomial:
- Since the lane lines do not move much from frame to frame, use the previous frame's polynomial to search around for pixels in the current frame.

#### Line Class

To facilitate the holding of individual line data and properties I created a Line class.  The class implementation is in the file `p2.py` from line 80 to line 142.  This class holds the pixels detected, the line coefficients obtained from `polyfit()`, the line curvature, and derivatives taken at 3 points along the line.  A method is also provided to detect if the line is valid.


#### Line Validation

Several properties of the line are checked for validity. The `isValid()` function of the `Line` class handles this determination.  Line slopes, coefficients, the number of pixels detected, and curvature factor into validation.

A visualization of the process can be seen in the lower right quadrant of this video frame:

![visualization][image5]

---
### 5. Measuring curvature and offset.

#### Curvature

To measure curvature I used the following formula

![Equation][image4]

A Pixel space to world space conversion was made using standard lane widths and a polynomial was refit to the new world space values.  The curvature equation was then evaluated at the lowest point of the line(bottom of screen).

#### Offset

Offset was calculated under the assumption that the camera was recording from the center point of the car.  The difference of the center of the image from the center point between the detected lane lines was calculated and converted from pixel space to world space.

---

### 6. Render over image
The final step is the fill the lane area detected and render it back over the original image.  This is done by the function `fillLane()`( file `p2.py`, lines 304 through 341).

OpenCV's `fillpoly()` was used to fill the lane boundaries and `warp_image()` was used in reverse to warp the image back into a first-person perspective.

![Fill_lane][image6]

---

### Videos

Both the `project_video` and the `challenge_video` went well. The `harder_challenge_video` exposed many weaknesses in my approach.  The many shadows and bright spots on the road confused my algorithm, as did the bright grass off the right shoulder of the road.  Even so, the lane detection does recover up until the last very brightly lit hairpin turn at the end.

Here's a [link to my video result for the project video][video1]

Here's a [link to my video result for the challenge video](./video_outputs/challenge_video.mp4)

Here's a [link to my video result for the fiendishly difficult challenge video](./video_outputs/harder_challenge_video.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
