## Writeup

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

[image1]: ./camera_cal/calibration2.jpg "Distorted"
[image1u]: ./output_images/calibration2_undist.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road"
[image2u]: ./output_images/undist_test1.jpg "Road Transformed"
[image3]: ./output_images/bin_test3.jpg "Binary Example"
[image4]: ./output_images/straight_lines1_warpsrc.jpg "Warp Source"
[image4w]: ./output_images/straight_lines1_warpdst.jpg "Warped"
[image5]: ./output_images/tr_test2.jpg "Sliding window"
[image6]: ./output_images/lane_test5.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[image7]: ./output_images/tr_test7.jpg "Fail"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function `calibrate()` (lines 17 through 40 of the file called `alf.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world.
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each
calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy
of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the
(x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using
the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the
`cv2.undistort()` function and obtained this result: 

Distorted             |  Undistorted
:--------------------:|:-------------------------:
 ![Distorted][image1] | ![Undistorted][image1u]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

Distorted             |  Undistorted
:--------------------:|:-------------------------:
 ![Distorted][image2] | ![Undistorted][image2u]

Distortion correction is done by calling `cv2.undistort` with the camera matrix and distortion coefficients obtained from the
previous step. The code for this is contained in the function `process_single_image()` in `alf.py`.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in function `image2LaneBinary()` in `LaneFinder.py`).  Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is included in the constructor of the class `LaneFinder` to create the
transformation matrix and the function `perspectiveTransform()` to do the actual transformation.
The `perspectiveTransform()` function takes as inputs an image (`img`) and returns the warped image.
The source (`src`) and destination (`dst`) points for calculation the matrix are defined in the first lines of
`LaneFinder.py`. I chose to hardcode the source and destination points in the following manner:

```python
src = np.float32([[200, 719], [588, 454], [692, 454], [1100, 719]])
dst = np.float32([[300, 719], [300, 0], [1000, 0], [1000, 719]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 719      | 300, 719        | 
| 588, 454      | 300, 0      |
| 692, 454     | 1000, 0      |
| 1100, 719      | 1000, 719        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image
and its warped counterpart to verify that the lines appear parallel in the warped image. This is done in the function
`warp_image()` in `alf.py`.

Undistorted             |  Warped
:--------------------:|:-------------------------:
 ![Distorted][image4] | ![Undistorted][image4w]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

There are two functions in the `LaneFinder` class, `findLanePixels()` and `findLanePixelsFromPrevious()`.

The first function will take the warped binary image and calculates a histogram to find the starting positions
of the lane-lines. From these positions it used a sliding window approach to follow the lines forward. It returns the x
and y coordinates of all found pixels.

![alt text][image5]


The second function also takes the warped binary image but will use the polynomials from the previous calculation
as corridor to find the relevant pixels.

From the list of x and y coordinates of relevant line pixels, a corresponding 2nd order polynomial is approximated in
the `setNewLinePixels()` function in the class `Line`.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the function `setNewLinePixels()` of class `Line`. Here `line_base_pos` is set to the distance of the line
to the center in meters by evaluating the polynomial at the bottom of the warped space. The distance to the center is
then multiplied by `xm_per_pix` to get the result in meters.

To finally get the position of the vehicle with respect to center, the distance of the right line to the center is
subtracted from the distance of the left line to the center and then divided by 2 in `LaneFinder.py` in line 266.

The radius of the curvature is calculated by the formula from the quizzes in the function `setNewLinePixels()` of class
`Line`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in function `drawLane()` of class `LaneFinder`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and
how I might improve it if I were going to pursue this project further.  

The image processing for finding the lane pixels is basically copied from the quizzes. And here I see the biggest room
for improvements. Especially the far line pixels are not recognized very well. As example, I processed the first image
of the challenge video:

![alt text][image7]

While the right line is detected quite well, the left line ends already in the middle of the image. Therefore, the
sliding window jumps to the outer border and even fails to follow this finally.

Also, the sliding window approach does not perform very well for small curvature radius's as we see in the picture above.
A possible improvement would be to already fit a polynomial to the first found pixels and then follow and refit this
forward through the image. This would take a lot of processing power however, so maybe it's not the most practical
solution.

To be a little robust against such line detection errors, I added sanity checks (function `sanityCheck()` of
`LaneFinder`), and a confidence counter. Everytime the lane-lines are detected and sanity checks are passed, the
confidence counter is increased up to a maximum value of 10. If the lane-lines are not detected, or the sanity checks
failed, the confidence counter is decreased, and the previous detected lane is still used. If the confidence counter
reaches 0, the lane-lines are detected by the histogram and sliding window method again.

To smoothen the output, I use the moving average over the last 3 detected polynomials (see function `acceptCurrentFit()` in `Line.py`)

While this works quite well on the project video, there is still room for fine-tuning. 10 as maximum value of confidence
might be too high for real world since in 10 frames, the lane-lines might be nowhere near the previous ones.

Also we could detect the case that one line (left or right) is found but the other is not. In this case we could more
trust the line that changed less compared to the previous frame and calculate the other line by maintaining distance and
curvature to the trusted line.
