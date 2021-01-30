import numpy as np
import matplotlib.pyplot as plt
import cv2
import Line

class LaneFinder:
  def __init__(self, mtx, dist):
    self.mtx = mtx
    self.dist = dist
    self.sobelKernel = 3
    self.LeftLine = Line.Line()
    self.RightLine = Line.Line()

    #calculate perspective transformation matrix and inverse
    src = np.float32([[200, 719], [590, 454], [690, 454], [1100, 719]])
    dst = np.float32([[200, 719], [200, 0], [980, 0], [980, 719]])
    self.M = cv2.getPerspectiveTransform(src, dst)
    self.Minv = cv2.getPerspectiveTransform(dst, src)


  def image2LaneBinary(self, img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    #cv2.imwrite("output_images/scsobel.jpg", scaled_sobel)

    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    #plt.imshow(color_binary)
    #plt.show()

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    #plt.imshow(combined_binary)

    return combined_binary

  def perspectiveTransform(self, img):
    #linar = src.astype(np.int32).reshape((-1,1,2))
    #cv2.polylines(img, [linar], True, [255,0,0],1)
    #plt.imshow(img)
    warped = cv2.warpPerspective(img, self.M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped

  def find_lane_pixels(self, binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
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
    window_height = np.int(binary_warped.shape[0] // nwindows)
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
      win_y_low = binary_warped.shape[0] - (window + 1) * window_height
      win_y_high = binary_warped.shape[0] - window * window_height
      ### TO-DO: Find the four below boundaries of the window ###
      win_xleft_low = leftx_current - margin  # Update this
      win_xleft_high = leftx_current + margin  # Update this
      win_xright_low = rightx_current - margin  # Update this
      win_xright_high = rightx_current + margin  # Update this

      # Draw the windows on the visualization image
      cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                    (win_xleft_high, win_y_high), (0, 255, 0), 2)
      cv2.rectangle(out_img, (win_xright_low, win_y_low),
                    (win_xright_high, win_y_high), (0, 255, 0), 2)

      ### TO-DO: Identify the nonzero pixels in x and y within the window ###
      good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                        (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
      good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

      # Append these indices to the lists
      left_lane_inds.append(good_left_inds)
      right_lane_inds.append(good_right_inds)

      ### If found > minpix pixels, recenter next window ###
      ### (`right` or `leftx_current`) on their mean position ###
      if len(good_left_inds) > minpix:
        leftx_current = np.sum(nonzerox[good_left_inds]) // len(good_left_inds)
      if len(good_right_inds) > minpix:
        rightx_current = np.sum(nonzerox[good_right_inds]) // len(good_right_inds)

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

  def findLanePixelsFromPrevious(self, binary_warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_fit = self.LeftLine.current_fit
    right_fit = self.RightLine.current_fit

    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = (nonzerox > left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy + left_fit[2] - margin) & \
                     (nonzerox < left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy + left_fit[2] + margin)
    right_lane_inds = (nonzerox > right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy + right_fit[2] - margin) & \
                      (nonzerox < right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy + right_fit[2] + margin)

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

  def recalcNeeded(self):
    return not (self.LeftLine.detected and self.RightLine.detected)

  def drawLane(self, warped, undist):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    #left_fitx = self.LeftLine.current_fit[0]*ploty**2 + self.LeftLine.current_fit[1]*ploty + self.LeftLine.current_fit[2]
    #right_fitx = self.RightLine.current_fit[0]*ploty**2 + self.RightLine.current_fit[1]*ploty + self.RightLine.current_fit[2]
    left_fitx = self.LeftLine.best_fit[0]*ploty**2 + self.LeftLine.best_fit[1]*ploty + self.LeftLine.best_fit[2]
    right_fitx = self.RightLine.best_fit[0]*ploty**2 + self.RightLine.best_fit[1]*ploty + self.RightLine.best_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, self.Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    #plt.imshow(result)
    return result

  def processImage(self, img):
    undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
    bin = self.image2LaneBinary(undist)
    per = self.perspectiveTransform(bin)

    if self.recalcNeeded():
      leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(per)

      #plt.imshow(out_img)

      self.LeftLine.setNewLinePixels(leftx, lefty)
      self.RightLine.setNewLinePixels(rightx, righty)

    else:
      leftx, lefty, rightx, righty = self.findLanePixelsFromPrevious(per)

      self.LeftLine.setNewLinePixels(leftx, lefty)
      self.RightLine.setNewLinePixels(rightx, righty)

    outimg = self.drawLane(per, undist)

    radius = (self.LeftLine.radius_of_curvature + self.RightLine.radius_of_curvature) / 2
    cv2.putText(outimg, "Radius = {:.2f} km".format(radius/1000), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))

    #plt.imshow(outimg)

    return outimg

