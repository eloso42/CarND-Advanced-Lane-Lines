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

  def image2LaneBinary(self, img):
    undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    hls = cv2.cvtColor(undist, cv2.COLOR_BGR2HLS)
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
    src = np.float32([[200, 719],[590, 454], [690, 454], [1100, 719]])
    dst = np.float32([[200,719], [200,0], [1080, 0], [1080, 719]])
    #linar = src.astype(np.int32).reshape((-1,1,2))
    #cv2.polylines(img, [linar], True, [255,0,0],1)
    #plt.imshow(img)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped

  def processImage(self, img):
    bin = self.image2LaneBinary(img)
    per = self.perspectiveTransform(bin)

