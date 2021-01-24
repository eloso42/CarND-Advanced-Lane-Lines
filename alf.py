import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import LaneFinder

"""
Calibrate camera

mostly copied from the lessons
"""
def calibrate():
  nx = 9
  ny = 6

  # Arrays to store object points and image points from all the images.
  objpoints = []  # 3d points in real world space
  imgpoints = []  # 2d points in image plane.

  # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
  objp = np.zeros((ny * nx, 3), np.float32)
  objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

  #iterate through the calibration images
  for filename in os.listdir("camera_cal/"):
    #print(filename)
    img = cv2.imread("camera_cal/"+filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret == True:
      objpoints.append(objp)
      imgpoints.append(corners)

  et, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
  return mtx, dist

# undistort sample image for writeup
def undistort_image(mtx, dist):
  img = cv2.imread("camera_cal/calibration2.jpg")
  dst = cv2.undistort(img, mtx, dist, None, mtx)
  cv2.imwrite("output_images/calibration2_undist.jpg", dst)

def process_single_image(filename, mtx, dist):
  print(filename)
  img = cv2.imread("test_images/"+filename)
  laneFinder = LaneFinder.LaneFinder(mtx, dist)
  bin = laneFinder.image2LaneBinary(img)
  gray = bin * 255
  cv2.imwrite("output_images/bin_" + filename, gray)
  # plt.imshow(grey, cmap='gray')
  rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  trans = laneFinder.perspectiveTransform(bin)
  #plt.imshow(rgb)
  cv2.imwrite("output_images/tr_" + filename, trans*255)

  laneFinder.processImage(img)


def process_single_images(mtx, dist):
  for filename in os.listdir("test_images/"):
    process_single_image(filename, mtx, dist)

if __name__ == "__main__":
  mtx, dist = calibrate()
  undistort_image(mtx, dist)
  process_single_images(mtx, dist)
  #process_single_image("straight_lines2.jpg", mtx, dist)
  img = cv2.imread("test_images/straight_lines1.jpg")
  dst = cv2.undistort(img, mtx, dist, None, mtx)
  dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
  #plt.imshow(dst)
  plt.show()
  print(mtx)
  print("hello")

