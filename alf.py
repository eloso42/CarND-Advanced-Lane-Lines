import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
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

def warp_image(mtx, dist):
  img = cv2.imread("test_images/straight_lines1.jpg")
  rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  laneFinder = LaneFinder.LaneFinder(mtx, dist)
  undist = cv2.undistort(rgb, mtx, dist, None, mtx)

  #warp
  warp = laneFinder.perspectiveTransform(undist)

  #draw src lines to undistorted image
  linar = LaneFinder.src.astype(np.int32).reshape((-1, 1, 2))
  cv2.polylines(undist, [linar], True, [255, 0, 0], 3)

  #draw dst lines to warped image
  linar = LaneFinder.dst.astype(np.int32).reshape((-1, 1, 2))
  cv2.polylines(warp, [linar], True, [255, 0, 0], 3)

  cv2.imwrite("output_images/straight_lines1_warpsrc.jpg", cv2.cvtColor(undist, cv2.COLOR_RGB2BGR))
  cv2.imwrite("output_images/straight_lines1_warpdst.jpg", cv2.cvtColor(warp, cv2.COLOR_RGB2BGR))


def process_single_image(filename, mtx, dist):
  print(filename)
  laneFinder = LaneFinder.LaneFinder(mtx, dist)

  #read image
  img = cv2.imread("test_images/"+filename)
  #convert to rgb
  rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  #undistort
  undist = cv2.undistort(rgb, mtx, dist, None, mtx)
  cv2.imwrite("output_images/undist_"+filename, cv2.cvtColor(undist, cv2.COLOR_RGB2BGR))

  bin = laneFinder.image2LaneBinary(undist)
  gray = bin * 255
  cv2.imwrite("output_images/bin_" + filename, gray)
  # plt.imshow(grey, cmap='gray')
  trans = laneFinder.perspectiveTransform(bin)

  #plt.imshow(rgb)

  leftx, lefty, rightx, righty, transLane = laneFinder.findLanePixels(trans)
  cv2.imwrite("output_images/tr_" + filename, transLane)


  outimg = laneFinder.processImage(rgb)
  cv2.imwrite("output_images/lane_"+filename, cv2.cvtColor(outimg, cv2.COLOR_RGB2BGR))


def process_single_images(mtx, dist):
  for filename in os.listdir("test_images/"):
    process_single_image(filename, mtx, dist)

def process_video_image(laneFinder, img):
  return laneFinder.processImage(img)


def processVideo(mtx, dist):
  #laneFinder = LaneFinder.LaneFinder(mtx, dist)
  #clip = VideoFileClip("project_video.mp4")
  #outclip = clip.fl_image(lambda img: process_video_image(laneFinder, img))
  #outclip.write_videofile("output_video/project_video.mp4", audio=False)

  laneFinder = LaneFinder.LaneFinder(mtx, dist)
  clip = VideoFileClip("challenge_video.mp4")
  outclip = clip.fl_image(lambda img: process_video_image(laneFinder, img))
  outclip.write_videofile("output_video/challenge_video.mp4", audio=False)

  laneFinder = LaneFinder.LaneFinder(mtx, dist)
  clip = VideoFileClip("harder_challenge_video.mp4")
  outclip = clip.fl_image(lambda img: process_video_image(laneFinder, img))
  outclip.write_videofile("output_video/harder_challenge_video.mp4", audio=False)


if __name__ == "__main__":
  mtx, dist = calibrate()
  undistort_image(mtx, dist)
  warp_image(mtx, dist)
  process_single_images(mtx, dist)
  #process_single_image("test4.jpg", mtx, dist)
  #processVideo(mtx, dist)

  img = cv2.imread("test_images/straight_lines1.jpg")
  dst = cv2.undistort(img, mtx, dist, None, mtx)
  dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
  #plt.imshow(dst)
  plt.show()
  print(mtx)
  print("hello")

