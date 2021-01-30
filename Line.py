import numpy as np

YMAX = 719
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 780  # meters per pixel in x dimension
detectionThreshold = 50
nRecent = 3

# Define a class to receive the characteristics of each line detection
class Line():
  def __init__(self):
    # was the line detected in the last iteration?
    self.detected = False
    # x values of the last n fits of the line
    self.recent_xfitted = []
    # average x values of the fitted line over the last n iterations
    self.bestx = None
    # polynomial coefficients averaged over the last n iterations
    self.best_fit = None
    # polynomial coefficients for the most recent fit
    self.current_fit = [np.array([False])]
    # radius of curvature of the line in some units
    self.radius_of_curvature = None
    # distance in meters of vehicle center from the line
    self.line_base_pos = None
    # difference in fit coefficients between last and new fits
    self.diffs = np.array([0, 0, 0], dtype='float')
    # x values for detected line pixels
    self.allx = None
    # y values for detected line pixels
    self.ally = None

  def setNewLinePixels(self, fitx, fity):
    miny = min(fity)
    if len(fity)> detectionThreshold and miny < YMAX // 2:
      self.current_fit = np.polyfit(fity, fitx, 2)

      if self.best_fit is not None:
        self.best_fit = self.best_fit + (self.current_fit - self.best_fit) / nRecent
      else:
        self.best_fit = self.current_fit

      self.line_base_pos = self.current_fit[2]*YMAX**2 + self.current_fit[1]*YMAX + self.current_fit[0]
      self.radius_of_curvature = (1 + (2 * self.current_fit[0] * YMAX * ym_per_pix + self.current_fit[1]) ** 2) ** (3 / 2) / (
        abs(2 * self.current_fit[0]))

      self.detected = True
    else:
      self.detected = False
