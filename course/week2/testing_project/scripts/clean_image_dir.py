"""
Utilities for processing photos of handwritten digits from 
raw photos to a clean version that neural networks expect. 

NOTE: You will not need to edit anything in this file although
we encourage you to understand the material here. 
"""

import os
import cv2
import math
import numpy as np
from scipy import ndimage


def get_best_shift(img):
  cy, cx = ndimage.measurements.center_of_mass(img)

  rows, cols = img.shape
  shiftx = np.round(cols / 2.0 - cx).astype(int)
  shifty = np.round(rows / 2.0 - cy).astype(int)

  return shiftx, shifty


def shift(img, sx, sy):
  rows, cols = img.shape
  M = np.float32([[1, 0, sx], [0, 1, sy]])
  shifted = cv2.warpAffine(img, M, (cols, rows))
  return shifted


def clean_photo_image(path, out_dir):
  """The photo we have taken by hand is quite different than a MNIST image.
  We will need to process them! 

  If you are curious, we perform the following operations:

  1. The photo is reshaped to 28 x 28 in grayscale.
  2. The photo is thresholded to increase contrast between black and white portions.
  3. The photo is cropped to remove any rows or columns of pixels that are 
    completely black.
  4. The crop is reshaped to 20 x 20 pixels and cented in a 28 x 28 pixe 
    box by center of mass. 

  Steps 3-4 are important for normalizing the size of the digit; otherwise you 
  might show digits of different sizes to the neural network. This sort of 
  standardization is very important for good performance.

  Credit: https://opensourc.es/blog/tensorflow-mnist for this function.
  """
  # load the raw image
  gray = cv2.imread(path, 0)
  # rescale it
  gray = cv2.resize(255 - gray, (28, 28))
  # better black and white version
  (thresh, gray) = cv2.threshold(gray, 128, 255, 
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  
  # remove all rows and columns at the sides of images that are 
  # completely black.
  while np.sum(gray[0]) == 0:
    gray = gray[1:]

  while np.sum(gray[:,0]) == 0:
    gray = np.delete(gray,0,1)

  while np.sum(gray[-1]) == 0:
    gray = gray[:-1]

  while np.sum(gray[:,-1]) == 0:
    gray = np.delete(gray,-1,1)

  rows,cols = gray.shape

  # MNIST images are normalized to fit in a 20x20 pixel box and
  # are centered in a 28x28 image using a center of mass. It is 
  # important we do this same thing!
  if rows > cols:
    factor = 20.0 / rows
    rows = 20
    cols = int(round(cols * factor))
    # first cols than rows
    gray = cv2.resize(gray, (cols, rows))
  else:
    factor = 20.0 / cols
    cols = 20
    rows = int(round(rows * factor))
    # first cols than rows
    gray = cv2.resize(gray, (cols, rows))

  # repad with black colums and rows
  cols_padding = (
    int(math.ceil((28 - cols) / 2.0)),
    int(math.floor((28 - cols) / 2.0)))
  rows_padding = (
    int(math.ceil((28 - rows) / 2.0)),
    int(math.floor((28 - rows) / 2.0)))

  gray = np.lib.pad(gray, (rows_padding, cols_padding), 'constant')

  # center of mass shifting
  shiftx,shifty = get_best_shift(gray)
  shifted = shift(gray,shiftx,shifty)
  gray = shifted

  # save to folder
  name = os.path.basename(path)
  out_file = os.path.join(out_dir, name)
  cv2.imwrite(out_file, gray)


if __name__ == "__main__":
  from glob import glob
  from os.path import join
  from pathlib import Path

  root = join(Path(__file__).resolve().parent, 'images/integration')
  raw_dir = 'digits-raw'
  out_dir = 'digits-processed'

  raw_paths = glob(join(root, raw_dir, '*.png'))

  for raw_path in raw_paths:
    clean_photo_image(raw_path, join(root, out_dir))
