import cv2 as cv
import numpy as np


def normalize(img):
    dpth = img
    dpth[dpth == 0] = 999
    nmin = np.min(dpth)
    dpth[dpth == 999] = nmin
    dpth = np.subtract(dpth, nmin)
    dpth = dpth / np.max(dpth)
    dpth[dpth == 0] = 1
    dpth = 1 - dpth
    # dpth[dpth == 0.] = 0.1
    return dpth


def make_mask(img):
    return np.where(img == 0, 0, 1.0)


rgb = cv.imread('data/cm_data/1-rgb.png')
# dpth = np.load('data/data/1-np_dpth.npy')
# rgb = cv.imread('data/olddata/rgb-5.png')

dpth = np.load('data/cm_data/1-np_dpth.npy')
mask = make_mask(dpth)
# dpth = normalize(dpth)
for i in range(20):
    temp = np.load('data/cm_data/' + str(i + 1) + '-np_dpth.npy')
    dpth += temp
    mask += make_mask(temp)
    # cv.imshow("dpth", normalize(dpth))
    # cv.imshow("mask", mask)
    temp_mask = np.copy(mask)
    temp_mask[temp_mask == 0] = 1
    cv.imshow("dpth", normalize(np.divide(dpth, temp_mask)))
    cv.waitKey()

# cv.imshow("rgb", rgb)
mask[mask == 0] = 1
cv.imshow("dpth", normalize(np.divide(dpth, mask)))
cv.imshow("mask", mask)
cv.imshow("rgb", rgb)

cv.waitKey()
