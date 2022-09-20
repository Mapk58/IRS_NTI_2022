import DominationCamera as dc
import contours as ct
import cv2 as cv
import time
from OperateCamera import OperateCamera
from contours import *
import numpy as np

# rgb, dpth = get_images()

# # Поиск зон для уничтожения (центр и радиус из карты глубины и высоты в мм)
rgb = np.load('data/now/7-np_rgb.npy')
dpth = np.load('data/now/7-np_dpth.npy')

# dpth = dc.depth_multiplyer('data')

dc.height_calibration(dpth)

cv.imshow('rgb', rgb)
cv.imshow('dpth', dc.normalize(dpth))
cv.imshow('mask', dc.normalize(dc.table_grip))

a = dc.get_sorted_coords(rgb, dpth)
print(a)
cv.waitKey()

# print(dc.recieve_demolition_order(dpth, 20))

# rgb, dpth = dc.get_images()
# dc.height_calibration(dpth)

# # SAVE
# count = 2
# np.save("data/dt_data/"+str(count)+"-np_rgb", rgb)
# np.save("data/dt_data/"+str(count)+"-np_dpth", dpth)
# cv.imwrite("data/dt_data/"+str(count)+"-np_rgb.png", rgb)

# print(dc.get_sorted_coords(rgb, dpth))
# cv.waitKey()


# rgb = np.load('data/angle_data/1-np_rgb.npy')
# dpth = dc.depth_multiplyer('data/angle_data')
# a = dc.get_sorted_coords(rgb, dpth)
# print(a)
#########
# cv.imshow('a', rgb)
# cv.waitKey()
# print(rgb_to_3d(sorted_data[0]['pos'][0], sorted_data[0]['pos'][1], dpth), end='')
# print(sorted_data[0]['angle'])

# двигать в точку с вращением по Z, захват элемента (х, у, ориентация), повернуть, взять, положить в полочку
