import time
import pyrealsense2 as rs
from math import pi
import numpy as np
import cv2 as cv
from OperateCamera import OperateCamera
from DominationCamera import rgb_to_3d
from contours import *

cam = OperateCamera()

count = 1
while (True):
    # # Taking data frame from camera (RGBD format)
    # frame = cam.catch_frame()
    # rgb = np.asarray(cam.rgb)
    # dpth = np.asarray(cam.depth)
    rgb, dpth = cam.catch_frame()
    rgb = np.asarray(rgb)
    dpth = np.asarray(dpth)

    # SHITPOST

    frame = rgb
    red_data, red_img = find_obj(frame, hsv_red, "red")
    blue_data, blue_img = find_obj(frame, hsv_blue, "blue")
    data, col_img = detect_collision(frame, red_data, blue_data)
    sorted_data, dst_img = sort_by_dst(frame, data)

    #########

    # if count == 1:
    #     cv.imwrite("data/cm_data/"+str(count)+"-rgb.png", rgb)
    #     # cv.imwrite("data/cm_data/"+str(count)+"-dpth.png", dpth)
    #     np.save("data/cm_data/"+str(count)+"-np_rgb", rgb)
    #     cam.save("data/cm_data/"+str(count)+"-PCL.ply")
    # np.save("data/cm_data/"+str(count)+"-np_dpth", dpth)

    # np.save("data/yaroslav/"+str(count)+"-np_rgb", rgb)
    # cv.imwrite("data/yaroslav/" + str(count) + "-rgb.png", rgb)
    # cv.imwrite("data/yaroslav/" + str(count) + "-rgb.bmp", rgb)

    count += 1
    # cam.visualization_of_points(cam.pcd)
    # print(count)
    # cv.imshow('dpth', normalize(dpth))

    print(rgb_to_3d(sorted_data[0]['pos'][0], sorted_data[0]['pos'][1], dpth))
    # print(sorted_data[0]['angle'][0])
    size = (720, 360)
    # cv2.imshow('Collisions', cv2.resize(col_img, size, interpolation=cv2.INTER_AREA))
    # cv2.imshow('Red', cv2.resize(red_img, size, interpolation=cv2.INTER_AREA))
    cv2.imshow('Blue', cv2.resize(blue_img, size, interpolation=cv2.INTER_AREA))
    # cv2.imshow('Dist2cntr', cv2.resize(dst_img, size, interpolation=cv2.INTER_AREA))
    cv.waitKey(1)
    time.sleep(0.001)