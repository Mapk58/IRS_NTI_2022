import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from enum import IntEnum
import matplotlib.pyplot as plt

def rgb_to_3d(x, y, depth):
    _intrinsics = rs.intrinsics()
    _intrinsics.width = 640
    _intrinsics.height = 480
    _intrinsics.fx = 918.05371094
    _intrinsics.fy = 917.69604492
    _intrinsics.ppx = 642.13146973
    _intrinsics.ppy = 350.59375

    # fx fy cx cy
    # 640, 480, 918.05371094, 917.69604492, 642.13146973, 350.59375
    # _intrinsics.model = cameraInfo.distortion_model
    result = rs.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth[y][x])
    return result[0], -result[1], result[2]


def make_mask(img):
    return np.where(img == 0, 0, 1.0)


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


def make_3d_map(depth):
    new_points = []
    for i in range(1280):
        for j in range(720):
            result = rgb_to_3d(i, j, depth)
            new_points.append([result[0], result[1], result[2]])
    new_points = np.array(new_points)
    return new_points
