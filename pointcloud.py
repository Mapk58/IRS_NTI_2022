import pyrealsense2 as rs
from enum import IntEnum
import open3d as o3d
import numpy as np
from DominationCamera import normalize, make_mask
import cv2 as cv
import numpy as np

def visualization_of_points(points):
    o3d.visualization.draw_geometries([points], 'Demonstration', 1080, 720)

def depth_multiplyer():
    dpth = np.load('data/data/1-np_dpth.npy')
    mask = make_mask(dpth)
    for i in range(20):
        temp = np.load('data/data/' + str(i + 1) + '-np_dpth.npy')
        dpth += temp
        mask += make_mask(temp)
        # cv.imshow("dpth", normalize(dpth))
        # cv.imshow("mask", mask)
        temp_mask = np.copy(mask)
        temp_mask[temp_mask == 0] = 1
        # cv.imshow("dpth", normalize(np.divide(dpth, temp_mask)))
        # cv.waitKey()

    # cv.imshow("rgb", rgb)
    mask[mask == 0] = 1
    dpth = np.divide(dpth, mask)
    return dpth


rgb = np.load('data/data/1-np_rgb.npy')

color_image = o3d.geometry.Image(rgb)
depth_image = o3d.geometry.Image((depth_multiplyer()).astype(np.uint16))
# depth_image = o3d.geometry.Image(np.load('data/data/2-np_dpth.npy').astype(np.uint16))

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_image,
    depth_image,
    depth_scale=1.0/0.0010000000474974513,
    depth_trunc=2,
    convert_rgb_to_intensity=False)

intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, 918.05371094, 917.69604492, 642.13146973, 350.59375)

temp = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image, intrinsic)
flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
temp.transform(flip_transform)
pcd = o3d.geometry.PointCloud()
pcd.points = temp.points
pcd.colors = temp.colors
visualization_of_points(pcd)
o3d.io.write_point_cloud('data/data/full-PCL.ply', pcd)

