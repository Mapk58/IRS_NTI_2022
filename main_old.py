import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from OperateCamera_old import OperateCamera, open, visualization_of_points, o3d

def convert_depth_to_phys_coord_using_realsense(x, y, depth):
  _intrinsics = rs.intrinsics()
  _intrinsics.width = 640
  _intrinsics.height = 480
  _intrinsics.fx = 918.05371094
  _intrinsics.fy = 917.69604492
  _intrinsics.ppx = 642.13146973
  _intrinsics.ppy = 350.59375

  # fx fy cx cy
  # 640, 480, 918.05371094, 917.69604492, 642.13146973, 350.59375
  #_intrinsics.model = cameraInfo.distortion_model
  result = rs.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth[y][x])
  return result[0], -result[1], result[2]

def full_depth():
    dpth = np.load('data/angle_data/1-np_dpth.npy')
    mask = np.where(dpth == 0, 0, 1.0)
    for i in range(20):
        temp = np.load('data/angle_data/' + str(i + 1) + '-np_dpth.npy')
        dpth += temp
        mask += np.where(temp == 0, 0, 1.0)
        temp_mask = np.copy(mask)
        temp_mask[temp_mask == 0] = 1
    mask[mask == 0] = 1
    dpth = np.divide(dpth, mask)
    return dpth

pcd = open("data/cm_data/full-PCL.ply")

points = np.asarray(pcd.points)
# colors = np.array(pcd.colors)

# depth = full_depth()
# new_points = []
# for i in range(1280):
#     for j in range(720):
#         result = convert_depth_to_phys_coord_using_realsense(i, j, depth)
#         new_points.append([result[0]/1000, result[1]/1000, result[2]/1000])
# new_points = np.array(new_points)

# points = points[points[:, 2] > -0.615]
# points = points[points[:, 1] > -0.15]
# points = points[points[:, 1] < -0.10]
# points = points[points[:, 0] > 0.25]
# points = points[points[:, 0] < 0.3]


plt.scatter(points[:, 0], points[:, 1], s=0.0001)
plt.show()
# pcd.points = points
# pcd.colors = colors
# o3d.visualization.draw_geometries([pcd], 'Demonstration', 1920, 1080)

# b = a[:,2]
# print(a)

