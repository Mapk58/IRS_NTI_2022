import pyrealsense2
import numpy as np

def convert_depth_to_phys_coord_using_realsense(x, y, depth):
  _intrinsics = pyrealsense2.intrinsics()
  _intrinsics.width = 1280
  _intrinsics.height = 720
  _intrinsics.fx = 918.05371094
  _intrinsics.fy = 917.69604492
  _intrinsics.ppx = 642.13146973
  _intrinsics.ppy = 350.59375
  _intrinsics.model = pyrealsense2.distortion.none
  # fx fy cx cy
  # 640, 480, 918.05371094, 917.69604492, 642.13146973, 350.59375
  #_intrinsics.model = cameraInfo.distortion_model
  result = pyrealsense2.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth[y][x])
  #result[0]: right, result[1]: down, result[2]: forward
  return result[2], -result[0], -result[1]

dpth = np.load('data/cm_data/1-np_dpth.npy')

print(convert_depth_to_phys_coord_using_realsense(1280-5, 720-5, dpth))