import math
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from enum import IntEnum
import matplotlib.pyplot as plt
import cv2 as cv
import contours as ct
from OperateCamera import OperateCamera
import time

# cv.namedWindow('collisions')
# cam = OperateCamera()
corners_for_calibration = (536.5, 545.75, 547.75, 566.5)
table_grip = np.array(np.linspace(np.linspace(corners_for_calibration[0], corners_for_calibration[1], 1280),
                                  np.linspace(corners_for_calibration[2], corners_for_calibration[3], 1280), 720))


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
    dpth = np.copy(img)
    dpth[dpth == 0] = 999
    nmin = np.min(dpth)
    dpth[dpth == 999] = nmin
    dpth = np.subtract(dpth, nmin)
    dpth = dpth / np.max(dpth)
    dpth[dpth == 0] = 1
    dpth = 1 - dpth
    return dpth


def make_3d_map(depth):
    new_points = []
    for i in range(1280):
        for j in range(720):
            result = rgb_to_3d(i, j, depth)
            new_points.append([result[0], result[1], result[2]])
    new_points = np.array(new_points)
    return new_points


def depth_multiplyer(path):
    dpth = np.load(path + '/1-np_dpth.npy')
    mask = make_mask(dpth)
    for i in range(20):
        temp = np.load(path + '/' + str(i + 1) + '-np_dpth.npy')
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


def get_images():
    global cam
    rgb, dpth = cam.catch_frame()
    fld = 'images_data'
    count = time.strftime("%d-%H-%M-%S")
    np.save(fld + "/" + str(count) + "-np_rgb", rgb)
    np.save(fld + "/" + str(count) + "-np_dpth", dpth)
    cv.imwrite(fld + "/" + str(count) + "-np_rgb.png", rgb)
    return rgb, dpth


def get_corners_heights(dpth):
    border = 5
    left_top = rgb_to_3d(border, border, dpth)[2]
    left_bot = rgb_to_3d(border, 720 - border, dpth)[2]
    rght_top = rgb_to_3d(1280 - border, border, dpth)[2]
    rght_bot = rgb_to_3d(1280 - border, 720 - border, dpth)[2]
    return left_top, rght_top, left_bot, rght_bot


def get_corners_heights_average(dpth):
    a, b, c = 1, 5, 1
    left_top, rght_top, left_bot, rght_bot = 0, 0, 0, 0
    for border in range(a, b, c):
        left_top += rgb_to_3d(border, border, dpth)[2]
        left_bot += rgb_to_3d(border, 720 - border, dpth)[2]
        rght_top += rgb_to_3d(1280 - border, border, dpth)[2]
        rght_bot += rgb_to_3d(1280 - border, 720 - border, dpth)[2]
    amount = len(range(a, b, c))
    return left_top / amount, rght_top / amount, left_bot / amount, rght_bot / amount


def height_calibration(dpth):
    # a, b, c, d = get_corners_heights(dpth)
    a, b, c, d = get_corners_heights_average(dpth)
    global table_grip
    table_grip = np.array(np.linspace(np.linspace(a, b, 1280),
                                      np.linspace(c, d, 1280), 720))
    print("heights of angles calibration:")
    print(a, b, c, d)
    print()
    # cv.imshow('a', normalize(table_grip))
    # cv.waitKey()


# remove table from heights
def get_objects_heights(dpth):
    diff = dpth - table_grip
    mask = np.copy(diff)
    mask[mask > -150] = 0
    mask[mask < -150] = 1
    mask = cv.bitwise_not(mask)
    buffer = cv.bitwise_and(diff, mask)
    return buffer


def _show_objects_heights(dpth):
    buffer = get_objects_heights(dpth)
    for edge in range(10, 30):
        diff = np.copy(buffer)
        diff[diff > -edge] = 0
        diff[diff < -edge] = 1
        print(edge)
        cv.imshow('c', diff)
        cv.waitKey(0)


def get_heights_mask(dpth, height):
    h_mask = get_objects_heights(dpth)
    h_mask[h_mask >= -height] = 0
    h_mask[h_mask < -height] = 255
    return h_mask.astype(np.uint8)


def recieve_demolition_order_pixels(dpth, height):
    mask = get_heights_mask(dpth, height)

    kernel_size = 10
    mask = cv.erode(mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size)), iterations=2)
    mask = cv.dilate(mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size)), iterations=10)
    mask = cv.erode(mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size)), iterations=5)
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.imshow('mask', mask)
    # cv.waitKey()
    coords = []
    for i in contours:
        bEllipse = cv.fitEllipse(i)
        pos = np.int0(bEllipse[0])
        zone = [pos[0], pos[1], int(max(bEllipse[1]))]
        coords.append(zone)
        cv.line(mask, (pos[0], pos[1]), (pos[0], pos[1]), 128, thickness=int(zone[2]))
    cv.imshow('mask', mask)
    # cv.waitKey()
    return coords


def recieve_demolition_order(dpth, height=27, type=1):
    zones_in_pixels = recieve_demolition_order_pixels(dpth, height)
    if type == 0 or type == 1:
        points_in_pixels = np.array(destroyer(zones_in_pixels)).astype(np.uint16)
    if type == 2 or type == 3:
        points_in_pixels = np.array(snake_destroyer(zones_in_pixels)).astype(np.uint16)
    # print(points_in_pixels)
    path_in_pixels = np.reshape(points_in_pixels, (-1, 2))
    zones = []
    for i in path_in_pixels:
        zones.append([rgb_to_3d(i[0], i[1], table_grip)[0],
                      rgb_to_3d(i[0], i[1], table_grip)[1],
                      0])
    if (type % 2):
        return zones
    else:
        return zones[::-1]


def get_sorted_coords(rgb, dpth):
    frame = np.copy(rgb)
    red_data, red_img = ct.find_obj(frame, ct.hsv_red, "red", True)
    blue_data, blue_img = ct.find_obj(frame, ct.hsv_blue, "blue", True)
    data, col_img = ct.detect_collision(frame, red_data, blue_data, True)

    data = ct.grab_able(frame, data)
    cv.imshow('collisions', col_img)
    cv.waitKey(1)
    coords = []
    colors = []
    others_big = []
    others_small = []

    for d in data:
        if (-get_objects_heights(dpth)[d['pos'][1], d['pos'][0]] > 30):
            print(int(-get_objects_heights(dpth)[d['pos'][1], d['pos'][0]]), end=' ')
            print(d['type'])
            d['type'] = 'OTHER_BIG'

    # закинуть красные
    for d in data:
        if (d['color'][0] == 'b' or d['type'] != 'NORM'):
            continue
        if (d['isAble'] == 'n'):
            d['type'] = 'OTHER_SMALL'
            continue
        if rgb_to_3d(d['pos'][0], d['pos'][1], dpth)[0] == 0:
            continue
        colors.append(d['color'][0])
        xyz = [rgb_to_3d(d['pos'][0], d['pos'][1], dpth)[0], rgb_to_3d(d['pos'][0], d['pos'][1], dpth)[1],
               d['gripAng'], d['putAng']]
        coords.append(xyz)

    # закинуть синие
    order = ['QUAD', 'LONG', 'NORM']
    for type in order:
        for d in data:
            if (d['color'][0] == 'r'):
                continue
            if (d['isAble'] == 'n'):
                d['type'] = 'OTHER_SMALL'
                continue
            if (d['type'] == type):
                if rgb_to_3d(d['pos'][0], d['pos'][1], dpth)[0] == 0:
                    continue
                colors.append(d['color'][0])
                xyz = [rgb_to_3d(d['pos'][0], d['pos'][1], dpth)[0], rgb_to_3d(d['pos'][0], d['pos'][1], dpth)[1],
                       d['gripAng'], d['putAng']]
                coords.append(xyz)

    order = ['OTHER_BIG', 'OTHER_SMALL']
    for type in order:
        for d in data:
            if (d['type'] == type):
                if rgb_to_3d(d['pos'][0], d['pos'][1], dpth)[0] == 0:
                    continue
                xyz = [rgb_to_3d(d['pos'][0], d['pos'][1], dpth)[0], rgb_to_3d(d['pos'][0], d['pos'][1], dpth)[1],
                       d['angle']]
                if (d['type'] == 'OTHER_BIG'):
                    others_big.append(xyz)
                else:
                    others_small.append(xyz)

    return np.array(coords, dtype=float), colors, others_small, others_big


def destroyer(arr):
    debug = False
    out = []
    xMin = 1
    xMax = 1279
    yMin = 1
    yMax = 719
    x_center = (xMax + xMin) / 2
    y_center = (yMax + yMin) / 2
    for elem in arr:
        x = elem[0]
        y = elem[1]
        r = elem[2]

        k = (y - y_center) / (x - x_center)
        b = y_center - k * x_center
        if x > x_center and y > y_center:
            if (x - x_center) / (x_center) > (y - y_center) / y_center:
                beginPoint = [xMax, k * xMax + b]
                if debug: print("1")
            else:
                beginPoint = [(yMax - b) / k, yMax]
                if debug: print("2")
        if x < x_center and y > y_center:
            if (x_center - x) / (x_center) < (y - y_center) / y_center:
                beginPoint = [(yMax - b) / k, yMax]
                if debug: print("3")
            else:
                beginPoint = [xMin, k * xMin + b]
                if debug: print("4")
        if x < x_center and y < y_center:
            if (x_center - x) / (x_center) > (y_center - y) / y_center:
                beginPoint = [xMin, k * xMin + b]
                if debug: print("5")
            else:
                beginPoint = [(yMin - b) / k, yMin]
                if debug: print("6")
        if x > x_center and y < y_center:
            if (x - x_center) / (x_center) < (y_center - y) / y_center:
                beginPoint = [(yMin - b) / k, yMin]
                if debug: print("7")
            else:
                beginPoint = [xMax, k * xMax + b]
                if debug: print("8")

        out.append([beginPoint, [x_center, y_center]])

    return out


def snake_destroyer(arr):
    debug = False
    out = np.array([0, 0, 0, 0])
    xMin = 1
    xMax = 1279
    yMin = 1
    yMax = 719
    x_center = (xMax + xMin) / 2
    y_center = (yMax + yMin) / 2
    final = []
    for elem in arr:
        x = elem[0]
        y = elem[1]
        r = elem[2]
        k = (y - y_center) / (x - x_center)
        b = y_center - k * x_center
        if x > x_center and y > y_center:
            if (x - x_center) / (x_center) > (y - y_center) / y_center:
                beginPoint = [xMax, k * xMax + b]
                if debug: print("1")
            else:
                beginPoint = [(yMax - b) / k, yMax]
                if debug: print("2")
        if x < x_center and y > y_center:
            if (x_center - x) / (x_center) < (y - y_center) / y_center:
                beginPoint = [(yMax - b) / k, yMax]
                if debug: print("3")
            else:
                beginPoint = [xMin, k * xMin + b]
                if debug: print("4")
        if x < x_center and y < y_center:
            if (x_center - x) / (x_center) > (y_center - y) / y_center:
                beginPoint = [xMin, k * xMin + b]
                if debug: print("5")
            else:
                beginPoint = [(yMin - b) / k, yMin]
                if debug: print("6")
        if x > x_center and y < y_center:
            if (x - x_center) / (x_center) < (y_center - y) / y_center:
                beginPoint = [(yMin - b) / k, yMin]
                if debug: print("7")
            else:
                beginPoint = [xMax, k * xMax + b]
                if debug: print("8")

        alpha = math.atan(k)
        out = [[beginPoint[0], beginPoint[1]]]
        dh = r / 2
        T = np.array([[math.cos(alpha), math.sin(alpha)], [-math.sin(alpha), math.cos(alpha)]])
        for i in range(int(2 * r / dh + 1)):
            high = np.array([-r + i * dh, r])
            high = np.dot(T, high)
            high[0] += x
            high[1] += y

            out.append(high)
            # print(T,"\n", high,"\n", np.dot(T,high).astype(np.uint16))
            low = np.array([-r + i * dh, -r])
            low = np.dot(T, low)
            low[0] += x
            low[1] += y
            out.append(low)
        np_out = np.array(out)
        final.append(np_out)
    final = np.array(final)
    final[final <= 0] = 1
    final = final.astype(np.uint16)
    return final
