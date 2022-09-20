import cv2
from cv2 import bitwise_and
from cv2 import bitwise_not
import numpy as np
from sklearn.metrics import zero_one_loss

hsv_red = ((104, 58, 93), (255, 255, 255))
hsv_blue = ((0, 58, 93), (77, 255, 255))
frameCntr = (640, 360)
hsv_lines = ((0, 176, 0), (255, 255, 141))


def find_obj(_image, hsv_constr, color, debug=False):
    out = []
    show = _image.copy()  # копия изображения для отрисовки
    image = _image.copy() / 255
    image **= 0.5  # коэф контрастности
    image = (image * 255).astype(np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # lines_mask = cv2.inRange(hsv, hsv_lines[0], hsv_lines[1]) # hsv маска
    # lines_mask = cv2.erode(lines_mask, np.ones((2,2), np.uint8), iterations=1)
    # cv2.imshow("Lines_mask", lines_mask)
    # kernel = np.ones((5,5), np.uint8)
    # lines_mask = cv2.dilate(lines_mask, kernel, iterations=4)
    # lines_mask = cv2.erode(lines_mask, kernel, iterations=3)
    # cv2.imshow("Lines_mask2", lines_mask)
    mask = cv2.inRange(hsv, hsv_constr[0], hsv_constr[1])  # hsv маска

    # mask = mask - lines_mask
    # lines_mask = cv2.bitwise_not(lines_mask)
    # mask = cv2.bitwise_and(mask, lines_mask)
    # mask[mask < 0] = 0
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for i in contours:
        area = cv2.contourArea(i)
        if len(i) < 5:
            continue
        bEllipse = cv2.fitEllipse(i)
        hull = cv2.convexHull(i, True)
        hull = cv2.approxPolyDP(hull, 15, True)
        if (not cv2.isContourConvex(hull)):
            continue

        pos = np.int0(bEllipse[0])

        rect = cv2.minAreaRect(i)  # описанный прямоугольник
        c_points = np.int0(cv2.boxPoints(rect))  # точки углов
        dst1 = np.linalg.norm(c_points[0] - c_points[1])
        dst2 = np.linalg.norm(c_points[0] - c_points[3])
        if dst1 > dst2:
            angle = np.arctan((c_points[0][0] - c_points[1][0]) / (c_points[0][1] - c_points[1][1]))
        else:
            angle = np.arctan((c_points[0][0] - c_points[3][0]) / (c_points[0][1] - c_points[3][1]))
        angle += np.pi / 2
        ang2center = np.arctan2((pos[0] - frameCntr[0]), (pos[1] - frameCntr[1])) + np.pi
        out.append({"pos": pos,
                    "area": area,
                    "corners_count": len(hull),
                    "angle": angle,
                    "ang2center": ang2center,
                    "corners": c_points,
                    "color": color,
                    "contour": i
                    })
        # some draw
        if debug:
            x2 = int(pos[0] + 40 * np.cos(angle))
            y2 = int(pos[1] - 40 * np.sin(angle))
            cv2.line(show, pos, (x2, y2), (0, 255, 255), 2)
            cv2.circle(show, pos, 2, (0, 255, 255), 1)
            cv2.putText(show, "Pos: {0}, {1} area: {2} c_num: {3} ang: {4}".format(*pos, area, len(hull),
                                                                                   int(np.rad2deg(angle))), pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            cv2.drawContours(show, contours, -1, (0, 255, 0), 3)
            cv2.ellipse(show, bEllipse, (255, 0, 255), 1, cv2.LINE_AA)
            cv2.drawContours(show, [c_points], 0, (0, 255, 255), 2)
    if debug:
        return out, show
    return out


def detect_collision(_image, red_data, blue_data, debug=False):
    show = _image.copy()
    data = red_data.copy() + blue_data.copy()
    data = sorted(data, key=lambda x: np.linalg.norm(x['pos'] - frameCntr))
    for ind, f in enumerate(data):
        cv2.drawContours(show, [f["corners"]], -1, (0, 255, 0), 3)
        cv2.drawContours(show, [f["contour"]], -1, (255, 255, 0), 3)
        dst1 = np.linalg.norm(f['corners'][0] - f['corners'][1])
        dst2 = np.linalg.norm(f['corners'][0] - f['corners'][3])
        if f['color'] == "blue" and (dst1 * dst2) / f['area'] < 1.3 and 6500 < f['area'] < 15000 and max(dst1,
                                                                                                         dst2) / min(
                dst1, dst2) <= 1.2 and f[
            'corners_count'] == 4:  # определяем квадрат по соотношению площадей и соотношению длин сторон
            cv2.putText(show, "QUAD", f['pos'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            data[ind]['type'] = "QUAD"

        # KOEFFS OF AREA for typical block
        elif 2500 < f['area'] < 5600 and 1.5 < max(dst1, dst2) / min(dst1, dst2) < 3:
            cv2.putText(show, "NORM", f['pos'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            data[ind]['type'] = "NORM"

        # KOEFFS OF AREA and PROPORTIONAL for long block
        elif f['color'] == "blue" and (dst1 * dst2) / f['area'] < 1.3 and 5000 < f['area'] < 15000 and 3 < max(dst1,
                                                                                                               dst2) / min(
                dst1, dst2) < 6:
            cv2.putText(show, "LONG", f['pos'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            data[ind]['type'] = "LONG"
        else:
            if f['area'] > 15000:
                cv2.putText(show, "OTHER_BIG", f['pos'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                data[ind]['type'] = "OTHER_BIG"
            else:
                cv2.putText(show, "OTHER_SMALL", f['pos'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                data[ind]['type'] = "OTHER_SMALL"
        if debug:
            cv2.circle(show, frameCntr, int(np.linalg.norm(f['pos'] - frameCntr)), (200, 200, 200), 2)
            cv2.putText(show, str(ind) + " angle: " + str(int(np.rad2deg(f['angle']))), f['pos'] + (0, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    if debug:
        return data, show
    return data


'''
def sort_by_dst(_image, _data):
    data = _data.copy()
    show = _image.copy()
    for ind, f in enumerate(data):
        cv2.circle(show, frameCntr, int(np.linalg.norm(f['pos'] - frameCntr)), (0, 255, 255), 2)
        cv2.putText(show, str(ind), f['pos'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 127, 255))
        data[ind]['dst2cntr'] = masknp.linalg.norm(f['pos'] - frameCntr)
        data[ind]['ang2cntr'] = np.arctan2((f['pos'][0] - frameCntr[0]), (f['pos'][1] - frameCntr[1])) + np.pi
        cv2.putText(show, "     " + str(np.rad2deg(data[ind]['ang2cntr'])), f['pos'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 127, 255))
        cv2.line(show, f['pos'], frameCntr, (0, 0, 255), 2)
    return data, show
'''


def grab_able(_image, _data, debug=False):
    gripper_len = (230, 50)
    data = _data.copy()
    show = _image.copy()
    mask_all = np.zeros((720, 1280))
    for ind, f in enumerate(data):
        cv2.fillPoly(mask_all, pts=[f['contour']], color=255)
    for ind, f in enumerate(data):
        enemys_mask = mask_all.copy()
        current_obj_mask = np.zeros((720, 1280))
        cv2.fillPoly(current_obj_mask, pts=[f['contour']], color=255)
        enemys_mask = cv2.bitwise_and(cv2.bitwise_not(current_obj_mask), enemys_mask)
        ang = f['angle'] + np.pi / 2
        p1 = np.int0((f['pos'][0] - gripper_len[0] / 2 * np.cos(ang), f['pos'][1] + gripper_len[0] / 2 * np.sin(ang)))
        p2 = np.int0((f['pos'][0] + gripper_len[0] / 2 * np.cos(ang), f['pos'][1] - gripper_len[0] / 2 * np.sin(ang)))
        gripper_mask = np.zeros((720, 1280))
        cv2.line(gripper_mask, p1, p2, 255, gripper_len[1])
        collisions_mask = bitwise_and(gripper_mask, enemys_mask)
        if np.sum(collisions_mask) / 255 < 100:
            data[ind]['isAble'] = "y"
            data[ind]['gripAng'] = ang - np.pi / 2
        else:
            ang = f['angle']
            p1 = np.int0(
                (f['pos'][0] - gripper_len[0] / 2 * np.cos(ang), f['pos'][1] + gripper_len[0] / 2 * np.sin(ang)))
            p2 = np.int0(
                (f['pos'][0] + gripper_len[0] / 2 * np.cos(ang), f['pos'][1] - gripper_len[0] / 2 * np.sin(ang)))
            gripper_mask = np.zeros((720, 1280))
            cv2.line(gripper_mask, p1, p2, 255, gripper_len[1])
            collisions_mask = bitwise_and(gripper_mask, enemys_mask)
            if np.sum(collisions_mask) / 255 < 100 and f['type'] != "LONG":
                data[ind]['isAble'] = "p"
                data[ind]['gripAng'] = ang - np.pi / 2
            else:
                data[ind]['isAble'] = "n"
        if debug:

            if data[ind]['isAble'] == "y":
                ang = f['angle'] + np.pi / 2
                p1 = np.int0(
                    (f['pos'][0] - gripper_len[0] / 2 * np.cos(ang), f['pos'][1] + gripper_len[0] / 2 * np.sin(ang)))
                p2 = np.int0(
                    (f['pos'][0] + gripper_len[0] / 2 * np.cos(ang), f['pos'][1] - gripper_len[0] / 2 * np.sin(ang)))
                cv2.line(show, p1, p2, (100, 100, 100), gripper_len[1])
                cv2.putText(show,
                            "isAble: " + data[ind]['isAble'].upper() + " grip_angle: " + str(int(np.rad2deg(ang))),
                            f['pos'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            elif data[ind]['isAble'] == "p":
                ang = f['angle']
                p1 = np.int0(
                    (f['pos'][0] - gripper_len[0] / 2 * np.cos(ang), f['pos'][1] + gripper_len[0] / 2 * np.sin(ang)))
                p2 = np.int0(
                    (f['pos'][0] + gripper_len[0] / 2 * np.cos(ang), f['pos'][1] - gripper_len[0] / 2 * np.sin(ang)))
                cv2.line(show, p1, p2, (100, 100, 100), gripper_len[1])
                cv2.putText(show,
                            "isAble: " + data[ind]['isAble'].upper() + " grip_angle: " + str(int(np.rad2deg(ang))),
                            f['pos'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            else:
                cv2.putText(show, "isAble: " + data[ind]['isAble'].upper(), f['pos'], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255))
        if data[ind]['type'] == "QUAD":
            data[ind]['putAng'] = np.pi / 4
        elif data[ind]['isAble'] == "y":
            data[ind]['putAng'] = 0
        elif data[ind]['isAble'] == "p":
            data[ind]['putAng'] = np.pi / 2
        else:
            data[ind]['putAng'] = 0
        if debug:
            cv2.putText(show, "Ind: " + str(ind) + " putAng: " + str(int(np.rad2deg(data[ind]['putAng']))),
                        f['pos'] + (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        # cv2.imshow("Gripper", gripper_mask)
        # cv2.imshow("Collisions", collisions_mask)

        # cv2.imshow("Enemys", enemys_mask)
        # cv2.waitKey(0)
    if debug:
        return data, show
    return data


if __name__ == "__main__":
    from pathlib import Path

    pathlist = Path("rgb_npy").rglob('*.npy')
    for i in pathlist:
        # frame = np.load("images_data/23-10-50-54-np_rgb.npy")
        # frame = np.load("images_data/23-10-55-17-np_rgb.npy")
        # frame = np.load("images_data/23-10-44-37-np_rgb.npy")
        frame = np.load(str(i))
        red_data, red_img = find_obj(frame, hsv_red, "red", True)
        blue_data, blue_img = find_obj(frame, hsv_blue, "blue", True)
        data, col_img = detect_collision(frame, red_data, blue_data, True)
        # friends = list(filter(lambda x: x['type'] in ("NORM", "QUAD", "LONG"), data)) # кубики, которые можно брать
        # enemys = list(filter(lambda x: x['type'] != "NORM", data)) # нужно разбить
        data, grab_img = grab_able(frame, data, True)

        cv2.imshow('Red', red_img)
        cv2.imshow('Blue', blue_img)
        cv2.imshow('Collisions', col_img)
        cv2.imshow('Grab_able', grab_img)
        cv2.waitKey(0)