from shutil import move
from time import time
import numpy as np
import math
import time
#from OperateCamera import OperateCamera
from OperateRobot import OperateRobot


#Another library>
from urx import Robot
import math3d as m3d
#robot = Robot("172.31.1.25")



# Connection to the robot
#rob = OperateRobot("172.31.1.25")






#sniat xc, yc, zc

### MEASURING EVERY DAY
#coordinates of the work zone
xc = -0.681
yc = -0.357
zc = 0.342
#zc = 0.338

#COORDINATES OF THE 2ND POINT
x1 = -0.660
y1 = -0.0
z1 = 0.336

#?coordinates of the 3rd point
x2 = 0
y2 = 0
z2 = 0 


### END OF MEASURES


grab_orient = np.array([1.176,2.917,-0.],dtype=float)
rob_orient = np.array([1.176,2.917,-0.],dtype=float)
cam_orient = np.array([1.487,3.536,-0.669],dtype=float)
calibrate_coords = np.array([x1,y1,z1],dtype=float)
theta = -math.atan2(z1-zc,y1-yc)
alpha = math.atan2(z2-zc,x2-xc)
print(theta)
cur_coords = np.array([0,0,0],dtype= float)


red_num = 0
blue_num = 0




work_coords = np.array([xc,yc,zc],dtype=float)

rot = math.pi/2
Tmatr = np.array([[math.cos(rot), -math.sin(rot), 0, work_coords[0]],
                [math.sin(rot), math.cos(rot), 0, work_coords[1]],
                [0, 0, 1, work_coords[2]],
                 [0, 0, 0, 1]],dtype=float)

T2 = np.array([[math.cos(theta), 0, math.sin(theta), 0],
             [0, 1, 0, 0],
             [-math.sin(theta), 0, math.cos(theta), 0],
             [0, 0, 0, 1]])


T2 = np.array([[math.cos(theta), 0, math.sin(theta), 0],
             [0, 1, 0, 0],
             [-math.sin(theta), 0, math.cos(theta), 0],
             [0, 0, 0, 1]])

T3 = np.array([[1, 0, 0, 0],
             [0, math.cos(alpha), -math.sin(alpha), 0],
             [0, math.sin(alpha), math.cos(alpha), 0],
             [0, 0, 0, 1]])



def move_to_point(point_coords, rot_z = 0):
    global rob_orient
    global cur_coords
    point_coords = np.append(point_coords,1)
    vec_points = np.dot(Tmatr,point_coords)
    trans = robot.get_pose() 
    trans.orient.rotate_zb(rot_z)
 
    rob_orient[0] = trans.pose_vector[3]
    rob_orient[1] = trans.pose_vector[4]
    rob_orient[2] = trans.pose_vector[5]
    new_vec = np.dot(np.dot(Tmatr,T2),point_coords)
    moving_coordinates = {"x": new_vec[0], "y": new_vec[1], "z": new_vec[2], "rx": rob_orient[0], "ry": rob_orient[1], "rz": rob_orient[2]}
    rob.movel(moving_coordinates)
    #not tested part
    new_new_vec = np.dot(T2,new_vec)
    #print(new_new_vec)
    #cur_coords = new_vec[0:3]
    

def orientate_to_grip():
    trans = robot.get_pose() 
    moving_coordinates = {"x": trans.pose_vector[0], "y": trans.pose_vector[1], "z": trans.pose_vector[2], "rx": grab_orient[0], "ry": grab_orient[1], "rz": grab_orient[2]}
    rob.movel(moving_coordinates)


def orientate_to_cam():
    trans = robot.get_pose() 
    moving_coordinates = {"x": trans.pose_vector[0], "y": trans.pose_vector[1], "z": trans.pose_vector[2], "rx": cam_orient[0], "ry": cam_orient[1], "rz": cam_orient[2]}
    rob.movel(moving_coordinates)


def move_to_picture():
    orientate_to_cam()
    pic_coords = np.array([0.15,0.08,0.38],dtype=float)
    move_to_point(pic_coords)


def get_coords_from_camera(coords):
    x_cal = 0
    y_cal = 0
    x = -coords[0]/1000 + 0.15+0.023
    y = -coords[1]/1000 + 0.1+0.03
    gamma = coords[2] + math.pi/2 #maybe not PI/2 
    return(np.array([x,y,gamma],dtype=float))
    

def grab_brick():
    center = np.array([0.2,0.2,0],dtype=float)
    up_vec = np.array([0,0,0.15],dtype=float)
    move_to_point(center+up_vec)
    rob.open_gripper()
    #time.sleep(1.)
    move_to_point(center)
    time.sleep(1)
    rob.close_gripper()
    time.sleep(0.5)
    move_to_point(center+up_vec)


def grab_brick_coord(coords):
    orientate_to_grip()
    vec = np.array([coords[0], coords[1], 0],dtype=float)

    up_vec = np.array([0,0,0.15],dtype=float)
    move_to_point(vec+up_vec,coords[2])
    rob.open_gripper()
    #time.sleep(1.)
    move_to_point(vec)
    time.sleep(1)
    rob.close_gripper()
    time.sleep(0.5)
    move_to_point(vec+up_vec,-coords[2])


def move_up_center():
    move_to_point(np.array([0.2,0.15,0.15],dtype=float))


def smash_1():
    #rob.open_gripper()
    print("AASDSAD")
    move_to_point(np.array([0.05, 0.05,0],dtype=float))
    smash_action(np.array([0.4,0.3,0],dtype=float))
    smash_action(np.array([0.,0.3,0],dtype=float))
    smash_action(np.array([0.4,0.,0],dtype=float))
    smash_action(np.array([0.4,0.3,0],dtype=float))


def smash_2():
    rob.open_gripper()
    move_to_point(np.array([0.0, 0.0,0],dtype=float))
    smash_action(np.array([0.2,0.15,0],dtype=float))
    move_up_center()
    move_to_point(np.array([0.4, 0.0,0],dtype=float))
    smash_action(np.array([0.2,0.15,0],dtype=float))
    move_up_center()
    move_to_point(np.array([0.4, 0.3,0],dtype=float))
    smash_action(np.array([0.2,0.15,0],dtype=float))
    move_up_center()
    move_to_point(np.array([0., 0.3,0],dtype=float))
    smash_action(np.array([0.2,0.15,0],dtype=float))
    move_up_center()


def smash_action(point):
    new_point = (point+cur_coords)/2
    move_to_point(new_point,math.pi/2)
    move_to_point(point,-math.pi/2)
    

def put_red():
    global red_num
    #coordinates relative to work zone (measure with lineyka)
    red_zone = np.array([0.585+0.04,0.+0.04,0.005],dtype=float)
    up_vec = np.array([0,0,0.15],dtype=float)
    move_to_point(red_zone+up_vec)
    up_vec_red = np.array([0,0,0.02*red_num],dtype=float)
    move_to_point(red_zone+up_vec_red)
    
    rob.open_gripper()
    time.sleep(0.5)
    move_to_point(red_zone+up_vec)
    rob.close_gripper()
    red_num = red_num + 1


def put_blue():
    global blue_num
    #coordinates relative to work zone (measure with lineyka)
    blue_zone = np.array([0.585+0.04,0.15+0.04,0.005],dtype=float)
    up_vec = np.array([0,0,0.15],dtype=float)
    move_to_point(blue_zone+up_vec)
    up_vec_red = np.array([0,0,0.02*red_num],dtype=float)
    move_to_point(blue_zone+up_vec_red)
    
    rob.open_gripper()
    time.sleep(0.5)
    move_to_point(blue_zone+up_vec)
    rob.close_gripper()
    red_num = red_num + 1

def init_robot():
    point = np.array([0.2,0.2,0.3],dtype=float)
    orientate_to_grip()
    move_to_point(point)
    rob.close_gripper()

    

def test_kinematics():
    
    rob.close_gripper()
    #move_to_point(point)

    #point = np.array([0.2,0.2,0.3],dtype=float)
    #move_to_point(point)
    grab_brick()
    put_red()


def check_calibration():
    init_robot()
    point = np.array([0.0,0.0,0.0],dtype=float)
    move_to_point(point)
    point = np.array([0.4,0.0,0.0],dtype=float)
    move_to_point(point)
    point = np.array([0.4,0.4,0.0],dtype=float)
    move_to_point(point)
    point = np.array([0.0,0.4,0.0],dtype=float)
    move_to_point(point)
    point = np.array([0.0,0.0,0.0],dtype=float)

#raskidka elementov s massiva Marka
def raskidka(points_position, points_colour):
    for i in range(points_position.shape[1]):
        point = get_coords_from_camera(points_position[i,:])
        grab_brick_coord(point)
        if(points_colour[i]=="b"):
            put_blue()
        else:
            put_red()






#print("inited")
#smash_1()
#print("SADASDSA")
#smash_2()
#move_to_picture()
#rientate_to_cam()

#check_calibration()
#point = np.array([0.15, 0.1,0.1],dtype = float)
#move_to_point(point)
#point = np.array([0.15, 0.1,0.05],dtype=float)
#move_to_point(point)

###
#p


#main
#init_robot()
#move_to_picture()
point2 = np.array([23, 30,0.05],dtype=float)
point3 = get_coords_from_camera(point2)
print(point3)
grab_brick_coord(point2)



#robpoints_position.close()