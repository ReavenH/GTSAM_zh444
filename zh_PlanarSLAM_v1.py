#!usr/bin/python

'''

Test version 1 of the planar SLAM based on factor graph and GTSAM optimizer.

Input: pseudo SLAM visual data (camera stream), with 4x7 chessboard shot on iPhone.
Output: positions being continuously updated.

Components:
    Initial States: observed from first few camera frames.
    Odometry Update: calculated from pre-integrated measurements between 10 camera frames (should not be faster than the iSAM speed)
    Factor Tree: between factors (odometry: dx, dy, dyaw) and nodes (robot state: x,y,yaw)
    Landmarks: the only chessboard.
    Loop Closing
    iSAM2 optimizer

'''

import cv2
import gtsam
from gtsam.symbol_shorthand import L, X
import os
import numpy as np
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import math

# camera calibration params
# 加载相机校准参数
calibration_data = np.load("calibration_data.npz")
mtx = calibration_data['mtx']
dist = calibration_data['dist']

# frame size to be processed
frame_size = (806, 605)

# chessboard pattern size
pattern_size = (4, 7)

# initialize the corner points in 3d and 2d worlds
# 用于存储棋盘格角点的空数组
objpoints = []  # 在3D空间中的棋盘格角点
imgpoints = []  # 在2D图像平面中的图像角点

# coordinate plot definition
# 创建用于绘制坐标轴的虚拟点
axis_points = np.float32([[0,0,0], [0,50,0], [50,0,0], [0,0,-50]]).reshape(-1,3)

# 生成棋盘格的3D坐标，以毫米为单位
square_size = 25 # mm
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

# 视频文件路径
# The chessboard should be vertical to the ground.
video_path = '/mnt/hgfs/Hiwonder_SharedFiles/testimgs7/1.mp4'

# TODO initialize a gtsam factor tree and related components
#   1) define noise models (params should be tuned)
#   2) create an empty factor tree
#   3) create empty arrays to hold X and L
#   4) add 1 prior to the X_all
#   5) initialize the initial_estimate
#       NOTE: initial_estimate is for optimization, along with graph

# Create noise models
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.2]))

# Create an empty nonlinear factor graph
graph = gtsam.NonlinearFactorGraph()

# empty arrays to hold X and L
X_all = []
L_all = []
X_all.append(X(1))
# L_all.append(L(1)) # L(1) is the only landmark since there's only one chessboard.

# add prior state X1 at the origin
graph.add(gtsam.PriorFactorPose2(X_all[0], gtsam.Pose2(0.0, 0.0, 0.0), PRIOR_NOISE))

# initialize the initial estimate
# the initial estimate of L1 should be added in the iteration part
initial_estimate = gtsam.Values()

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 获取视频的帧速率和帧大小
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# 创建视频编写器
# 如果你想将处理后的帧保存为新视频，可以使用 cv2.VideoWriter
# out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

count_frame = 0
count_detected = 0
do_SLAM = True

def DistEuler(frame, ret, disp = True):
    # 在图像上绘制棋盘格角点
    cv2.drawChessboardCorners(frame, pattern_size, corners, ret)

    # 执行相机姿势估计
    ret_PnP, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)

    # 投影3D点到图像平面
    imgpts, jac = cv2.projectPoints(axis_points, rvecs, tvecs, mtx, dist)

    # 获取欧拉角
    rmat, _ = cv2.Rodrigues(rvecs)
    # angles, _, _, _, _, _, = cv2.decomposeProjectionMatrix(np.hstack((rmat, tvecs)))
    # 使用旋转矩阵计算欧拉角
    angles = cv2.RQDecomp3x3(rmat)[0]
    print(angles)

    # 打印欧拉角
    text_euler = "Frame: {}".format(count_frame) + " Euler Angles: {:.2f}, {:.2f}, {:.2f}".format(angles[0],
                                                                                                  angles[1],
                                                                                                  angles[2])

    # 计算相机到棋盘格的距离
    # TODO the vertical distance or the straight-line distance
    # ChatGPT says this is a straight-line distance
    distance = math.sqrt(tvecs[0] ** 2 + tvecs[1] ** 2 + tvecs[2] ** 2) / 10  # 测算距离

    if disp:
        # 绘制坐标轴
        frame = cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0, 0, 255), 5)
        frame = cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0, 255, 0), 5)
        frame = cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (255, 0, 0), 5)

        print(text_euler)
        cv2.putText(frame, text_euler, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 打印距离
        print("相机到目标的距离: {:.2f} cm".format(distance))

        # 将距离信息添加到图像
        text = "Distance: {:.2f} cm".format(distance)
        cv2.putText(frame, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # 显示原始图像
        # cv2.imwrite(save_path + image_file, frame)
        cv2.imshow('Chessboard Corners', frame)
        cv2.waitKey(np.floor(1000//fps))  # 显示图像1秒钟

        # 如果想要保存处理后的帧到新视频，可以使用下面的代码
        # out.write(frame)

    return angles, distance

while True:
    # 逐帧读取视频
    ret, frame = cap.read()

    # 检查是否成功读取帧
    if not ret:
        break

    count_frame += 1

    # 在这里添加你的处理逻辑，例如图像处理、特征提取等
    frame = cv2.resize(frame, (frame_size[0], frame_size[1]))

    # 将图像转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret_corner, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    print("Frame ", count_frame, " Corner: ", ret)

    if ret_corner:

        count_detected += 1

        # 将棋盘格角点添加到objpoints和imgpoints中
        objpoints.append(objp)
        imgpoints.append(corners)

        # get euler angles and distances
        angle, distance = DistEuler(frame, ret_corner)
        yaw = angle[0]

        # calculate the x and y of the robot at the world's coordinate
        x = np.sin(np.radians(yaw)) * distance
        y = np.cos(np.radians(yaw)) * distance

        # TODO: 1) pre-integrate 10 frame's measurements to get an odometry update
        #       2) if count_detected % 10 == 0, update odometry and add new node (close loop detection), add X-Landmark observation

        if do_SLAM:
            # see if landmark initial should be added.
            if count_detected == 1:
                L_all.append(L(count_detected)) # add a landmark
                graph.add(
                    gtsam.BearingRangeFactor2D(X_all[0], L_all[0], gtsam.Rot2.fromDegrees(yaw), # should determine which angle to use.
                                               distance, MEASUREMENT_NOISE)
                )
                # consider the landmark as the fixed origin
                # add initial state and initial landmark

                initial_estimate.insert(X_all[0], gtsam.Pose2(x, y, yaw)) # add initial to X1
                initial_estimate.insert(L_all[0], gtsam.Point2(0.0, 0.0))
                # store the old measurements
                yaw_previous = yaw
                distance_previous = distance
                x_previous = x
                y_previous = y

            elif count_detected > 1:
                # TODO consider loop closure. Consider IMU data for x and y odometry
                if count_detected % 10 == 0:
                    # pre-integrate is to ignore the reads during the 10 frames.
                    dyaw = yaw - yaw_previous # the odometry of yaw
                    # dx and dy should be based on the orientation of robot
                    dx = x - x_previous




    # 检测按键 'q'，如果按下则退出循环
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()

# 如果使用了 VideoWriter，也要释放它
# out.release()

# 关闭所有窗口
cv2.destroyAllWindows()