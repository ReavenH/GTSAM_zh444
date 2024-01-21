import os
import numpy as np
import cv2
import math

# 加载相机校准参数
calibration_data = np.load("calibration_data.npz")
mtx = calibration_data['mtx']
dist = calibration_data['dist']

# 设置棋盘格的内部角点数目
pattern_size = (3, 3)  # 在你的棋盘格上的角点数目, default 6*9

# 设置棋盘格的每个方格的实际大小（单位：毫米）
square_size = 25  # 棋盘格方格大小，这里假设每个方格大小为25.4毫米（1英寸）

# 用于存储棋盘格角点的空数组
objpoints = []  # 在3D空间中的棋盘格角点
imgpoints = []  # 在2D图像平面中的图像角点

# 生成棋盘格的3D坐标，以毫米为单位
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

# 创建用于绘制坐标轴的虚拟点
axis_points = np.float32([[0,0,0], [0,50,0], [50,0,0], [0,0,-50]]).reshape(-1,3)

# Folder Path
# 读取照片文件夹中的所有照片
folder_path = "/mnt/hgfs/Hiwonder_SharedFiles/testimgs6/"
save_path = folder_path + "euler_and_dist/"

# 获取文件夹中的所有图像文件
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    # 读取图像
    image_path = os.path.join(folder_path, image_file)
    frame = cv2.imread(image_path)
    # rescale the image.
    frame = cv2.resize(frame, (806, 605))

    # 将图像转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    print("File ", image_file, " Corner: ", ret)
    # 如果找到棋盘格角点
    if ret:

        # 将棋盘格角点添加到objpoints和imgpoints中
        objpoints.append(objp)
        imgpoints.append(corners)

        # 在图像上绘制棋盘格角点
        cv2.drawChessboardCorners(frame, pattern_size, corners, ret)

        # 执行相机姿势估计
        ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)

        # 投影3D点到图像平面
        imgpts, jac = cv2.projectPoints(axis_points, rvecs, tvecs, mtx, dist)

        # 绘制坐标轴
        frame = cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0,0,255), 5)
        frame = cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0,255,0), 5)
        frame = cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (255,0,0), 5)

        # 获取欧拉角
        rmat, _ = cv2.Rodrigues(rvecs)
        # angles, _, _, _, _, _, = cv2.decomposeProjectionMatrix(np.hstack((rmat, tvecs)))
        # 使用旋转矩阵计算欧拉角
        angles = cv2.RQDecomp3x3(rmat)[0]
        print(angles)

        # 打印欧拉角
        text_euler = "Image: {}".format(image_file) + " Euler Angles: {:.2f}, {:.2f}, {:.2f}".format(angles[0], angles[1], angles[2])
        print(text_euler)
        cv2.putText(frame, text_euler, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 计算相机到棋盘格的距离
        distance = math.sqrt(tvecs[0] ** 2 + tvecs[1] ** 2 + tvecs[2] ** 2) / 10  # 测算距离

        # 打印距离
        print("相机到目标的距离: {:.2f} cm".format(distance))

        # 将距离信息添加到图像
        text = "Distance: {:.2f} cm".format(distance)
        cv2.putText(frame, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # 显示原始图像
        cv2.imwrite(save_path + image_file, frame)
        cv2.imshow('Chessboard Corners', frame)
        cv2.waitKey(1000)  # 显示图像1秒钟W

cv2.destroyAllWindows()