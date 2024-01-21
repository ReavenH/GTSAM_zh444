import os
import numpy as np
import cv2

# 设置棋盘格的内部角点数目
pattern_size = (6, 9)  # 在你的棋盘格上的角点数目

# 设置棋盘格的每个方格的实际大小（单位：毫米）
square_size = 25.4  # 棋盘格方格大小，这里假设每个方格大小为25.4毫米（1英寸）

# 用于存储棋盘格角点的空数组
objpoints = []  # 在3D空间中的棋盘格角点
imgpoints = []  # 在2D图像平面中的图像角点

# 读取照片文件夹中的所有照片
folder_path = "/mnt/hgfs/Hiwonder_SharedFiles/chessboard_calibration/"
save_path = folder_path + "calibrated_imgs/"
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    # 读取照片
    image_path = os.path.join(folder_path, image_file)
    print("Image Path: ", image_path)
    img = cv2.imread(image_path)
    # rescale the image size.
    img = cv2.resize(img, (806, 605))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # 如果找到棋盘格角点
    if ret:
        print("Corner Found.")
        # 将棋盘格角点添加到objpoints和imgpoints中
        objpoints.append(np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32))
        objpoints[-1][:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
        imgpoints.append(corners)

        # 在图像上绘制棋盘格角点
        cv2.drawChessboardCorners(img, pattern_size, corners, ret)
        cv2.imwrite(save_path + image_path[-5:], img)
# 执行相机校准
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 打印相机矩阵和失真系数
print("相机矩阵:\n", mtx)
print("失真系数:\n", dist)

# 保存相机矩阵和失真系数
np.savez("calibration_data_6x9.npz", mtx=mtx, dist=dist)

# display the images with chessboard corners.
cv2.imshow('Chessboard Corners', img)

# kill all windows.

cv2.waitKey(0)
cv2.destroyAllWindows()