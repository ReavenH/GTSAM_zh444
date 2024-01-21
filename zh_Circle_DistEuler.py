import cv2
import numpy as np
import os

# Camera calibration
pattern_size = (4, 11)  # Rows and columns of the circle pattern
objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# Load calibration results
calibration_data = np.load('calibration_data_circle.npz')
mtx, dist = calibration_data['mtx'], calibration_data['dist']
print("mtx: ", mtx)
print("dist: ", dist)

# 创建用于绘制坐标轴的虚拟点
axis_points = np.float32([[0,0,0], [0,50,0], [50,0,0], [0,0,-50]]).reshape(-1,3)

folder_path = "/mnt/hgfs/Hiwonder_SharedFiles/circle_calibration/"
save_path = folder_path + "calibrated_imgs/"
# while(found < num):  # Here, 10 can be changed to whatever number you like to choose
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    # Undistort an image
    image_path = os.path.join(folder_path, image_file)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (806, 605))
    img = cv2.undistort(img, mtx, dist, None, mtx)
    undistorted_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    undistorted_img = cv2.equalizeHist(undistorted_img)



    # Circle pattern detection
    ret, circles = cv2.findCirclesGrid(undistorted_img, pattern_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)

    if ret:
        # Pose estimation
        ret, rvec, tvec, inliers = cv2.solvePnPRansac(objp, circles, mtx, dist)

        # 投影3D点到图像平面
        imgpts, jac = cv2.projectPoints(axis_points, rvec, tvec, mtx, dist)

        # 绘制坐标轴
        img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0, 0, 255), 5)
        img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0, 255, 0), 5)
        img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (255, 0, 0), 5)

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Print translation vector (position)
        print("Translation Vector (tvec):", tvec)

        # Convert rotation matrix to Euler angles
        euler_angles = cv2.RQDecomp3x3(rotation_matrix)[0]
        print("Euler Angles (radians):", euler_angles)

        '''# Convert Euler angles to degrees
        euler_angles_deg = np.degrees(euler_angles)
        print("Euler Angles (degrees):", euler_angles_deg)'''

        # Calculate distance
        distance = np.linalg.norm(tvec)
        print("Distance:", distance)

        # put info on the image
        text_euler = "Image: {}".format(image_file) + " Euler Angles: {:.2f}, {:.2f}, {:.2f}".format(euler_angles[0],euler_angles[1],euler_angles[2])
        cv2.putText(img, text_euler, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # 将距离信息添加到图像
        text = "Distance: {:.2f} cm".format(distance)
        cv2.putText(img, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        print("Circle pattern not detected.")

    # Display the undistorted image with detected circle pattern
    cv2.imshow('Undistorted Image with Circle Pattern', img)
    cv2.imwrite(save_path + image_file, img)
    cv2.waitKey(1000)

cv2.destroyAllWindows()