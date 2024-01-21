import cv2
import numpy as np

# 读取图像
image = cv2.imread('/mnt/hgfs/Hiwonder_SharedFiles/testimgs/7.png')

# Down-sample the image.
image = cv2.resize(image, (680, 430))
# 将图像从BGR色彩空间转换为HSV色彩空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义红色范围的阈值（在HSV色彩空间中）
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

# threshold values for blue.
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

# 创建一个二值图像，其中红色部分被设置为白色（255），其他部分为黑色（0）
mask_red = cv2.inRange(hsv, lower_red, upper_red)

# mask the blue regions.
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)


# 查找红色区域的轮廓
_, contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# find the blue contours
_, contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制红色区域的边界
for contour in contours_red:
    area = cv2.contourArea(contour)
    if area > 1100:  # 过滤掉面积较小的轮廓，以防止噪声, default=100
        print("Shape of red contour: ", contour.shape)
        # calculate the moment of the contour
        M = cv2.moments(contour)
        # calculate the Center of Mass
        # 计算质心坐标
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # print the COM
            cv2.circle(image, (cx, cy), 5, (0, 100, 255), -1)
        cv2.drawContours(image, [contour], 0, (0, 0, 255), 2)

# Blue region boundaries.
for contour in contours_blue:
    area = cv2.contourArea(contour)
    if area > 2000:  # 过滤掉面积较小的轮廓，以防止噪声, default = 100
        print("Shape of blue contour: ", contour.shape)
        # calculate the moment of the contour
        M = cv2.moments(contour)
        # calculate the Center of Mass
        # 计算质心坐标
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # print the COM
            cv2.circle(image, (cx, cy), 5, (255, 100, 0), -1)
        cv2.drawContours(image, [contour], 0, (255, 0, 0), 2)


# 显示结果
cv2.imshow('Detected Red and Blue Tapes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()