import cv2
import numpy as np

# 读取图像
# 读取图像
image = cv2.imread('/mnt/hgfs/Hiwonder_SharedFiles/testimgs/7.png')

# Down-sample the image.
image = cv2.resize(image, (680, 430))

# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义蓝色范围
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

# 创建蓝色掩码
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 在原始图像上应用掩码
blue_region = cv2.bitwise_and(image, image, mask=blue_mask)

# 寻找蓝色区域的轮廓
_, contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 选择最大的轮廓
if contours:
    largest_contour = max(contours, key=cv2.contourArea)

    # 计算轮廓的二阶中心矩
    M = cv2.moments(largest_contour)
    mu20 = M['mu20']
    mu02 = M['mu02']
    mu11 = M['mu11']

    # 在图像上绘制箭头表示二阶中心矩的方向
    center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    scale_factor = 50  # 调整箭头长度
    arrow_color = (0, 255, 255)  # 箭头颜色，这里使用黄色
    '''
    cv2.arrowedLine(image, center, int(center[0] + int(mu20*scale_factor), center[1]), arrow_color, 2)
    cv2.arrowedLine(image, center, int(center[0], center[1] + int(mu02*scale_factor)), arrow_color, 2)
    cv2.arrowedLine(image, center, int(center[0] + int(mu11*scale_factor), center[1] + int(mu11*scale_factor)), arrow_color, 2)
    '''
    # 绘制轮廓
    cv2.drawContours(image, [largest_contour], 0, (0, 255, 0), 2)

    # 显示结果图像
    cv2.imshow('Detected Blue Region with Arrowed Moments', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No blue region detected.")
