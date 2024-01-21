import cv2
import numpy as np

# 输入视频文件名
video1_file = '/mnt/hgfs/Hiwonder_SharedFiles/IMU_filtering/1205/Jackal_MATLAB_output_video.mp4'
video2_file = '/mnt/hgfs/Hiwonder_SharedFiles/IMU_filtering/1205/Jackel_Video_Result.mp4'

# 打开视频文件
video1 = cv2.VideoCapture(video1_file)
video2 = cv2.VideoCapture(video2_file)

# 获取视频的帧率、宽度和高度
fps = int(video1.get(cv2.CAP_PROP_FPS))
width = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建 VideoWriter 对象，设置输出视频的文件名和格式
output_file = '/mnt/hgfs/Hiwonder_SharedFiles/IMU_filtering/1205/Jackal_Concat_Video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_file, fourcc, fps, (2*width, height))

while True:
    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()

    # 如果其中一个视频结束，就退出循环
    if not ret1 or not ret2:
        break

    # 将两个帧横向拼接在一起
    concatenated_frame = np.concatenate((frame1, frame2), axis=1)

    # 写入输出视频
    output_video.write(concatenated_frame)

# 释放资源
video1.release()
video2.release()
output_video.release()

print('视频拼接完成，输出文件：', output_file)