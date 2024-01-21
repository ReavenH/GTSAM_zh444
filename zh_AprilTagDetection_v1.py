import numpy as np
import apriltag
import cv2
import math
from scipy.spatial.transform import Rotation as R

read_path = "/mnt/hgfs/Hiwonder_SharedFiles/IMU_filtering/1205/output_1205_test_distance_unknown_deg_87cm.avi"
save_path = "/mnt/hgfs/Hiwonder_SharedFiles/IMU_filtering/1205/"
save_name = "result_output_1205_test_distance_unknown_deg_87cm.mp4"
cap = cv2.VideoCapture(read_path)

# img = cv2.imread("color.png")
# intrinsics of PuppyPi camera
mtx = np.matrix([[619.063979, 0, 302.560920],
                 [0, 613.745352, 237.714934],
                 [0, 0, 1]])
dist = 0.033
cam_params = np.array([619.063979, 613.745352, 302.560920, 237.714934])
frame_size = (640, 480)  # frame size of the PuppyPi

# define a video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vidout = cv2.VideoWriter(save_path + save_name, fourcc, 13.0, (frame_size[0], frame_size[1]))

# define an apriltag detector
tag_size = 0.16  # the size of actual tag is 16 centimeters
apriltag_options = apriltag.DetectorOptions(families='tag36h11', quad_decimate=1.0)
apriltag_detector = apriltag.Detector(apriltag_options)
object_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).astype('float32')  # object reference points
object_points *= tag_size  # convert the object points to the actual size

# define a container for distances and angles
distance_sample = []
angle_sample = []  # should be in radians

if not cap.isOpened():
    print("Video is not opened, exiting...")
    exit(1)

while True:
    # 逐帧读取视频
    ret, frame_raw = cap.read()

    # 检查是否成功读取帧
    if not ret:
        print("No returning frame, exiting the loop.")
        break

    # frame_raw = cv2.resize(frame_raw, (frame_size[0], frame_size[1]))
    frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2GRAY)
    frame = cv2.undistort(frame, mtx, dist)  # anti-distortion
    at_detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11', quad_decimate=1.0))
    apriltag_result = at_detector.detect(frame)
    #print("tags: {}\n".format(tags))

    for each_tag in apriltag_result:
        each_tag_corners = each_tag.corners  # extract the corner points of each tag
        cv2.polylines(frame, [each_tag_corners.astype(int)], isClosed=True, color=(0, 255, 0),
                      thickness=2)  # draw the contour
        each_tag_center = each_tag.center  # get the coordinate of center of apriltag
        # calculate the distance and rotation angles
        # print("object_points: ", object_points)
        # print("each_tag_corners: ", each_tag_corners)

        each_tag_corners = each_tag_corners.astype('float32')
        ret, rvec, tvec = cv2.solvePnP(object_points, each_tag_corners, mtx, dist)
        each_tag_distance = np.linalg.norm(tvec)  # 距离
        # each_tag_angles = cv2.Rodrigues(rvec)[0].round()  # 旋转角度, deprecated
        each_tag_corners = each_tag_corners.astype(int)

        # euler angle detection
        homo = each_tag.homography
        num, Rs, Ts, Ns = cv2.decomposeHomographyMat(homo, mtx)
        r = R.from_dcm(Rs[1].T)
        eulerangle = r.as_euler('xyz').T * 180 / math.pi
        each_tag_yaw = eulerangle[2]

        # put the ID, distance and angle offset in the frame
        cv2.putText(frame_raw, f"ID: {each_tag.tag_id}", (each_tag_corners[0, 0], each_tag_corners[0, 1] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame_raw, f"Distance: {each_tag_distance:.2f} meters",
                    (each_tag_corners[0, 0], each_tag_corners[0, 1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame_raw, f"Angles: {each_tag_yaw}", (each_tag_corners[0, 0], each_tag_corners[0, 1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # draw the pointing line of the angle direction
        dir_angle = (each_tag_yaw - 5) * math.pi / 180 * 1.8
        ARROW_LENGTH = 80
        delta_x = math.sin(dir_angle) * ARROW_LENGTH
        delta_y = ARROW_LENGTH / 2 * math.cos(dir_angle)
        new_center = each_tag.center + np.array([delta_x, delta_y])
        cv2.circle(frame_raw, tuple(new_center.astype(int)), 8, (255, 0, 0), 5)
        cv2.line(frame_raw, tuple(new_center.astype(int)), tuple(each_tag.center.astype(int)), (255, 0, 0), 2)

        # get the angle offset of the current Apriltag
        each_tag_angle_offset = math.atan(
            ((frame_size[0] / 2) - each_tag_center[0]) / cam_params[0])  # radian angle offset
        each_tag_angle_offset_degree = math.degrees(each_tag_angle_offset)  # angle offset in degrees
        cv2.putText(frame_raw, f"Angle Offset: {each_tag_angle_offset_degree}", (each_tag_corners[0, 0],
                                                                                 each_tag_corners[0, 1] - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if each_tag.tag_id == 0:
            angle_sample.append(each_tag_angle_offset)
            distance_sample.append(each_tag_distance)

    cv2.imshow('AprilTag Detection', frame_raw)
    vidout.write(frame_raw)

distance_sigma = np.std(distance_sample)
angle_sigma = np.std(angle_sample)

print("Distance Sigma: ", distance_sigma)
print("Angle Sigma: ", angle_sigma)

cap.release()
vidout.release()
cv2.destroyAllWindows()