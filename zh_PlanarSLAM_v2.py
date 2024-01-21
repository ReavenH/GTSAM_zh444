import cv2
import numpy as np
import os
import apriltag
import gtsam
from gtsam.symbol_shorthand import L, X
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import math
from scipy.spatial.transform import Rotation as R
import csv

'''

Debug Problem (Nov. 30, 23:00):
The yaw is changing, while the origin remains the same (when landmarks are not incorporated).

Postulated Reasons:
1. Between Pose Factors and Pose Nodes:
    the prior error of the x and y are too big.
2. Measurement Nodes:
    1) the measurement error prior is too small.
    2) in landmark slam, ground truth for landmarks is needed.
    3) the AprilTag yaw is not suitable for relative measurements.
    
TODO Dec. 4:
1. Adapt the measurements calculation and update.
2. Measure the sigma of measurements and update the noise prior.
3. Calibrate the IMU data.
'''

# define path and read mp4 data
read_path = "/mnt/hgfs/Hiwonder_SharedFiles/IMU_filtering/1130/puppy_test_1130/output_1130_test_7.avi"
cap = cv2.VideoCapture(read_path)
save_path = "/mnt/hgfs/Hiwonder_SharedFiles/IMU_filtering/1130/puppy_test_1130/"
save_name = "ndydxdyaw_1130_test_7_result.mp4"
display_frame = True
do_landmark_slam = False

# read csv odometry files.
IMU_odometry_path = "/mnt/hgfs/Hiwonder_SharedFiles/IMU_filtering/1130/puppy_test_1130/ndydxdyaw_1130_test_7.csv"
IMU_odometry = []

# open CSV file and read. note that each odometry should be stored as a tuple
with open(IMU_odometry_path, newline='') as csvfile:
    # instanciate a csv reader
    csv_reader = csv.reader(csvfile)

    # skip the hearders
    header = next(csv_reader)

    # read the csv file by each line and append the data into a numpy array
    # the format of each row: [frame ID, dx, dy, dyaw].
    for row in csv_reader:
        row = [float(element) for element in row]  # convert each element in a row to float
        IMU_odometry = np.append(IMU_odometry, row, axis=0)  # append one line to the np array, each line is a tuple

IMU_odometry = np.reshape(IMU_odometry, (-1, 4))

# frame size to be processed
# frame_size = (806, 605)
frame_size = (640, 480)  # frame size of the PuppyPi

# define a video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vidout = cv2.VideoWriter(save_path + save_name, fourcc, 13.0, (frame_size[0], frame_size[1]))

if not cap.isOpened():
    print("Video is not opened, exiting...")
    exit(1)

# read the calibration parameters from file
# calibration_data = np.load("calibration_data_6x9.npz")
# mtx = calibration_data['mtx']  # camera matrix
# dist = calibration_data['dist']  # distortion matrix

# intrinsics of PuppyPi
mtx = np.matrix([[619.063979, 0,          302.560920],
                        [0,          613.745352, 237.714934],
                        [0,          0,          1]])
dist = 0.033
cam_params = np.array([619.063979, 613.745352, 302.560920, 237.714934])

# define an apriltag detector
tag_size = 0.16  # 16 centimeters
apriltag_options = apriltag.DetectorOptions(families='tag36h11', quad_decimate=1.0)
apriltag_detector = apriltag.Detector(apriltag_options)
object_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).astype('float32')  # object reference points
object_points *= tag_size

# initialize the dictionary to store the mapping from tagID to Landmark ID.
tag2landmark = {}  # keys: tagID, values: landmark ID.

# TODO: modify noise params.
# Declare the 2D translational standard deviations of the prior factor's Gaussian model, in meters.
prior_xy_sigma = 0.05  # default 0,3 meters

# Declare the 2D rotational standard deviation of the prior factor's Gaussian model, in degrees.
prior_theta_sigma = 4  # default 5 degrees

# Declare the 2D translational standard deviations of the odometry factor's Gaussian model, in meters.
odometry_xy_sigma = 0.02  # default 0.2 meters

# Declare the 2D rotational standard deviation of the odometry factor's Gaussian model, in degrees.
odometry_theta_sigma = 4  # default 5 degrees

# Declare the 2D measurement error priors
measurement_distance_sigma = 0.1
measurement_angle_sigma = 50

# Although this example only uses linear measurements and Gaussian noise models, it is important
# to note that iSAM2 can be utilized to its full potential during nonlinear optimization. This example
# simply showcases how iSAM2 may be applied to a Pose2 SLAM problem.
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([prior_xy_sigma,
                                                        prior_xy_sigma,
                                                        prior_theta_sigma*np.pi/180]))

ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([odometry_xy_sigma,
                                                            odometry_xy_sigma,
                                                            odometry_theta_sigma*np.pi/180]))

MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([measurement_distance_sigma, measurement_angle_sigma]))

# Create a Nonlinear factor graph as well as the data structure to hold state estimates.
graph = gtsam.NonlinearFactorGraph()
initial_estimate = gtsam.Values()  # initialize init estimate

# add the initials to the graph
graph.push_back(gtsam.PriorFactorPose2(X(1), gtsam.Pose2(0, 0, 0), PRIOR_NOISE))
initial_estimate.insert(X(1), gtsam.Pose2(0.0, 0.0, 0.0)) # default: 0.5, 0.0, 0.2

# Initialize the current estimate which is used during the incremental inference loop.
current_estimate = initial_estimate

# Create iSAM2 parameters which can adjust the threshold necessary to force relinearization and how many
# update calls are required to perform the relinearization.
parameters = gtsam.ISAM2Params()
parameters.setRelinearizeThreshold(0.1)
parameters.setRelinearizeSkip(1)
isam = gtsam.ISAM2(parameters)

# iterate over all frames and update the factor graph.
count_frame = 0  # frame counter
# current_LM_ID = 0 # initialize the landmark ID counter
new_LM_ID = 0

# define the LC detction function
def determine_loop_closure(odom: np.ndarray, current_estimate: gtsam.Values,
    key: int, xy_tol=0.6, theta_tol=17) -> int:
    """Simple brute force approach which iterates through previous states
    and checks for loop closure.

    Args:
        odom: Vector representing noisy odometry (x, y, theta) measurement in the body frame.
        current_estimate: The current estimates computed by iSAM2.
        key: Key corresponding to the current state estimate of the robot.
        xy_tol: Optional argument for the x-y measurement tolerance, in meters.
        theta_tol: Optional argument for the theta measurement tolerance, in degrees.
    Returns:
        k: The key of the state which is helping add the loop closure constraint.
            If loop closure is not found, then None is returned.
    """

    if current_estimate:
        prev_est = current_estimate.atPose2(X(key+1))
        rotated_odom = prev_est.rotation().matrix() @ odom[:2]
        curr_xy = np.array([prev_est.x() + rotated_odom[0],
                            prev_est.y() + rotated_odom[1]])
        curr_theta = prev_est.theta() + odom[2]
        for k in range(1, key+1):
            pose_xy = np.array([current_estimate.atPose2(X(k)).x(),
                                current_estimate.atPose2(X(k)).y()])
            pose_theta = current_estimate.atPose2(X(k)).theta()
            if (abs(pose_xy - curr_xy) <= xy_tol).all() and \
                (abs(pose_theta - curr_theta) <= theta_tol*np.pi/180):
                    return k

# define the progress report function
def report_on_progress(graph: gtsam.NonlinearFactorGraph, current_estimate: gtsam.Values,
                        key: int):
    """Print and plot incremental progress of the robot for 2D Pose SLAM using iSAM2."""

    # Print the current estimates computed using iSAM2.
    print("*"*50 + f"\nInference after State {key+1}:\n")
    print(current_estimate)

    # Compute the marginals for all states in the graph.
    marginals = gtsam.Marginals(graph, current_estimate)

    # Plot the newly updated iSAM2 inference.
    fig = plt.figure(0)
    axes = fig.gca()
    plt.cla()

    i = 1
    while current_estimate.exists(X(i)):
        gtsam_plot.plot_pose2(0, current_estimate.atPose2(X(i)), 0.5, marginals.marginalCovariance(X(i)))
        i += 1

    plt.axis('equal')
    axes.set_xlim(-1, 5)
    axes.set_ylim(-1, 3)
    plt.pause(1)

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

    '''
    TODO: udpate the odometry data. Assume the csv structure is 
    [dx (+: left), dy (+: backwards), dyaw (+: anti-clockwise)]
    '''

    # acquire the odometry data from the specific line of csv
    # each line of IMU_odometry is a tuple of 3 elements (dx, dy, dyaw)
    [_, noisy_odom_x, noisy_odom_y, noisy_odom_theta] = IMU_odometry[count_frame]

    # detect loop closure, TODO: adjust the xy_tol(tolerance in meters), and theta tolerance
    loop = determine_loop_closure(IMU_odometry[count_frame], current_estimate, count_frame, xy_tol=0.8, theta_tol=25)

    # Add a binary factor in between two existing states if loop closure is detected.
    # Otherwise, add a binary factor between a newly observed state and the previous state.
    if loop:
        print("Loop Closure Detected: X", loop)
        graph.push_back(gtsam.BetweenFactorPose2(X(count_frame + 1), X(loop),
                                                 gtsam.Pose2(noisy_odom_x, noisy_odom_y, noisy_odom_theta),
                                                 ODOMETRY_NOISE))
        count_frame -= 1 # no new Pose node is generated

    else:
        graph.push_back(gtsam.BetweenFactorPose2(X(count_frame + 1), X(count_frame + 2),
                                                 gtsam.Pose2(noisy_odom_x, noisy_odom_y, noisy_odom_theta),
                                                 ODOMETRY_NOISE))

        # Compute and insert the initialization estimate for the current pose using the noisy odometry measurement.
        computed_estimate = current_estimate.atPose2(X(count_frame + 1)).compose(gtsam.Pose2(noisy_odom_x,
                                                                                noisy_odom_y,
                                                                                noisy_odom_theta))
        initial_estimate.insert(X(count_frame + 2), computed_estimate)

    # TODO: update the measurement data
    # detect the tags in the current frame
    apriltag_result = apriltag_detector.detect(frame)

    # iterate over each tag detected
    for each_tag in apriltag_result:
        each_tag_corners = each_tag.corners  # extract the corner points of each tag
        cv2.polylines(frame, [each_tag_corners.astype(int)], isClosed=True, color=(0, 255, 0),
                      thickness=2)  # draw the contour

        # calculate the distance and rotation angles
        # print("object_points: ", object_points)
        # print("each_tag_corners: ", each_tag_corners)

        each_tag_corners = each_tag_corners.astype('float32')
        ret, rvec, tvec = cv2.solvePnP(object_points, each_tag_corners, mtx, dist)
        each_tag_distance = np.linalg.norm(tvec)  # 距离
        # each_tag_angles = cv2.Rodrigues(rvec)[0].round()  # 旋转角度, deprecated
        each_tag_corners = each_tag_corners.astype(int)

        # angle detection
        homo = each_tag.homography
        num, Rs, Ts, Ns = cv2.decomposeHomographyMat(homo, mtx)
        '''r = R.from_dcm(Rs[0].T)
        eulerangle = r.as_euler('xyz').T * 180 / math.pi
        angledim1.append(eulerangle[2])'''
        r = R.from_dcm(Rs[1].T)
        eulerangle = r.as_euler('xyz').T * 180 / math.pi
        each_tag_yaw = eulerangle[2]
        '''angledim2.append(eulerangle[2]) # Z axis angle offset (yaw)
        r = R.from_dcm(Rs[2].T)
        eulerangle = r.as_euler('xyz').T * 180 / math.pi
        angledim3.append(eulerangle[2])
        r = R.from_dcm(Rs[3].T)
        eulerangle = r.as_euler('xyz').T * 180 / math.pi
        angledim4.append(eulerangle[2])'''

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

        # TODO update the Apriltag Measurement (check the distance and angle units); LM ground truth is needed for LM SLAM
        if do_landmark_slam:
            # check if the current tag has been stored as a landmark
            if each_tag.tag_id not in tag2landmark:  # new landmark
                # create a new ID for the new landmark
                new_LM_ID += 1
                # store the new landmark ID into the dictionary's value
                tag2landmark[each_tag.tag_id] = new_LM_ID
                graph.add(gtsam.BearingRangeFactor2D(X(count_frame + 1), L(new_LM_ID), gtsam.Rot2.fromDegrees(each_tag_yaw),
                                                     each_tag_distance, MEASUREMENT_NOISE))
                # initial_estimate.insert(L(new_LM_ID))  # add initial for the firstly seen landmark

            else: # existing landmark
                # update the measurement between current node and the existing
                graph.add(gtsam.BearingRangeFactor2D(X(count_frame + 1), L(tag2landmark[each_tag.tag_id]),
                                                     gtsam.Rot2.fromDegrees(each_tag_yaw),
                                                     each_tag_distance, MEASUREMENT_NOISE))


    # write the detected frame into a video output
    vidout.write(frame_raw)

    # display result
    if display_frame:
        cv2.imshow('AprilTag Detection', frame_raw)

    # check if exit
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    # TODO: use iSAM2 to update the current estimate,
    # also draw the trajectory

    # Perform incremental update to iSAM2's internal Bayes tree, optimizing only the affected variables.
    isam.update(graph, initial_estimate)
    current_estimate = isam.calculateEstimate()

    # Report all current state estimates from the iSAM2 optimzation.
    report_on_progress(graph, current_estimate, count_frame)
    initial_estimate.clear()

    # accumulate the counter
    count_frame += 1

    # print the current mapping from TagID to LandmarkID
    print("tag2landmark\n", tag2landmark.keys())

# Print the final covariance matrix for each pose after completing inference on the trajectory.
marginals = gtsam.Marginals(graph, current_estimate)
i = 1
for i in range(1, len(IMU_odometry.shape[0]) + 1):
    print(f"X{i} covariance:\n{marginals.marginalCovariance(i)}\n")

plt.ioff()
plt.show()

cap.release()
vidout.release()
cv2.destroyAllWindows()