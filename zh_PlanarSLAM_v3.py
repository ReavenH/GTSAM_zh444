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
import scipy.io as io

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

TODO Dec. 5:
1. Adapt the measurements calculation and update. 
    1.1 check how to get the current estimation by printing x, y, yaw;
        DONE
    1.2 write a function to calculate the estimated global x, y (Point2 typedef) for newly added landmarks;
        DONE (BR2Global function using rotation matrix)
        PS: the global estimate is based on the noisy current pose estimate, where the noisy pose estimate is
            the odometry plus the previous calculated pose.
    1.3 adapt the initial_estimate codes for landmarks.
        DONE
2. Measure the sigma of measurements and update the noise prior.
    2.1 Range: input camera videos from the Puppy of different distances from the Apriltag, 
        get the sigma of each distance, calculate the mean sigma as the range noise;
    2.2 Bearing: input camera videos from the Puppy from different view angles of the same distance, get the sigmas.
        Test 1: 66cm + 0 degree. 
            Range_sigma = 4.793458e-05
            Bearing_sigma = 2.461924e-05 (in radians)
        Test 2: 100cm + 0 degree.
            Range_sigma = 0.0001670088
            Bearing_sigma = 3.200517e-05 (in radians)
        Test 3: 87cm + unknown angle offset
            Range_sigma = 0.0001219077 
            Bearing_sigma = 2.823240e-05 (in radians)
        Overall Average Data:
            Range_sigma = 0.00012493309621653700 
            Bearing_sigma = 2.828560333333333e-05 (in radians)
3. Measure the odometry noise sigma.
    3.1 walk 1 meter for 10 times, get the IMU odometries over the period, calculate the sigma over the 10 samples;
        DONE
    3.2 (to be updated).
    
TODO Dec. 8
1. The Landmark initial estimate update concept is correct.
2. The apriltag has a large mis-detection rate, affecting the accuracy of upcoming landmarks.
    compare the results of MATLAB and our algorithm.
    Checked. Detection rates are similar.
3. Mapping the origin of the offset angle from the camera to the IMU.
4. The distance calculation is incorrect. 
    Checked. No obvious empacts.
5. The translation from quaternions to euler angles might not be correct. There are some 180~-180 swings.
'''

# define path and read mp4 data
'''
read_path = "/mnt/hgfs/Hiwonder_SharedFiles/IMU_filtering/1130/puppy_test_1130/output_1130_test_7.avi"
save_path = "/mnt/hgfs/Hiwonder_SharedFiles/IMU_filtering/1130/puppy_test_1130/"
save_name = "ndydxdyaw_1130_test_7_result.mp4"
'''
read_path = "/mnt/hgfs/Hiwonder_SharedFiles/IMU_filtering/1205/Jackal_Video.mp4"
save_path = "/mnt/hgfs/Hiwonder_SharedFiles/IMU_filtering/1205/"
save_name = "Jackel_Video_Result.mp4"
cap = cv2.VideoCapture(read_path)
display_frame = True
do_landmark_slam = True  # will not affect writing the Apriltag detection result video

# read csv odometry files.
IMU_type = 'Jackal'  # choose from 'PippyPi' or 'Jackal'.
# PuppyPi odometry path.
'''
IMU_odometry_path = "/mnt/hgfs/Hiwonder_SharedFiles/IMU_filtering/1130/puppy_test_1130/ndydxdyaw_1130_test_7.csv"
'''

# Jackal odometry path.
# standard deviation of each column: [0.01381677 0.00656759 0.00171639 0.00020382 0.00681032 0.0058156 0.02099308]
IMU_odometry_path = "/mnt/hgfs/Hiwonder_SharedFiles/IMU_filtering/1205/Jackal_Between_Factor.mat"

IMU_odometry = []

# open CSV file and read. note that each odometry should be stored as a tuple
if IMU_type == 'PuppyPi':
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
    print("Using PuppyPi IMU Odometry.")
elif IMU_type == 'Jackal':
    IMU_odometry_raw = io.loadmat(IMU_odometry_path)['betweenFactorOut']  # load as a dictionary.
    print("Using Jackal IMU Raw Odometry, shape: ", IMU_odometry_raw.shape)
    # transform the quaternions into standard yaw, pitch, roll.
    IMU_odometry = np.zeros((IMU_odometry_raw.shape[0], 6))  # create an empty container
    IMU_odometry[:, 0:3] = IMU_odometry_raw[:, 0:3]
    rotation_obj = R.from_quat(IMU_odometry_raw[:, 3:])
    IMU_odometry[:, 3:] = np.degrees(rotation_obj.as_euler('ZYX'))  # denotes the order of output
    # the units of Jackal IMU: meters, radians
    print("Transformed raw quaternions into euler angles, new odometry shape is:", IMU_odometry.shape)


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

# intrinsics of PuppyPi camera
'''
mtx = np.matrix([[619.063979, 0, 302.560920],
                 [0, 613.745352, 237.714934],
                 [0, 0, 1]])
dist = 0.033  # TODO: needs to be revised to a 1x5 vector.
cam_params = np.array([619.063979, 613.745352, 302.560920, 237.714934])
'''

# intrinsics of Jackal Robot camera
mtx = np.matrix([[564.5156, 0, 312.0826],
                 [0, 566.4349, 251.8228],
                 [0, 0, 1]])
cam_params = np.array([564.5156, 566.4349, 312.0806, 251.8228])
# distortion coeffs of Jackal Robot camera: k1, k2, p1, p2 (, k3 optional).
# The Jackal uses a simplified pinhole camera model
dist = np.array([-0.3162, 0.1189, 0, 0])

# define an apriltag detector
tag_family = 'tag36h11'  # Jackal uses tag36h11
# tag_size = 0.16  # the size of actual tag is 16 centimeters
tag_size = 0.13665  # Jackal uses a size of 136.65mm tag size
apriltag_options = apriltag.DetectorOptions(families=tag_family, quad_decimate=1.0)
apriltag_detector = apriltag.Detector(apriltag_options)
object_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).astype('float32')  # object reference points
object_points *= tag_size  # convert the object points to the actual size

# initialize the dictionary to store the mapping from tagID to Landmark ID.
tag2landmark = {}  # keys: tagID, values: landmark ID.

# TODO: modify noise params.
# Declare the 2D translational standard deviations of the prior factor's Gaussian model, in meters.
prior_xy_sigma = 0.05  # default 0.3 meters

# Declare the 2D rotational standard deviation of the prior factor's Gaussian model, in degrees.
prior_theta_sigma = 4  # default 5 degrees

# Declare the 2D translational standard deviations of the odometry factor's Gaussian model, in meters.
odometry_xy_sigma = 0.00012493309621653700   # default 0.2 meters

# Declare the 2D rotational standard deviation of the odometry factor's Gaussian model, in degrees.
odometry_theta_sigma = 0.0019227266412239800000   # default 5 degrees, 0.19227266412239800000

# Declare the 2D measurement error priors
# measurement_distance_sigma = 0.00016941248333333331
measurement_distance_sigma = 0.00016941248333333331 * 10
# measurement_angle_sigma = math.radians(2.828560333333333/10000)  # in radians
measurement_angle_sigma = math.radians(2.828560333333333/1000)  # in radians
# Although this example only uses linear measurements and Gaussian noise models, it is important
# to note that iSAM2 can be utilized to its full potential during nonlinear optimization. This example
# simply showcases how iSAM2 may be applied to a Pose2 SLAM problem.
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([prior_xy_sigma,
                                                         prior_xy_sigma,
                                                         prior_theta_sigma * np.pi / 180]))

ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([odometry_xy_sigma,
                                                            odometry_xy_sigma,
                                                            odometry_theta_sigma * np.pi / 180]))

MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([measurement_distance_sigma, measurement_angle_sigma]))

# Create a Nonlinear factor graph as well as the data structure to hold state estimates.
graph = gtsam.NonlinearFactorGraph()
initial_estimate = gtsam.Values()  # initialize init estimate
initial_estimate_LM = np.array([[0, 1],[1, 1], [2, 1], [3, 1], [4, 1]])  # pseudo landmark ground truth

# add the initials to the graph
graph.push_back(gtsam.PriorFactorPose2(X(1), gtsam.Pose2(0, 0, 0), PRIOR_NOISE))
initial_estimate.insert(X(1), gtsam.Pose2(0.0, 0.0, 0.0))  # default: 0.5, 0.0, 0.2
# initial_estimate.insert(L(1), gtsam.BearingRange2D(gtsam.Rot2.fromDegrees(10), 0.0))

# Initialize the current estimate which is used during the incremental inference loop.
current_estimate = initial_estimate

# Create iSAM2 parameters which can adjust the threshold necessary to force relinearization and how many
# update calls are required to perform the relinearization.
parameters = gtsam.ISAM2Params()
parameters.setRelinearizeThreshold(0.1)
parameters.setRelinearizeSkip(1)

# define an optimizer, choose either one as needed
isam = gtsam.ISAM2(parameters)  # iSAM optimizer
LM_optimizer_params = gtsam.LevenbergMarquardtParams()  # LM optimizer
LM_optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, LM_optimizer_params)

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
        prev_est = current_estimate.atPose2(X(key + 1))
        rotated_odom = prev_est.rotation().matrix() @ odom[:2]
        curr_xy = np.array([prev_est.x() + rotated_odom[0],
                            prev_est.y() + rotated_odom[1]])
        curr_theta = prev_est.theta() + odom[2]
        for k in range(1, key + 1):
            pose_xy = np.array([current_estimate.atPose2(X(k)).x(),
                                current_estimate.atPose2(X(k)).y()])
            pose_theta = current_estimate.atPose2(X(k)).theta()
            if (abs(pose_xy - curr_xy) <= xy_tol).all() and \
                    (abs(pose_theta - curr_theta) <= theta_tol * np.pi / 180):
                return k

# define the progress report function
def report_on_progress(graph: gtsam.NonlinearFactorGraph, current_estimate: gtsam.Values,
                       key: int, verbose = False):
    """Print and plot incremental progress of the robot for 2D Pose SLAM using iSAM2."""
    if verbose:
        # Print the current estimates computed using iSAM2.
        print("*" * 50 + f"\nInference after State {key + 1}:\n")
        print(current_estimate)

    # Compute the marginals for all states in the graph.
    marginals = gtsam.Marginals(graph, current_estimate)

    # Plot the newly updated iSAM2 inference.
    fig = plt.figure(0)
    axes = fig.gca()
    plt.cla()

    i = 1
    while current_estimate.exists(X(i)):
        gtsam_plot.plot_pose2(0, current_estimate.atPose2(X(i)), 0.1, marginals.marginalCovariance(X(i)))
        i += 1

    # plot landmark in the graph
    i = 1
    while current_estimate.exists(L(i)):
        plt.figure(0)
        plt.scatter(current_estimate.atPoint2(L(i))[0], current_estimate.atPoint2(L(i))[1], marker='o')
        i += 1

    print("Pose nodes number: ", i - 1)

    plt.axis('equal')
    plt.grid('True')
    axes.set_xlim(-7, 7)
    axes.set_ylim(-7, 7)
    plt.pause(0.2)  # default value = 1, the delay of each plotting

# define the function to convert the bearing/range measurements to estimated global Pose2 position
# bearing (input) should be in degrees
def BR2Global(bearing: float, range: float, current_estimate: gtsam.Values, current_pose_no: int):
    print("In function BR2Global")
    print("The current node is: ", current_pose_no)
    print("Bearing: ", bearing)
    print("Range: ", range)
    if current_pose_no == 1:  # the current pose is the initial pose (0,0,0)
        x_lm = - range * math.sin(bearing * math.pi / 180)  # bearing: left=positive, right=negative, in radians
        y_lm = range * math.cos(bearing * math.pi / 180)
        lm_coords = np.array([[x_lm], [y_lm]])
    else:
        # rotation direction: theta positive <-> anti-clockwise
        current_pose = np.array([current_estimate.atPose2(X(current_pose_no)).x(),
                                 current_estimate.atPose2(X(current_pose_no)).y(),
                                 current_estimate.atPose2(X(current_pose_no)).theta()])  # theta in radians
        # calculate the landmark coordinate in the pose node's coordinate system
        x_lm = - range * math.sin(bearing * math.pi / 180)
        y_lm = range * math.cos(bearing * math.pi / 180)
        lm_coords = np.array([[x_lm], [y_lm]])
        # use -dyaw to indicate the mapping from the previous node coordinate system
        current_pose_yaw = current_pose[2]
        rotation_matrix = np.array([[math.cos(current_pose_yaw), - math.sin(current_pose_yaw)],
                                    [math.sin(current_pose_yaw), math.cos(current_pose_yaw)]])
        lm_coords = np.matmul(rotation_matrix, lm_coords)  # rotate the current node's coordinate system
        # map the landmark coordinates relative to pose node to the global coordinate system
        lm_coords = lm_coords.reshape(2) + current_pose[0:-1]  # check if it is a column vector
    return lm_coords

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
    if IMU_type == 'PuppyPi':
        [_, noisy_odom_x, noisy_odom_y, noisy_odom_theta] = IMU_odometry[count_frame]
    elif IMU_type == 'Jackal':
        # the reading is a quarternion measure.
        [noisy_odom_x, noisy_odom_y, noisy_odom_z,
         noisy_odom_yaw, noisy_odom_pitch, noisy_odom_roll] = IMU_odometry[count_frame]
        '''[noisy_odom_y, noisy_odom_x, noisy_odom_z,
         noisy_odom_yaw, noisy_odom_pitch, noisy_odom_roll] = IMU_odometry[count_frame]'''
        noisy_odom_theta = noisy_odom_yaw
        # noisy_odom_y = - noisy_odom_y
        # noisy_odom_theta = - noisy_odom_yaw
    # detect loop closure, TODO: adjust the xy_tol(tolerance in meters), and theta tolerance
    # default: xy_tol=0.05, theta_tol=3
    loop = determine_loop_closure(IMU_odometry[count_frame], current_estimate, count_frame, xy_tol=0.00005, theta_tol=0.003)

    # Add a binary factor in between two existing states if loop closure is detected.
    # Otherwise, add a binary factor between a newly observed state and the previous state.
    if loop:
        print("Loop Closure Detected: X", loop)
        graph.push_back(gtsam.BetweenFactorPose2(X(count_frame + 1), X(loop),
                                                 gtsam.Pose2(noisy_odom_x, noisy_odom_y, noisy_odom_theta),
                                                 ODOMETRY_NOISE))
        current_pose_no = count_frame + 1
        count_frame -= 1  # no new Pose node is generated

    else:
        graph.push_back(gtsam.BetweenFactorPose2(X(count_frame + 1), X(count_frame + 2),
                                                 gtsam.Pose2(noisy_odom_x, noisy_odom_y, noisy_odom_theta),
                                                 ODOMETRY_NOISE))

        # Compute and insert the initialization estimate for the current pose using the noisy odometry measurement.
        computed_estimate = current_estimate.atPose2(X(count_frame + 1)).compose(gtsam.Pose2(noisy_odom_x,
                                                                                             noisy_odom_y,
                                                                                             noisy_odom_theta))
        initial_estimate.insert(X(count_frame + 2), computed_estimate)
        # the first observed node (except the initial) is X(2)
        current_pose_no = count_frame + 2

    # TODO: update the measurement data
    # detect the tags in the current frame
    apriltag_result = apriltag_detector.detect(frame)

    # iterate over each tag detected
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
        # TODO newly added
        # each_tag_distance = each_tag_distance * 1.7
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
        '''dir_angle = (each_tag_yaw - 5) * math.pi / 180 * 1.8
        ARROW_LENGTH = 80
        delta_x = math.sin(dir_angle) * ARROW_LENGTH
        delta_y = ARROW_LENGTH / 2 * math.cos(dir_angle)
        new_center = each_tag.center + np.array([delta_x, delta_y])
        cv2.circle(frame_raw, tuple(new_center.astype(int)), 8, (255, 0, 0), 5)
        cv2.line(frame_raw, tuple(new_center.astype(int)), tuple(each_tag.center.astype(int)), (255, 0, 0), 2)'''

        # get the angle offset of the current Apriltag
        each_tag_angle_offset = math.atan(((frame_size[0]/2) - each_tag_center[0])/cam_params[0])  # radian angle offset
        each_tag_angle_offset_degree = math.degrees(each_tag_angle_offset)  # angle offset in degrees
        cv2.putText(frame_raw, f"Angle Offset: {each_tag_angle_offset_degree}", (each_tag_corners[0, 0],
                    each_tag_corners[0, 1] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # TODO update the Apriltag Measurement (check the distance and angle units); LM ground truth is needed for LM SLAM
        if do_landmark_slam:
            # check if the current tag has been stored as a landmark
            if each_tag.tag_id not in tag2landmark:  # new landmark
                # create a new ID for the new landmark
                new_LM_ID += 1
                print("found new landmark, ", new_LM_ID)
                # store the new landmark ID into the dictionary's value
                tag2landmark[each_tag.tag_id] = new_LM_ID
                graph.add(
                    gtsam.BearingRangeFactor2D(X(count_frame + 1), L(new_LM_ID),
                                               gtsam.Rot2.fromDegrees(each_tag_angle_offset_degree),
                                               each_tag_distance, MEASUREMENT_NOISE))
                # add initial for the firstly seen landmark
                '''
                initial_estimate.insert(L(new_LM_ID),
                                        gtsam.BearingRange2D(gtsam.Rot2.fromDegrees(each_tag_angle_offset_degree),
                                                     each_tag_distance))
                '''
                # calculate the initial estimate of the LM global coordinates
                LM_global_coords = BR2Global(each_tag_angle_offset_degree, each_tag_distance,
                                             initial_estimate, current_pose_no).reshape(2)
                # put text on the image
                cv2.putText(frame_raw, f"Initial Estimate: {LM_global_coords[0], LM_global_coords[1]}", (each_tag_corners[0, 0],
                    each_tag_corners[0, 1] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                print("New landmark estimated global coordinates: ", LM_global_coords)
                initial_estimate.insert(L(new_LM_ID),  # insert 2D poses of landmarks (should be measured)
                                         gtsam.Point2(LM_global_coords[0], LM_global_coords[1]))
                print("Inserted initial estimate for new landmark No.", new_LM_ID)
            else:  # existing landmark
                # update the measurement between current node and the existing
                graph.add(gtsam.BearingRangeFactor2D(X(count_frame + 1), L(tag2landmark[each_tag.tag_id]),
                                                     gtsam.Rot2.fromDegrees(each_tag_angle_offset_degree),
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
    # current_estimate = initial_estimate
    # Report all current state estimates from the iSAM2 optimzation.
    report_on_progress(graph, current_estimate, count_frame)
    initial_estimate.clear()

    # accumulate the counter
    count_frame += 1

    # print the current mapping from TagID to LandmarkID
    print("tag2landmark\n", tag2landmark)

# Print the final covariance matrix for each pose after completing inference on the trajectory.
marginals = gtsam.Marginals(graph, current_estimate)
i = 1
for i in range(1, len(IMU_odometry.shape[0]) + 1):
    print(f"X{i} covariance:\n{marginals.marginalCovariance(i)}\n")

plt.ioff()
plt.show()

cap.release()
vidout.release()
cv2.waitKey('q')
cv2.destroyAllWindows()