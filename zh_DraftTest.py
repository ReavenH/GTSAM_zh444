import numpy as np
import math
import gtsam
from gtsam.symbol_shorthand import L, X

current_estimate = gtsam.Values()
current_estimate.insert(X(1),gtsam.Pose2(0, 0, 0))
current_estimate.insert(X(2), gtsam.Pose2(1, 0, np.radians(-45)))

# green arrow points towards the front of the robot
BR = np.array([[0, 1], [90, math.sqrt(2)]]).astype('float')

def BR2Global(bearing: float, range: float, current_estimate: gtsam.Values, current_pose_no: int):
    print("In function BR2Global")
    print("The current node is: ", current_pose_no)
    print("Bearing: ", bearing)
    print("Range: ", range)
    if current_pose_no == 1:  # the current pose is the initial pose (0,0,0)
        x_lm = - range * math.sin(bearing)  # bearing: left=positive, right=negative angles
        y_lm = range * math.cos(bearing)
        lm_coords = np.array([[x_lm], [y_lm]])
    else:
        current_pose = np.array([current_estimate.atPose2(X(current_pose_no)).x(),
                                 current_estimate.atPose2(X(current_pose_no)).y(),
                                 current_estimate.atPose2(X(current_pose_no)).theta()])
        # calculate the landmark coordinate in the pose node's coordinate system
        x_lm = - range * math.sin(bearing * math.pi / 180)
        y_lm = range * math.cos(bearing * math.pi / 180)
        print("LM coods relative to the pose coordinate: ", x_lm, y_lm)
        lm_coords = np.array([[x_lm], [y_lm]])
        # use -dyaw to indicate the mapping from the previous node coordinate system
        print("current_pose[2] \n", current_pose[2])
        current_pose_yaw = current_pose[2]
        print("current_pose_yaw \n", current_pose_yaw)
        rotation_matrix = np.array([[math.cos(current_pose_yaw), - math.sin(current_pose_yaw)],
                                    [math.sin(current_pose_yaw), math.cos(current_pose_yaw)]])
        print("Rotation Matrix \n", rotation_matrix)
        lm_coords = np.matmul(rotation_matrix, lm_coords)  # rotate the current node's coordinate system
        print("LM coords after rotation: ", lm_coords[0], lm_coords[1])
        # map the landmark coordinates relative to pose node to the global coordinate system
        print(current_pose[0:-1].T, lm_coords.reshape(2))
        lm_coords = lm_coords.reshape(2) + current_pose[0:-1]  # check if it is a column vector
        print(lm_coords)
    return lm_coords

for i in range(1, 3):
    lm_coords = BR2Global(BR[i - 1, 0], BR[i - 1, 1], current_estimate, i).reshape(2)
    print("Landmark coordinates: ", lm_coords[0], lm_coords[1])
