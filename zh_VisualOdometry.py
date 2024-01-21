import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to compute visual odometry using ORB features
def compute_visual_odometry(prev_frame, current_frame):
    # Convert frames to grayscale
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    keypoints1, descriptors1 = orb.detectAndCompute(prev_frame_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(current_frame_gray, None)

    # Create BFMatcher (Brute Force Matcher) object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort them in ascending order of distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Select top N matches
    N = 50
    selected_matches = matches[:N]

    # Get corresponding keypoints
    prev_points = np.float32([keypoints1[match.queryIdx].pt for match in selected_matches]).reshape(-1, 1, 2)
    current_points = np.float32([keypoints2[match.trainIdx].pt for match in selected_matches]).reshape(-1, 1, 2)

    # Compute the Essential matrix
    E, _ = cv2.findEssentialMat(prev_points, current_points)

    # Recover pose from the Essential matrix
    _, R, _, _ = cv2.recoverPose(E, prev_points, current_points)

    # Convert rotation matrix to rotation angles
    angles, _ = cv2.Rodrigues(R)

    return angles

# Function to plot the camera trajectory
def plot_trajectory(trajectory):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], marker='o')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Camera Trajectory')
    plt.show()

def plot_trajectory_with_orientation(trajectory, angles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], marker='o', label='Camera Trajectory')

    # Plot arrows for camera orientation
    for i in range(len(trajectory)):
        ax.quiver(trajectory[i, 0], trajectory[i, 1], trajectory[i, 2],
                  np.cos(angles[i, 0]), np.sin(angles[i, 0]), 0.0,
                  color='red', length=0.2, normalize=True, arrow_length_ratio=0.1)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Camera Trajectory with Orientation')
    ax.legend()
    plt.show()

# Function to perform bundle adjustment
def bundle_adjustment(points_3d, points_2d, camera_matrix, dist_coeffs, rotation_matrix, translation_vector):
    _, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, points_2d, camera_matrix, dist_coeffs,
                                               rvec=rotation_matrix, tvec=translation_vector,
                                               useExtrinsicGuess=True)
    return rvec, tvec

# camera calibration params
# 加载相机校准参数
calibration_data = np.load("calibration_data.npz")
camera_matrix = calibration_data['mtx']
dist_coeffs = calibration_data['dist']

# Initialize video capture
cap = cv2.VideoCapture('/mnt/hgfs/Hiwonder_SharedFiles/testimgs7/3.mp4')

# Read the first frame
ret, prev_frame = cap.read()

# Initialize trajectory array
trajectory = np.zeros((0, 3))
angles_list = []
points_3d = np.zeros((0, 3)).astype('float32')
points_2d = np.zeros((0, 2)).astype('float32')

count_frame = 0

while True:
    # Read the current frame
    ret, current_frame = cap.read()

    if not ret:
        break

    count_frame += 1
    if count_frame % 5 != 0: # calculate the camera motion every 2 frames
        continue
    count_frame = 0

    # Compute visual odometry
    angles = compute_visual_odometry(prev_frame, current_frame)
    angles_list.append(angles)
    print("Angles list: ", angles_list)

    # Update the previous frame for the next iteration
    prev_frame = current_frame.copy()

    # Accumulate translation vectors to estimate camera trajectory
    translation_vector = np.array([angles[0, 0], angles[1, 0], angles[2, 0]]).astype('float32')
    trajectory = np.vstack((trajectory, translation_vector))

    '''# Add 3D points for bundle adjustment
    points_3d = np.vstack((points_3d, translation_vector)).astype('float32')
    # Add 2D points for bundle adjustment (use some feature detection method)
    # Example: ORB feature detection
    orb = cv2.ORB_create()
    keypoints, _ = orb.detectAndCompute(current_frame, None)
    if keypoints:
        points_2d = np.vstack((points_2d, np.array([keypoint.pt for keypoint in keypoints]))).astype('float32')

    # Perform bundle adjustment every N frames (adjust based on your requirements)
    N = 2
    if len(angles_list) % N == 0:
        # Perform bundle adjustment
        rotation_matrix = cv2.Rodrigues(angles_list[-1])[0]
        rvec, tvec = bundle_adjustment(points_3d, points_2d, camera_matrix, dist_coeffs, rotation_matrix,
                                       translation_vector)
        # Update the rotation and translation vectors
        angles_list[-1] = rvec.flatten()
        translation_vector = tvec.flatten()
'''
    # Display the result
    # print("Rotation Angles:", angles)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# Plot the camera trajectory with orientation
angles_array = np.vstack(angles_list)
print("angles_array shape: ", angles_array.shape)
print("trajectory shape: ", trajectory.shape)
plot_trajectory_with_orientation(trajectory, angles_array)
