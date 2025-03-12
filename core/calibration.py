import cv2
import numpy as np
import glob
import os
import json
import matplotlib.pyplot as plt

def calibrate_camera(image_folder, chessboard_size=(7, 9), square_size=0.025, save_results=True):
    """
    Calibrate a single camera and compute intrinsic and extrinsic parameters.
    :param image_folder: Path to the folder containing calibration images.
    :param chessboard_size: Tuple indicating the number of inner corners per row and column.
    :param square_size: Size of a square in meters.
    :param save_results: If True, saves calibration results to a file.
    :return: Camera matrix, distortion coefficients, rotation and translation vectors, reprojection error.
    """
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    images = glob.glob(os.path.join(image_folder, '*.jpg')) + glob.glob(os.path.join(image_folder, '*.png'))

    if not images:
        print("No images found in the folder.")
        return None

    fig, axes = plt.subplots(1, min(len(images), 5), figsize=(15, 5))  # Display up to 5 images
    if len(images) == 1:
        axes = [axes]

    for i, fname in enumerate(images[:5]):  # Limiting to first 5 images for calibration
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)
        axes[i].set_title(f"Image {i+1}")
        axes[i].axis("off")

    plt.show()

    if not objpoints or not imgpoints:
        print("Chessboard pattern not found in images.")
        return None

    # Camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Compute reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    reprojection_error = total_error / len(objpoints)

    print("Camera Matrix:\n", camera_matrix)
    print("\nDistortion Coefficients:\n", dist_coeffs)
    print("\nReprojection Error:\n", reprojection_error)

    if save_results:
        # Saving calibration results
        calib_data = {
            'camera_matrix': camera_matrix.tolist(),
            'dist_coeffs': dist_coeffs.tolist(),
            'reprojection_error': reprojection_error
        }
        with open(os.path.join(image_folder, 'calibration_results.json'), 'w') as f:
            json.dump(calib_data, f)

    return camera_matrix, dist_coeffs, rvecs, tvecs, reprojection_error
