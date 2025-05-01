import os
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# ------------------ 1. Load Camera Parameters & Images ------------------ #
def read_camera_parameters(file_path):
    camera_params = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        data = line.split()
        if len(data) < 22:
            continue
        img_name = data[0]
        K = np.array(data[1:10], dtype=float).reshape(3, 3)
        R = np.array(data[10:19], dtype=float).reshape(3, 3)
        t = np.array(data[19:22], dtype=float).reshape(3, 1)
        Rt = np.hstack((R, t))
        P = K @ Rt
        camera_params[img_name] = {"K": K, "R": R, "t": t, "P": P}
    return camera_params

def load_images(image_dir, num_images):
    images = []
    for i in range(0, num_images):
        img_name = f"ll{i}.png"
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is not None:
            images.append((img_name, img))
    return images

# ------------------ 2. Feature Detection & Matching ------------------ #
def detect_and_compute_keypoints(img, detector):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_features(desc1, desc2, ratio=0.75):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = [m for m, n in matches if m.distance < ratio * n.distance]
    return good_matches

def draw_matches(img1, kp1, img2, kp2, matches):
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
    plt.close('all')

    plt.figure(figsize=(15, 8))
    plt.title("Feature Matches")
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# ------------------ 3. F, E, Pose Estimation, Testing ------------------ #
def estimate_fundamental_matrix(kp1, kp2, matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    return F, pts1[mask.ravel() == 1], pts2[mask.ravel() == 1]

def compute_essential_matrix(F, K1, K2):
    return K2.T @ F @ K1

def recover_pose(E, pts1, pts2, K):
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

def test_fundamental_matrix(F, pts1, pts2):
    print("Testing Fundamental Matrix...")
    errors = []
    for i in range(len(pts1)):
        x1 = np.append(pts1[i], 1)
        x2 = np.append(pts2[i], 1)
        error = np.abs(x2.T @ F @ x1)
        errors.append(error)
    mean_error = np.mean(errors)
    print(f"Mean Epipolar Constraint Error: {mean_error:.6f}\n")

def test_essential_matrix(E):
    print("Testing Essential Matrix...")
    U, S, Vt = np.linalg.svd(E)
    print(f"Singular values: {S}")
    if np.isclose(S[2], 0, atol=1e-5):
        print("✅ Essential matrix has correct rank (2).")
    else:
        print("❌ Essential matrix does NOT have correct rank!")
    det = np.linalg.det(E)
    print(f"Det(E) = {det}")
    if np.isclose(det, 0, atol=1e-5):
        print("✅ Essential matrix satisfies det(E) = 0.")
    else:
        print("❌ Essential matrix does NOT satisfy det(E) = 0.\n")

# ------------------ 4. Triangulation & 3D Visualization ------------------ #
def triangulate_points(P1, P2, pts1, pts2, K):
    pts1_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), K, None)
    pts2_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), K, None)
    pts_4d = cv2.triangulatePoints(P1, P2, pts1_norm, pts2_norm)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T
    return pts_3d

def visualize_3d_points(points_3d,colors):
    print("Visualizing 3D points...")
    pcd = o3d.geometry.PointCloud()
    print(len(points_3d),len(colors))
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(colors)
    o3d.visualization.draw_geometries([pcd])
