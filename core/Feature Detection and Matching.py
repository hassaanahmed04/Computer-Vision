import cv2
import numpy as np
import matplotlib.pyplot as plt

# Feature Detection and Matching by using Different Algorithms

# surf = cv2.SURF_create()  # Updated SURF implementation
# bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

akaze = cv2.AKAZE_create()
bf_akaze = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# sift = cv2.SIFT_create()
# flann = cv2.FlannBasedMatcher()

# orb = cv2.ORB_create()
# bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Detect keypoints and descriptors in both images
kp1, des1 = akaze.detectAndCompute(rectified_img1, None)
kp2, des2 = akaze.detectAndCompute(rectified_img2, None)

# Match features using FLANN-based matcher
matches = bf_akaze.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance

# Set a threshold to filter out weak matches
# This threshold can be adjusted based on your needs
threshold = 60 # Matches with distance are considered strong
good_matches = [m for m in matches if m.distance < threshold]

# Print number of good matches
print(f"Total matches found: {len(matches)}")
print(f"Total strong matches after filtering: {len(good_matches)}")

# Extract the matched keypoints for the good matches
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Use RANSAC to compute the Fundamental Matrix and filter out outliers
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=3)

# Select inliers (good matches after RANSAC)
pts1_inliers = pts1[mask.ravel() == 1]
pts2_inliers = pts2[mask.ravel() == 1]


# Visualize the matched features after filtering with RANSAC
matched_img = cv2.drawMatches(rectified_img1, kp1, rectified_img2, kp2, good_matches, None, 
                              matchesMask=mask.ravel().tolist(), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matched features
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
plt.title("Strong Feature Matches")
plt.axis("off")
plt.show()

# # Save the matched features image
cv2.imwrite("strong_feature_matches_with_ransac.jpg", matched_img)

# Compute Essential Matrix
E = K.T @ F @ K
print("Essential Matrix:")
print(E)

# Decompose Essential Matrix to get Rotation and Translation
U, S, Vt = np.linalg.svd(E)
W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

# Possible solutions
R1 = U @ W @ Vt
R2 = U @ W.T @ Vt
t = U[:, 2]

print("Possible Rotation Matrices:")
print(R1)
print(R2)
print("Translation Vector:")
print(t)
