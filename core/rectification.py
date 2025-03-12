import cv2
import numpy as np
import matplotlib.pyplot as plt

def rectify_images(img1, img2, K, dist_coeffs, T):
    # Extract rotation (R) and translation (t)
    R = T[:3, :3]  # Extract rotation from transformation matrix
    t = T[:3, 3]   # Extract translation

    print(t)
    print(R)

    # Perform stereo rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, dist_coeffs,  # Camera 1 intrinsics and distortion
        K, dist_coeffs,  # Camera 2 intrinsics and distortion
        img1.shape[:2][::-1],  # Image size (width, height)
        R, t, 
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=1  # 1 = No cropping
    )

    # Compute rectification maps
    map1_x, map1_y = cv2.initUndistortRectifyMap(K, dist_coeffs, R1, P1, img1.shape[:2][::-1], cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(K, dist_coeffs, R2, P2, img2.shape[:2][::-1], cv2.CV_32FC1)

    # Rectify images
    img1_rectified = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(img2, map2_x, map2_y, cv2.INTER_LINEAR)

    return img1_rectified, img2_rectified

# Load undistorted images
img1 = cv2.imread("2.jpg")  # First image
img2 = cv2.imread("1.jpg")  # Second image

# Camera parameters (already defined)
# K - Camera intrinsic matrix
# dist_coeffs - Distortion coefficients

# Define transformation matrix
T = np.eye(4)
T[:3, 3] = np.array([0.12, 0, 0])  # Translation: 12 cm along x-axis

# Rectify images
rectified_img1, rectified_img2 = rectify_images(img1, img2, K, dist, T)

# Save or display rectified images
cv2.imwrite('rectified_img1.jpg', rectified_img1)
cv2.imwrite('rectified_img2.jpg', rectified_img2)

plt.imshow(cv2.cvtColor(rectified_img1, cv2.COLOR_BGR2RGB))
plt.title("Rectified Left Image")
plt.axis('off')
plt.show()

plt.imshow(cv2.cvtColor(rectified_img2, cv2.COLOR_BGR2RGB))
plt.title("Rectified Right Image")
plt.axis('off')
plt.show()
