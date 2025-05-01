import numpy as np
import cv2
import json

def show_parameters(E, F, R1, R2, t, pts1, pts2):
    output = []

    U, S, Vt = np.linalg.svd(E)
    output.append(f"Singular values of E: {S}")

    if np.isclose(S[2], 0, atol=1e-5):
        output.append("Essential matrix has the correct rank (2).")
    else:
        output.append("Essential matrix does NOT have the correct rank!")

    det_E = np.linalg.det(E)
    output.append(f"Det(E) = {det_E}")
    if np.isclose(det_E, 0, atol=1e-5):
        output.append("Essential matrix satisfies det(E) = 0 condition.")
    else:
        output.append("Essential matrix does NOT satisfy det(E) = 0 condition!")

    det_R1 = np.linalg.det(R1)
    det_R2 = np.linalg.det(R2)
    output.append(f"Det(R1) = {det_R1}, Det(R2) = {det_R2}")

    identity_matrix = np.eye(3)
    R1_deviation = np.linalg.norm(R1 - identity_matrix)
    R2_deviation = np.linalg.norm(R2 - identity_matrix)
    output.append(f"Deviation of R1 from Identity: {R1_deviation}")
    output.append(f"Deviation of R2 from Identity: {R2_deviation}")

    errors = []
    for i in range(len(pts1)):
        x1 = np.append(pts1[i], 1)
        x2 = np.append(pts2[i], 1)
        error = np.abs(x2.T @ F @ x1)
        errors.append(error)
    mean_error = np.mean(errors)
    output.append(f"Mean Epipolar Constraint Error: {mean_error}")

    return "\n".join(map(str, output))

def show_triangulation(E, R1, R2, t, pts1, pts2, K):
    if t.shape == (3,):
        t = t.reshape(3, 1)

    identity_matrix = np.eye(3)
    R = R1 if np.linalg.norm(R1 - identity_matrix) < np.linalg.norm(R2 - identity_matrix) else R2

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    if pts1.ndim == 3:
        pts1 = pts1.reshape(-1, 2)
    if pts2.ndim == 3:
        pts2 = pts2.reshape(-1, 2)

    pts1_hom = np.vstack((pts1.T, np.ones((1, pts1.shape[0]))))
    pts2_hom = np.vstack((pts2.T, np.ones((1, pts2.shape[0]))))

    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_hom[:2], pts2_hom[:2])
    points_3d = points_4d_hom[:3] / points_4d_hom[3]

    valid_indices = points_3d[2] > 0
    points_3d_filtered = points_3d[:, valid_indices]

    result = "Reconstructed 3D Points (filtered):\n"
    result += str(points_3d_filtered.T[:10])  # Show first 10 for brevity
    return result,points_3d
