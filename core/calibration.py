import os
import json
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter.filedialog
import customtkinter
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
def update_current_json(calib_data):
    
    current_json_path = f"{parent_dir}/data/current.json"
    
    if os.path.exists(current_json_path):
        with open(current_json_path, 'r') as f:
            current_data = json.load(f)
    else:
        current_data = {}
    
    current_data.update(calib_data)
    
    with open(current_json_path, 'w') as f:
        json.dump(current_data, f, indent=4)
    
    print("Updated current.json with new calibration data.")
def format_matrix(matrix):
    if matrix is None:
        return "Not Available"
    return "\n".join([" ".join([f"{num:.6e}" for num in row]) for row in matrix])

def format_distortion(dist):
    if dist is None or dist.size == 0:
        return "Not Available"
    return "[ " + "  ".join([f"{num:.6e}" for num in dist.ravel()]) + " ]"

def calibrate_camera(image_folder, chessboard_size=(7, 9), square_size=1, save_results=True):
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []  
    imgpoints = [] 

    images = glob.glob(os.path.join(image_folder, '*.jpg')) + glob.glob(os.path.join(image_folder, '*.png'))
    
    if not images:
        print("No images found in the folder.")
        return None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
    
    if not objpoints or not imgpoints:
        print("Chessboard pattern not found in images.")
        return None

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    reprojection_error = total_error / len(objpoints)

    calib_data = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist(),
        'reprojection_error': reprojection_error
    }

    if save_results:
        with open(os.path.join(image_folder, 'calibration_results.json'), 'w') as f:
            json.dump(calib_data, f, indent=4)
        
    update_current_json(calib_data)
    
    return camera_matrix, dist_coeffs, rvecs, tvecs, reprojection_error
def load_current_calibration():
    current_json_path = "current.json"
    if os.path.exists(current_json_path):
        with open(current_json_path, 'r') as f:
            return json.load(f)
    return {"camera_matrix": "Not Available", "dist_coeffs": "Not Available", "reprojection_error": "Not Available"}

def open_calibration_dialog():
    dialog = customtkinter.CTk()
    dialog.geometry("600x500")
    dialog.title("Select Calibration Option")
    
    current_params = load_current_calibration()

    customtkinter.CTkLabel(dialog, text="Current Calibration Parameters", font=("Arial", 16, "bold")).pack(pady=10)

    param_frame = customtkinter.CTkFrame(dialog)
    param_frame.pack(pady=10, padx=10, fill="both", expand=True)

    customtkinter.CTkLabel(param_frame, text="Camera Matrix (K):", font=("Arial", 14, "bold")).pack(anchor="w", padx=10)
    customtkinter.CTkLabel(param_frame, text=f"[[{format_matrix(current_params['camera_matrix'])}]]", font=("Arial", 12)).pack(anchor="w", padx=10, pady=5)

    customtkinter.CTkLabel(param_frame, text="Distortion Coefficients:", font=("Arial", 14, "bold")).pack(anchor="w", padx=10)
    customtkinter.CTkLabel(param_frame, text=format_distortion(current_params['dist_coeffs']), font=("Arial", 12)).pack(anchor="w", padx=10, pady=5)

    customtkinter.CTkLabel(param_frame, text=f"Reprojection Error: {current_params['reprojection_error']:.6f}", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=5)

    def upload_calibration_file():
        file_path = tkinter.filedialog.askopenfilename(title="Select Calibration JSON", filetypes=[("JSON files", "*.json")])
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r') as f:
                calib_data = json.load(f)
            update_current_json(calib_data)
            print(f"Calibration loaded from {file_path} and updated current.json")
        dialog.destroy()

    def select_folder_and_calibrate():
        folder = tkinter.filedialog.askdirectory(title="Select Folder with Calibration Images")
        if folder:
            json_file = os.path.join(folder, "calibration_results.json")
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    calib_data = json.load(f)
                update_current_json(calib_data)
                print(f"Loaded existing calibration from {json_file}")
            else:
                print("No existing calibration file found. Running calibration...")
                calibrate_camera(folder)
        dialog.destroy()

    customtkinter.CTkButton(dialog, text="Upload Calibration JSON", command=upload_calibration_file).pack(pady=10)
    customtkinter.CTkButton(dialog, text="Run New Calibration from Images", command=select_folder_and_calibrate).pack(pady=10)

    dialog.mainloop()
