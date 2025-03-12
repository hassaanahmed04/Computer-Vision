import cv2
import numpy as np
import os
import glob
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from core.calibration import calibrate_camera


def browse_folder():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        entry_var.set(folder_selected)
    else:
        entry_var.set("./data")


def run_calibration():
    folder_path = entry_var.get()
    result = calibrate_camera(folder_path)
    if result:
        camera_matrix, dist_coeffs, rvecs, tvecs, reprojection_error = result
        result_text.set(f"Calibration Successful!\n\nCamera Matrix:\n{camera_matrix}\n\nDistortion Coefficients:\n{dist_coeffs}\n\nReprojection Error: {reprojection_error:.4f}")
        show_sample_images(folder_path, camera_matrix, dist_coeffs)
    else:
        result_text.set("Calibration Failed! Check console for errors.")


def show_sample_images(image_folder, camera_matrix, dist_coeffs):
    images = glob.glob(os.path.join(image_folder, '*.jpg')) + glob.glob(os.path.join(image_folder, '*.png'))

    if not images:
        return

    fig, axes = plt.subplots(1, min(len(images), 3), figsize=(15, 5))  # Show up to 3 images
    if len(images) == 1:
        axes = [axes]

    for i, fname in enumerate(images[:3]):
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        
        # Undistort the image
        undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)
        img_rgb = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)

        axes[i].imshow(img_rgb)
        axes[i].set_title(f"Image {i+1}")
        axes[i].axis("off")

    canvas = FigureCanvasTkAgg(fig, master=calibration_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()


# Main GUI Setup
root = tk.Tk()
root.title("Stereo Vision System")
root.geometry("800x600")

# Frame to hold everything
frame = ttk.Frame(root)
frame.pack(fill="both", expand=True)

# Creating a notebook for tabs (if you want to add more later)
notebook = ttk.Notebook(frame)
notebook.pack(expand=True, fill="both")

# Calibration Tab
calibration_frame = ttk.Frame(notebook)
notebook.add(calibration_frame, text="Calibration")

# Create a left frame for buttons and input fields
left_frame = ttk.Frame(calibration_frame)
left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill="both", expand=True)

# Entry for the folder path
entry_var = tk.StringVar(value="./data")
entry = tk.Entry(left_frame, textvariable=entry_var, width=40)
entry.grid(row=0, column=0, padx=5, pady=10, sticky="w")

# Browse Button
browse_button = tk.Button(left_frame, text="Browse", command=browse_folder)
browse_button.grid(row=0, column=1, padx=5, pady=10)

# Calibrate Button
calibrate_button = tk.Button(left_frame, text="Calibrate Camera", command=run_calibration)
calibrate_button.grid(row=1, column=0, columnspan=2, pady=10)

# Create a frame for displaying images
image_frame = ttk.Frame(calibration_frame)
image_frame.pack(side=tk.LEFT, padx=10, pady=10, fill="both", expand=True)

# Label to show results
result_text = tk.StringVar()
result_label = tk.Label(calibration_frame, textvariable=result_text, justify=tk.LEFT)
result_label.pack(side=tk.BOTTOM, padx=10, pady=10, fill="both", expand=True)


# Rectification Tab (Placeholder)
rectification_frame = ttk.Frame(notebook)
notebook.add(rectification_frame, text="Rectification")
tk.Label(rectification_frame, text="Rectification Module Coming Soon").pack(pady=50)

# Reconstruction Tab (Placeholder)
reconstruction_frame = ttk.Frame(notebook)
notebook.add(reconstruction_frame, text="Reconstruction")
tk.Label(reconstruction_frame, text="Reconstruction Module Coming Soon").pack(pady=50)

root.mainloop()
