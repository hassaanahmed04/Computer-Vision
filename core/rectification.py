import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import numpy as np
from customtkinter import (
    CTk,
    CTkLabel,
    CTkButton,
    CTkFrame
)
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from pypylon import pylon
    PYLON_AVAILABLE = True
except ImportError:
    PYLON_AVAILABLE = False


def load_json():
    try:
        with open(f"{parent_dir}/data/current.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}


def save_json(data):
    with open(f"{parent_dir}/data/current.json", "w") as file:
        json.dump(data, file, indent=4)


def show_image_on_canvas(file_path, container):
    for widget in container.winfo_children():
        widget.destroy()

    img = mpimg.imread(file_path)
    fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=100)
    ax.imshow(img)
    ax.axis('off')

    canvas = FigureCanvasTkAgg(fig, master=container)
    canvas.draw()
    canvas.get_tk_widget().pack()


def select_image(side, label, canvas_container):
    file_path = filedialog.askopenfilename(
        title=f"Select {side.capitalize()} Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if file_path:
        label.configure(text=os.path.basename(file_path))
        show_image_on_canvas(file_path, canvas_container)
        data = load_json()
        data[f"{side}_image"] = file_path
        save_json(data)


def capture_from_basler():
    try:
        if not PYLON_AVAILABLE:
            raise ImportError("Pylon library not installed")

        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()

        if len(devices) < 2:
            raise Exception(f"Expected 2 cameras, but found {len(devices)}.")

        cameras = pylon.InstantCameraArray(2)
        for i, cam in enumerate(cameras):
            cam.Attach(tl_factory.CreateDevice(devices[i]))

        cameras.Open()

        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        for i in range(2):
            window_name = f'Camera {i + 1}'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 720, 560)

        print("Press 'q' to capture images...")

        while True:
            frames = []
            for i, cam in enumerate(cameras):
                cam.StartGrabbingMax(1)
                with cam.RetrieveResult(2000) as result:
                    if result.GrabSucceeded():
                        image = converter.Convert(result)
                        img = image.GetArray()
                        frames.append(img)
                        cv2.imshow(f'Camera {i + 1}', img)
                    else:
                        print(f"Failed to grab image from Camera {i + 1}")
                        frames.append(None)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') and all(f is not None for f in frames):
                file_paths = []
                for i, img in enumerate(frames):
                    file_path = f"basler_cam_{i + 1}.jpg"
                    cv2.imwrite(file_path, img)
                    file_paths.append((img, file_path))
                break
            elif key==27:
                break
        cameras.StopGrabbing()
        cameras.Close()
        cv2.destroyAllWindows()
        return file_paths if file_paths else None

    except Exception as e:
        print(f"Error capturing from Basler cameras: {e}")
        return []

def open_rectification_dialog():
    data = load_json()

    if "camera_matrix" not in data or "dist_coeffs" not in data:
        warning = CTk()
        warning.geometry("400x200")
        warning.title("Warning")
        CTkLabel(warning, text="Please do the calibration first.", font=("Arial", 14)).pack(pady=20)
        CTkButton(warning, text="OK", command=warning.destroy).pack(pady=10)
        warning.mainloop()
        return

    dialog = CTk()
    dialog.geometry("750x700")
    dialog.title("Rectification")

    left_label = CTkLabel(dialog, text="No Left Image Selected", font=("Arial", 12))
    left_label.pack(pady=5)
    left_canvas_container = CTkFrame(dialog)
    left_canvas_container.pack(pady=5)
    CTkButton(dialog, text="Select Left Image",
              command=lambda: select_image("left", left_label, left_canvas_container)).pack(pady=5)

    right_label = CTkLabel(dialog, text="No Right Image Selected", font=("Arial", 12))
    right_label.pack(pady=5)
    right_canvas_container = CTkFrame(dialog)
    right_canvas_container.pack(pady=5)
    CTkButton(dialog, text="Select Right Image",
              command=lambda: select_image("right", right_label, right_canvas_container)).pack(pady=5)

    def capture_and_show_images():
        results = capture_from_basler()
        print(results)
        if len(results) == 2:
            (img_left, path_left), (img_right, path_right) = results
            data = load_json()
            data["left_image"] = path_left
            data["right_image"] = path_right
            save_json(data)
            left_label.configure(text=os.path.basename(path_left))
            right_label.configure(text=os.path.basename(path_right))
            show_image_on_canvas(path_left, left_canvas_container)
            show_image_on_canvas(path_right, right_canvas_container)

    CTkButton(dialog, text="Capture from Basler Cameras", command=capture_and_show_images).pack(pady=10)

    CTkButton(dialog, text="Done", command=dialog.destroy).pack(pady=20)

    return dialog  


def rectify_images(img1, img2, T,camera_mode):
    R = T[:3, :3]
    t = T[:3, 3]
    with open(f"{parent_dir}/data/current.json", "r") as f:
            data = json.load(f)
    if camera_mode == "single":
        K1 = np.array(data.get("camera_matrix"))
        K2 = np.array(data.get("camera_matrix"))
        dist_coeffs1=np.array(data.get("dist_coeffs"))
        dist_coeffs2=np.array(data.get("dist_coeffs"))
    else:

        K1 = np.array([[3.56487210e+03, 0.00000000e+00, 1.25615010e+03],
    [0.00000000e+00, 3.54243868e+03, 9.94841216e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        K2 = np.array([[3.95627076e+03, 0.00000000e+00, 1.41195677e+03],
    [0.00000000e+00, 3.93491245e+03, 1.00009426e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        dist_coeffs1 = np.array([-2.23291029e-01,  7.09951120e-02, -2.82582523e-03,  5.79697269e-04,
    -6.58818514e-01])
        dist_coeffs2 = np.array([-2.29653887e-01, -3.31835273e-02, -5.15949943e-04, -1.19966003e-03,
    6.63499465e-01])

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, dist_coeffs1,
        K2, dist_coeffs2,
        img1.shape[:2][::-1], R, t,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=1
    )

    map1_x, map1_y = cv2.initUndistortRectifyMap(K1, dist_coeffs1, R1, P1, img1.shape[:2][::-1], cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(K1, dist_coeffs2, R2, P2, img2.shape[:2][::-1], cv2.CV_32FC1)

    img1_rectified = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(img2, map2_x, map2_y, cv2.INTER_LINEAR)

    cv2.imwrite(f"{parent_dir}/data/pictures/rectified_img1.jpg", img1_rectified)
    cv2.imwrite(f"{parent_dir}/data/pictures/rectified_img2.jpg", img2_rectified)
    cv2.imwrite(f"{parent_dir}/data/pictures/rectified_img1.png", img1_rectified)
    cv2.imwrite(f"{parent_dir}/data/pictures/rectified_img2.png", img2_rectified)
    return img1_rectified, img2_rectified, Q
