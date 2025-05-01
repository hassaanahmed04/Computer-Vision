import customtkinter
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os 
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

    
class DisparityFrame(customtkinter.CTkScrollableFrame):
    def __init__(self, parent):
        super().__init__(parent, label_text="Disparity Parameters")
        self.grid_columnconfigure(0, weight=1)
        
        self.parameters = {
            "min_disparity": 0,
            "num_disparities": 128,
            "block_size": 5,
            "uniqueness_ratio": 10,
            "speckle_window_size": 100,
            "speckle_range": 32,
            "disp12_max_diff": 1,
            "pre_filter_cap": 63
        }
        
        self.load_parameters()
        self.entries = {}
        row = 0
        for param, value in self.parameters.items():
            label = customtkinter.CTkLabel(self, text=param.replace("_", " ").title())
            label.grid(row=row, column=0, padx=10, pady=5, sticky="w")
            
            entry = customtkinter.CTkEntry(self)
            entry.insert(0, str(value))
            entry.grid(row=row, column=1, padx=10, pady=5)
            
            self.entries[param] = entry
            row += 1

        self.save_button = customtkinter.CTkButton(self, text="Save", command=self.save_parameters)
        self.save_button.grid(row=row, column=0, columnspan=2, pady=10)

    def save_parameters(self):
        for param in self.entries:
            try:
                self.parameters[param] = int(self.entries[param].get())
            except ValueError:
                self.parameters[param] = 0
        
        with open(f"{parent_dir}/data/disparity_parameters.json", "w") as file:
            json.dump(self.parameters, file, indent=4)
        # with open("current.json", "w") as file:
        #     json.dump(self.parameters, file, indent=4)
        print("Parameters saved successfully!")

    def load_parameters(self):    
        try:
            with open(f"{parent_dir}/data/disparity_parameters.json", "r") as file:
                self.parameters.update(json.load(file))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
def calculate_disparity(min_disparity,num_disparities,block_size,uniqueness_ratio,speckle_window_size,speckle_range,disp12_max_diff,pre_filter_cap,filter_name,Q):
    print(filter_name)
    left_img = cv2.imread(f"{parent_dir}/data/pictures/rectified_img1.png", cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    right_img = cv2.imread(f"{parent_dir}/data/pictures/rectified_img2.png", cv2.IMREAD_GRAYSCALE)

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        disp12MaxDiff=disp12_max_diff,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        preFilterCap=pre_filter_cap,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        
    )
    if filter_name=="wls":
        stereo_right = cv2.ximgproc.createRightMatcher(stereo)  

        disparity_left = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
        disparity_right = stereo_right.compute(right_img, left_img).astype(np.float32) / 16.0
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
        wls_filter.setLambda(8000)  
        wls_filter.setSigmaColor(1.5) 

        disparity = wls_filter.filter(disparity_left, left_img, disparity_map_right=disparity_right)

        disparity_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disparity_norm = np.uint8(disparity_norm)
        plt.close('all')
        plt.figure(figsize=(10, 5))
        plt.imshow(disparity_norm, cmap="jet")
        plt.colorbar()
        plt.title(f"Disparity Map ({filter_name} Filter)")
        plt.show()
        
        points_3D = cv2.reprojectImageTo3D(disparity, Q)

        mask = (disparity > 10) & np.isfinite(points_3D[:, :, 0])

        points = points_3D[mask]
        colors = left_img[mask] 

        colors = colors.astype(np.float32) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.colors = o3d.utility.Vector3dVector(colors)  # Assign colors to point cloud


        # o3d.io.write_point_cloud("output.ply", pcd)
        o3d.visualization.draw_geometries([pcd])

        print("3D point cloud saved as output.ply")

    elif filter_name=="all":
        stereo_right = cv2.ximgproc.createRightMatcher(stereo) 

        disparity_left = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
        disparity_right = stereo_right.compute(right_img, left_img).astype(np.float32) / 16.0

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
        wls_filter.setLambda(8000)  
        wls_filter.setSigmaColor(1.5)  
        filtered_disparity_wls = wls_filter.filter(disparity_left, left_img, disparity_map_right=disparity_right)

        filtered_disparity_bilateral = cv2.bilateralFilter(disparity_left, d=9, sigmaColor=75, sigmaSpace=75)

        filtered_disparity_guided = cv2.ximgproc.guidedFilter(guide=left_img, src=disparity_left, radius=9, eps=0.01)


        filtered_disparity_median = cv2.medianBlur(disparity_left, ksize=5)

        fgs_filter = cv2.ximgproc.createFastGlobalSmootherFilter(guide=left_img, lambda_=1000, sigma_color=0.1)
        filtered_disparity_fgs = fgs_filter.filter(disparity_left)

        def normalize_disparity(disparity):
            return cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        filtered_disparity_wls_norm = normalize_disparity(filtered_disparity_wls)
        filtered_disparity_bilateral_norm = normalize_disparity(filtered_disparity_bilateral)
        filtered_disparity_guided_norm = normalize_disparity(filtered_disparity_guided)
        filtered_disparity_median_norm = normalize_disparity(filtered_disparity_median)
        filtered_disparity_fgs_norm = normalize_disparity(filtered_disparity_fgs)

        plt.close('all')
        plt.figure(figsize=(20, 10))

        plt.subplot(2, 3, 1)
        plt.imshow(filtered_disparity_wls_norm, cmap="jet")
        plt.colorbar()
        plt.title("WLS Filter")

        plt.subplot(2, 3, 2)
        plt.imshow(filtered_disparity_bilateral_norm, cmap="jet")
        plt.colorbar()
        plt.title("Bilateral Filter")

        plt.subplot(2, 3, 3)
        plt.imshow(filtered_disparity_guided_norm, cmap="jet")
        plt.colorbar()
        plt.title("Guided Filter")

        plt.subplot(2, 3, 5)
        plt.imshow(filtered_disparity_median_norm, cmap="jet")
        plt.colorbar()
        plt.title("Median Filter")

        plt.subplot(2, 3, 6)
        plt.imshow(filtered_disparity_fgs_norm, cmap="jet")
        plt.colorbar()
        plt.title("Fast Global Smoother")

        plt.tight_layout()
        plt.show()
    else:
            disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

            disparity_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            disparity_norm = np.uint8(disparity_norm)
    
            plt.close('all')
            plt.figure(figsize=(10, 5))
            plt.imshow(disparity_norm, cmap="jet")
            plt.colorbar()
            plt.title(f"Disparity Map ({filter_name} Filter)")
            plt.show()
            points_3D = cv2.reprojectImageTo3D(disparity, Q)

            mask = (disparity > 10) & np.isfinite(points_3D[:, :, 0])

            points = points_3D[mask]
            colors = left_img[mask] 

            colors = colors.astype(np.float32) / 255.0

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            # pcd.colors = o3d.utility.Vector3dVector(colors)  # Assign colors to point cloud


            # o3d.io.write_point_cloud("output.ply", pcd)
            o3d.visualization.draw_geometries([pcd])


