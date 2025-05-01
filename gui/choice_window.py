import customtkinter as ctk
import json
import cv2
import numpy as np
from core.disparity import *
from core.sparse import *
import open3d as o3d
import os 
from core.daaata import *
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
class ChoiceDialog:
    def __init__(self, rectified_img1=None, rectified_img2=None,Q=None):
        self.rectified_img1 = rectified_img1
        self.rectified_img2 = rectified_img2
        self.Q=Q
        self.root = ctk.CTk()
        self.root.title("3D Reconstruction Choice")
        self.root.geometry("900x800")

        self.filter_var = None
        self.matcher_var = None
        self.descriptor_var = None

        self.create_widgets()
        self.root.mainloop()

    def create_widgets(self):
        label = ctk.CTkLabel(self.root, text="What do you want to do?", font=("Arial", 16))
        label.pack(pady=20)

        sparse_button = ctk.CTkButton(self.root, text="Sparse 3D Reconstruction", command=self.sparse_reconstruction)
        sparse_button.pack(pady=10)

        dense_button = ctk.CTkButton(self.root, text="Dense 3D Reconstruction", command=self.dense_reconstruction)
        dense_button.pack(pady=10)

        self.options_frame = ctk.CTkFrame(self.root)
        self.options_frame.pack(fill="both", expand=True, pady=10)

    def clear_options(self):
        for widget in self.options_frame.winfo_children():
            widget.destroy()

    def sparse_reconstruction(self):
        self.clear_options()
        print("Sparse 3D Reconstruction Selected")

        self.matcher_var = ctk.StringVar(value="bf_akaze")
        matcher_label = ctk.CTkLabel(self.options_frame, text="Select Feature Matcher:")
        matcher_label.pack(pady=5)

        matcher_options = ["bf_akaze", "bf_knn", "flann", "bf_orb"]
        matcher_menu = ctk.CTkOptionMenu(self.options_frame, variable=self.matcher_var, values=matcher_options)
        matcher_menu.pack(pady=5)

        self.descriptor_var = ctk.StringVar(value="akaze")
        descriptor_label = ctk.CTkLabel(self.options_frame, text="Select Descriptor:")
        descriptor_label.pack(pady=5)

        descriptor_options = ["akaze", "sift", "orb",]
        descriptor_menu = ctk.CTkOptionMenu(self.options_frame, variable=self.descriptor_var, values=descriptor_options)
        descriptor_menu.pack(pady=5)

        # Threshold Entry Field
        threshold_label = ctk.CTkLabel(self.options_frame, text="Set Match Distance Threshold:")
        threshold_label.pack(pady=5)

        self.threshold_entry = ctk.CTkEntry(self.options_frame, placeholder_text="e.g. 100")
        self.threshold_entry.pack(pady=5)

        match_button = ctk.CTkButton(self.options_frame, text="Run Sparse Reconstruction", command=self.run_sparse_reconstruction)
        match_button.pack(pady=10)

        # Persistent Buttons Frame
        self.button_frame = ctk.CTkFrame(self.options_frame)
        self.button_frame.pack(pady=10)

        self.feature_button = ctk.CTkButton(self.button_frame, text="Check Feature Matching", command=self.show_feature_matching)
        self.feature_button.pack(side="left", padx=10)

        self.param_button = ctk.CTkButton(self.button_frame, text="Parameters", command=self.show_parameters)
        self.param_button.pack(side="left", padx=10)

        self.triangulate_button = ctk.CTkButton(self.button_frame, text="Triangulation", command=self.show_triangulation)
        self.triangulate_button.pack(side="left", padx=10)
        self.epipole_lines_button = ctk.CTkButton(self.button_frame, text="Epipole Lines", command=self.show_eipole_lines)
        self.epipole_lines_button.pack(side="left", padx=10)

        self.open_3d_button = ctk.CTkButton(self.options_frame, text="Open in 3D", command=self.open_in_3d)
        self.open_3d_button.pack(pady=10)

        # Text display area (initially empty)
        self.text_output = ctk.CTkTextbox(self.options_frame, width=580, height=400)
        self.text_output.pack(pady=10)

    def run_sparse_reconstruction(self):
        try:
            threshold = float(self.threshold_entry.get())
        except ValueError:
            print("Invalid threshold input. Using default 100.")
            threshold = 100

        with open(f"{parent_dir}/data/current.json", "r") as f:
            data = json.load(f)
        # K = np.array(data.get("camera_matrix"), dtype=np.float64)
        K = np.array([[3.56487210e+03, 0.00000000e+00, 1.25615010e+03],
       [0.00000000e+00, 3.54243868e+03, 9.94841216e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],dtype=float).reshape(3, 3)
        rectified_img1 = self.rectified_img1
        rectified_img2 = self.rectified_img2
        matcher = self.matcher_var.get()
        descriptor = self.descriptor_var.get()

        print(f"Selected Feature Matcher: {matcher}")
        print(f"Selected Descriptor: {descriptor}")
        print(f"Threshold: {threshold}")

        if descriptor == "akaze":
            descriptor_method = cv2.AKAZE_create()
        elif descriptor == "sift":
            descriptor_method = cv2.SIFT_create()
        elif descriptor == "orb":
            descriptor_method = cv2.ORB_create()
        else:
            print("Invalid descriptor selected!")
            return

        if matcher == "bf_akaze":
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif matcher == "bf_orb":
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif matcher == "bf_knn":
            bf = cv2.BFMatcher()
            
        elif matcher == "bf":
            norm_type = cv2.NORM_L2 if descriptor in ["sift", "akaze"] else cv2.NORM_HAMMING
            bf = cv2.BFMatcher(norm_type, crossCheck=True)
        elif matcher == "flann":
            if descriptor in ["sift", "akaze"]:
                index_params = dict(algorithm=1, trees=10)
                search_params = dict(checks=50)
                bf = cv2.FlannBasedMatcher(index_params, search_params)
            else:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            print("Invalid matcher selected!")
            return

        if matcher=="bf_knn" and descriptor_method=="sift":
            with open(f"{parent_dir}/data/current.json", "r") as f:
              data = json.load(f)
        
            img1 = data.get("left_image")
            img2 = data.get("right_image")
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            kp1, des1 = cv2.SIFT_create().detectAndCompute(gray_img1, None)
            kp2, des2 = cv2.SIFT_create().detectAndCompute(gray_img2, None)
            
            good_matches =match_features(des1,des2)            
            draw_matches(img1, kp1, img2, kp2, good_matches)
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            matched_img = cv2.drawMatches(rectified_img1, kp1, rectified_img2, kp2, good_matches, None,
                                        flags=2)
            F, inliers1, inliers2 = estimate_fundamental_matrix(kp1, kp2, good_matches)            
            E = compute_essential_matrix(F, K, K)
            T1 = np.array([0, 0, 0],dtype=float).reshape(3, 1)  # Translation vector
            R = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]],dtype=float).reshape(3, 3)
            Rt = np.hstack((R, T1))
            # Projection matrix
            P1 = K @ Rt
            # Recover Pose
            R, t = recover_pose(E, inliers1, inliers2, K)

            Rt2 = np.hstack((R, t))
            P2 = K @ Rt2

            points_3d = triangulate_points(P1, P2, inliers1, inliers2, K)
            self.E = E
            self.F = F
            self.t = t
            self.pts1 = pts1.reshape(-1, 2)
            self.pts2 = pts2.reshape(-1, 2)
            print(f"Triangulated {len(points_3d)} 3D points from first two views.")
            pts1=pts1.reshape(-1, 2)
            colors = []
            for pt in pts1:
                x, y = int(pt[0]), int(pt[1])  # Convert to integer coordinates
                if 0 <= x < img1_rgb.shape[1] and 0 <= y < img1_rgb.shape[0]:  # Check if within image bounds
                    color = img1_rgb[y, x] / 255.0  # Normalize to [0, 1] for Open3D
                    colors.append(color)
                else:
                    colors.append([0, 0, 0])  # Default to black if out of bounds
            colors = np.array(colors)
            visualize_3d_points(points_3d,colors)

        else:
            kp1, des1 = descriptor_method.detectAndCompute(rectified_img1, None)
            kp2, des2 = descriptor_method.detectAndCompute(rectified_img2, None)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance
             # Matches with distance < 50 are considered strong
            good_matches = [m for m in matches if m.distance < threshold]

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
            cv2.imwrite(f"{parent_dir}/data/strong_feature_matches_with_ransac.jpg", matched_img)

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

            self.E = E
            self.F = F
            self.R1 = R1
            self.R2 = R2
            self.t = t
            self.pts1 = pts1.reshape(-1, 2)
            self.pts2 = pts2.reshape(-1, 2)

            # Optional: show confirmation message
            self.text_output.delete("0.0", "end")
            self.text_output.insert("0.0", f"Sparse reconstruction completed.\nMatches found: {len(matches)}\nGood matches: {len(good_matches)}")
            self.show_parameters()

    def show_feature_matching(self, max_width=1000, max_height=800):
        matched_img = cv2.imread(f"{parent_dir}/data/strong_feature_matches_with_ransac.jpg")

        if matched_img is None:
            print("Image not found.")
            return

        # Resize if needed
        height, width = matched_img.shape[:2]
        if width > max_width or height > max_height:
            scale_w = max_width / width
            scale_h = max_height / height
            scale = min(scale_w, scale_h)
            new_size = (int(width * scale), int(height * scale))
            matched_img = cv2.resize(matched_img, new_size, interpolation=cv2.INTER_AREA)

        # Convert BGR to RGB for matplotlib
        matched_img_rgb = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)

        # Force closing all other figures (optional but clean)
        plt.close('all')

        # Create and show a specific figure by number (reuse if already exists)
        fig = plt.figure(num=999, figsize=(10, 8))  # Using a unique number unlikely to collide
        fig.clf()  # Clear previous contents of this figure (if reusing)
        plt.imshow(matched_img_rgb)
        plt.title("Feature Matching")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def show_parameters(self):
        result = show_parameters(self.E, self.F, self.R1, self.R2, self.t, self.pts1, self.pts2)
        # self.display_text(result)
        # self.text_output.delete("0.0", "end")
        self.text_output.insert("end",f"\n{result}")

    def show_triangulation(self):
        # with open("current.json", "r") as f:
        #     data = json.load(f)
        # K = np.array(data.get("camera_matrix"), dtype=np.float64)
        K = np.array([[3564.65081, 0, 1256.39072],
                   [0, 3542.14646, 997.879134],
                   [0, 0, 1]])

        result, self.points_3d = show_triangulation(self.E, self.R1, self.R2, self.t, self.pts1, self.pts2, K)
        self.display_text(result)


    def open_in_3d(self):
        with open(f"{parent_dir}/data/current.json", "r") as f:
            data = json.load(f)
        
        # Get the paths from the JSON file
        left_img_path = data.get("left_image")
        right_img_path = data.get("right_image")

        # Check if the image paths are valid
        if not os.path.exists(left_img_path):
            raise ValueError(f"Left image not found at {left_img_path}")
        if not os.path.exists(right_img_path):
            raise ValueError(f"Right image not found at {right_img_path}")
        
        # Load the images
        K = np.array([[3564.65081, 0, 1256.39072],
                   [0, 3542.14646, 997.879134],
                   [0, 0, 1]])
        result, self.points_3d = show_triangulation(self.E, self.R1, self.R2, self.t, self.pts1, self.pts2, K)

        
        img1 = cv2.imread(left_img_path)
        img2 = cv2.imread(right_img_path)

        
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Reshape pts1 and pts2 to (N, 2) format
        pts1 = self.pts1.reshape(-1, 2)
        pts2 = self.pts2.reshape(-1, 2)

        # Extract colors from the first image (img1)
        colors = []
        for pt in pts1:
            x, y = int(pt[0]), int(pt[1])  # Convert to integer coordinates
            if 0 <= x < img1_rgb.shape[1] and 0 <= y < img1_rgb.shape[0]:  # Check if within image bounds
                color = img1_rgb[y, x] / 255.0  # Normalize to [0, 1] for Open3D
                colors.append(color)
            else:
                colors.append([0, 0, 0])  # Default to black if out of bounds

        # Convert colors to a numpy array
        colors = np.array(colors)

        # Create Open3D point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(self.points_3d.T)  # Assign 3D points
        point_cloud.colors = o3d.utility.Vector3dVector(colors)  # Assign colors
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # P1 = K [I | 0]

        # Step 2: Reproject 3D points onto the first image
        points_4d_hom = np.vstack((self.points_3d, np.ones((1, self.points_3d.shape[1]))))  # Convert to homogeneous coordinates
        reprojected_pts = P1 @ points_4d_hom  # Project to 2D
        reprojected_pts = reprojected_pts[:2] / reprojected_pts[2]  # Normalize to get (u, v) coordinates


        # Step 3: Visualize the reprojected points on the original image
        img1_with_points = img1.copy()  # Make a copy of the original image

        # Draw the reprojected points on the image
        for pt in reprojected_pts.T:
            u, v = int(pt[0]), int(pt[1])  # Convert to integer pixel coordinates
            cv2.circle(img1_with_points, (u, v), radius=5, color=(0, 255, 0), thickness=-1)  # Green points
        plt.close('all')
        # Display the image with reprojected points
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(img1_with_points, cv2.COLOR_BGR2RGB))
        plt.title("Original Image with Reprojected 3D Points")
        plt.axis('off')
        plt.show()
        # Visualize the colored point cloud
        o3d.visualization.draw_geometries([point_cloud])
    def show_eipole_lines(self):
        if not hasattr(self, 'F') or self.F is None:
            print("Fundamental matrix not computed yet.")
            return

        img1 = self.rectified_img1.copy()
        img2 = self.rectified_img2.copy()
        pts1 = self.pts1
        pts2 = self.pts2
        F = self.F

        # Filter only inlier matches (assuming RANSAC mask was used earlier)
        if hasattr(self, 'mask') and self.mask is not None:
            inlier_mask = self.mask.ravel().astype(bool)
            pts1 = pts1[inlier_mask]
            pts2 = pts2[inlier_mask]

        # Compute epilines
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

        def draw_epilines(img, lines, pts):
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
            h, w = img_color.shape[:2]
            for r, pt in zip(lines, pts):
                color = tuple(np.random.randint(0, 255, 3).tolist())
                a, b, c = r
                if b != 0:
                    x0, y0 = 0, int(-c / b)
                    x1, y1 = w, int(-(a * w + c) / b)
                    if 0 <= y0 < h and 0 <= y1 < h:
                        cv2.line(img_color, (x0, y0), (x1, y1), color, 1)
                cv2.circle(img_color, tuple(np.int32(pt)), 5, color, -1)
            return img_color

        img1_with_lines = draw_epilines(img1, lines1, pts1)
        img2_with_lines = draw_epilines(img2, lines2, pts2)

        plt.close('all')
        plt.figure(figsize=(14, 6))

        plt.subplot(121)
        plt.imshow(cv2.cvtColor(img1_with_lines, cv2.COLOR_BGR2RGB))
        plt.title("Epipolar Lines in Image 1")
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(cv2.cvtColor(img2_with_lines, cv2.COLOR_BGR2RGB))
        plt.title("Epipolar Lines in Image 2")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def dense_reconstruction(self):
        self.clear_options()
        self.filter_var = ctk.StringVar(value="no")
        filter_label = ctk.CTkLabel(self.options_frame, text="Choose a filter:")
        filter_label.pack(pady=5)

        filter_options = ["No Filters", "wls", "all"]
        for option in filter_options:
            filter_button = ctk.CTkButton(self.options_frame, text=option.upper(),
                                          command=lambda opt=option: self.run_dense_reconstruction(opt))
            filter_button.pack(pady=2)

    def run_dense_reconstruction(self, filter_name):
        with open(f"{parent_dir}/data/disparity_parameters.json", "r") as f:
            data = json.load(f)

        self.disparity_params = {
            "min_disparity": data.get("min_disparity"),
            "num_disparities": data.get("num_disparities"),
            "block_size": data.get("block_size"),
            "uniqueness_ratio": data.get("uniqueness_ratio"),
            "speckle_window_size": data.get("speckle_window_size"),
            "speckle_range": data.get("speckle_range"),
            "disp12_max_diff": data.get("disp12_max_diff"),
            "pre_filter_cap": data.get("pre_filter_cap"),
            "filter_name": filter_name
            
        }
        show_button = ctk.CTkButton(
            self.options_frame,
            text="Show Disparity",
            command=self.show_disparity_result
        )
        show_button.pack(pady=10)

        print(f"Dense 3D Reconstruction Selected with {filter_name} filter")

    def show_disparity_result(self):
        with open(f"{parent_dir}/data/disparity_parameters.json", "r") as f:
            data = json.load(f)
        self.disparity_param = {
            "min_disparity": data.get("min_disparity"),
            "num_disparities": data.get("num_disparities"),
            "block_size": data.get("block_size"),
            "uniqueness_ratio": data.get("uniqueness_ratio"),
            "speckle_window_size": data.get("speckle_window_size"),
            "speckle_range": data.get("speckle_range"),
            "disp12_max_diff": data.get("disp12_max_diff"),
            "pre_filter_cap": data.get("pre_filter_cap")}
        
        g = self.disparity_param
        p=self.disparity_params
        calculate_disparity(
            g["min_disparity"], g["num_disparities"], g["block_size"],
            g["uniqueness_ratio"], g["speckle_window_size"], g["speckle_range"],
            g["disp12_max_diff"], g["pre_filter_cap"], p["filter_name"],self.Q
        )
        print("Disparity result displayed.")


