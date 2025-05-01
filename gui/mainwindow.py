
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import time
import threading
import tkinter
import tkinter.messagebox
import customtkinter
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from core.calibration import *
from core.disparity import *
from core.rectification import *
from .canvas_images import *
from .choice_window import *
from PIL import Image
import webbrowser
import random
customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("3d Reconstructor")
        self.geometry(f"{1100}x{580}")

        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        image_path = f"{parent_dir}/data/pictures/logo.png" 
        logo_image = customtkinter.CTkImage(
            light_image=Image.open(image_path),
            dark_image=Image.open(image_path),
            size=(80, 80) 
        )
        self.logo_img_label = customtkinter.CTkLabel(self.sidebar_frame, image=logo_image, text="")
        self.logo_img_label.grid(row=0, column=0, pady=(20, 5), padx=20)

        # Add the "ELITE-SNAKES" label below the image
        self.logo_label = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="ELITE-SNAKES",
            font=customtkinter.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=1, column=0, padx=20, pady=(0, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Automate",command=self.automate)
        self.sidebar_button_1.grid(row=2, column=0, padx=20, pady=10)

       
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["50","80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        
        self.textbox = customtkinter.CTkTextbox(self, width=250)
        self.textbox.grid(row=0, column=1, rowspan=2, padx=(20, 0), pady=(20, 0), sticky="nsew")

        # create tabview
        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.tabview.add("Options")

        # Configure the single column in the "Options" tab to expand
        self.tabview.tab("Options").grid_columnconfigure(0, weight=1)

        # Label for the dropdown
        self.filter_var = ctk.StringVar(value="No Filter")

        self.filter_label = customtkinter.CTkLabel(self.tabview.tab("Options"), text="Select Filter:")
        self.filter_label.grid(row=0, column=0, padx=20, pady=(10, 0), sticky="w")

        # OptionMenu itself
        self.optionmenu_1 = customtkinter.CTkOptionMenu(
            self.tabview.tab("Options"),
            dynamic_resizing=False,
            variable=self.filter_var,
            values=["No Filter", "WLS", "All Filters"]
        )
        self.optionmenu_1.grid(row=1, column=0, padx=20, pady=(5, 10), sticky="ew")


        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("Options"), text="Enter baseLine",
                                                        command=self.open_input_dialog_event)
        self.string_input_button.grid(row=2, column=0, padx=20, pady=(10, 10), sticky="ew")

        self.radiobutton_frame = customtkinter.CTkFrame(self)
        self.radiobutton_frame.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.label_team_title = customtkinter.CTkLabel(master=self.radiobutton_frame, text="Our Team", font=("Arial", 16, "bold"))
        self.label_team_title.grid(row=0, column=0, columnspan=2, pady=(10, 20))

        # Team member info
        team_members = [
            {"name": "Hassaan Ahmed", "linkedin": "https://www.linkedin.com/in/hassaanahmed04/"},
            {"name": "Ureed Hussain", "linkedin": "https://www.linkedin.com/in/muhammad-ureed-hussain-86947219b/"},
            {"name": "Danial Ahmed", "linkedin": "https://www.linkedin.com/in/malik-danial-ahmed-07a744217/"},
        ]
        random.shuffle(team_members)

        # Load LinkedIn icon
        linkedin_icon = customtkinter.CTkImage(light_image=Image.open(f"{parent_dir}/data/pictures/linkedin_logo.png"), size=(20, 20))

        # Add each member to UI
        for i, member in enumerate(team_members, start=1):
            label_name = customtkinter.CTkLabel(master=self.radiobutton_frame, text=member["name"])
            label_name.grid(row=i, column=0, padx=(10, 5), pady=5, sticky="w")

            button_linkedin = customtkinter.CTkButton(
                master=self.radiobutton_frame,
                image=linkedin_icon,
                text="",
                width=30,
                fg_color="transparent",
                hover_color="#e5e5e5",
                command=lambda url=member["linkedin"]: webbrowser.open(url)
            )
            button_linkedin.grid(row=i, column=1, padx=(0, 10), pady=5, sticky="e")



        self.scrollable_frame = DisparityFrame(self)
        self.scrollable_frame.grid(row=1, column=2,columnspan=2, padx=(20, 0), pady=(20, 0), sticky="nsew")

        
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")
        self.optionmenu_1.set("CTkOptionmenu")
        # self.combobox_1.set("CTkComboBox")

        self.textbox.insert("0.0", "Logs\n\n" )


    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Type in a number:", title="Baseline between cameras")
        baseline_value = dialog.get_input()
        
        if baseline_value:  # Only proceed if user enters something
            try:
                baseline_float = float(baseline_value)  # Ensure it's a number
                # Load current.json
                current_data = {}
                if os.path.exists(f"{parent_dir}/data/current.json"):
                    with open("current.json", "r") as f:
                        current_data = json.load(f)

                # Update and save baseline
                current_data["baseline"] = baseline_float
                with open(f"{parent_dir}/data/current.json", "w") as f:
                    json.dump(current_data, f, indent=4)

                print(f"Baseline saved: {baseline_float}")
                self.textbox.insert("end", f"Baseline saved: {baseline_float}\n")

            except ValueError:
                print("Invalid input. Please enter a valid number.")
                self.textbox.insert("end", "Invalid input. Please enter a valid number.\n")
       
    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)
    


    def automate(self):
        try:
            camera_mode = self.camera_mode_dialog()

            folder_path = filedialog.askdirectory(title="Select Folder for Calibration")
            if not folder_path:
                return

            result = calibrate_camera(folder_path)
            if result is None:
                self.textbox.insert("end", "Calibration failed. No valid images found.\n")
                return

            camera_matrix, dist_coeffs, rvecs, tvecs, reprojection_error = result

            formatted_output = "=== Calibration Results ===\n\n"
            formatted_output += "Camera Matrix (K):\n" + format_matrix(camera_matrix) + "\n\n"
            formatted_output += "Distortion Coefficients:\n" + format_distortion(dist_coeffs) + "\n\n"
            formatted_output += f"Reprojection Error: {reprojection_error:.6f}\n"
            self.textbox.delete("0.0", "end")
            self.textbox.insert("end", formatted_output)

            self.textbox.insert("end", "\nOpening Rectification Dialog...\n")
            
            # Open and WAIT for dialog to close
            rect_dialog = open_rectification_dialog()
            if rect_dialog:
                rect_dialog.wait_window()  #  Waits here until closed

            self.textbox.insert("end", "\n Rectification dialog closed. Proceeding...\n")

            with open(f"{parent_dir}/data/current.json", "r") as f:
                data = json.load(f)

            left_path = data.get("left_image")
            right_path = data.get("right_image")

            if camera_mode == "single":
                left_path, right_path = right_path, left_path

            if not left_path or not right_path or not os.path.exists(left_path) or not os.path.exists(right_path):
                self.textbox.insert("end", " Left or Right image path not found in current.json.\n")
                return

            self.textbox.insert("end", f"Left: {left_path}\nRight: {right_path}\n")
            self.textbox.insert("end", "\n Doing Undistortion...\n")

            K = np.array(camera_matrix)
            dist = np.array(dist_coeffs)

            left_img = cv2.imread(left_path)
            right_img = cv2.imread(right_path)

            undistorted_left = cv2.undistort(left_img, K, dist)
            undistorted_right = cv2.undistort(right_img, K, dist)

            self.textbox.insert("end", " Undistortion Complete.\n\n")
            
            self.textbox.insert("end", " Starting Rectification.\n\n")

            T = np.eye(4)
            T[:3, 3] = np.array([0.11, 0, 0])
            rectified_img1, rectified_img2, Q = rectify_images(left_img, right_img, T,camera_mode)
            

            self.textbox.insert("end", f"Value of Q: {Q}\n\n")

            self.textbox.insert("end", " Rectification Complete.\n\n")


            # Show them in a 2-column grid
            images_with_titles = [
                # (undistorted_left, "Undistorted Left"),
                # (undistorted_right, "Undistorted Right"),
                (rectified_img1, "Rectified Left"),
                (rectified_img2, "Rectified Right")
            ]
            show_images_grid(images_with_titles, columns=2)
            ChoiceDialog(rectified_img1, rectified_img2,Q)

            # calculate_disparity(data.get("min_disparity"),data.get("num_disparities"),data.get("block_size"),data.get("uniqueness_ratio"),data.get("speckle_window_size"),data.get("speckle_range"),data.get("disp12_max_diff"),data.get("pre_filter_cap"),left_img,right_img)
     
        except Exception as e:
            self.textbox.insert("end", f" Error: {str(e)}\n")


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, filepath):
        self.filepath = filepath

    def on_modified(self, event):
        if event.src_path == self.filepath:
            print(f"File {event.src_path} modified. Restarting application...")
            os.execl(sys.executable, sys.executable, *sys.argv)


def start_watcher(filepath):
    event_handler = FileChangeHandler(filepath)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(filepath), recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    file_path = os.path.abspath(__file__)

    # Run the file watcher in a separate thread
    watcher_thread = threading.Thread(target=start_watcher, args=(file_path,), daemon=True)
    watcher_thread.start()

    app = App()
    app.mainloop()

# def main():
#     file_path = os.path.abspath(__file__)

#     # Run the file watcher in a separate thread
#     watcher_thread = threading.Thread(target=start_watcher, args=(file_path,), daemon=True)
#     watcher_thread.start()

#     app = App()
#     app.mainloop()

# if __name__ == "__main__":
#     main()
