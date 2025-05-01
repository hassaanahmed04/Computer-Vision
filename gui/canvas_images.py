import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter
import cv2

def show_images_grid(images_with_titles, columns=2):
    """
    Display images in a grid using CTkToplevel and matplotlib.
    
    Args:
        images_with_titles (list of tuples): [(image, title), ...]
        columns (int): Number of columns in the grid.
    """
    num_images = len(images_with_titles)
    rows = math.ceil(num_images / columns)

    # Create a new CTkToplevel window
    win = customtkinter.CTkToplevel()
    win.title("Image Grid View")
    win.geometry("1000x800")

    # Create figure and axes grid
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
    
    # Flatten axes array for uniform access
    axes = axes.flatten() if num_images > 1 else [axes]

    # Loop through images and place them
    for idx, (img, title) in enumerate(images_with_titles):
        axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(title)
        axes[idx].axis("off")

    # Hide any extra axes
    for i in range(len(images_with_titles), len(axes)):
        axes[i].axis("off")

    # Add canvas to CTk window
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
