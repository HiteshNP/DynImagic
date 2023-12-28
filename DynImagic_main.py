import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

def extract_color(image_path, lower_bound, upper_bound, output_name):
    """
    Extract the specified color from the image and save the largest ROI without the border.

    Args:
        image_path (str): Path to the input image.
        lower_bound (numpy.ndarray): Lower bound of the color range in HSV format.
        upper_bound (numpy.ndarray): Upper bound of the color range in HSV format.
        output_name (str): Name of the output file.

    Returns:
        None
    """
    original_img_path = image_path

    # Load the group image
    group_image = cv2.imread(original_img_path)
    if group_image is None:
        print(f"Error: Unable to read the group image from {original_img_path}")
        return None

    # Convert to HSV
    hsv_image = cv2.cvtColor(group_image, cv2.COLOR_BGR2HSV)

    # Create a mask for the specified color
    color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to store bounding boxes
    bounding_boxes = []

    # Draw bounding boxes around the detected regions and store bounding box coordinates
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Ignore small contours
        if w * h < 15500:
            continue

        bounding_boxes.append((x, y, w, h))
        cv2.rectangle(group_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Extract the largest ROI based on bounding boxes
    largest_roi = extract_largest_roi(group_image, bounding_boxes)

    # Save the largest ROI without the border
    if largest_roi is not None:
        cv2.imwrite(output_name, largest_roi)


def extract_black_color(image_path):
    original_img_path = image_path

    # Load the group image
    group_image = cv2.imread(original_img_path)
    if group_image is None:
        print(f"Error: Unable to read the group image from {original_img_path}")
    else:
        # Convert to HSV for black color
        lower_black = np.array([90, 50, 50])
        upper_black = np.array([130, 255, 255])

        # Create a mask for black color
        black_mask = cv2.inRange(group_image, lower_black, upper_black)

        # Find contours in the mask
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # List to store bounding boxes
        bounding_boxes = []
        # Draw bounding boxes around the detected regions and store bounding box coordinates
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Ignore small contours
            if w * h < 15500:
                continue

            bounding_boxes.append((x, y, w, h))
            cv2.rectangle(group_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the image with bounding boxes for black color
        cv2.imwrite('black_saree_with_boxes.jpg', group_image)

        # Extract the largest ROI based on bounding boxes for black color
        largest_roi = extract_largest_roi(group_image, bounding_boxes)

        # Save the largest ROI without the border for black color
        if largest_roi is not None:
            cv2.imwrite('largest_black_roi_without_border.jpg', largest_roi)

def extract_largest_roi(original_image, bounding_boxes):
    """
    Extract the largest region of interest (ROI) from the list of bounding boxes.

    Args:
        original_image (numpy.ndarray): Original image.
        bounding_boxes (list): List of bounding boxes in the format (x, y, w, h).

    Returns:
        numpy.ndarray: Extracted ROI.
    """
    if not bounding_boxes:
        return None

    # Find the index of the largest bounding box based on area
    largest_index = np.argmax([w * h for x, y, w, h in bounding_boxes])

    x, y, w, h = bounding_boxes[largest_index]

    # Adjust ROI coordinates to exclude the border
    border_thickness = 2
    x += border_thickness
    y += border_thickness
    w -= 2 * border_thickness
    h -= 2 * border_thickness

    roi = original_image[y:y + h, x:x + w]
    return roi

class ColorExtractorApp:
    def __init__(self, root):
        """
        Initialize the GUI application.

        Args:
            root (tk.Tk): Tkinter root object.
        """
        self.root = root
        self.root.title("Image Generator App")

        # Variables to store image paths
        self.group_image_path = ""
        self.single_image_path = ""

        # Create buttons
        self.group_button = tk.Button(root, text="Upload Group Image", command=self.upload_group_image)
        self.group_button.pack(pady=10)

        self.single_button = tk.Button(root, text="Upload Single Image", command=self.upload_single_image)
        self.single_button.pack(pady=10)

        # Create color buttons
        colors = ["Yellow", "White", "Black", "Purple", "Blue", "Blue2", "Darkmagenta", "Darkpink", "Pink", "Green"]
        for color in colors:
            color_button = tk.Button(root, text=color, command=lambda c=color: self.generate_color_image(c))
            color_button.pack(side=tk.LEFT, pady=5)

        # Display area for color images
        self.color_images_frame = tk.Frame(root)
        self.color_images_frame.pack(pady=10)

        # Display area for the largest ROI image
        self.largest_roi_label = tk.Label(self.color_images_frame)
        self.largest_roi_label.pack(pady=10)

    def upload_group_image(self):
        """
        Open a file dialog to upload the group image.
        """
        self.group_image_path = filedialog.askopenfilename(title="Select Group Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    def upload_single_image(self):
        """
        Open a file dialog to upload the single image.
        """
        self.single_image_path = filedialog.askopenfilename(title="Select Single Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    def generate_color_image(self, color_name):
        """
        Generate an image based on the selected color.

        Args:
            color_name (str): Name of the selected color.

        Returns:
            None
        """
        if not self.group_image_path or not self.single_image_path:
            messagebox.showwarning("Error", "Please upload both group and single images.")
            return
        
        elif color_name == "Black":
            extract_black_color(self.group_image_path)

        else:
            color_bounds = {
                "Yellow": (np.array([20, 100, 100]), np.array([30, 255, 255])),
                "White": (np.array([0, 0, 200]), np.array([180, 30, 255])),
                "Purple": (np.array([150, 40, 40]), np.array([160, 255, 255])),
                "Blue": (np.array([90, 50, 50]), np.array([120, 255, 255])),
                "Blue2": (np.array([90, 50, 50]), np.array([130, 255, 255])),
                "Darkmagenta": (np.array([140, 50, 50]), np.array([160, 255, 150])),
                "Darkpink": (np.array([170, 40, 40]), np.array([180, 255, 255])),
                "Pink": (np.array([120, 20, 200]), np.array([200, 255, 255])),
                "Green": (np.array([40, 40, 40]), np.array([60, 255, 255])),
            }

            lower_bound, upper_bound = color_bounds[color_name]

            extract_color(self.group_image_path, lower_bound, upper_bound, f'largest_{color_name.lower()}_roi_without_border.jpg')

        self.display_image(f'largest_{color_name.lower()}_roi_without_border.jpg')

    def display_image(self, image_path):
        """
        Display the image in the GUI.

        Args:
            image_path (str): Path to the image.

        Returns:
            None
        """ 
        img = Image.open(image_path)
        img = ImageTk.PhotoImage(img)
        self.largest_roi_label.config(image=img)
        self.largest_roi_label.image = img


def main():
    root = tk.Tk()
    app = ColorExtractorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
