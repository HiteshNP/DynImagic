from PIL import Image
import numpy as np
import cv2
import os

os.environ['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '10'  # Set timeout to 10 seconds
import threading
from sklearn.cluster import KMeans

# Example usage
# Load the original image with all sarees
original_img_path = 'image_group.jpg'
target_img_path = 'image_single.jpg'


# Function to get channel averages
def get_ch_avgs(im):
    if im is None:
        print("Error: Unable to read the image.")
        return None

    # Convert image to float32 to avoid data type issues
    im = im.astype(np.float32)

    non_zero_c = cv2.countNonZero(im.sum(axis=2))
    ch_avgs = np.array([im[:, :, 0].sum(), im[:, :, 1].sum(), im[:, :, 2].sum()])
    ch_avgs = ch_avgs / non_zero_c
    return ch_avgs


# Function to extract the largest ROI from the original image based on bounding boxes
def extract_largest_roi(original_image, bounding_boxes):
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


# Load the group image
group_image = cv2.imread(original_img_path)
if group_image is None:
    print(f"Error: Unable to read the group image from {original_img_path}")
else:
    # Convert to HSV for black color
    lower_black = np.array([90, 55, 55])
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
