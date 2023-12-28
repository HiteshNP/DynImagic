# Individual Color Generator Application README

## Introduction
This Python application utilizes AI techniques for color separation in a group image, generating individual images with distinct colors. 

The application is built using the tkinter library for the graphical user interface (GUI).

The Python Imaging Library(PIL) and OpenCV for image processing.

## Setup and Usage
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/HiteshNP/DynImagic.git
   cd DynImagic

Install Dependencies:

bash
Copy code
pip install pillow opencv-python
Run the Application:

bash
Copy code
python color_extractor.py
Usage:

Click on "Upload Group Image" to select the group image.
Click on "Upload Single Image" to select a single image.
Choose a color from the available buttons to extract and generate an image based on that color.
AI Techniques for Color Separation
The color separation is achieved using the following AI techniques:

Color Masking: The OpenCV library is used to convert the image to the HSV color space and create a mask for the specified color range.
Contour Detection: Contours are identified in the mask to locate regions of the specified color.
Bounding Boxes: Bounding boxes are drawn around the detected regions, and the largest region is extracted as the Region of Interest (ROI).
Border Removal: A border around the ROI is removed to obtain the final individual color image.
Example
Below is an example of how to run the application and generate an individual color image:

bash
Copy code
# Clone the repository
git clone https://github.com/your-username/color-extractor.git
cd color-extractor

# Install dependencies
pip install pillow opencv-python

# Run the application
python color_extractor.py
Important Note
Ensure that both the group image and the single image are uploaded before generating individual color images.
