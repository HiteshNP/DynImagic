# Individual Colored Image Generator Application README

## Introduction
This Python application utilizes AI techniques for color separation in a group image, generating individual images with distinct colors. 

The application is built using the tkinter library for the graphical user interface (GUI).

The Python Imaging Library(PIL) and OpenCV for image processing.

## Setup and Run Instructions

1. **Clone/Download ZIP of the Repository:**
   ```bash
   git clone https://github.com/HiteshNP/DynImagic.git
   cd DynImagic

2. **Install Dependencies:**
   ```bash
   pip install tk
   pip install Pillow
   pip install opencv-python

3. **Run the Application:**
   ```bash
   python DynImagic_main.py

4. **Usage:**
- Click "Upload Group Image" and select a group image.
- Click "Upload Single Image" and select a single image.
- Choose a color button to generate individual images based on the selected color.
- The generated images will be displayed in the GUI.

## AI Techniques for Color Separation

The application uses the following AI techniques for color separation:

- **HSV Color Space:** The input images are converted from the RGB color space to the HSV (Hue, Saturation, Value) color space, which is more suitable for color-based segmentation.

- **Color Masking:** A mask is created for the specified color range in the HSV color space. This mask isolates the pixels within the desired color range.

- **Contour Detection:** Contours are detected in the color mask, representing regions of the specified color.

- **Bounding Boxes:** Bounding boxes are drawn around the detected contours. Small contours are ignored to filter out noise.

- **Largest ROI Extraction:** The application extracts the largest Region of Interest (ROI) based on the bounding boxes. The extracted ROI represents the individual item of the specified color.

- **Image Display:** The GUI displays the generated images, including the largest ROI without the border.

