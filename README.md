# 📷 Image Stitching Project

This is the final project for **CSE 5524: Computer Vision for Human-Computer Interaction**.

## Description

This project implements a panoramic image stitching pipeline that combines multiple overlapping images into a seamless composite.

The core steps include:

- **Feature detection and matching** using **SIFT (Scale-Invariant Feature Transform)**

- **Robust homography estimation** using **RANSAC (Random Sample Consensus)**

- **Image warping** and **multi-band blending** for smooth transitions

## Project Structure

    ```bash
    ├── Image_Stitching/
    │   ├── image_stitching.ipynb    # Final notebook implementation
    │   ├── legacy/                  # Old development-stage Python files
    │   ├── assets/
    │   │   ├── inputs/              # Input image sets
    │   │   ├── outputs/             # Stitched and cropped results
    │   │   └── demo_imgs_and_diagrams/ # Diagrams and README visuals
    │   ├── requirements.txt
    │   └── README.md
    ```

## Demo

1. **Input images** (3-view capture):

    ![These are the 3 input images to be stitched](assets/demo_imgs_and_diagrams/inputs.png)

2. **Stitched image output** (before cropping):

    ![This is the stitched output image](assets/demo_imgs_and_diagrams/output.png)

3. **Final cropped result**:

    ![This is the final result after manually cropping](assets/demo_imgs_and_diagrams/final_result.png)

## Installation

1. Clone the repository to your local machine and navigate to the project directory.

    ```bash
    git clone https://github.com/JasonSu14/Image_Stitching.git
    cd Image_Stitching
    ```

2. Install the required dependencies.

    ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Add images you wish to stitch to the `assets/inputs/` directory.
2. Modify the file paths in the notebook (`image_stitching.ipynb`) as needed:

    ```python
    # Example
    img1_path = "assets/inputs/room_left.png"
    img2_path = "assets/inputs/room_center.png"
    img3_path = "assets/inputs/room_right.png"
    ```

3. Run all notebook cells. Final outputs will be saved to `assets/outputs/`.

## Contributors

- Jason Su [su.925@osu.edu]
- Joe Quinn [quinn.450@osu.edu]
