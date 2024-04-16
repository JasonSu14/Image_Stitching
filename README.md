# Image Stitching Project

This is the final project for CSE 5524: Computer Vision for Human-Computer Interaction at The Ohio State University.

## Description

This image stitching application is designed to combine multiple photographic with overlapping fields of view to produce a single larger image, also widely known as a panorama. We will be using SIFT (Scale-Invariant Feature Transform) to detect and match features between images, RANSAC (Random Sample Consensus) to estimate the homography between images, and image warping and blending to combine the images.

## Installation

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies.

## Usage

[Provide instructions on how to use your project.]

## Contributors

Jason Su [su.925@osu.edu]
Joe Quinn [quinn.450@osu.edu]

## Work Breakdown

1. Feature Detection and Matching (using SIFT) -- (OpenCV)
2. Homography Estimation (using RANSAC) -- Jason: get H, Joe: do RANSAC
3. Image Warping -- Joe
4. Image Blending -- Jason

## License
