import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
import scipy
from skimage import io, morphology, color, util, filters


def image_stitcher(*image_paths):
    images = [skimage.io.imread(path) for path in image_paths]
    # Display images
    fig, axes = plt.subplots(1, len(images), figsize=(10, 5))
    for i, (img, path) in enumerate(zip(images, image_paths)):
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(path)
    plt.show()
    # Get total number of vertical pixels
    y_stitched = sum(img.shape[0] for img in images)  # Sum of heights
    x_stitched = sum(img.shape[1] for img in images)  # Sum of widths
    channel_stitch = images[0].shape[2]  # Set the channel (color)

    # Create an array for the stitched image (initializing to 0)
    stitched_img = np.zeros((y_stitched, x_stitched, channel_stitch), dtype=images[0].dtype)
    num_images = len(images) - 1  # number of images to be stitched
    middle_img_idx = num_images // 2  # index of the center image
    print("Middle Image: ", middle_img_idx)

    # Setting the reference image (image to stitch around)
    ref_img = images[middle_img_idx]
    # Size of the ref image
    ref_img_height = ref_img.shape[0]
    ref_img_width = ref_img.shape[1]
    print("Ref Image Height: ", ref_img_height)
    print("Ref Image Width: ", ref_img_width)
    # Calculate the starting position  ensuring they fit within the dimensions of the stitched image (floor division is done using, //)
    ref_start_x = max(0, min((x_stitched - ref_img_width) // 2, x_stitched - ref_img_width))
    ref_start_y = max(0, min((y_stitched - ref_img_height) // 2, y_stitched - ref_img_height))
    print("Ref Start Y: ", ref_start_y)
    print("Ref Start X: ", ref_start_x)
    # Initialize binary mask for tracking filled areas (0 is not filled, otherwise filled)
    # Gets updated each time new area is filled
    stitch_mask = np.zeros((y_stitched, x_stitched))
    print("Mask Shape", stitch_mask.shape)
    print("Stitched image dimensions:", stitched_img.shape)
    print("Reference image dimensions:", ref_img.shape)
    print("Start Y:", ref_start_y, "End Y:", ref_start_y + ref_img.shape[0])
    print("Start X:", ref_start_x, "End X:", ref_start_x + ref_img.shape[1])
    # Testing the starting image
    stitched_img[ref_start_y:ref_start_y + ref_img_height, ref_start_x:ref_start_x + ref_img_width, :] = ref_img

    stitch_mask[ref_start_y:ref_start_y + ref_img_height, ref_start_x:ref_start_x + ref_img_width] = 1

    # for idx, img in enumerate(images):
    #     if idx == middle_img_idx:
    #         continue
    #     # Generate SIFT matches from indexed image and the current stitched image
    #     get_sift_points(stitched_img, img)
    # 
    # 
    return stitch_mask


# Testing images
img1_path = "room_left.jpeg"
img2_path = "room_center.jpeg"
img3_path = "room_right.jpeg"

result_img = image_stitcher(img1_path, img2_path, img3_path)
plt.imshow(result_img)
plt.axis('off')
plt.show()
