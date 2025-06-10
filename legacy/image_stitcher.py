# This file contains the image stitching function that stitches together multiple images.

import numpy as np
from numpy.linalg import inv
from legacy.sift_keypoints import get_sift_points
from legacy.run_ransac import run_ransac
from legacy.warping import backward_warping
from legacy.blending import blend_img
from legacy.cropping import crop_image, remove_black_edges

def image_stitcher(*image_paths):
    # ensure the input images are either all color or all grayscale (stitching color and grayscale images makes no sense)
    assert max(img.shape[2] for img in image_paths) == min(img.shape[2] for img in image_paths), "All images should be either grayscale or color"

    # Calculate the dimensions of the stitched image
    total_height_of_stitched_img = sum(img.shape[0] for img in image_paths)
    total_width_of_stitched_img = sum(img.shape[1] for img in image_paths)
    num_of_color_channels_of_stitched_img = image_paths[0].shape[2]

    # Create an empty array for the stitched image (serve as the canvas onto which the images will be stitched)
    stitched_img = np.zeros((total_height_of_stitched_img, total_width_of_stitched_img, num_of_color_channels_of_stitched_img))

    # Calculate the total number of images
    num_of_imgs = len(image_paths)

    # Calculate the index of the center image
    center_img_idx = round((num_of_imgs + 1) / 2)

    # Set the reference index to the center image index (image to stitch around)
    ref_idx = center_img_idx

    # Select the reference image
    ref_img = image_paths[ref_idx]

    # Get the height and width of the reference image
    ref_img_height = ref_img.shape[0]
    ref_img_width = ref_img.shape[1]

    # Calculate the starting coordinates for the reference image
    ref_start_x = 1 + int((total_width_of_stitched_img - ref_img_width) / 2)
    ref_start_y = 1 + int((total_height_of_stitched_img - ref_img_height) / 2)

    # Paste the reference image into the stitched image
    stitched_img[ref_start_y - 1 : ref_start_y + ref_img_height - 1, ref_start_x - 1 : ref_start_x + ref_img_width - 1] = ref_img

    # Create a logical mask of the same size as the stitched image - to keep track of the filled areas on the canvas (stitched image)
    # defaulted to false because of np.zeros(): 0 == False
    stitch_mask = np.zeros((total_height_of_stitched_img, total_width_of_stitched_img), dtype=bool)

    # Update the mask to indicate where the reference image has been pasted
    stitch_mask[ref_start_y - 1 : ref_start_y + ref_img_height - 1, ref_start_x - 1 : ref_start_x + ref_img_width - 1] = True

    # Iterate over each image
    for n in range(num_of_imgs):
        # Skip the reference image
        if n == ref_idx:
            continue

        # Select the current image
        current_img = image_paths[n]

        # Generate SIFT matches
        kp_stitched, kp_n = get_sift_points(stitched_img, current_img)

        # Run RANSAC to estimate a homography along with the inliers (correct matches)
        inliers_id, H_3x3 = run_ransac(kp_n, kp_stitched, 300, 1)

        # Warp the current image to the stitched image's perspective
        mask, dest_img = backward_warping(current_img, inv(H_3x3), (total_width_of_stitched_img, total_height_of_stitched_img))

        # Blend the warped image into the stitched image
        stitched_img = blend_img(dest_img, mask, stitched_img, stitch_mask, 'blend')

        # Update the mask
        stitch_mask = np.logical_or(stitch_mask, mask)

    # Crop the stitched image
    stitched_img = crop_image(stitched_img, ref_start_y, ref_img_height)
    stitched_img = remove_black_edges(stitched_img)

    # Return the stitched image
    return stitched_img

##################################################
# # Implementation from Joe
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import skimage
# import scipy
# from skimage import io, morphology, color, util, filters

# def image_stitcher(*image_paths):
#     images = [skimage.io.imread(path) for path in image_paths]
#     # Display images
#     fig, axes = plt.subplots(1, len(images), figsize=(10, 5))
#     for i, (img, path) in enumerate(zip(images, image_paths)):
#         axes[i].imshow(img)
#         axes[i].axis('off')
#         axes[i].set_title(path)
#     plt.show()
#     # Get total number of vertical pixels
#     y_stitched = sum(img.shape[0] for img in images)  # Sum of heights
#     x_stitched = sum(img.shape[1] for img in images)  # Sum of widths
#     channel_stitch = images[0].shape[2]  # Set the channel (color)

#     # Create an array for the stitched image (initializing to 0)
#     stitched_img = np.zeros((y_stitched, x_stitched, channel_stitch), dtype=images[0].dtype)
#     num_images = len(images) - 1  # number of images to be stitched
#     middle_img_idx = num_images // 2  # index of the center image
#     print("Middle Image: ", middle_img_idx)

#     # Setting the reference image (image to stitch around)
#     ref_img = images[middle_img_idx]
#     # Size of the ref image
#     ref_img_height = ref_img.shape[0]
#     ref_img_width = ref_img.shape[1]
#     print("Ref Image Height: ", ref_img_height)
#     print("Ref Image Width: ", ref_img_width)
#     # Calculate the starting position  ensuring they fit within the dimensions of the stitched image (floor division is done using, //)
#     ref_start_x = max(0, min((x_stitched - ref_img_width) // 2, x_stitched - ref_img_width))
#     ref_start_y = max(0, min((y_stitched - ref_img_height) // 2, y_stitched - ref_img_height))
#     print("Ref Start Y: ", ref_start_y)
#     print("Ref Start X: ", ref_start_x)
#     # Initialize binary mask for tracking filled areas (0 is not filled, otherwise filled)
#     # Gets updated each time new area is filled
#     stitch_mask = np.zeros((y_stitched, x_stitched))
#     print("Mask Shape", stitch_mask.shape)
#     print("Stitched image dimensions:", stitched_img.shape)
#     print("Reference image dimensions:", ref_img.shape)
#     print("Start Y:", ref_start_y, "End Y:", ref_start_y + ref_img.shape[0])
#     print("Start X:", ref_start_x, "End X:", ref_start_x + ref_img.shape[1])
#     # Testing the starting image
#     stitched_img[ref_start_y:ref_start_y + ref_img_height, ref_start_x:ref_start_x + ref_img_width, :] = ref_img

#     stitch_mask[ref_start_y:ref_start_y + ref_img_height, ref_start_x:ref_start_x + ref_img_width] = 1

#     # for idx, img in enumerate(images):
#     #     if idx == middle_img_idx:
#     #         continue
#     #     # Generate SIFT matches from indexed image and the current stitched image
#     #     get_sift_points(stitched_img, img)
#     # 
#     # 
#     return stitch_mask


##################################################
# def image_stitcher(*image_paths):
#     images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in image_paths]
#     # Convert BGR image to RGB image
#     images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
#     # Display images
#     fig, axes = plt.subplots(1, len(images), figsize=(10, 5))
#     for i, (img, path) in enumerate(zip(images, image_paths)):
#         axes[i].imshow(img)
#         axes[i].axis('off')
#         axes[i].set_title(path)
#     plt.show()
#     # Get total number of vertical pixels
#     y_stitched = sum(img.shape[0] for img in images)  # Sum of heights
#     x_stitched = sum(img.shape[1] for img in images)  # Sum of widths
#     print("y_stitched: ", y_stitched)
#     print("x_stitched: ", x_stitched)
#     channel_stitch = images[0].shape[2]  # Set the channel (color)

#     # Create an array for the stitched image (initializing to 0)
#     stitched_img = np.zeros((y_stitched, x_stitched, channel_stitch), dtype=images[0].dtype)
#     num_images = len(images) - 1  # number of images to be stitched
#     middle_img_idx = num_images // 2  # index of the center image

#     # Setting the reference image (image to stitch around)
#     ref_img = images[middle_img_idx]
#     # Size of the ref image
#     ref_img_height = ref_img.shape[0]
#     ref_img_width = ref_img.shape[1]
#     # Calculate the starting position  ensuring they fit within the dimensions of the stitched image (floor division is done using, //)
#     ref_start_x = max(0, min((x_stitched - ref_img_width) // 2, x_stitched - ref_img_width))
#     ref_start_y = max(0, min((y_stitched - ref_img_height) // 2, y_stitched - ref_img_height))
#     # Initialize binary mask for tracking filled areas (0 is not filled, otherwise filled) and gets updated each time new area is filled
#     stitch_mask = np.zeros((y_stitched, x_stitched))
#     stitch_mask[ref_start_y:ref_start_y + ref_img_height, ref_start_x:ref_start_x + ref_img_width] = 1
#     # Create the starting image stitch (starts with center image)
#     stitched_img[ref_start_y:ref_start_y + ref_img_height, ref_start_x:ref_start_x + ref_img_width, :] = ref_img

#     for idx, img in enumerate(images):
#         # Skip past the middle image
#         if idx == middle_img_idx:
#             continue
#         # Generate SIFT matches from indexed image and the current stitched image (Generates kp for stitched image and current image)
#         """--------- Without OpenCV ------------ """
#         # kp_stitch, kp_img = get_sift_points(stitched_img, img)
#         #inliers_id, best_H = run_ransac(kp_img, kp_stitch, 300, 1)
#         """ ----- With OpenCV (get sift points needs to return homography matrix) ---- """
#         best_H = get_sift_points_cv(stitched_img, img)
#         print("H Shape", best_H.shape)
#         if (best_H is None):
#             print("Can not stitch images")
            
#         # Get the new mask and new destination image
#         # mask, dest_img = backward_warp(img, best_H, y_stitched, x_stitched)
#         mask, dest_img = warp_cv(img, stitched_img, best_H, x_stitched, y_stitched)
#         print("mask shape", mask.shape)
#         print("dest_img shape", dest_img.shape)
#         # stitched_img = blend_img(dest_img, mask, stitched_img, stitch_mask, 'blend')
#         # Or the mask
#         plt.imshow(dest_img)
#         plt.show()
#         stitched_img = dest_img
#         stitch_mask = np.logical_or(stitch_mask, mask)
#         plt.imshow(stitch_mask, cmap="gray")
#         plt.show()

#     stitched_img = crop_image(stitched_img, ref_start_y, ref_img_height)
#     stitched_img = remove_black_edges(stitched_img)
#     # stitched_img = remove_black_cropping(stitched_img)
#     return stitched_img


# # Testing images
# img1_path = "room_left.jpeg"
# img2_path = "room_center.jpeg"
# img3_path = "room_right.jpeg"

# result_img = image_stitcher(img1_path, img2_path, img3_path)
# plt.imshow(result_img)
# cv2.imwrite('final_stitch.jpg', result_img)
# plt.axis('off')
# plt.show()

