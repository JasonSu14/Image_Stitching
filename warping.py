# This file is performing backward image warping.

import numpy as np
from apply_homography import apply_homography

def backward_warping(source_image, inverse_H, output_img_shape):
    # Get the dimensions of the source image and the output image
    source_img_width, source_img_height, source_img_channels = source_image.shape[0], source_image.shape[1], source_image.shape[2]
    output_img_width, output_img_height = output_img_shape[0], output_img_shape[1]
    
    # Initialize the output image
    output_image = np.zeros((output_img_width, output_img_height, source_img_channels), dtype=np.uint8)
    mask = np.zeros((output_img_width, output_img_height))

    output_Y, output_X = np.meshgrid(np.arange(output_img_height), np.arange(output_img_width), indexing='ij')

    # Map the output image coordinates to the source image coordinates
    source_points = apply_homography(inverse_H, np.stack([output_X.ravel(), output_Y.ravel()], axis=-1))
    source_X = np.round(source_points[:, 0]).reshape(output_img_height, output_img_width)
    source_Y = np.round(source_points[:, 1]).reshape(output_img_height, output_img_width)

    for i in range(output_img_height):
        for j in range(output_img_width):
            x = int(source_X[i, j])
            y = int(source_Y[i, j])
            if x >= 0 and y >= 0 and x < source_img_width and y < source_img_height:
                output_image[i, j, :] = source_image[y, x, :]
                mask[j, i] = True
            else:
                output_image[i, j, :] = [0, 0, 0]

    return mask, output_image


############################################
# second implementation
# Function to applyHomography
# def applyHomography(H, pts):
#     # get points
#     t_pts = np.dot(H, np.array([pts[0], pts[1], 1]))
#     return t_pts[0] / t_pts[2], t_pts[1] / t_pts[2]


# def backward_warp(src_img, resultToSrc_H, dest_width, dest_height):
#     src_height, src_width = src_img.shape[:2]
#     # Initialize output image and mask, using the src_img channels
#     result_img = np.zeros((dest_height, dest_width, src_img.shape[2]), dtype=np.uint8)
#     mask = np.zeros((dest_height, dest_width), dtype=np.uint8)  # set mask to 0's

#     # Go through all points in the output image
#     for i in range(dest_height):
#         for j in range(dest_width):
#             # Map point to source image
#             x, y = applyHomography(resultToSrc_H, (i, j))
#             # Checking if point lies in source image
#             if x >= 0 and y >= 0 and x < src_width and y < src_height:
#                 result_img[i, j, :] = cv2.getRectSubPix(src_img, (1, 1), (x, y))
#                 mask[i, j] = 1 # update mask at proper position in new image

#     cv2.imshow('Result Image', result_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.imwrite('bward.jpg', result_img)
#     return mask, result_img

############################################
# third implementation
# def warp_cv(src_img, stitched_img, resultToSrc_H, dest_width, dest_height):
#     # Define size of the output image
#     size = (dest_width, dest_height)

#     # Determine the inverse homography matrix
#     inv_H = np.linalg.inv(resultToSrc_H)

#     # Perform the warp using inverse homography
#     result_img = cv2.warpPerspective(src_img, inv_H, size)

#     # Blending the warped image with the second image using alpha blending
#     alpha = 0.5  # blending factor
#     blended_image = cv2.addWeighted(result_img, alpha, stitched_img, 1 - alpha, 0)
#     # Create a mask based on where we have non-zero pixels in the result
#     mask = np.sum(blended_image, axis=2) > 0
#     cv2.imwrite('bward.jpg', blended_image)
#     return mask, blended_image