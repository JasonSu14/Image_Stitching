# This file contains the function that runs the RANSAC algorithm to find the best homography matrix between two sets of points

import numpy as np
import cv2
from compute_homography import compute_homography

# x_src are the points from the source image; x_dest are the destination points
# ransac_itr is the number of iterations; threshold is the threshold for the RANSAC algorithm
def run_ransac(x_src, x_dest, ransac_itr, threshold):
    num_pts = len(x_src)
    pts_id = np.arange(num_pts)
    max_M = 0
    best_H = None
    inliers_id = np.array([])

    for iter in range(ransac_itr):
        inds = np.random.choice(num_pts, 4, replace=False)
        pts_src = x_src[inds, :]
        pts_dst = x_dest[inds, :]

        # Compute the homography matrix
        H = compute_homography(pts_src, pts_dst)

        # Project source points to destination using homography
        ones = np.ones((num_pts, 1))

        # Convert to homogenous coordinates
        x_src_homog = np.hstack((x_src, ones))
        projected_pts = H @ x_src_homog.T
        projected_pts /= projected_pts[2, :]  # Convert from homogenous to Cartesian coords
        
        # Calculate distances
        temp = projected_pts[:2, :].T
        dist = np.sqrt(np.sum((temp - x_dest) ** 2, axis=1))

        # Determine inliers
        i = dist < threshold
        num_inliers = np.sum(i)

        # Update best model if current model is better
        if num_inliers > max_M:
            max_M = num_inliers
            best_H = H
            inliers_id = np.where(i)[0]  # Store the indices of the inliers

    return inliers_id, best_H


###############################################
# Functon runRANSAC: x_dest are the destination points, x_src are the points from source, ransac_itr is the number of iterations, threshold.
# def run_ransac(x_src, x_dest, ransac_itr, threshold):
#     # H, mask = cv2.findHomography(np.float32(kp_img), np.float32(kp_stitch), cv2.RANSAC, threshold)
#     num_pts = len(x_src)
#     pts_id = np.arange(num_pts)
#     max_M = 0
#     best_H = None
#     inliers_id = np.array([])
#     print("Keypoint shapes", x_src.shape, x_dest.shape)

#     for iter in range(ransac_itr):
#         inds = np.random.choice(num_pts, 4, replace=False)
#         pts_src = x_src[inds, :]
#         pts_dst = x_dest[inds, :]
#         # Compute the homography matrix
#         ######### USING CV2 ############
#         # H, _ = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC)
#         ######## HW Homography Function ###########
#         H = compute_homography(pts_src, pts_dst)
#         # Project source points to destination using homography
#         ones = np.ones((num_pts, 1))
#         # Convert to homogenous coordinates
#         x_src_homog = np.hstack((x_src, ones))
#         projected_pts = H @ x_src_homog.T
#         projected_pts /= projected_pts[2, :]  # Convert from homogenous to Cartesian coords
#         # Calculate distances
#         temp = projected_pts[:2, :].T
#         print("Shape of temp:", temp.shape)
#         print("Shape of x_dest:", x_dest.shape)
#         dist = np.sqrt(np.sum((temp - x_dest) ** 2, axis=1))

#         # Determine inliers
#         i = dist < threshold
#         num_inliers = np.sum(i)

#         # Update best model if current model is better
#         if num_inliers > max_M:
#             max_M = num_inliers
#             best_H = H
#             inliers_id = np.where(i)[0]  # Store the indices of the inliers

#     if max_M > 8:  # at least 4 point correspondences
#         x_src_inliers = x_src[inliers_id, :]
#         x_dest_inliers = x_dest[inliers_id, :]
#         best_H = compute_homography(x_src_inliers, x_dest_inliers)

#     return inliers_id, best_H


# def run_ransac_cv(x_src, x_dest, threshold):
#     # Convert points to the required data type
#     pts_src = x_src.astype('float32')
#     pts_dst = x_dest.astype('float32')

#     # Compute the homography matrix using RANSAC
#     H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, threshold)

#     # Find the inliers
#     inliers_cv_id = np.where(mask.flatten() == 1)[0]

#     return inliers_cv_id, H


###########################################
# code to test the ransac script
# img1_path = "room_center.jpeg"
# img2_path = "room_left.jpeg"
# img1 = cv2.imread(img1_path)
# img2 = cv2.imread(img2_path)
# kp_stitched, kp_img = get_sift_points(img1_path, img2_path)
# # inliers_id, best_H = run_ransac(kp_img, kp_stitched, 500, 5)
# inliers_id, best_H = run_ransac_cv(kp_img, kp_stitched, 500)


# def draw_matches(img1, kp_img, img2, kp_stitched, inliers):
#     """
#     Draws lines between matching keypoints of two images.
#     Only inliers are drawn.
#     img1, img2: source and destination images
#     kp1, kp2: keypoints from both images
#     inliers: array of indices of inliers to be drawn
#     """
#     # Create a blank image that fits both the input images
#     h1, w1 = img1.shape[:2]
#     h2, w2 = img2.shape[:2]
#     nWidth = w1 + w2
#     nHeight = max(h1, h2)
#     hdif = (h1 - h2) // 2
#     new_img = np.zeros((nHeight, nWidth, 3), dtype=np.uint8)
#     new_img[hdif:hdif + h2, :w2] = img2
#     new_img[:h1, w2:w1 + w2] = img1

#     # Draw lines for inliers
#     for i in inliers:
#         pt1 = (int(kp_img[i][0] + w2), int(kp_img[i][1]))
#         pt2 = (int(kp_stitched[i][0]), int(kp_stitched[i][1]) + hdif)
#         # color = tuple(np.random.randint(0, 255, 3).tolist())
#         color = (0, 255, 0)
#         cv2.line(new_img, pt1, pt2, color, 1)

#     plt.figure(figsize=(10, 5))
#     plt.imshow(new_img)
#     cv2.imwrite("ransac.jpg", new_img)
#     plt.show()


# draw_matches(img1, kp_img, img2, kp_stitched, inliers_id)
