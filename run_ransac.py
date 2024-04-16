import numpy as np
import cv2


# Functon runRANSAC: x_dest are the destination points, x_src are the points from source, ransac_itr is the number of
# iterations, threshold.
def run_ransac(x_src, x_dest, ransac_itr, threshold):
    # H, mask = cv2.findHomography(np.float32(kp_img), np.float32(kp_stitch), cv2.RANSAC, threshold)
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
        H, _ = cv2.findHomography(pts_src, pts_dst, method=0)
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
