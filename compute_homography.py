# This file contains the function that computes the homography between two images

import numpy as np

def transform(img):
    x, y = img[:,0], img[:,1]
    s = np.sqrt(2) / np.mean(np.sqrt((x - np.mean(x))**2 + (y - np.mean(y))**2))
    T = np.zeros((3,3))
    T[0,0] = s
    T[1,1] = s
    T[2,2] = 1
    T[0,2] = -s * np.mean(x)
    T[1,2] = -s * np.mean(y)
    return T

def compute_homography(keypoints1, keypoints2):
    # get the number of keypoints
    num_of_kps = len(keypoints1)

    # perform Normalized Direct Linear Transformation on the keypoints
    Ta = transform(keypoints1)
    Tb = transform(keypoints2)
    keypoints1_normalized = np.transpose(np.dot(Ta, np.transpose(np.column_stack((keypoints1, np.ones(num_of_kps))))))
    keypoints2_normalized = np.transpose(np.dot(Tb, np.transpose(np.column_stack((keypoints2, np.ones(num_of_kps))))))

    # create the A matrix for Ah = 0
    A = []
    for i in range(num_of_kps):
        x1, y1, _ = keypoints1_normalized[i]
        x2, y2, _ = keypoints2_normalized[i]
        A.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])

    A = np.array(A)  # shape = 2N x 9
    B = np.dot(np.transpose(A), A)   # B = A^T*A

    # eigen analysis
    eigenvalue, eigenvector = np.linalg.eig(B)
    h = eigenvector[:, np.argmin(eigenvalue)]   # set h equal to eigenvector of A^T*A corresponding to smallest eigenvalue
    h /= np.linalg.norm(h)   # normalize h to unit vector to enforce 8 DoF (Degrees of Freedom)

    # unrasterize h to give homography matrix H_tilde
    H_tilde = h.reshape((3,3))

    # Remove normalization on H_tilde to get the final homography matrix H
    H = np.dot(np.dot(np.linalg.inv(Tb), H_tilde), Ta)

    # print('Final homography H:')
    # print(H)

    return H

########################################
# second implementation
# This file contains the function that computes the homography between two images
# def compute_homography(keypoint1, keypoint2):
#     # perform Normalized Direct Linear Transformation on the keypoints

#     print("Shape of keypoints 1: ", keypoint1.shape)
#     print("Shape of keypoints 2: ", keypoint2.shape)
#     num_of_kps = keypoint1.shape[0]

#     print("Shape of op:", np.transpose(np.column_stack((keypoint1, np.ones(num_of_kps)))).shape)
#     keypoint1_homogeneous = np.column_stack((keypoint1, np.ones(num_of_kps)))
#     keypoint2_homogeneous = np.column_stack((keypoint2, np.ones(num_of_kps)))

#     # create the A matrix for Ah = 0
#     A = []
#     for i in range(num_of_kps):
#         x1, y1, _ = keypoint1_homogeneous[i]
#         x2, y2, _ = keypoint2_homogeneous[i]
#         A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
#         A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])

#     A = np.array(A)  # shape = 2N x 9
#     # B = np.dot(np.transpose(A), A)  # B = A^T*A

#     _, _, V = np.linalg.svd(A)
#     H = V[-1].reshape((3, 3))

#     H = H / H[-1, -1]

#     print('Final homography H:')
#     print(H)

#     return H