import cv2
import numpy as np
import matplotlib.pyplot as plt


# From OpenCV tutorial
def get_sift_points(query_img, train_img):
    img1 = cv2.imread(query_img, cv2.IMREAD_GRAYSCALE)  # queryImage
    img2 = cv2.imread(train_img, cv2.IMREAD_GRAYSCALE)  # trainImage
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp_img, des1 = sift.detectAndCompute(img1, None)
    kp_stitch, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
        if len(good) == 50:
            break
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp_img, img2, kp_stitch, good, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3), plt.show()
    cv2.imwrite("matches.jpg", img3)
    return kp_stitch, kp_img
