# This file performs the SIFT keypoint matching between two images

import cv2
import numpy as np
import matplotlib.pyplot as plt

# From OpenCV tutorial
def get_sift_points(query_img, train_img):
    img1 = cv2.imread(query_img, cv2.IMREAD_GRAYSCALE)  # queryImage
    img2 = cv2.imread(train_img, cv2.IMREAD_GRAYSCALE)  # trainImage
    img_color1 = cv2.imread(query_img, cv2.IMREAD_COLOR)  # queryImage
    img_color2 = cv2.imread(train_img, cv2.IMREAD_COLOR)  # trainImage
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

    kp_img_list = [kp.pt for kp in kp_img]
    kp_stitch_list = [kp.pt for kp in kp_stitch]

    kp_img_np = np.array(kp_img_list)
    kp_stitch_np = np.array(kp_stitch_list)

    return kp_img_np, kp_stitch_np


##################################################
# second implementation
# def get_sift_points_cv(query_img, train_img):
#     img1 = cv2.cvtColor(query_img, cv2.COLOR_RGB2GRAY)  # queryImage
#     img2 = cv2.cvtColor(train_img, cv2.COLOR_RGB2GRAY)  # trainImage
#     # img_color1 = cv2.imread(query_img, cv2.IMREAD_COLOR)  # queryImage
#     # img_color2 = cv2.imread(train_img, cv2.IMREAD_COLOR)  # trainImage
#     # Initiate SIFT detector
#     sift = cv2.SIFT_create()

#     # find the keypoints and descriptors with SIFT
#     kp_img, des1 = sift.detectAndCompute(img1, None)
#     kp_stitch, des2 = sift.detectAndCompute(img2, None)

#     MIN_MATCH_COUNT = 10
#     FLANN_INDEX_KDTREE = 1
#     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#     search_params = dict(checks=50)
#     flann = cv2.FlannBasedMatcher(index_params, search_params)
#     matches = flann.knnMatch(des1, des2, k=2)
#     # store all the good matches as per Lowe's ratio test.
#     good = []
#     for m, n in matches:
#         if m.distance < 0.7 * n.distance:
#             good.append(m)

#     if len(good) > MIN_MATCH_COUNT:
#         src_pts = np.float32([kp_img[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#         dst_pts = np.float32([kp_stitch[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#         M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#         matchesMask = mask.ravel().tolist()
#         h, w = img1.shape
#         pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
#         dst = cv2.perspectiveTransform(pts, M)
#         img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
#     else:
#         print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
#         matchesMask = None
#         M = None

#     draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
#                        singlePointColor=None,
#                        matchesMask=matchesMask,  # draw only inliers
#                        flags=2)

#     # Display the image with RANSAC lines
#     img3 = cv2.drawMatches(img1, kp_img, img2, kp_stitch, good, None, **draw_params)
#     plt.imshow(img3, 'gray'), plt.show()

#     plt.imshow(img3), plt.show()
#     cv2.imwrite("matches.jpg", img3)

#     return M


##################################################
# current implementation in ipynb
# From OpenCV tutorial
# def get_sift_points(query_img, train_img):
#     img1 = cv2.imread(query_img, cv2.IMREAD_GRAYSCALE)  # queryImage
#     img2 = cv2.imread(train_img, cv2.IMREAD_GRAYSCALE)  # trainImage
#     img_color1 = cv2.imread(query_img, cv2.IMREAD_COLOR)  # queryImage
#     img_color2 = cv2.imread(train_img, cv2.IMREAD_COLOR)  # trainImage
#     # Initiate SIFT detector
#     sift = cv2.SIFT_create()

#     # find the keypoints and descriptors with SIFT
#     kp_img, des1 = sift.detectAndCompute(img1, None)
#     kp_stitch, des2 = sift.detectAndCompute(img2, None)

#     ########### TEST 1 #############

#     # BFMatcher with default params
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(des1, des2, k=2)
#     # Apply ratio test
#     good = []
#     for m, n in matches:
#         if m.distance < 0.75 * n.distance:
#             good.append([m])

#     points_query = np.float32([kp_img[m.queryIdx].pt for m, n in matches if m.distance < 0.75 * n.distance])
#     points_train = np.float32([kp_stitch[m.trainIdx].pt for m, n in matches if m.distance < 0.75 * n.distance])

#     # cv.drawMatchesKnn expects list of lists as matches.
#     img3 = cv2.drawMatchesKnn(img_color1, kp_img, img_color2, kp_stitch, good, None,
#                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    ########### TEST 2 (WITH RANSAC) ###########

    # MIN_MATCH_COUNT = 10
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks = 50)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1,des2,k=2)
    # # store all the good matches as per Lowe's ratio test.
    # good = []
    # for m,n in matches:
    #     if m.distance < 0.7*n.distance:
    #         good.append(m)
    # 
    # if len(good)>MIN_MATCH_COUNT:
    #     src_pts = np.float32([ kp_img[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    #     dst_pts = np.float32([ kp_stitch[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    #     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    #     matchesMask = mask.ravel().tolist()
    #     h,w = img1.shape
    #     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    #     dst = cv2.perspectiveTransform(pts,M)
    #     img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    # else:
    #     print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    #     matchesMask = None
    # 
    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #                    singlePointColor = None,
    #                    matchesMask = matchesMask, # draw only inliers
    #                    flags = 2)
    # img3 = cv2.drawMatches(img1,kp_img,img2,kp_stitch,good,None,**draw_params)
    # plt.imshow(img3, 'gray'),plt.show()

    ####### TEST 3 (not quite working) #########

    # # find the keypoints and descriptors with SIFT
    # kp_img, des1 = sift.detectAndCompute(img1, None)
    # kp_stitch, des2 = sift.detectAndCompute(img2, None)
    # 
    # # initialize the Brute Force matcher (BFMatcher)
    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # 
    # # match the descriptors
    # matches = bf.match(des1, des2)
    # 
    # # sort the matches based on distance
    # matches = sorted(matches, key = lambda x:x.distance)
    # img3 = cv2.drawMatches(img1, kp_img, img2, kp_stitch, matches[:300], None, flags=2)

    ############ TEST 4 IMAGE POINTS ################
    # # Initiate SIFT detector
    # sift = cv2.SIFT_create()
    # 
    # # find the keypoints with SIFT
    # kp1 = sift.detect(img_color1, None)
    # kp2 = sift.detect(img_color2, None)
    # 
    # # Draw keypoints on the images
    # img_keypoints1 = cv2.drawKeypoints(img_color1, kp1, img_color1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # img_keypoints2 = cv2.drawKeypoints(img_color2, kp2, img_color2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # 
    #     # find the descriptors with SIFT
    # kp1, des1 = sift.compute(img_color1, kp1)
    # kp2, des2 = sift.compute(img_color2, kp2)
    # 
    # good = good[:50]
    # 
    # # Use green color for match lines
    # img3 = cv2.drawMatchesKnn(img_keypoints1, kp1, img_keypoints2, kp2, good, None,
    #                           matchColor=(0, 255, 0),
    #                           flags=2)

    # plt.imshow(img3), plt.show()
    # cv2.imwrite("matches.jpg", img3)

    # # Convert kp object into numpy array
    # kp_img_list = [kp.pt for kp in kp_img]
    # kp_stitch_list = [kp.pt for kp in kp_stitch]

    # kp_img_np = np.array(kp_img_list)
    # kp_stitch_np = np.array(kp_stitch_list)

    # return points_query, points_train

##################################################
# test for the get sift points function
# img1_path = "room_center.jpeg"
# img2_path = "room_left.jpeg"
# img1 = cv2.imread(img1_path)
# img2 = cv2.imread(img2_path)
# # kp_pts, kp_im_pts = get_sift_points(img2_path, img1_path)
# M = get_sift_points_cv(img2, img1)
# print(kp_pts.shape)