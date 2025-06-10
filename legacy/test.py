import cv2
import numpy as np

# Load the images
image1 = cv2.imread('mountain_left.jpg')
image2 = cv2.imread('mountain_right.jpg')

# Convert the images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect and compute keypoints and descriptors
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Initialize Brute Force matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Compute Homography
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Use homography to warp image
height, width, channels = image2.shape
image1_warped = cv2.warpPerspective(image1, M, (width, height))

# Create a mask from the warped image
_, mask = cv2.threshold(cv2.cvtColor(image1_warped, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)

# Stitch the two images together using the mask
stitched = cv2.bitwise_and(image1_warped, image1_warped, mask=mask)
stitched[mask == 0] = image2[mask == 0]

# Calculate the size of the new image
h1, w1 = image1.shape[:2]
h2, w2 = image2.shape[:2]
pts = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
pts = cv2.perspectiveTransform(pts, M)
pts = np.concatenate((pts, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)), axis=0)
[umin, vmin] = np.int32(pts.min(axis=0).ravel() - 0.5)
[umax, vmax] = np.int32(pts.max(axis=0).ravel() + 0.5)
uoff, voff = -umin, -vmin
Ht = np.array([[1, 0, uoff], [0, 1, voff], [0, 0, 1]])  # translate

# Draw the two images together using the new size
result = cv2.warpPerspective(image1, Ht.dot(M), (umax-umin, vmax-vmin))
result[voff:voff+h2, uoff:uoff+w2] = image2

# Display the result
cv2.imshow('Test 1 Stitched Image', result)
cv2.imwrite('test_1_stitched_image.jpg', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
