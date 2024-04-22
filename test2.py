import cv2
import numpy as np

# Read the two images
image1 = cv2.imread('mountain_left.jpg')
image2 = cv2.imread('mountain_right.jpg')

# Convert the images to grayscale
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Find the keypoints and descriptors with SIFT
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(image1_gray, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2_gray, None)

# Match features between the two images
matcher = cv2.BFMatcher()
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

# Select the good points
good_points = []
for m, n in matches:
    if m.distance < 0.6*n.distance:
        good_points.append(m)

# Stitch the images together using the good match points
result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_points, None)

# Save the result
cv2.imwrite('test_2_result.jpg', result)

# Display the result
cv2.imshow('Test 2 Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()