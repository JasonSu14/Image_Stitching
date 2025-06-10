# This file runs the image stitching application

import matplotlib.pyplot as plt
import cv2
from legacy.image_stitcher import image_stitcher

##################################################
# Image Set 1 - Joe's Room                       #
##################################################

# Load images to be stitched
room_left = "room_left.png"
room_center = "room_center.png"
room_right = "room_right.png"

# Stitch the images together
room_stitched_img = image_stitcher(room_left, room_center, room_right)
plt.imshow(room_stitched_img)
cv2.imwrite("room_stitched.png", room_stitched_img)
plt.axis('off')
plt.show()


##################################################
# Image Set 2 - Mountain                         #
##################################################

# Load images to be stitched
mountain_left = "mountain_left.jpg"
mountain_right = "mountain_right.jpg"

# Stitch the images together
mountain_stitched_img = image_stitcher(mountain_left, mountain_right)
plt.imshow(mountain_stitched_img)
cv2.imwrite("mountain_stitched.jpg", mountain_stitched_img)
plt.axis('off')
plt.show()
