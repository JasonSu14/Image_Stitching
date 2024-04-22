import numpy as np
import cv2

def crop_image():
    return 0


############################################
def remove_black_cropping(img):
    # Convert to grayscale and binary
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find non-zero points
    points = np.nonzero(binary)
    points = np.transpose(np.vstack(points))

    # Compute minimal bounding box
    rect = cv2.boundingRect(points)

    result = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    return result

def remove_black_cropping_vertical(img):
    # Convert to grayscale and binary
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Generate list of binary values for each row by checking if there's any non-black pixel in the row
    rows_with_value = [any(row) for row in binary]

    # Compute the top and bottom bounds
    top_bound = next((i for i, x in enumerate(rows_with_value) if x), None)
    bottom_bound = len(rows_with_value) - next((i for i, x in enumerate(reversed(rows_with_value)) if x), None)

    # Remove blank pixels on top and bottom
    result = img[top_bound:bottom_bound, :]

    return result

def crop_image(image, ref_start_y, ref_img_height):
    cropped_image = image[ref_start_y: ref_start_y + ref_img_height, :]
    return cropped_image

def remove_black_edges(image):
    # Find the first column that is not entirely black (all values are 0)
    non_black_columns = np.where(image.min(axis=(0, 2)) > 0)[0]
    leftmost_non_black_column = non_black_columns[0]
    
    # Crop the image to remove the black pixels on the left
    cropped_image = image[:, leftmost_non_black_column:]
    return cropped_image
