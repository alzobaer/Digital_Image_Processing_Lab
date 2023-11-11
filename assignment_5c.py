import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

def perform_erosion(image, structuring_element_size):
    # Define the structuring element (kernel)
    kernel = np.ones((structuring_element_size, structuring_element_size), np.uint8)

    # Get image dimensions
    rows, cols = image.shape

    # Create an empty array for the result
    eroded_image = np.zeros_like(image)

    # Perform Dilation manually
    for i in range(structuring_element_size // 2, rows - structuring_element_size // 2):
        for j in range(structuring_element_size // 2, cols - structuring_element_size // 2):
            region = image[i - structuring_element_size // 2:i + structuring_element_size // 2 + 1,
                           j - structuring_element_size // 2:j + structuring_element_size // 2 + 1]
            eroded_image[i, j] = np.min(region * kernel)

    return eroded_image


# load an image
original_image = cv2.imread('./images/boundary_extraction.jpg', cv2.IMREAD_GRAYSCALE)
structuring_element_size = 3

# Perform Erosion operation
eroded_image = perform_erosion(original_image, structuring_element_size)
boundary_image = original_image - eroded_image

# display images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
pl.title("Original Binary Image")
plt.imshow(original_image, cmap='gray')

plt.subplot(1, 2, 2)
pl.title("Boundary Extracted from Original Image")
plt.imshow(boundary_image, cmap='gray')

plt.tight_layout()
plt.show()
