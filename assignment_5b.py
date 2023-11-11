import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl


def perform_erosion(image, structuring_element_size):
    kernel = np.ones((structuring_element_size, structuring_element_size), np.uint8)  # structuring element
    rows, cols = image.shape    # Get image dimensions
    eroded_image = np.zeros_like(image)  # Create an empty array for the result

    # Perform Erosion manually
    for i in range(structuring_element_size // 2, rows - structuring_element_size // 2):
        for j in range(structuring_element_size // 2, cols - structuring_element_size // 2):
            region = image[i - structuring_element_size // 2:i + structuring_element_size // 2 + 1,
                           j - structuring_element_size // 2:j + structuring_element_size // 2 + 1]
            eroded_image[i, j] = np.min(region * kernel)

    return eroded_image


def perform_dilation(image, structuring_element_size):
    kernel = np.ones((structuring_element_size, structuring_element_size), np.uint8)  # structuring element
    rows, cols = image.shape  # Get image dimensions
    dilated_image = np.zeros_like(image)    # Create an empty array for the result

    # Perform Dilation manually
    for i in range(structuring_element_size // 2, rows - structuring_element_size // 2):
        for j in range(structuring_element_size // 2, cols - structuring_element_size // 2):
            region = image[i - structuring_element_size // 2:i + structuring_element_size // 2 + 1,
                           j - structuring_element_size // 2:j + structuring_element_size // 2 + 1]
            dilated_image[i, j] = np.max(region * kernel)

    return dilated_image


def perform_opening(image, structuring_element_size):
    # Perform Erosion followed by Dilation (Opening)
    opened_image = perform_erosion(image, structuring_element_size)
    opened_image = perform_dilation(opened_image, structuring_element_size)

    return opened_image

def perform_closing(image, structuring_element_size):
    # Perform Dilation followed by Erosion (Closing)
    closed_image = perform_dilation(image, structuring_element_size)
    closed_image = perform_erosion(closed_image, structuring_element_size)

    return closed_image


image1 = cv2.imread('./images/opening.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('./images/closing.jpg', cv2.IMREAD_GRAYSCALE)

structuring_element_size1 = 11   # for morphological opening operation
structuring_element_size2 = 5   # for morphological closing operation

# Perform Opening
opened_image = perform_opening(image1, structuring_element_size1)
# Perform Closing
closed_image = perform_closing(image2, structuring_element_size2)

# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
pl.title("Original Binary Image1")
plt.imshow(image1, cmap='gray')

plt.subplot(2, 2, 2)
pl.title("Morphological Opening (erosion+dilation)")
plt.imshow(opened_image, cmap='gray')

plt.subplot(2, 2, 3)
pl.title("Original Image2")
plt.imshow(image2, cmap='gray')

plt.subplot(2, 2, 4)
pl.title("Morphological Closing (dilation+erosion)")
plt.imshow(closed_image, cmap='gray')

plt.tight_layout()
plt.show()
