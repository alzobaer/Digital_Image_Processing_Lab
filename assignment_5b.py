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

    # Perform Erosion manually
    for i in range(structuring_element_size // 2, rows - structuring_element_size // 2):
        for j in range(structuring_element_size // 2, cols - structuring_element_size // 2):
            region = image[i - structuring_element_size // 2:i + structuring_element_size // 2 + 1,
                           j - structuring_element_size // 2:j + structuring_element_size // 2 + 1]
            eroded_image[i, j] = np.min(region)

    return eroded_image


def perform_dilation(image, structuring_element_size):
    # Define the structuring element (kernel)
    kernel = np.ones((structuring_element_size, structuring_element_size), np.uint8)

    # Get image dimensions
    rows, cols = image.shape

    # Create an empty array for the result
    dilated_image = np.zeros_like(image)

    # Perform Dilation manually
    for i in range(structuring_element_size // 2, rows - structuring_element_size // 2):
        for j in range(structuring_element_size // 2, cols - structuring_element_size // 2):
            region = image[i - structuring_element_size // 2:i + structuring_element_size // 2 + 1,
                           j - structuring_element_size // 2:j + structuring_element_size // 2 + 1]
            dilated_image[i, j] = np.max(region)

    return dilated_image


def perform_opening(image, structuring_element_size):
    # Define the structuring element (kernel)
    kernel = np.ones((structuring_element_size, structuring_element_size), np.uint8)

    # Perform Erosion followed by Dilation (Opening)
    opened_image = perform_erosion(image, structuring_element_size)
    opened_image = perform_dilation(opened_image, structuring_element_size)

    return opened_image

def perform_closing(image, structuring_element_size):
    # Define the structuring element (kernel)
    kernel = np.ones((structuring_element_size, structuring_element_size), np.uint8)

    # Perform Dilation followed by Erosion (Closing)
    closed_image = perform_dilation(image, structuring_element_size)
    closed_image = perform_erosion(closed_image, structuring_element_size)

    return closed_image


# Example usage
image1 = cv2.imread('./images/finger_print.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('./images/closing.png', cv2.IMREAD_GRAYSCALE)
structuring_element_size = 10

# Perform Opening
opened_image = perform_opening(image1, structuring_element_size)

# Perform Closing
closed_image = perform_closing(image2, structuring_element_size)

# Display the results
plt.subplot(2, 2, 1)
pl.title("Original Image1")
plt.imshow(image1, cmap='gray')

plt.subplot(2, 2, 2)
pl.title("Morphological Opening")
plt.imshow(opened_image, cmap='gray')

plt.subplot(2, 2, 3)
pl.title("Original Image2")
plt.imshow(image2, cmap='gray')

plt.subplot(2, 2, 4)
pl.title("Morphological Closing")
plt.imshow(closed_image, cmap='gray')

plt.tight_layout()
plt.show()
