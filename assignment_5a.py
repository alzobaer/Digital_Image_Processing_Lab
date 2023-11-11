import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl


def perform_erosion(image, structuring_element_size):
    rows, cols = image.shape
    kernel = np.ones((structuring_element_size, structuring_element_size), np.uint8)
    eroded_image = np.zeros_like(image)

    for i in range(structuring_element_size // 2, rows - structuring_element_size // 2):
        for j in range(structuring_element_size // 2, cols - structuring_element_size // 2):
            region = image[i - structuring_element_size // 2: i + structuring_element_size // 2 + 1,
                           j - structuring_element_size // 2: j + structuring_element_size // 2 + 1]
            eroded_image[i, j] = np.min(region * kernel)  # structuring element fits entirely within the image for each pixel.
    return eroded_image


def perform_dilation(image, structuring_element_size):
    rows, cols = image.shape
    kernel = np.ones((structuring_element_size, structuring_element_size), np.uint8)
    dilated_image = np.zeros_like(image)

    for i in range(structuring_element_size // 2, rows - structuring_element_size // 2):
        for j in range(structuring_element_size // 2, cols - structuring_element_size // 2):
            region = image[i - structuring_element_size // 2: i + structuring_element_size // 2 + 1,
                           j - structuring_element_size // 2: j + structuring_element_size // 2 + 1]
            dilated_image[i, j] = np.max(region * kernel)
    return dilated_image


image1 = cv2.imread('./images/erosion.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('./images/dilation.png', cv2.IMREAD_GRAYSCALE)

structuring_element_size1 = 15  # for morphological opening operation
structuring_element_size2 = 11  # for morphological closing operation

# Perform Erosion
eroded_image = perform_erosion(image1, structuring_element_size1)
# Perform Dilation
dilated_image = perform_dilation(image2, structuring_element_size2)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
pl.title("Original Image1")
plt.imshow(image1, cmap='gray')

plt.subplot(2, 2, 2)
pl.title("Erosion Operation")
plt.imshow(eroded_image, cmap='gray')

plt.subplot(2, 2, 3)
pl.title("Orignal Image2")
plt.imshow(image2, cmap='gray')

plt.subplot(2, 2, 4)
pl.title("Delation Operation")
plt.imshow(dilated_image, cmap='gray')

plt.tight_layout()
plt.show()
