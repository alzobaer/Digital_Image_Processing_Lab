import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl


def perform_erosion(image, structuring_element):
    rows, cols = image.shape
    eroded_image = np.zeros_like(image)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = image[i - 1:i + 2, j - 1:j + 2]
            eroded_image[i, j] = np.min(region * structuring_element)

    return eroded_image


def perform_dilation(image, structuring_element):
    rows, cols = image.shape
    dilated_image = np.zeros_like(image)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = image[i - 1:i + 2, j - 1:j + 2]
            dilated_image[i, j] = np.max(region * structuring_element)

    return dilated_image


# Define the kernel
structuring_element = np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]], dtype=np.uint8)

image1 = cv2.imread('./images/closing.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('./images/dilation.png', cv2.IMREAD_GRAYSCALE)

eroded_image = perform_erosion(image1, structuring_element)
dilated_image = perform_dilation(image2, structuring_element)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
pl.title("Original Image1")
plt.imshow(image1, cmap='gray')

plt.subplot(2, 2, 2)
pl.title("Erosion Operation")
plt.imshow(eroded_image, cmap='gray')

plt.subplot(2, 2, 3)
pl.title("Original Image2")
plt.imshow(image2, cmap='gray')

plt.subplot(2, 2, 4)
pl.title("Dilation Operation")
plt.imshow(dilated_image, cmap='gray')

plt.tight_layout()
plt.show()
