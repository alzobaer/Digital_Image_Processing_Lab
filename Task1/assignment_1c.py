import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Function to generate a histogram from an image
def histogram_generate(image):
    histogram = np.zeros(256, dtype=int)
    for row in image:
        for pixel_value in row:
            histogram[pixel_value] += 1
    return histogram


# Load the grayscale image
gray_image = cv.imread('images/img4.jpg', 0)

# Generate a histogram for the grayscale image
histogram = histogram_generate(gray_image)

# Set the threshold for binary thresholding
threshold = 128

# Apply binary thresholding to create a segmented binary image
segmented_image = (gray_image > threshold).astype(np.uint8) * 255

# Create a 2x2 subplot grid for displaying images and the histogram
plt.figure(figsize=(8, 7))

# Display the histogram of the grayscale image
plt.subplot(2, 2, (1, 2))
plt.bar(range(256), histogram)
plt.title('Histogram')

# Display the original grayscale image
plt.subplot(2, 2, 3)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')

# Display the segmented binary image
plt.subplot(2, 2, 4)
plt.imshow(segmented_image, cmap='gray')
plt.title(f'Binary Threshold : {threshold}')

plt.tight_layout()  # Adjust layout for better visualization
plt.show()  # Show the plot
