import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the input image in grayscale
img1 = cv2.imread('./images/img4.jpg', 0)   # take image of size 512x512

# Get the dimensions of the image (height and width)
[m, n] = img1.shape     # m = height, n = width
print('Image Shape:', m, n)

# Initialize the down-sampling rate
f = 2
# Iterate through 8 levels of down-sampling
for k in range(8):
    # Down-sample the image
    # resized_image = img1[::f, ::f]
    resized_image = np.zeros((m // f, n // f))

    # Iterate through the original image pixels and copy every f-th pixel
    for i in range(0, m, f):
        for j in range(0, n, f):
            resized_image[i // f][j // f] = img1[i][j]

    # Plot the down-sampled image
    plt.subplot(2, 4, k+1)
    plt.title(f'({m//f}x{n//f})')  # Set the title with the dimensions
    plt.imshow(resized_image, cmap='gray')

    # Increase the down-sampling rate when any other key is pressed
    f = f * 2

# Arrange the subplots neatly
plt.tight_layout()
# Display the plot
plt.show()
