import cv2
import matplotlib.pyplot as plt

# Load the grayscale image
gray_image = cv2.imread('./images/img4.jpg', 0)

# Create a list to store sampled images
sampled_image = [gray_image.copy()]

# Create a figure for displaying the images
plt.figure(figsize=(9, 7))

# Display the original 8-bit image
plt.subplot(2, 4, 1)
plt.title('8 bit')
plt.imshow(gray_image, cmap='gray')  # Use cmap='gray' for grayscale images

# Iterate through decreasing bit depths
for k in range(7):
    # Right-shift each pixel by 1 bit to reduce bit depth
    gray_image //= 2
    sampled_image.append(gray_image.copy())

    # Display the sampled images with their respective bit depths
    plt.subplot(2, 4, k + 2)
    plt.title(f'({8 - (k + 1)} bit)')  # Adjusted the bit depth title
    plt.imshow(sampled_image[k + 1], cmap='gray')  # Use cmap='gray' for grayscale images

# Adjust layout for better visualization
plt.tight_layout()

# Show the plot
plt.show()
