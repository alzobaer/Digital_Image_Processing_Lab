import cv2
import matplotlib.pyplot as plt

# Load a grayscale image of size 512x512 (you need to replace 'your_image_path' with the actual path to your image)
gray_image = cv2.imread('./images/moon.jpg', cv2.IMREAD_GRAYSCALE)

# Define the range of gray levels to enhance (e.g., 50 to 150)
min_range = 0
max_range = 40

# Apply brightness enhancement to the specified range
enhanced_image = gray_image.copy()
enhanced_image[(gray_image >= min_range) & (gray_image <= max_range)] += 90  # You can adjust the enhancement value
# enhanced_image [enhanced_image>100] = 255
# enhanced_image [enhanced_image<100] = 0

# display size
plt.figure(figsize=(12, 6))

# display the original image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(gray_image, cmap='gray')

# display the enhanced image
plt.subplot(1, 2, 2)
plt.title('Enhanced Image')
plt.imshow(enhanced_image, cmap='gray')

plt.tight_layout()
plt.show()
