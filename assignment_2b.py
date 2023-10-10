import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image (you need to replace 'your_image_path' with the actual path to your image)
original_image1 = cv2.imread('./images/city.png', cv2.IMREAD_GRAYSCALE)
original_image2 = cv2.imread('./images/point_light.png', cv2.IMREAD_GRAYSCALE)
original_image3 = cv2.imread('./images/Einstain_white.jpg', cv2.IMREAD_GRAYSCALE)

# Define the parameters for the power-law transform
gamma = 5.0  # Adjust this value to control the transformation
power_law_image = np.power(original_image1 / 255.0, gamma) * 255.0
power_law_image = power_law_image.astype(np.uint8)

# Define the parameters for the logarithmic transform
c1 = 255.0 / np.log10(1 + original_image2.max())   # Adjust this value to control the transformation
log_image = c1 * np.log10(1 + original_image2)
log_image = log_image.astype(np.uint8)

# Define the parameters for the inverse logarithmic transform
c2 = 255.0 / np.log10(1 + original_image3.max())   # Adjust this value to control the transformation
inverse_log_image = np.power(10, (original_image3 / c2)) - 1
inverse_log_image = inverse_log_image.astype(np.uint8)

# Display the original image and both transformed images
plt.figure(figsize=(12, 8))

# display original image for power transformation
plt.subplot(3, 2, 1)
plt.title('Original Image')
plt.imshow(original_image1, cmap='gray')
# display power-law transformed image
plt.subplot(3, 2, 2)
plt.title('Power Law Transform (gamma={:.2f})'.format(gamma))
plt.imshow(power_law_image, cmap='gray')

# display original image for power transformation
plt.subplot(3, 2, 3)
plt.title('Original Image 2')
plt.imshow(original_image2, cmap='gray')
# display original image for log transformation
plt.subplot(3, 2, 4)
plt.title('Logarithmic Transform (c1={:.2f})'.format(c1))
plt.imshow(log_image, cmap='gray')

# display original image for power transformation
plt.subplot(3, 2, 5)
plt.title('Original Image 3')
plt.imshow(original_image3, cmap='gray')
# display original image for log transformation
plt.subplot(3, 2, 6)
plt.title('Inverse Logarithmic Transform (c2={:.2f})'.format(c2))
plt.imshow(inverse_log_image, cmap='gray')


plt.tight_layout()
plt.show()
