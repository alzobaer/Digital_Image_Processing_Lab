import cv2
import matplotlib.pyplot as plt
# Load the grayscale image
original_image = cv2.imread('./images/einstein.jpeg', cv2.IMREAD_GRAYSCALE)

# Keep only the last three bits (most significant bits)
last_three_bits_image = original_image & 0xE0  # 0xE0 is the binary mask(MSB) 11100000b or 224d

# Compute the difference image
difference_image = original_image - last_three_bits_image

plt.figure(figsize=(10, 8))
# display original image
plt.subplot(2, 2, 1)
plt.title('original image')
plt.imshow(original_image, cmap='gray')

# display last 3 bits image
plt.subplot(2, 2, 2)
plt.title('3 bit sliced image')
plt.imshow(last_three_bits_image, cmap='gray')

# display the difference image
plt.subplot(2, 2, (3, 4))
plt.title('difference image')
plt.imshow(difference_image, cmap='gray')


plt.tight_layout()
plt.show()
