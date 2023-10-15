import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
image = cv2.imread('./images/einstein.jpeg', cv2.IMREAD_GRAYSCALE)


# Add salt and pepper noise
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    total_pixels = image.size
    num_salt = int(total_pixels * salt_prob)
    num_pepper = int(total_pixels * pepper_prob)

    # Add salt noise
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    # Add pepper noise
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image


salt_prob = 0.02  # Adjust the probability as needed
pepper_prob = 0.02  # Adjust the probability as needed

noisy_image = add_salt_and_pepper_noise(image, salt_prob, pepper_prob)


# Function to apply average filter with a specified kernel size
def apply_average_filter(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))


# Calculate PSNR between two images
def calculate_psnr(original, noisy):
    original = original.astype(np.float64)
    noisy = noisy.astype(np.float64)
    mse = np.mean((original - noisy) ** 2)
    max_pixel_value = 255
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr


# Apply average filter with different kernel sizes and calculate PSNR
kernel_sizes = [3, 5, 7]

for kernel_size in kernel_sizes:
    filtered_image = apply_average_filter(noisy_image, kernel_size)
    psnr = calculate_psnr(image, filtered_image)
    print(f'PSNR for {kernel_size}x{kernel_size} filter: {psnr}')

# Display the results
plt.figure(figsize=(12, 8))

# Display the original image
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

# Display the noisy image
plt.subplot(2, 2, 2)
plt.title('Noisy Image')
plt.imshow(noisy_image, cmap='gray')

# Display the filtered images with different kernel sizes
for i, kernel_size in enumerate(kernel_sizes):
    filtered_image = apply_average_filter(noisy_image, kernel_size)
    psnr = calculate_psnr(image, filtered_image)
    plt.subplot(2, 2, i + 2)
    plt.title(f'{kernel_size}x{kernel_size} Filter (PSNR = {psnr:.2f} dB)')
    plt.imshow(filtered_image, cmap='gray')

plt.tight_layout()
plt.show()
