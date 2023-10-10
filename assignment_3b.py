import cv2
import numpy as np

# Load the grayscale image (replace 'your_image.jpg' with the actual file path)
original_image = cv2.imread('./images/img4.jpg', cv2.IMREAD_GRAYSCALE)


# Add salt and pepper noise to the original image
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    total_pixels = image.size

    # Add salt noise
    num_salt = np.ceil(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    # Add pepper noise
    num_pepper = np.ceil(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image


salt_and_pepper_prob = 0.02  # Adjust the probability as needed
noisy_image = add_salt_and_pepper_noise(original_image, salt_and_pepper_prob, salt_and_pepper_prob)

# Define the kernel sizes for average filtering
kernel_sizes = [3, 5, 7]

# Initialize a dictionary to store PSNR values for each filter
psnr_values = {}

# Apply average filtering with different kernel sizes
for kernel_size in kernel_sizes:
    # Apply average filtering
    filtered_image = cv2.blur(noisy_image, (kernel_size, kernel_size))

    # Calculate PSNR
    mse = np.mean((original_image - filtered_image) ** 2)
    psnr = 10 * np.log10((255 ** 2) / mse)

    # Store PSNR value in the dictionary
    psnr_values[f'PSNR {kernel_size}x{kernel_size}'] = psnr

    # Display the filtered image
    cv2.imshow(f'Filtered Image {kernel_size}x{kernel_size}', filtered_image)

# Display the original noisy image
cv2.imshow('Noisy Image', noisy_image)

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print PSNR values
for kernel_size, psnr in psnr_values.items():
    print(f'{kernel_size}: {psnr} dB')
