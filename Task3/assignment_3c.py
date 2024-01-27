import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt


# Function to add salt and pepper noise to the image
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    total_pixels = image.size
    num_salt = int(total_pixels * salt_prob)
    num_pepper = int(total_pixels * pepper_prob)

    # Add salt noise
    salt_coords = [np.random.randint(0, size, num_salt) for size in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    # Add pepper noise
    pepper_coords = [np.random.randint(0, size, num_pepper) for size in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image


# Function to apply harmonic mean filter with a specified kernel size
def apply_harmonic_mean_filter(image, kernel_size):
    height, width = image.shape
    filtered_image = np.zeros_like(image, dtype=np.float64)

    # Calculate the kernel radius
    kernel_radius = kernel_size // 2

    for i in range(kernel_radius, height - kernel_radius):
        for j in range(kernel_radius, height - kernel_radius):
            values = []

            # Iterate over the neighborhood
            for x in range(i - kernel_radius, i + kernel_radius + 1):
                for y in range(j - kernel_radius, j + kernel_radius + 1):
                    if image[x, y] != 0:
                        values.append(1 / image[x, y])
            # Calculate the harmonic mean
            if values:
                filtered_image[i, j] = int(len(values) / np.sum(values))

    return filtered_image


# Function to apply geometric mean filter with a specified kernel size
def apply_geometric_mean_filter(image, kernel_size):
    height, width = image.shape
    filtered_image = np.zeros_like(image)

    # Calculate the kernel radius
    kernel_radius = kernel_size // 2

    for i in range(kernel_radius, height - kernel_radius):
        for j in range(kernel_radius, width - kernel_radius):
            values = []
            # Iterate over the neighborhood
            for x in range(i - kernel_radius, i + kernel_radius + 1):
                for y in range(j - kernel_radius, j + kernel_radius + 1):
                    if image[x, y] != 0:
                        values.append(image[x, y])
            # Calculate the geometric mean
            if values:
                product = np.prod(values)
                filtered_image[i, j] = product ** (1 / len(values))

    return filtered_image


# Function to calculate PSNR between two images
def calculate_psnr(original_img, noisy_image):
    original_img = original_img.astype(np.float64)
    noisy_image = noisy_image.astype(np.float64)
    # calculate the mean square error between the original and noisy image
    mse = np.mean((original_img - noisy_image) ** 2)
    max_pixel_value = 255  # maximum possible pixel value
    # Calculate PSNR = 10log((max^2)/mse)
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr


# Load the grayscale image
image = cv2.imread('./images/img4.jpg', cv2.IMREAD_GRAYSCALE)

salt_prob = 0.05  # Adjust the probability as needed
pepper_prob = 0.05  # Adjust the probability as needed

noisy_image = image.copy()
noisy_image = add_salt_and_pepper_noise(noisy_image, salt_prob, pepper_prob)

# List of kernel sizes for filtering
kernel_sizes = [3, 5, 7]

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

# Display the filtered images with harmonic and geometric mean filters
for i, kernel_size in enumerate(kernel_sizes):
    image = image.astype(np.float64)
    noisy_image = noisy_image.astype(np.float64)
    harmonic_filtered_image = apply_harmonic_mean_filter(noisy_image, kernel_size)
    geometric_filtered_image = apply_geometric_mean_filter(noisy_image, kernel_size)

    psnr_harmonic = calculate_psnr(image, harmonic_filtered_image)
    psnr_geometric = calculate_psnr(image, geometric_filtered_image)

    plt.subplot(2, 2, i + 2)
    plt.title(f'{kernel_size}x{kernel_size} Filters\nHarmonic PSNR = {psnr_harmonic:.2f} dB\nGeometric PSNR = {psnr_geometric:.2f} dB')
    plt.imshow(harmonic_filtered_image, cmap='gray')
    plt.imshow(geometric_filtered_image, cmap='gray')

plt.tight_layout()
plt.show()
