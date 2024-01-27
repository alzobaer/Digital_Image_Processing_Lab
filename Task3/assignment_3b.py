import random
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Add salt and pepper noise
def add_noise(img):
    row, col = img.shape
    # salt noise
    number_of_pixels = random.randint(300, 10000)  # Pick a random number between 300 and 10000 to be white pixel
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)  # Pick a random y coordinate
        x_coord = random.randint(0, col - 1)  # Pick a random x coordinate
        img[y_coord][x_coord] = 255  # Color that pixel to white
    # pepper noise
    number_of_pixels = random.randint(300, 10000)  # Pick a random number between 300 and 10000
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)  # Pick a random y coordinate
        x_coord = random.randint(0, col - 1)  # Pick a random x coordinate
        img[y_coord][x_coord] = 0  # Color that pixel to black
    return img


# Function to apply average filter with a specified kernel size
def average_filter(image, kernel_size):
    height, width = image.shape
    filtered_image = image.copy()

    # Define the padding size based on the kernel size
    padding = kernel_size // 2
    mask = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

    for i in range(padding, height - padding):
        for j in range(padding, width - padding):
            # Extract the neighborhood around the pixel
            neighborhood = image[i - padding: i + padding + 1, j - padding: j + padding + 1]
            conv_res = neighborhood * mask  # convolution result

            # Calculate the average value of the neighborhood
            average_value = np.sum(conv_res)

            # Set the filtered pixel value to the average
            filtered_image[i, j] = average_value
    return filtered_image


# Calculate PSNR between two images
def calculate_psnr(original_img, noisy_image):
    original_img = original_img.astype(np.float64)
    noisy_image = noisy_image.astype(np.float64)
    # calculate the mean square error between the original and noisy image
    mse = np.sum((original_img - noisy_image) ** 2) * (1 / (original_img.shape[0] * original_img.shape[1]))
    max_pixel_value = 255  # maximum possible pixel value
    # Calculate PSNR = 10log((max^2)/mse)
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr


# Load the grayscale image
image = cv2.imread('./images/img4.jpg', cv2.IMREAD_GRAYSCALE)
noisy_image = image.copy()
noisy_image = add_noise(noisy_image)


# Apply average filter with different kernel sizes and calculate PSNR
kernel_sizes = [3, 5, 7, 9]


# Display the results
plt.figure(figsize=(12, 8))

# Display the original image
plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

# Display the noisy image
plt.subplot(2, 3, 2)
psnr = calculate_psnr(image, noisy_image)
plt.title(f'Noisy Image Filter (PSNR = {psnr:.2f} dB)')
plt.imshow(noisy_image, cmap='gray')

# Display the filtered images with different kernel sizes
for i, kernel_size in enumerate(kernel_sizes):
    filtered_image = average_filter(noisy_image, kernel_size)
    # print(kernel_size)
    psnr = calculate_psnr(image, filtered_image)
    plt.subplot(2, 3, i + 3)
    plt.title(f'{kernel_size}x{kernel_size} Filter (PSNR = {psnr:.2f} dB)')
    plt.imshow(filtered_image, cmap='gray')

plt.tight_layout()
plt.show()
