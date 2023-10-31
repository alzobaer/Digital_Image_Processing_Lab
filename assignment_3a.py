import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import median


# apply salt and pepper noise in the image
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


# calculate Peak signal-to-noise ratio (PSNR)
def cal_psnr(original_img, noisy_image):
    original_img = original_img.astype(np.float64)
    noisy_image = noisy_image.astype(np.float64)
    # calculate the mean square error between the original and noisy image
    mse = np.mean((original_img - noisy_image) ** 2)
    max_pixel_value = 255  # maximum possible pixel value
    # Calculate PSNR = 10log((max^2)/mse)
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr


# apply average filtering to the noisy image with kernel size 5
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


# apply median filtering to the noisy image with kernel size 5
def median_filter(image, kernel_size):
    height, width = image.shape
    filtered_image = image.copy()

    # Define the padding size based on the kernel size
    padding = kernel_size // 2

    for i in range(padding, height - padding):
        for j in range(padding, width - padding):
            # Extract the neighborhood around the pixel
            neighborhood = image[i - padding:i + padding + 1, j - padding:j + padding + 1]

            # Calculate the median value of the neighborhood and set it as the filtered pixel value
            median_value = median(neighborhood)
            filtered_image[i, j] = median_value

    return filtered_image


# read a grayscale image
original_img = cv2.imread('./images/img4.jpg', cv2.IMREAD_GRAYSCALE)

# make an image with salt and pepper noise
noisy_image = original_img.copy()  # storing the noisy image
noisy_image = add_noise(noisy_image)  # add noise to the original image

# calculate signal-to-noise ratio for original and noisy image
psnr_noisy = cal_psnr(original_img, noisy_image)

# calculate average filter
avg_filtered_img = average_filter(noisy_image, 5)
psnr_avg = cal_psnr(original_img, avg_filtered_img)

# calculate median filter
median_filtered_img = median_filter(noisy_image, 5)
psnr_median = cal_psnr(original_img, median_filtered_img)

# Display the results
plt.figure(figsize=(12, 8))
# display original image
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(original_img, cmap='gray')

# display noisy image
plt.subplot(2, 2, 2)
plt.title('Noisy Image (PSNR = {:.2f} DB)'.format(psnr_noisy))
plt.imshow(noisy_image, cmap='gray')

# display average filtered image
plt.subplot(2, 2, 3)
plt.title('Average Filtered Image (PSNR = {:.2f} DB)'.format(psnr_avg))
plt.imshow(avg_filtered_img, cmap='gray')

# display median filtered image
plt.subplot(2, 2, 4)
plt.title('Median filtered Image (PSNR = {:.2f} DB)'.format(psnr_median))
plt.imshow(median_filtered_img, cmap='gray')

plt.tight_layout()
plt.show()
