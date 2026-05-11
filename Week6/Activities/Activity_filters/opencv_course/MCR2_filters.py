import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# -------------------------------
# Section 1: Sharpening Filter
# -------------------------------

# Load and verify image
img = cv.imread('images/Puzzlebot_hand.png')
assert img is not None, "Image not found!"
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Define sharpening kernel
sharpen_kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])

# Apply sharpening filter using filter2D
sharpened_img = cv.filter2D(img, -1, sharpen_kernel)

# Display original and sharpened image
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(sharpened_img, cv.COLOR_BGR2RGB))
plt.title('Sharpened')
plt.axis('off')
plt.tight_layout()
plt.show()

# -------------------------------
# Section 2: Salt & Pepper Noise + Median Filter
# -------------------------------

# Function to add salt and pepper noise to an image
def add_salt_pepper_noise(image, prob=0.02):
    noisy = image.copy()
    total_pixels = image.size // 3  # For RGB
    num_salt = int(prob * total_pixels / 2)
    num_pepper = int(prob * total_pixels / 2)

    # Add white (salt) pixels
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = [255, 255, 255]

    # Add black (pepper) pixels
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = [0, 0, 0]

    return noisy

# Add salt and pepper noise
noisy_sp_img = add_salt_pepper_noise(img_rgb, prob=0.02)

# Denoise using Median Blur
median_denoised = cv.medianBlur(noisy_sp_img, 5)

# Display results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(noisy_sp_img)
plt.title('Noisy (Salt & Pepper)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(median_denoised)
plt.title('Denoised (Median Blur)')
plt.axis('off')
plt.tight_layout()
plt.show()

# -------------------------------
# Section 3: Gaussian Noise + Gaussian Blur
# -------------------------------

# Function to add Gaussian noise
def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

# Add Gaussian noise
gaussian_noisy_img = add_gaussian_noise(img_rgb)

# Denoise using Gaussian Blur
gaussian_denoised_img = cv.GaussianBlur(gaussian_noisy_img, (5, 5), 0)

# Display results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(gaussian_noisy_img)
plt.title('Noisy (Gaussian)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gaussian_denoised_img)
plt.title('Denoised (Gaussian Blur)')
plt.axis('off')
plt.tight_layout()
plt.show()

# -------------------------------
# Section 4: Gaussian Noise + Averaging Filter
# -------------------------------

# Reuse previous Gaussian noisy image
# Denoise using Averaging Filter (mean filter)
average_denoised_img = cv.blur(gaussian_noisy_img, (11, 11))

# Display results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(gaussian_noisy_img)
plt.title('Noisy (Gaussian)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(average_denoised_img)
plt.title('Denoised (Average Blur)')
plt.axis('off')
plt.tight_layout()
plt.show()