import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
img = cv2.imread('images/line_1.jpg')
assert img is not None, "Image not found!"

# Desired image dimensions
target_width, target_height = 640, 480

# Resize image if necessary
height, width = img.shape[:2]
if (width, height) != (target_width, target_height):
    resized_img = cv2.resize(img, (target_width, target_height))
    print("Image resized to 640x480.")
else:
    resized_img = img
    print("Image already 640x480, no resizing needed.")

# Extract bottom 25% region of interest (ROI)
roi = resized_img[int(target_height * 0.75):, :]

# Convert ROI to grayscale
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

# Threshold to binary inverse image (high contrast for edges)
_, binary_inv = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

# Apply morphological operations to enhance features
kernel = np.ones((3, 3), np.uint8)
morph = cv2.erode(binary_inv, kernel, iterations=3)
morph = cv2.dilate(morph, kernel, iterations=3)

# Calculate vertical sum (projection on horizontal axis)
detection = np.sum(morph, axis=0, dtype=np.float32)

# Compute first and second gradients
first_grad = np.gradient(detection)
second_grad = np.gradient(first_grad)

# Set dynamic thresholds to filter out insignificant edges
line_thresh = 0.1
thresh_pos = np.mean(first_grad) + line_thresh * (np.max(first_grad) - np.min(first_grad))
thresh_neg = np.mean(first_grad) - line_thresh * (np.max(first_grad) - np.min(first_grad))

# Identify potential left and right edges based on thresholds
right_grad = np.where(first_grad > thresh_pos, second_grad, 0)
left_grad = np.where(first_grad < thresh_neg, second_grad, 0)

# Zero negative gradient values (removing false edges)
right_grad[right_grad < 0] = 0
left_grad[left_grad < 0] = 0

# Detect edge positions by comparing with shifted gradients
left_shift = np.roll(left_grad, 1)
right_shift = np.roll(right_grad, 1)
left_shift[0], right_shift[0] = 0, 0

left_edges = np.argwhere(left_shift < left_grad).flatten()
right_edges = np.argwhere(right_shift < right_grad).flatten()

# Visualization: Image processing steps
plt.figure(figsize=(14, 8))
titles = ['Original', 'Resized', 'Gray (ROI)', 'Blurred', 'Binary Threshold', 'Morphological Result']
images = [img, resized_img, gray, blurred, binary_inv, morph]

for i in range(6):
    plt.subplot(2, 3, i+1)
    if i < 2:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()

# Visualization: Gradient analysis and detected edges
plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.plot(detection)
plt.title('Vertical Projection (Detection)')

plt.subplot(2, 3, 2)
plt.plot(first_grad)
plt.title('First Gradient')

plt.subplot(2, 3, 3)
plt.plot(second_grad)
plt.title('Second Gradient')

plt.subplot(2, 3, 4)
plt.plot(right_grad, label='Right Gradient')
plt.plot(right_shift, '--', label='Right Shifted')
plt.title('Right Edge Detection')
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(left_grad, label='Left Gradient')
plt.plot(left_shift, '--', label='Left Shifted')
plt.title('Left Edge Detection')
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(left_grad, label='Left Gradient')
plt.plot(right_grad, label='Right Gradient')
plt.scatter(left_edges, [np.max(left_grad)*0.1]*len(left_edges), color='red', marker='o', label='Left Edges')
plt.scatter(right_edges, [np.max(right_grad)*0.1]*len(right_edges), color='blue', marker='x', label='Right Edges')
plt.title('Detected Edges')
plt.legend()

plt.tight_layout()
plt.show()