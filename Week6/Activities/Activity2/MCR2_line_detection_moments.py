# Re-import necessary libraries due to kernel reset
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reload image
img = cv2.imread("images/line_1.jpg")
target_width, target_height = 640, 480
img = cv2.resize(img, (target_width, target_height))

# Crop the ROI from the bottom part of the image
roi = img[int(target_height * 0.4):, :]
roi_height, roi_width = roi.shape[:2]

# Create trapezoidal mask
top_width = int(roi_width * 0.9)
trapezoid = np.array([[((roi_width - top_width) // 2, 0),
                       ((roi_width + top_width) // 2, 0),
                       (roi_width, roi_height),
                       (0, roi_height)]], dtype=np.int32)
mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
cv2.fillPoly(mask, trapezoid, 255)

# Apply the mask
roi_masked = cv2.bitwise_and(roi, roi, mask=mask)

# Convert to grayscale and apply thresholding
gray = cv2.cvtColor(roi_masked, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary_inv = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

# Morphological operations
kernel = np.ones((3, 3), np.uint8)
morph = cv2.erode(binary_inv, kernel, iterations=3)
morph = cv2.dilate(morph, kernel, iterations=3)

# Connected Components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morph, connectivity=4)

output = roi.copy()
total_weight = 0
weighted_sum = np.array([0.0, 0.0])
MIN_AREA, MAX_AREA = 1000, 100000

for i in range(1, num_labels):  # skip background
    x, y, w, h, area = stats[i]
    cx, cy = centroids[i]

    if MIN_AREA <= area <= MAX_AREA:
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(output, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        cv2.putText(output, f"({int(cx)},{int(cy)})", (int(cx) + 5, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        weighted_sum += np.array([cx, cy]) * area
        total_weight += area

# Compute weighted average center (setpoint)
if total_weight > 0:
    avg_cx, avg_cy = (weighted_sum / total_weight).astype(int)
    cv2.circle(output, (avg_cx, avg_cy), 6, (0, 255, 255), -1)
    cv2.putText(output, "Setpoint", (avg_cx + 10, avg_cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

# Plotting

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
plt.title('Trapezoidal ROI')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(roi_masked, cv2.COLOR_BGR2RGB))
plt.title('Trapezoidal Masked ROI')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(binary_inv, cmap='gray')
plt.title("Thresholded")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(morph, cmap='gray')
plt.title("Morphological Cleaned")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Connected Components + Setpoint")
plt.axis('off')


plt.tight_layout()
plt.show()
