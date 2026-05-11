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

   # Canny edge detection
canny_edges = cv2.Canny(morph, 50, 150)

    # Side crop mask
CROP_SIDE_PERCENT = 0.05
crop_x = int(roi_width * CROP_SIDE_PERCENT)
side_mask = np.zeros_like(canny_edges)
cv2.rectangle(side_mask, (crop_x, 0), (roi_width - crop_x, roi_height), 255, thickness=-1)
canny_edges = cv2.bitwise_and(canny_edges, canny_edges, mask=side_mask)

# Final mask with trapezoid
edges_roi = cv2.bitwise_and(canny_edges, canny_edges, mask=mask)

# Detect lines using HoughLinesP
lines = cv2.HoughLinesP(
    edges_roi,
    rho=1,
    theta=np.pi / 180,
    threshold=50,
    minLineLength=5,
    maxLineGap=50
)

output = roi.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 3)


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
plt.imshow(canny_edges, cmap='gray')
plt.title("Edges")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Hugh Lines")
plt.axis('off')

plt.tight_layout()
plt.show()
