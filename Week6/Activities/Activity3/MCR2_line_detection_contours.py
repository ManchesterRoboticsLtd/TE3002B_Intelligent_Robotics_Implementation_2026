import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
img = cv2.imread('images/line_1.jpg')
assert img is not None, "Image not found!"

# Set target dimensions
target_width, target_height = 640, 480

# Resize image if needed
height, width = img.shape[:2]
if (width, height) != (target_width, target_height):
    resized_img = cv2.resize(img, (target_width, target_height))
    print("Image resized to 640x480.")
else:
    resized_img = img.copy()
    print("Image already 640x480.")

# --- Step 1: Create trapezoidal mask ---

#roi = resized_img[int(target_height*0.0):, :]
roi = resized_img[int(target_height * 0.5):, :]

roi_height, roi_width = roi.shape[:2]

mask = np.zeros((roi_height, roi_width), dtype=np.uint8)

top_width = int(roi_width * 0.9)
trapezoid = np.array([[
    ((roi_width - top_width) // 2, int(roi_height * 0.0)),  # top-left
    ((roi_width + top_width) // 2, int(roi_height * 0.0)),  # top-right
    (roi_width, roi_height),                                # bottom-right
    (0, roi_height)                                            # bottom-left
]], dtype=np.int32)

cv2.fillPoly(mask, trapezoid, 255)
roi = cv2.bitwise_and(roi, roi, mask=mask)

# --- Step 2: Preprocess image ---
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary_inv = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

# Morphological operations
kernel = np.ones((3, 3), np.uint8)
morph = cv2.erode(binary_inv, kernel, iterations=3)
morph = cv2.dilate(morph, kernel, iterations=3)

# Canny edge detection
canny_edges = cv2.Canny(morph, 50, 150)

side_crop_percent = 0.05
crop_x = int(roi_width * side_crop_percent)

# Create a mask that only keeps the center part
side_mask = np.zeros_like(canny_edges)
cv2.rectangle(
    side_mask,
    (crop_x, 0),  # top-left corner
    (roi_width - crop_x, roi_height),  # bottom-right corner
    255,  # white region
    thickness=-1
)

# Apply side mask
canny_edges = cv2.bitwise_and(canny_edges, canny_edges, mask=side_mask)

# Apply trapezoidal mask to Canny output
all_edges_roi = cv2.bitwise_and(canny_edges, canny_edges, mask=mask)

# --- Step 3: Find and process contours ---
contours, _ = cv2.findContours(all_edges_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = roi.copy()
epsilon_factor = 0.1

for cnt in contours:
    # Approximate polygon
    epsilon = epsilon_factor * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # Bounding rectangle and center
    x, y, w, h = cv2.boundingRect(approx)
    cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box

    box_area = w * h

    # Filter by bounding box area
    if box_area < 1000 or box_area > 100000:
        continue

    cx = x + w // 2
    cy = y + h // 2
    cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)               # Red center
    cv2.putText(output, f"({cx},{cy})", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # Draw the approximated polygon
    cv2.polylines(output, [approx], isClosed=True, color=(0, 255, 0), thickness=2)

# Convert to RGB for matplotlib
output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

# --- Step 4: Display results ---
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
plt.title('Trapezoidal ROI')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(morph, cmap='gray')
plt.title('Morphological Result')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(all_edges_roi, cmap='gray')
plt.title('Masked Edges')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(output_rgb)
plt.title('Bounding Boxes + Centers')
plt.axis('off')

plt.tight_layout()
plt.show()