import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_ubyte

# --- Load image ---
img = cv2.imread("table.jpeg")
orig = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Step 1: Adaptive Thresholding ---
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 5)
thresh = cv2.medianBlur(thresh, 3)

# --- Step 2: Strong Dilation to connect faint lines ---
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
dilated = cv2.dilate(thresh, dilate_kernel, iterations=1)

# --- Step 3: Horizontal and Vertical Line Detection ---
height, width = dilated.shape

hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width//12, 1))
hor_lines = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, hor_kernel)

ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height//12))
ver_lines = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, ver_kernel)

# --- Step 4: Edge Detection + Skeletonization for curved lines ---
edges = cv2.Canny(gray, 50, 150)
edges_skeleton = skeletonize(edges//255)
edges_skeleton = img_as_ubyte(edges_skeleton * 255)

# --- Step 5: Combine all masks ---
combined_mask = cv2.add(hor_lines, ver_lines)
combined_mask = cv2.add(combined_mask, edges_skeleton)

# Optional: small dilation to strengthen combined mask
combined_mask = cv2.dilate(combined_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)

# --- Step 6: Convert mask to 3-channel for overlay ---
mask_color = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
mask_color[:, :, 1] = 0
mask_color[:, :, 0] = 0  # Red channel only

# --- Step 7: Overlay on original image ---
overlay = cv2.addWeighted(orig, 0.7, mask_color, 0.3, 0)


# --- Show result ---
cv2.imshow("Robust Table Overlay", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
