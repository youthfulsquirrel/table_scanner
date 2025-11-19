import cv2
import numpy as np

# ===== 1. Load & deskew (basic) =====
img = cv2.imread("table.jpg")
orig = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Light blur helps remove noise
gray_blur = cv2.GaussianBlur(gray, (5,5), 0)

# ===== 2. Detect strong vertical lines (10 groups) =====
th = cv2.adaptiveThreshold(gray_blur, 255,
                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv2.THRESH_BINARY_INV, 15, 3)

kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
v_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_v)

cnts, _ = cv2.findContours(v_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

x_positions = []
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if h > img.shape[0] * 0.2:     # real column divider
        x_positions.append(x)

x_positions = sorted(x_positions)

# Add image left & right edges
x_positions = [0] + x_positions + [img.shape[1]]

# ===== 3. Subdivide each visible col into 4 equal parts =====
col_bounds = []
for i in range(len(x_positions)-1):
    start = x_positions[i]
    end = x_positions[i+1]
    width = (end - start) / 4
    for k in range(4):
        col_bounds.append(int(start + k * width))
col_bounds.append(img.shape[1])  # last boundary

# Clean & sort
col_bounds = sorted(col_bounds)
COLS = 40     # total sub-columns
assert len(col_bounds) == COLS + 1

# ===== 4. Detect horizontal lines (rows) =====
kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
h_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_h)

cnts_h, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

y_positions = []
for c in cnts_h:
    x, y, w, h = cv2.boundingRect(c)
    if w > img.shape[1] * 0.5:  # long horizontal line
        y_positions.append(y)

y_positions = sorted(y_positions)
ROWS = len(y_positions) - 1     # row count inferred
row_bounds = sorted(y_positions)

# ===== 5. Create matrix and classify each cell =====
matrix = np.zeros((ROWS, COLS), dtype=int)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

for r in range(ROWS):
    y1 = row_bounds[r]
    y2 = row_bounds[r+1]

    for c in range(COLS):
        x1 = col_bounds[c]
        x2 = col_bounds[c+1]

        cell = hsv[y1:y2, x1:x2]

        # Compute mean saturation (colored cells very high)
        S = cell[:,:,1].mean()

        # Compute mean brightness
        V = cell[:,:,2].mean()

        # Threshold for any shaded region
        if S > 35 or V < 160:  
            matrix[r, c] = 1
        else:
            matrix[r, c] = 0

# ===== 6. Column totals =====
col_sums = matrix.sum(axis=0)
print("Column totals =", col_sums)