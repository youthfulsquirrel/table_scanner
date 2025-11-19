import cv2
import numpy as np

def warp_table_manual(image, corners):
    """
    Warp table so that the 4 clicked corners map to the edges of the corrected image.
    corners: np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]) in any order
    Returns: warped image
    """
    # Auto-sort corners to TL, TR, BR, BL
    s = corners.sum(axis=1)
    diff = np.diff(corners, axis=1)

    tl = corners[np.argmin(s)]
    br = corners[np.argmax(s)]
    tr = corners[np.argmin(diff)]
    bl = corners[np.argmax(diff)]

    ordered = np.array([tl, tr, br, bl], dtype="float32")

    # Compute width and height based on average of opposite sides
    widthTop = np.linalg.norm(tr - tl)
    widthBottom = np.linalg.norm(br - bl)
    maxWidth = int((widthTop + widthBottom)/2)

    heightLeft = np.linalg.norm(bl - tl)
    heightRight = np.linalg.norm(br - tr)
    maxHeight = int((heightLeft + heightRight)/2)

    # Destination rectangle
    dst = np.array([
        [0,0],
        [maxWidth-1,0],
        [maxWidth-1,maxHeight-1],
        [0,maxHeight-1]
    ], dtype="float32")

    # Perspective transform
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def process_table(image_bytes, corners, cols=48):
    """
    Digitize the table after perspective warp
    Returns: matrix, column sums, overlay image
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    # Warp table using the submitted corners
    warped = warp_table_manual(img, corners)

    # Compute number of rows based on column width
    height, width = warped.shape[:2]
    cell_width = width / cols
    rows = int(round(height / cell_width))
    row_height = height / rows
    col_width = width / cols

    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    matrix = np.zeros((rows, cols), dtype=int)

    for r in range(rows):
        y1 = int(r*row_height)
        y2 = int((r+1)*row_height)
        for c in range(cols):
            x1 = int(c*col_width)
            x2 = int((c+1)*col_width)
            cell = hsv[y1:y2, x1:x2]
            S = cell[:,:,1].mean()
            V = cell[:,:,2].mean()
            matrix[r,c] = 1 if S > 35 or V < 160 else 0

    col_sums = matrix.sum(axis=0)
    overlay_img = draw_overlay(warped, matrix)

    return matrix, col_sums, overlay_img

def draw_overlay(image, matrix, alpha=0.3, border_color=(255,255,255), border_thickness=1):
    """
    Draw overlay with shaded cells + grid borders
    """
    overlay_img = image.copy()
    mask = np.zeros_like(image, dtype=np.uint8)
    rows, cols = matrix.shape
    height, width = image.shape[:2]
    row_height = height / rows
    col_width = width / cols

    for r in range(rows):
        y1 = int(r*row_height)
        y2 = int((r+1)*row_height)
        for c in range(cols):
            x1 = int(c*col_width)
            x2 = int((c+1)*col_width)
            color = (0,255,0) if matrix[r,c]==1 else (0,0,255)
            cv2.rectangle(mask, (x1,y1), (x2,y2), color, -1)

    overlay_img = cv2.addWeighted(mask, alpha, overlay_img, 1-alpha, 0)

    # Draw grid borders
    for r in range(rows+1):
        y = int(r*row_height)
        cv2.line(overlay_img, (0,y), (width,y), border_color, border_thickness)
    for c in range(cols+1):
        x = int(c*col_width)
        cv2.line(overlay_img, (x,0), (x,height), border_color, border_thickness)

    return overlay_img
