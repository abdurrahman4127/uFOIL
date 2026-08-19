import cv2
import numpy as np


def canny_edges(
    img: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> np.ndarray:
    """
    args:
        img: BGR or grayscale image.
        low_threshold: lower hysteresis threshold.
        high_threshold: upper hysteresis threshold.

    returns:
        binary edge image, shape (H, W).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.Canny(gray, low_threshold, high_threshold)


def detect_horizontal_lines(
    edges: np.ndarray,
    min_line_length_ratio: float = 0.5,
) -> list[tuple[int, int, int, int]]:
    """
    find near-horizontal lines long enough to be table borders.

    args:
        edges: binary edge image, shape (H, W).
        min_line_length_ratio: minimum line length as a fraction of
            image width, shorter lines are ignored.

    returns:
        list of (x1, y1, x2, y2) line segments.
    """
    h, w = edges.shape
    min_length = int(w * min_line_length_ratio)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=min_length,
        maxLineGap=10,
    )

    if lines is None:
        return []

    horizontal = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 5:
            horizontal.append((x1, y1, x2, y2))

    return horizontal


def detect_table_boundary(edges: np.ndarray) -> int:
    """
    find the y-coordinate where the marks table starts.

    table header border is (in our case) the first long horizontal line
    in the bottom half of the page, so lines from the top half of the
    page are ignored to avoid mistaking the student-info box grid for
    the table.
        
    """
    h = edges.shape[0]
    lines = detect_horizontal_lines(edges)

    candidates = [y1 for (_, y1, _, _) in lines if y1 > h * 0.4]

    # y-coordinate of table's top boundary. 
    # defaults to 60% of img height if no clear boundary line is found.
    if not candidates:
        return int(h * 0.6)  

    return min(candidates)


def split_image(
    img: np.ndarray,
    boundary_y: int,
) -> tuple[np.ndarray, np.ndarray]:
    # split an image horizontally at the given y-coordinate.
    upper = img[:boundary_y, :]
    lower = img[boundary_y:, :]
    return upper, lower


def separate_sections(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = canny_edges(img)
    boundary_y = detect_table_boundary(edges)
    return split_image(img, boundary_y)