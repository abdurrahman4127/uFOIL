import cv2
import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# binarize an image using Otsu's method.
def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


# find external contours in a binary image.
def extract_contours(binary: np.ndarray) -> list[np.ndarray]:

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return list(contours)


def largest_contour(contours: list[np.ndarray]) -> np.ndarray | None:
    # pick contour with the biggest enclosed area.
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def crop_to_contour(
    img: np.ndarray,
    contour: np.ndarray,
    padding: int = 10,
) -> np.ndarray:
    x, y, w, h = cv2.boundingRect(contour)
    img_h, img_w = img.shape[:2]

    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(img_w, x + w + padding)
    y1 = min(img_h, y + h + padding)

    return img[y0:y1, x0:x1]


def isolate_script(img: np.ndarray, padding: int = 10) -> np.ndarray:
    gray = to_grayscale(img)
    binary = otsu_threshold(gray)
    contours = extract_contours(binary)

    page_contour = largest_contour(contours)
    if page_contour is None:
        return img

    return crop_to_contour(img, page_contour, padding=padding)