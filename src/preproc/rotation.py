import cv2
import numpy as np
import pytesseract


# handles the case where the whole photo is sideways/upside-down (90,
# 180, 270 degrees) - detect_skew below only catches small tilt within
# +-45 degrees, a fully rotated image falls outside that range entirely.
#
# originally this trusted pytesseract's OSD "Rotate:" value directly,
# but tested on real phone photos it was unreliable - two photos that
# both needed the same actual correction got opposite Rotate values
# from OSD, with low confidence on both. tesseract's OSD seems to
# struggle on busy backgrounds/table-heavy pages like these scripts.
# brute-forcing all 4 orientations and picking whichever one tesseract
# can actually read text from confidently is slower but much more
# reliable on this kind of image.
def _text_confidence_score(img: np.ndarray) -> float:
    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractError:
        return -1.0

    confs = [float(c) for c in data["conf"] if float(c) >= 0]
    return sum(confs) if confs else 0.0


def correct_orientation(img: np.ndarray) -> np.ndarray:
    candidates = {
        0: img,
        90: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(img, cv2.ROTATE_180),
        270: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }

    best_angle = 0
    best_score = -1.0
    for angle, candidate in candidates.items():
        score = _text_confidence_score(candidate)
        if score > best_score:
            best_score = score
            best_angle = angle

    return candidates[best_angle]


def detect_skew(edges: np.ndarray) -> float:
    """
    estimate skew angle from an edge image using Hough lines.

    args:
        edges: binary edge image, shape (H, W).

    returns:
        estimated skew angle in degrees.
    """
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)
    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = (theta * 180 / np.pi) - 90
        # only count near-horizontal lines, vertical ones skew the avg
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def compute_avg_text_region_angle(boxes: list[tuple[int, int, int, int]]) -> float:
    """
    compute the avg orientation of detected text bounding boxes.

    args:
        boxes: list of (x, y, w, h) boxes from an OCR detector.

    returns:
        avg angle in degrees, 0.0 if no boxes given.
    """
    if not boxes:
        return 0.0

    angles = [np.degrees(np.arctan2(h, w)) for (_, _, w, h) in boxes]
    return float(np.mean(angles))


def refine_rotation(
    img: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """
    apply a second rotation pass based on detected text box angles.
    """

    angle_avg = compute_avg_text_region_angle(boxes)
    return rotate_image(img, angle_avg)