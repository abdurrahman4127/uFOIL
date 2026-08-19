from dataclasses import dataclass

import numpy as np


@dataclass
class DetectedLabel:
    """
    text: recognized label text, e.g. "Name" or "ID".
    x, y: top-left corner of the label's bounding box.
    w, h: width and height of the label's bounding box.
    confidence: OCR confidence score for this detection, 0 to 1.
    """

    text: str
    x: int
    y: int
    w: int
    h: int
    confidence: float


KNOWN_LABELS = [
    "name",
    "id",
    "course code",
    "course title",
    "semester",
    "trimester",
    "section",
    "date of examination",
]


def normalize_label_text(text: str) -> str:
    """
    clean up raw OCR text so it can be matched against KNOWN_LABELS.
    lowercased text with punctuation and extra whitespace stripped.
    """
    cleaned = text.lower().strip()
    cleaned = cleaned.replace(":", "").replace(".", "")
    return " ".join(cleaned.split())


# check if OCR text matches one of the known field labels.
def match_known_label(text: str) -> str | None:
    normalized = normalize_label_text(text)

    for known in KNOWN_LABELS:
        if known in normalized or normalized in known:
            return known

    return None


def ocr_detect_labels(
    upper_img: np.ndarray,
    ocr_results: list[tuple[str, int, int, int, int, float]],
) -> list[DetectedLabel]:
    """
    filter raw OCR detections down to just the ones matching known
    field labels.

    args:
        upper_img: the upper section image the OCR ran on, used only
            for its shape.
        ocr_results: raw detections as (text, x, y, w, h, confidence)
            tuples, coming from whichever OCR model in the ensemble
            ran detection.
    """
    labels = []

    for text, x, y, w, h, conf in ocr_results:
        matched = match_known_label(text)
        if matched is not None:
            labels.append(DetectedLabel(matched, x, y, w, h, conf))

    return labels


# crop the handwritten value box next to a label.
def extract_handwritten_data(
    upper_img: np.ndarray,
    label: DetectedLabel,
    field_width: int = 200,
    y_margin: int = 5,
) -> np.ndarray:
    """
    in our data, the value box starts right after the label's
    bounding box ends and extend a fixed width to the right

    args:
        upper_img: image the label was detected on.
        label: the label whose adjacent value box we want.
        field_width: how wide the value box is, in pixels.
        y_margin: extra vertical padding above/below the label height.
    """
    img_h, img_w = upper_img.shape[:2]

    x0 = label.x + label.w
    y0 = max(0, label.y - y_margin)
    x1 = min(img_w, x0 + field_width)
    y1 = min(img_h, label.y + label.h + y_margin)

    if x0 >= img_w or y0 >= img_h or x1 <= x0 or y1 <= y0:
        return np.empty((0, 0), dtype=upper_img.dtype)

    return upper_img[y0:y1, x0:x1]


def extract_all_fields(
    upper_img: np.ndarray,
    labels: list[DetectedLabel],
    field_width: int = 200,
) -> dict[str, np.ndarray]:
    fields = {}
    for label in labels:
        fields[label.text] = extract_handwritten_data(
            upper_img, label, field_width=field_width
        )
    return fields