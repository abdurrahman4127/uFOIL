from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TableEntry:
    x: int
    y: int
    w: int
    h: int
    row: int


def binarize_lower_section(lower_img: np.ndarray) -> np.ndarray:
    gray = (
        cv2.cvtColor(lower_img, cv2.COLOR_BGR2GRAY)
        if lower_img.ndim == 3
        else lower_img
    )
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


# find contours in the binarized lower section.
def detect_table_contours(binary: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    return list(contours)


def largest_rectangular_contour(
    contours: list[np.ndarray],
    min_area_ratio: float = 0.1,
    img_area: int = 1,
) -> np.ndarray | None:
    """
    args:
        contours: contours from detect_table_contours.
        min_area_ratio: minimum contour area as a fraction of the
            whole image, filters out small noise contours.
        img_area: total pixel area of the source image.

    returns:
        the best matching contour, or None if nothing qualifies.
    """
    min_area = img_area * min_area_ratio
    candidates = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not candidates:
        return None

    return max(candidates, key=cv2.contourArea)


    # kernel_size: structuring element size for both operations.
def clean_binary_table(
    binary: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    # run erosion then dilation to strip noise and reconnect broken table lines.

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    eroded = cv2.erode(binary, kernel, iterations=1)
    cleaned = cv2.dilate(eroded, kernel, iterations=1)
    return cleaned


def find_connected_components(
    binary: np.ndarray,
    min_component_area: int = 50,
) -> list[tuple[int, int, int, int]]:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)

    boxes = []
    # label 0 is background, always skipped
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= min_component_area:
            boxes.append((x, y, w, h))

    return boxes


    # assign each detected box a row index based on vertical position.
def group_into_rows(
    boxes: list[tuple[int, int, int, int]],
    row_tolerance: int = 15,
) -> list[TableEntry]:
    """
    boxes with y-coordinates within row_tolerance of each other are
    treated as belonging to the same row, which matches how a table
    row's question number and mark cells line up.

    args:
        boxes: (x, y, w, h) boxes from find_connected_components.
        row_tolerance: max y-difference to still count as the same row.
    """
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda b: b[1])

    rows: list[list[tuple[int, int, int, int]]] = []
    current_row = [sorted_boxes[0]]
    current_y = sorted_boxes[0][1]

    for box in sorted_boxes[1:]:
        if abs(box[1] - current_y) <= row_tolerance:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
            current_y = box[1]
    rows.append(current_row)

    entries = []
    for row_idx, row_boxes in enumerate(rows):
        row_boxes_sorted = sorted(row_boxes, key=lambda b: b[0])
        for x, y, w, h in row_boxes_sorted:
            entries.append(TableEntry(x, y, w, h, row=row_idx))

    return entries


def segment_entries(lower_img: np.ndarray) -> list[TableEntry]:
    binary = binarize_lower_section(lower_img)
    cleaned = clean_binary_table(binary)
    boxes = find_connected_components(cleaned)
    return group_into_rows(boxes)