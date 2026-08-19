# tesseract wrapper, returns (text, x, y, w, h, conf) tuples same as the rest of the ensemble.

import pytesseract
from pytesseract import Output

import numpy as np


def recognize(img: np.ndarray, min_conf: float = 0.0) -> list[tuple[str, int, int, int, int, float]]:
    data = pytesseract.image_to_data(img, output_type=Output.DICT)

    results = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        conf_raw = float(data["conf"][i])

        if not text or conf_raw < 0:
            continue  # tesseract uses -1 conf for non-text regions

        conf = conf_raw / 100.0  # normalize to 0-1 to match the others
        if conf < min_conf:
            continue

        x, y, w, h = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )
        results.append((text, x, y, w, h, conf))

    return results
