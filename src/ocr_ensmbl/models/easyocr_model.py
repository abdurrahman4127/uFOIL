# easyocr wrapper, same (text, x, y, w, h, conf) output format as the rest.

import easyocr
import numpy as np

_reader = None  # lazy singleton, loading the reader is slow


def get_reader(langs: list[str] | None = None) -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(langs or ["en"], verbose=False)
    return _reader


def recognize(img: np.ndarray, min_conf: float = 0.0) -> list[tuple[str, int, int, int, int, float]]:
    reader = get_reader()
    raw = reader.readtext(img)

    results = []
    for box, text, conf in raw:
        if conf < min_conf or not text.strip():
            continue

        # easyocr gives 4 corner points, not x/y/w/h, convert here
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs) - x), int(max(ys) - y)

        results.append((text.strip(), x, y, w, h, float(conf)))

    return results
