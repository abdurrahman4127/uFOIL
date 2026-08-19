from collections import Counter
from dataclasses import dataclass, field

Box = tuple[int, int, int, int]
Detection = tuple[str, int, int, int, int, float]  # text, x, y, w, h, conf


@dataclass
class VotedEntry:
    text: str
    x: int
    y: int
    w: int
    h: int
    votes: int  # how many models contributed to this entry
    sources: list[str] = field(default_factory=list)


def box_iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)

    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0

    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def group_by_region(
    model_outputs: dict[str, list[Detection]],
    iou_threshold: float = 0.4,
) -> list[dict[str, Detection]]:
    # clusters detections from different models that overlap enough to
    # be "the same region", one dict per region: model name -> detection
    all_dets = []
    for model_name, dets in model_outputs.items():
        for d in dets:
            all_dets.append((model_name, d))

    groups: list[dict[str, Detection]] = []

    for model_name, det in all_dets:
        _, x, y, w, h, _ = det
        box = (x, y, w, h)

        matched_group = None
        for group in groups:
            # only need to check against one existing member's box
            existing = next(iter(group.values()))
            existing_box = (existing[1], existing[2], existing[3], existing[4])
            if box_iou(box, existing_box) >= iou_threshold:
                matched_group = group
                break

        if matched_group is not None:
            matched_group[model_name] = det
        else:
            groups.append({model_name: det})

    return groups


def char_majority_vote(texts: list[str]) -> str:
    # votes char by char across texts of possibly different lengths,
    # shorter ones padded with a blank so index errors don't happen
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]

    max_len = max(len(t) for t in texts)
    padded = [t.ljust(max_len, "\0") for t in texts]

    result_chars = []
    for i in range(max_len):
        col = [t[i] for t in padded]
        counts = Counter(c for c in col if c != "\0")

        if not counts:
            continue  # every model had nothing here, skip position

        winner, _ = counts.most_common(1)[0]
        result_chars.append(winner)

    return "".join(result_chars)


def vote_region(detections: dict[str, Detection]) -> VotedEntry:
    texts = [d[0] for d in detections.values()]
    voted_text = char_majority_vote(texts)

    # union box across all contributing models rather than trusting
    # just one model's box, since they rarely match pixel-for-pixel
    xs0 = [d[1] for d in detections.values()]
    ys0 = [d[2] for d in detections.values()]
    xs1 = [d[1] + d[3] for d in detections.values()]
    ys1 = [d[2] + d[4] for d in detections.values()]

    x, y = min(xs0), min(ys0)
    w, h = max(xs1) - x, max(ys1) - y

    return VotedEntry(
        text=voted_text,
        x=x,
        y=y,
        w=w,
        h=h,
        votes=len(detections),
        sources=list(detections.keys()),
    )


def fuse_ensemble_outputs(
    model_outputs: dict[str, list[Detection]],
    iou_threshold: float = 0.4,
) -> list[VotedEntry]:
    # model_outputs: {"tesseract": [...], "easyocr": [...], ...}
    # returns one VotedEntry per matched region, sorted top to bottom
    groups = group_by_region(model_outputs, iou_threshold=iou_threshold)
    entries = [vote_region(g) for g in groups]
    return sorted(entries, key=lambda e: (e.y, e.x))
