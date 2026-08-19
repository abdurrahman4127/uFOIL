from dataclasses import dataclass, field

import cv2
import numpy as np

from preproc.clahe import apply_dynamic_clahe, edge_preserving_filter
from preproc.rotation import detect_skew, rotate_image, refine_rotation, correct_orientation
from preproc.denoising_bm3d import apply_extended_bm3d

from segmnt.background_isolation import isolate_script
from segmnt.section_separation import separate_sections
from segmnt.label_field_detection import ocr_detect_labels, extract_all_fields
from segmnt.table_segmentation import segment_entries

from ocr_ensmbl.models import tesseract, easyocr_model, craft, trocr_model
from ocr_ensmbl.majority_voting import fuse_ensemble_outputs, VotedEntry

from postproc.field_validation import (
    validate_format,
    parse_marks_column,
    validate_total_marks,
    parse_filename,
    matches_ground_truth,
)


@dataclass
class ScriptResult:
    fields: dict[str, str] = field(default_factory=dict)          # name, id, course code, etc
    field_valid: dict[str, bool] = field(default_factory=dict)    # per-field format check
    marks: list[float] = field(default_factory=list)
    marks_valid: bool = False
    ground_truth: dict | None = None                              # parsed from filename, if given
    field_matches_ground_truth: dict[str, bool] = field(default_factory=dict)
    raw_upper_votes: list[VotedEntry] = field(default_factory=list)
    raw_lower_votes: list[VotedEntry] = field(default_factory=list)


def preprocess_image(img: np.ndarray, craft_weights_path: str) -> np.ndarray:
    upright = correct_orientation(img)
    contrast = apply_dynamic_clahe(upright)
    edges_filtered = edge_preserving_filter(contrast)

    gray = cv2.cvtColor(edges_filtered, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    skew_angle = detect_skew(edges)
    rotated = rotate_image(edges_filtered, skew_angle)

    # refine using CRAFT's own detected boxes as the text regions
    boxes = craft.detect_boxes(rotated, craft_weights_path)
    refined = refine_rotation(rotated, boxes)

    denoised = apply_extended_bm3d(refined)
    return denoised


def run_ocr_ensemble(
    img: np.ndarray,
    craft_weights_path: str,
    min_conf: float = 0.3,
) -> list[VotedEntry]:
    # each model runs independently, then majority_voting fuses them
    outputs = {
        "tesseract": tesseract.recognize(img, min_conf=min_conf),
        "easyocr": easyocr_model.recognize(img, min_conf=min_conf),
        "craft": craft.recognize(img, craft_weights_path, min_conf=min_conf),
        "trocr": trocr_model.recognize(img, min_conf=min_conf),
    }
    return fuse_ensemble_outputs(outputs)


def extract_fields_from_upper(
    upper_img: np.ndarray,
    voted_entries: list[VotedEntry],
) -> tuple[dict[str, str], dict[str, bool]]:
    # convert voted entries into the (text, x, y, w, h, conf) shape
    # ocr_detect_labels expects, then pull the value box for each
    ocr_results = [(e.text, e.x, e.y, e.w, e.h, 1.0) for e in voted_entries]
    labels = ocr_detect_labels(upper_img, ocr_results)

    fields = {}
    field_valid = {}
    for label in labels:
        crop = extract_all_fields(upper_img, [label])[label.text]
        if crop.size == 0:
            continue

        # value box text comes from whichever voted entry sits nearest
        # to this label, matched by finding the closest voted x/y
        value_entries = [e for e in voted_entries if e.x > label.x + label.w]
        if not value_entries:
            continue
        value_text = min(value_entries, key=lambda e: (e.y - label.y) ** 2 + (e.x - label.x) ** 2).text

        is_valid, cleaned = validate_format(label.text, value_text)
        fields[label.text] = cleaned
        field_valid[label.text] = is_valid

    return fields, field_valid


def extract_marks_from_lower(
    lower_img: np.ndarray,
    voted_entries: list[VotedEntry],
    claimed_total: float | None,
) -> tuple[list[float], bool]:
    table_entries = segment_entries(lower_img)
    if not table_entries:
        return [], False

    # marks column is whichever voted text falls inside a table cell box
    marks_text = []
    for entry in table_entries:
        matches = [
            e for e in voted_entries
            if e.x >= entry.x and e.x <= entry.x + entry.w
            and e.y >= entry.y and e.y <= entry.y + entry.h
        ]
        if matches:
            marks_text.append(matches[0].text)

    marks = parse_marks_column(marks_text)
    is_valid = validate_total_marks(marks, claimed_total) if claimed_total is not None else False
    return marks, is_valid


def process_script(
    img: np.ndarray,
    craft_weights_path: str,
    filename: str | None = None,
) -> ScriptResult:
    preprocessed = preprocess_image(img, craft_weights_path)
    script = isolate_script(preprocessed)
    upper, lower = separate_sections(script)

    upper_votes = run_ocr_ensemble(upper, craft_weights_path)
    lower_votes = run_ocr_ensemble(lower, craft_weights_path)

    fields, field_valid = extract_fields_from_upper(upper, upper_votes)

    # filename encodes IMG_<name>_<id>_<marks> - used both as the
    # claimed total for the marks-sum check and as ground truth to
    # compare the OCR-extracted name/id fields against
    ground_truth = parse_filename(filename) if filename else None
    claimed_total = ground_truth["marks"] if ground_truth else None
    marks, marks_valid = extract_marks_from_lower(lower, lower_votes, claimed_total)

    field_matches_ground_truth = {}
    if ground_truth is not None:
        if "name" in fields:
            field_matches_ground_truth["name"] = matches_ground_truth(
                fields["name"], ground_truth["name"]
            )
        if "id" in fields:
            field_matches_ground_truth["id"] = matches_ground_truth(
                fields["id"], ground_truth["id"]
            )

    return ScriptResult(
        fields=fields,
        field_valid=field_valid,
        marks=marks,
        marks_valid=marks_valid,
        ground_truth=ground_truth,
        field_matches_ground_truth=field_matches_ground_truth,
        raw_upper_votes=upper_votes,
        raw_lower_votes=lower_votes,
    )


def process_batch(
    image_paths: list[str],
    craft_weights_path: str,
) -> dict[str, ScriptResult]:
    results = {}
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"skipping unreadable file: {path}")
            continue
        results[path] = process_script(img, craft_weights_path, filename=path)
    return results
