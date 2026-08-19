import re

STUDENT_ID_PATTERN = re.compile(r"^\d{9,10}$")
COURSE_CODE_PATTERN = re.compile(r"^[A-Z]{2,4}\s?\d{3,4}$")


def validate_student_id(text: str) -> tuple[bool, str]:
    """
    strips whitespace and checks the 9-10 digit format. returns
    (is_valid, cleaned_text).
    """
    cleaned = re.sub(r"\s", "", text)
    return bool(STUDENT_ID_PATTERN.match(cleaned)), cleaned


def validate_course_code(text: str) -> tuple[bool, str]:
    cleaned = text.strip().upper()
    return bool(COURSE_CODE_PATTERN.match(cleaned)), cleaned


def validate_format(field_name: str, text: str) -> tuple[bool, str]:
    """dispatches to the right validator based on field name."""
    if field_name == "id":
        return validate_student_id(text)
    if field_name == "course code":
        return validate_course_code(text)

    # no pattern defined for this field, treat as-is
    return True, text.strip()


def normalize_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def matches_ground_truth(extracted: str, expected: str) -> bool:
    return normalize_for_compare(extracted) == normalize_for_compare(expected)


def parse_marks_column(marks_text: list[str]) -> list[float]:
    marks = []
    for raw in marks_text:
        cleaned = raw.strip().replace(",", ".")
        try:
            marks.append(float(cleaned))
        except ValueError:
            continue
    return marks


def validate_total_marks(
    question_marks: list[float],
    claimed_total: float,
    tolerance: float = 0.5,
) -> bool:
    computed = sum(question_marks)
    return abs(computed - claimed_total) <= tolerance


# real filenames follow IMG_<studentname>_<id>_<marks>.jpg/png -
# student name can itself contain underscores/multiple words, so name
# is everything between the leading "IMG" token and the last two
# tokens (id, marks).
def parse_filename(filename: str) -> dict | None:
    stem = filename.rsplit("/", 1)[-1]
    stem = stem.rsplit(".", 1)[0]  # drop extension
    parts = stem.split("_")

    if len(parts) < 4 or parts[0].upper() != "IMG":
        return None

    marks_str = parts[-1]
    id_str = parts[-2]
    name = " ".join(parts[1:-2])

    try:
        marks = float(marks_str)
    except ValueError:
        return None

    return {"name": name, "id": id_str, "marks": marks}


def total_from_filename(filename: str) -> float | None:
    parsed = parse_filename(filename)
    return parsed["marks"] if parsed else None