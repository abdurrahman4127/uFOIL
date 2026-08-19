from dataclasses import dataclass


@dataclass
class ModelVote:
    char: str
    prob: float  # this model's probability for this char
    total_chars: int  # total chars this model detected overall


def char_confidence(votes: list[ModelVote]) -> float:
    """
    weighted avg of per-model probs, weighted down for models that
    detected way more chars than others (usually means noisier output).
    """
    if not votes:
        return 0.0

    scores = [v.prob / max(v.total_chars, 1) for v in votes]
    return sum(scores) / len(scores)


def flag_low_confidence(
    chars: list[str],
    confidences: list[float],
    threshold: float = 0.5,
) -> list[int]:
    """returns indices of chars below threshold."""
    return [i for i, c in enumerate(confidences) if c < threshold]


def correct_recognition_error(
    char: str,
    candidates: list[tuple[str, float]],
) -> str:
    """
    swaps a low-confidence char for the highest-prob candidate from
    the ensemble, unless nothing beats the original.
    """
    if not candidates:
        return char

    best_char, best_prob = max(candidates, key=lambda c: c[1])
    current_prob = next((p for c, p in candidates if c == char), 0.0)

    return best_char if best_prob > current_prob else char


def apply_confidence_correction(
    text: str,
    per_char_confidence: list[float],
    per_char_candidates: list[list[tuple[str, float]]],
    threshold: float = 0.5,
) -> str:
    """runs the full flag -> correct loop over one string of text."""
    chars = list(text)
    low_conf_idx = flag_low_confidence(chars, per_char_confidence, threshold)

    for i in low_conf_idx:
        if i < len(per_char_candidates):
            chars[i] = correct_recognition_error(chars[i], per_char_candidates[i])

    return "".join(chars)