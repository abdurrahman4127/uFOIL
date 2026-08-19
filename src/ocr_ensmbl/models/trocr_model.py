"""
TrOCR wrapper. TrOCR does recognition only, no detection - reuses tesseract's boxes for region proposals, 
then reads each crop fresh with TrOCR so the two votes stay independent.
"""

from PIL import Image
import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ocr_ensmbl.models import tesseract

CHECKPOINT = "microsoft/trocr-large-handwritten"

_processor = None
_model = None


def get_model():
    global _processor, _model
    if _model is None:
        """
        use_fast=False dodges a transformers bug where the fast
        tokenizer path fails to build even with sentencepiece
        installed - confirmed on colab, this is the actual fix,
        not just a workaround around a missing dependency
        """
        _processor = TrOCRProcessor.from_pretrained(CHECKPOINT, use_fast=False)
        _model = VisionEncoderDecoderModel.from_pretrained(CHECKPOINT)
        _model.eval()
    return _processor, _model


def recognize(img: np.ndarray, min_conf: float = 0.0) -> list[tuple[str, int, int, int, int, float]]:
    processor, model = get_model()
    boxes = [(x, y, w, h) for (_, x, y, w, h, _) in tesseract.recognize(img)]

    results = []
    for x, y, w, h in boxes:
        crop = img[y : y + h, x : x + w]
        if crop.size == 0:
            continue

        pil_crop = Image.fromarray(crop).convert("RGB")
        pixel_values = processor(images=pil_crop, return_tensors="pt").pixel_values

        with torch.no_grad():
            out = model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
            )

        text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        if not text:
            continue

        conf = sequence_confidence(out)
        if conf < min_conf:
            continue

        results.append((text, x, y, w, h, conf))

    return results


def sequence_confidence(generate_output) -> float:
    # avg token probability across the generated sequence
    if not generate_output.scores:
        return 0.5

    probs = []
    for step_logits in generate_output.scores:
        step_probs = torch.softmax(step_logits, dim=-1)
        probs.append(float(step_probs.max()))

    return sum(probs) / len(probs)
