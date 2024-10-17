import numpy as np
import re

def calculate_confidence(predictions):
    unique, counts = np.unique(predictions, return_counts=True)
    confidence_scores = {u: c/len(predictions) for u, c in zip(unique, counts)}
    return [confidence_scores.get(pred, 0) for pred in predictions]

def correct_recognition_error(text):
    corrected_text = re.sub(r'\D', '', text)
    return corrected_text if len(corrected_text) in [9, 10] else "INVALID"

def post_processing(extracted_texts):
    processed_texts = []
    for texts in extracted_texts:
        predictions = majority_voting(texts)
        confidence_scores = calculate_confidence(predictions)

        for t, conf in zip(predictions, confidence_scores):
            if conf < 0.5:
                t = correct_recognition_error(t)
            processed_texts.append(t)
    return processed_texts
