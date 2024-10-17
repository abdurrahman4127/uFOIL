import cv2
import os
from ocr_models import OCRModel

models = [OCRModel('tesseract'), OCRModel('easyocr'), OCRModel('trocr')]

def extract_feature_maps(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def recognize_characters(feature_map):
    text_results = []
    for model in models:
        text = model.recognize(feature_map)
        text_results.append(text)
    return text_results

def text_extraction(images):
    extracted_texts = []
    for image in images:
        try:
            feature_map = extract_feature_maps(image)
            texts = recognize_characters(feature_map)
            extracted_texts.append(texts)
        except Exception as e:
            extracted_texts.append(["ERROR"])
    return extracted_texts
