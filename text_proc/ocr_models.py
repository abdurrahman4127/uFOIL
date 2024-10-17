import cv2
import pytesseract
import easyocr
import torch
import numpy as np

class OCRModel:
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name == 'easyocr':
            self.reader = easyocr.Reader(['en'])
        elif model_name == 'trocr':
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'microsoft/trocr-base-printed')
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'microsoft/trocr-base-printed')

    def recognize(self, image):
        if self.model_name == 'tesseract':
            return self.tesseract_recognize(image)
        elif self.model_name == 'easyocr':
            return self.easyocr_recognize(image)
        elif self.model_name == 'trocr':
            return self.trocr_recognize(image)
        return ""

    def tesseract_recognize(self, image):
        try:
            return pytesseract.image_to_string(image)
        except Exception as e:
            return f"Tesseract Error: {str(e)}"

    def easyocr_recognize(self, image):
        try:
            results = self.reader.readtext(image)
            return " ".join([result[1] for result in results])
        except Exception as e:
            return f"EasyOCR Error: {str(e)}"

    def trocr_recognize(self, image):
        try:
            image_tensor = self.preprocess(image)
            with torch.no_grad():
                output = self.model(image_tensor)
            return self.decode(output)
        except Exception as e:
            return f"TrOCR Error: {str(e)}"

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (320, 320))
        return torch.tensor(image).float().unsqueeze(0)

    def decode(self, output):
        return output  