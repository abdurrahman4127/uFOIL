import os
import cv2
import pytesseract

def ocr_detect_labels(image):
    # OCR to detect text labels
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    labels = pytesseract.image_to_string(gray_image).strip().split("\n")
    return labels

def process_label_detection(input_dir):
    image_filenames = [f for f in os.listdir(input_dir) if f.startswith('upper_') and f.endswith('.png')]
    for filename in image_filenames:
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        labels = ocr_detect_labels(image)
        print(f"Detected labels in {filename}: {labels}")

input_dir = "content/preprocessed"
process_label_detection(input_dir)
