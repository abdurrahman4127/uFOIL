import os
import cv2
import numpy as np

def detect_table_contours(lower_image):
    gray_image = cv2.cvtColor(lower_image, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def segment_entries(contours, lower_image):
    entries = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # Filtering out small contours
            entries.append(lower_image[y:y + h, x:x + w])
    return entries

def process_table_detection(input_dir):
    image_filenames = [f for f in os.listdir(input_dir) if f.startswith('lower_') and f.endswith('.png')]
    for filename in image_filenames:
        image_path = os.path.join(input_dir, filename)
        lower_image = cv2.imread(image_path)
        contours = detect_table_contours(lower_image)
        entries = segment_entries(contours, lower_image)
        for i, entry in enumerate(entries):
            entry_output_path = os.path.join(input_dir, f"entry_{i}_{filename}")
            cv2.imwrite(entry_output_path, entry)

input_dir = "content/preprocessed"
process_table_detection(input_dir)
