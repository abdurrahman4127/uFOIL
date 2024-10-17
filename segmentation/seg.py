import os
import cv2
import numpy as np
import pytesseract

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def otsu_threshold(image):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def extract_contours(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def crop_to_largest_contour(image, contours):
    if len(contours) == 0:
        return image
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return image[y:y + h, x:x + w]

def segment_exam_scripts(image):
    gray_image = convert_to_grayscale(image)
    binary_image = otsu_threshold(gray_image)
    contours = extract_contours(binary_image)
    segmented_image = crop_to_largest_contour(image, contours)
    return segmented_image

def process_images_in_directory(input_dir):
    image_filenames = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    for filename in image_filenames:
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        segmented_image = segment_exam_scripts(image)
        output_path = os.path.join(input_dir, f"segmented_{filename}")
        cv2.imwrite(output_path, segmented_image)

input_dir = "content/preprocessed"
process_images_in_directory(input_dir)
