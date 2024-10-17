import os
import cv2
from text_extraction import text_extraction
from majority_voting import majority_voting
from post_processing import post_processing
from validation import validate_structured_fields

def process_text_pipeline(input_dir):
    image_filenames = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    images = [cv2.imread(os.path.join(input_dir, f)) for f in image_filenames]

    extracted_texts = text_extraction(images)

    voted_texts = [majority_voting(texts) for texts in extracted_texts]

    processed_texts = post_processing(voted_texts)

    validated_texts = validate_structured_fields(processed_texts)

    structured_output = transform_to_structured_format(validated_texts)

    return structured_output

def transform_to_structured_format(texts):
    structured_output = {}
    structured_output["validated_texts"] = texts
    return structured_output

if __name__ == '__main__':
    input_dir = '/content/segmented'
    structured_data = process_text_pipeline(input_dir)
