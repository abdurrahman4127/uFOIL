import os
import cv2
import numpy as np

def load_real_images(image_directory):
    image_filenames = [f for f in os.listdir(image_directory) if f.endswith('.png')]
    images = []
    for filename in image_filenames:
        img_path = os.path.join(image_directory, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))  # Resize images to 256x256
        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1] range
        images.append(img)
    return np.array(images)

def preprocess_pipeline(input_dir, output_dir, generator):
    images = load_real_images(input_dir)

    images_contrast_adjusted = dynamic_contrast_adjustment(images)
    images_rotated = preprocess_images_for_rotation(images_contrast_adjusted)
    images_denoised = apply_denoising(images_rotated)
    augmented_dataset = augment_dataset_with_gan(images_denoised, generator)

    for i, img in enumerate(augmented_dataset):
        output_path = os.path.join(output_dir, f"preprocessed_{i}.png")
        cv2.imwrite(output_path, img * 255)

    print(f"Preprocessing complete. Images saved to {output_dir}")



input_dir = "content/dataset"
output_dir = "content/output"
os.makedirs(output_dir, exist_ok=True)

generator = build_generator()
preprocess_pipeline(input_dir, output_dir, generator)
