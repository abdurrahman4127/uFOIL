import cv2
import numpy as np
import pytesseract

def detect_skew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    angles = []

    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            angles.append(angle)

    if angles:
        return np.median(angles)
    return 0

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def compute_average_text_region_angle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = pytesseract.image_to_boxes(gray)
    angles = []

    for b in boxes.splitlines():
        b = b.split()
        x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        width = w - x
        height = h - y
        angle = np.degrees(np.arctan2(height, width))
        angles.append(angle)

    if angles:
        return np.mean(angles)
    return 0

def refine_rotation(image, average_angle):
    final_angle = detect_skew(image) + average_angle
    return rotate_image(image, final_angle)

def preprocess_images_for_rotation(images):
    rotated_images = []
    for img in images:
        skew_angle = detect_skew(img)
        aligned_image = rotate_image(img, skew_angle)
        average_angle = compute_average_text_region_angle(aligned_image)
        refined_image = refine_rotation(aligned_image, average_angle)
        rotated_images.append(refined_image)
    return rotated_images

def main_rotation_correction_pipeline(images):
    corrected_images = []
    for img in images:
        skew_angle = detect_skew(img)
        img_rotated = rotate_image(img, skew_angle)
        avg_angle = compute_average_text_region_angle(img_rotated)
        final_image = refine_rotation(img_rotated, avg_angle)
        corrected_images.append(final_image)
    return corrected_images

def visualize_rotation_correction(original_images, corrected_images):
    combined_images = []
    for original, corrected in zip(original_images, corrected_images):
        combined = np.hstack((original, corrected))
        combined_images.append(combined)
    return combined_images

def save_rotated_images(rotated_images, output_dir):
    for i, img in enumerate(rotated_images):
        cv2.imwrite(f"{output_dir}/rotated_image_{i + 1}.png", img)

def display_images(images, title="Images"):
    for img in images:
        cv2.imshow(title, img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(input_dir, output_dir):
    images = [cv2.imread(f"{input_dir}/image_{i + 1}.png") for i in range(len(os.listdir(input_dir)))]
    corrected_images = preprocess_images_for_rotation(images)
    save_rotated_images(corrected_images, output_dir)

if __name__ == "__main__":
    import os
    input_dir = "content/dataset"
    output_dir = "content/output"
    os.makedirs(output_dir, exist_ok=True)
    main(input_dir, output_dir)
