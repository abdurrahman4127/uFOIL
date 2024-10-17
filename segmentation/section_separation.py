import os
import cv2

def split_image(image, boundary):
    h, w = image.shape[:2]
    return image[0:boundary, 0:w], image[boundary:h, 0:w]

def detect_table_boundary(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(y1 - y2) < 10:  # horizontal lines
                return (y1 + y2) // 2  # Return the middle point of the line
    return image.shape[0] // 2  # Default to middle of the image if no lines found

def process_section_separation(input_dir):
    image_filenames = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    for filename in image_filenames:
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        edges = cv2.Canny(image, 50, 150)
        boundary = detect_table_boundary(edges)
        upper_section, lower_section = split_image(image, boundary)
        cv2.imwrite(os.path.join(input_dir, f"upper_{filename}"), upper_section)
        cv2.imwrite(os.path.join(input_dir, f"lower_{filename}"), lower_section)

input_dir = "content/preprocessed"
process_section_separation(input_dir)
