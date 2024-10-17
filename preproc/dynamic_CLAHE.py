import cv2
import numpy as np

def apply_dynamic_clahe(image, tile_grid_size, clip_limit, brightness_threshold=0.5, text_density_threshold=0.5):
    def compute_brightness(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        return np.mean(l_channel) / 255.0

    def compute_text_density(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        return np.sum(binary > 0) / binary.size

    def adjust_clahe_parameters(brightness, text_density):
        if brightness < brightness_threshold:
            adjusted_clip_limit = clip_limit * 1.5
        else:
            adjusted_clip_limit = clip_limit * 0.7

        if text_density > text_density_threshold:
            adjusted_tile_size = (tile_grid_size[0] // 2, tile_grid_size[1] // 2)
        else:
            adjusted_tile_size = tile_grid_size

        return adjusted_clip_limit, adjusted_tile_size

    brightness = compute_brightness(image)
    text_density = compute_text_density(image)
    adjusted_clip_limit, adjusted_tile_size = adjust_clahe_parameters(brightness, text_density)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=adjusted_clip_limit, tileGridSize=adjusted_tile_size)
    cl_l_channel = clahe.apply(l_channel)

    cl_lab = cv2.merge((cl_l_channel, a, b))
    return cv2.cvtColor(cl_lab, cv2.COLOR_LAB2BGR)

def dynamic_clahe_preprocessing(images, tile_grid_size=(8, 8), clip_limit=2.0, brightness_threshold=0.5, text_density_threshold=0.5):
    def compute_local_contrast(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var

    def region_wise_clahe(image):
        h, w = image.shape[:2]
        n_regions = 4
        region_size_x, region_size_y = h // n_regions, w // n_regions

        processed_image = np.zeros_like(image)
        for i in range(n_regions):
            for j in range(n_regions):
                x_start, y_start = i * region_size_x, j * region_size_y
                x_end, y_end = x_start + region_size_x, y_start + region_size_y
                region = image[x_start:x_end, y_start:y_end]

                brightness = compute_brightness(region)
                text_density = compute_text_density(region)
                adjusted_clip_limit, adjusted_tile_size = adjust_clahe_parameters(brightness, text_density)

                lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
                l_channel, a, b = cv2.split(lab)

                clahe = cv2.createCLAHE(clipLimit=adjusted_clip_limit, tileGridSize=adjusted_tile_size)
                cl_l_channel = clahe.apply(l_channel)

                cl_lab = cv2.merge((cl_l_channel, a, b))
                processed_region = cv2.cvtColor(cl_lab, cv2.COLOR_LAB2BGR)

                processed_image[x_start:x_end, y_start:y_end] = processed_region

        return processed_image

    def adaptive_contrast_adjustment(image):
        local_contrast = compute_local_contrast(image)
        if local_contrast < 50:
            clip_limit_factor = 1.5
        else:
            clip_limit_factor = 0.9

        return apply_dynamic_clahe(image, tile_grid_size, clip_limit * clip_limit_factor, brightness_threshold, text_density_threshold)

    enhanced_images = []
    for img in images:
        enhanced_image = adaptive_contrast_adjustment(img)
        enhanced_image = region_wise_clahe(enhanced_image)
        enhanced_images.append(enhanced_image)

    return enhanced_images
