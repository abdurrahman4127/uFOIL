import numpy as np
from scipy.fftpack import dct, idct

def bm3d_denoising(image, sigma_psd, block_size=8, search_window=21, hard_threshold=True):
    def dct2d(block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def idct2d(block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    def hard_thresholding(coeffs, threshold):
        return coeffs * (np.abs(coeffs) > threshold)

    def aggregate_blocks(blocks, weights, img_size):
        aggregated_image = np.zeros(img_size)
        weight_sum = np.zeros(img_size)
        for (i, j), block, weight in blocks:
            aggregated_image[i:i+block_size, j:j+block_size] += block * weight
            weight_sum[i:i+block_size, j:j+block_size] += weight
        return aggregated_image / weight_sum

    image = image.astype(np.float32) / 255.0
    noisy_image = image.copy()
    padded_img = np.pad(noisy_image, ((block_size // 2, block_size // 2), (block_size // 2, block_size // 2)), mode='reflect')

    blocks = []
    weights = []

    for i in range(0, padded_img.shape[0] - block_size + 1):
        for j in range(0, padded_img.shape[1] - block_size + 1):
            block = padded_img[i:i + block_size, j:j + block_size]
            block_dct = dct2d(block)

            if hard_threshold:
                threshold = 2.7 * sigma_psd * np.sqrt(2 * block_size)
                block_dct = hard_thresholding(block_dct, threshold)

            inverse_block = idct2d(block_dct)
            weight = 1 / (sigma_psd ** 2 + np.mean(block_dct ** 2))
            blocks.append(((i, j), inverse_block, weight))
            weights.append(weight)

    denoised_image = aggregate_blocks(blocks, weights, image.shape)
    return np.clip(denoised_image * 255, 0, 255).astype(np.uint8)

def apply_bm3d(images, sigma_psd=0.1):
    denoised_images = []
    for img in images:
        denoised_img = bm3d_denoising(img, sigma_psd)
        denoised_images.append(denoised_img)
    return denoised_images
