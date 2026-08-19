import bm3d
import cv2
import numpy as np


def split_into_blocks(
    img: np.ndarray,
    block_size: int = 8,
    overlap: int = 4,
) -> list[tuple[np.ndarray, int, int]]:
    """
    split an image into overlapping blocks.

    args:
        img: single-channel image, shape (H, W).
        block_size: block edge length in pixels.
        overlap: how many pixels adjacent blocks overlap by.

    returns:
        list of (block, x, y) tuples, x/y are the block's top-left
        corner in the original image.
    """
    h, w = img.shape
    stride = max(1, block_size - overlap)

    blocks = []
    for y in range(0, h - block_size + 1, stride):
        for x in range(0, w - block_size + 1, stride):
            block = img[y : y + block_size, x : x + block_size]
            blocks.append((block, x, y))

    return blocks


def block_feature(block: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(block, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(block, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx)

    mag_hist, _ = np.histogram(magnitude, bins=8, range=(0, 255))
    ori_hist, _ = np.histogram(orientation, bins=8, range=(-np.pi, np.pi))

    feature = np.concatenate([mag_hist, ori_hist]).astype(np.float32)
    norm = np.linalg.norm(feature)
    return feature / norm if norm > 0 else feature


def block_similarity(feat_a: np.ndarray, feat_b: np.ndarray) -> float:
    diff = feat_a - feat_b
    return float(np.dot(diff, diff))


# computing block_feature() inline inside find_similar_blocks meant
# recomputing the same feature thousands of times over one image -
# that's what made this unusably slow past tiny test sizes. compute
# every block's feature once up front instead.
def precompute_features(blocks: list[tuple[np.ndarray, int, int]]) -> list[np.ndarray]:
    return [block_feature(block) for block, _, _ in blocks]


def find_similar_blocks(
    blocks: list[tuple[np.ndarray, int, int]],
    features: list[np.ndarray],
    ref_idx: int,
    search_radius: int = 40,
    top_k: int = 8,
    max_distance: float = 0.5,
) -> list[int]:
    """
    find indices of blocks most similar to a reference block.

    only blocks within search_radius pixels are checked, matching how
    real BM3D limits its search window instead of scanning the whole
    image for every single block.

    args:
        blocks: output of split_into_blocks.
        features: precomputed features from precompute_features, same
            order/length as blocks.
        ref_idx: index of the reference block in `blocks`.
        search_radius: how far out (in pixels) to look for matches.
        top_k: max number of matches to return.
        max_distance: similarity distance cutoff, matches above this
            are dropped even if within top_k.

    returns:
        list of block indices, sorted by similarity, closest first.
    """
    ref_feat = features[ref_idx]
    _, ref_x, ref_y = blocks[ref_idx]

    scored = []
    for i, (_, x, y) in enumerate(blocks):
        if i == ref_idx:
            continue
        if abs(x - ref_x) > search_radius or abs(y - ref_y) > search_radius:
            continue
        dist = block_similarity(ref_feat, features[i])
        if dist <= max_distance:
            scored.append((dist, i))

    scored.sort(key=lambda pair: pair[0])
    return [idx for _, idx in scored[:top_k]]


def apply_bm3d_base(img: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    img_float = img.astype(np.float32) / 255.0
    denoised = bm3d.bm3d(img_float, sigma_psd=sigma / 255.0)
    denoised = np.clip(denoised * 255.0, 0, 255)
    return denoised.astype(np.uint8)


def apply_extended_bm3d(
    img: np.ndarray,
    block_size: int = 8,
    overlap: int = 4,
    sigma: float = 15.0,
) -> np.ndarray:
    """
    run the extended BM3D pass: base BM3D, then a light blend with a
    feature-matched block average to pull back stroke detail that
    plain BM3D tends to smear out.

    args:
        img: BGR or grayscale image, uint8.
        block_size: block size used for the feature-matching pass.
        overlap: overlap between adjacent blocks.
        sigma: noise sigma passed to the base BM3D pass.

    returns:
        denoised image, same shape and channel count as input.
    """
    is_color = img.ndim == 3
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if is_color else img

    base_denoised = apply_bm3d_base(gray, sigma=sigma)

    blocks = split_into_blocks(gray, block_size=block_size, overlap=overlap)
    features = precompute_features(blocks)
    refined = base_denoised.astype(np.float32).copy()
    weight_map = np.zeros_like(refined)

    for idx, (block, x, y) in enumerate(blocks):
        matches = find_similar_blocks(blocks, features, idx)
        if not matches:
            continue

        stacked = [block.astype(np.float32)]
        stacked += [blocks[m][0].astype(np.float32) for m in matches]
        block_avg = np.mean(stacked, axis=0)

        region = refined[y : y + block_size, x : x + block_size]
        region += block_avg
        weight_map[y : y + block_size, x : x + block_size] += 1.0

    # blend the feature-matched average back in where we had matches,
    # fall back to the plain base_denoised elsewhere
    mask = weight_map > 0
    refined[mask] = (
        base_denoised.astype(np.float32)[mask] * 0.5
        + (refined[mask] / (weight_map[mask] + 1.0)) * 0.5
    )
    refined[~mask] = base_denoised.astype(np.float32)[~mask]

    result = np.clip(refined, 0, 255).astype(np.uint8)

    if is_color:
        # apply the same denoised luma back onto the color image via
        # the L channel so color info isn't touched
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        _, a_channel, b_channel = cv2.split(lab)
        merged = cv2.merge((result, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return result