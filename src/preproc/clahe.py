import cv2
import numpy as np


# first, convert an img to single-channel grayscale if it isn't already.
def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def compute_local_text_density(
    gray: np.ndarray,
    tile_size: int = 32,
) -> np.ndarray:
    """
    text and handwriting produce a lot of edges relative to blank
    paper, so the fraction of edge pixels in a tile is a cheap proxy
    for how much text sits in that tile.

    args:
        gray: single-channel grayscale img, shape (H, W).
        tile_size: base tile size in pixels before density-based
            adjustment.

    returns:
        density_map: array of shape (H // tile_size, W // tile_size)
        with values in [0, 1], where 1.0 marks a fully edge-dense tile.
    """
    edges = cv2.Canny(gray, 50, 150)
    h, w = gray.shape
    n_rows, n_cols = h // tile_size, w // tile_size

    if n_rows == 0 or n_cols == 0:
        raise ValueError(
            f"img too small ({h}x{w}) for tile_size={tile_size}"
        )

    density_map = np.zeros((n_rows, n_cols), dtype=np.float32)

    for i in range(n_rows):
        for j in range(n_cols):
            tile = edges[
                i * tile_size : (i + 1) * tile_size,
                j * tile_size : (j + 1) * tile_size,
            ]
            density_map[i, j] = np.count_nonzero(tile) / tile.size

    return density_map


def compute_regional_stats(
    gray: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    splits the img into an n rows x n cols grid and computes mean intensity 
    and variance for each cell.

    args:
        gray: single-channel grayscale img, shape (H, W).
        n_rows: number of grid rows.
        n_cols: number of grid columns.

    returns:
        tuple of (brightness_map, variance_map), each of shape
        (n_rows, n_cols).
    """
    h, w = gray.shape
    tile_h, tile_w = h // n_rows, w // n_cols

    brightness_map = np.zeros((n_rows, n_cols), dtype=np.float32)
    variance_map = np.zeros((n_rows, n_cols), dtype=np.float32)

    for i in range(n_rows):
        for j in range(n_cols):
            region = gray[
                i * tile_h : (i + 1) * tile_h,
                j * tile_w : (j + 1) * tile_w,
            ]
            brightness_map[i, j] = float(np.mean(region))
            variance_map[i, j] = float(np.var(region))

    return brightness_map, variance_map


def density_to_tile_size(
    density: float,
    min_tile: int = 8,
    max_tile: int = 32,
) -> int:
    """
    map a density value to a CLAHE tile size.

    args:
        density: text density in [0, 1].
        min_tile: smallest allowed tile size, used at density = 1.0.
        max_tile: largest allowed tile size, used at density = 0.0.

    returns:
        tile size in pixels, even, clamped to [min_tile, max_tile].
    """
    density = float(np.clip(density, 0.0, 1.0))
    tile = int(round(max_tile - density * (max_tile - min_tile)))
    tile = max(min_tile, min(max_tile, tile))

    # cv2 wants an even tile grid size, bump up if odd
    if tile % 2 != 0:
        tile += 1

    return tile


def adaptive_clip_limit(
    brightness: float,
    variance: float,
    base_clip: float = 2.0,
    min_clip: float = 1.0,
    max_clip: float = 4.0,
) -> float:
    """
    bright, high-variance regions already have plenty of contrast, so
    the clip limit is pulled down to avoid amplifying noise.

    args:
        brightness: mean pixel intensity, in [0, 255].
        variance: pixel intensity variance.
        base_clip: baseline clip limit before scaling.
        min_clip: lower bound on the returned clip limit.
        max_clip: upper bound on the returned clip limit.

    returns:
        adjusted clip limit, clamped to [min_clip, max_clip].
    """
    brightness_norm = brightness / 255.0
    # cap variance contribution so a few very noisy tiles don't
    # dominate the whole scaling factor
    variance_penalty = min(variance / 5000.0, 1.0)

    clip_limit = (
        base_clip
        * (1.2 - brightness_norm)
        * (1.0 + 0.3 * (1.0 - variance_penalty))
    )

    return float(np.clip(clip_limit, min_clip, max_clip))


def apply_dynamic_clahe(
    img: np.ndarray,
    base_tile: int = 32,
) -> np.ndarray:
    """
    for color imgs, CLAHE is applied on the L channel of LAB color
    space only, so color information in a/b channels is untouched.

    args:
        img: input img, BGR (H, W, 3) or grayscale (H, W), uint8.
        base_tile: tile size used when computing the density map,
            before density-based tile size adjustment.

    returns:
        contrast-adjusted img, same shape and dtype as input.
    """
    gray = to_grayscale(img)
    density_map = compute_local_text_density(gray, tile_size=base_tile)
    n_rows, n_cols = density_map.shape

    brightness_map, variance_map = compute_regional_stats(
        gray, n_rows, n_cols
    )

    avg_density = float(np.mean(density_map))
    avg_brightness = float(np.mean(brightness_map))
    avg_variance = float(np.mean(variance_map))

    # cv2's CLAHE takes one grid size for the whole img, not a
    # per-tile varying grid, so the per-region maps above get
    # aggregated into single global parameters here
    tile_size = density_to_tile_size(avg_density)
    clip_limit = adaptive_clip_limit(avg_brightness, avg_variance)

    h, w = gray.shape
    grid_x = max(1, w // tile_size)
    grid_y = max(1, h // tile_size)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_x, grid_y),
    )

    if img.ndim == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = clahe.apply(l_channel)
        merged = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return clahe.apply(gray)


def edge_preserving_filter(
    img: np.ndarray,
    sigma_s: float = 40.0,
    sigma_r: float = 0.2,
) -> np.ndarray:
    """
    meant to run right after apply_dynamic_clahe(), since boosting
    contrast can also amplify small noise artifacts picked up by the
    cams.

    args:
        img: BGR img, uint8.
        sigma_s: spatial filter strength, larger values smooth over a
            wider neighborhood.
        sigma_r: range filter strength, smaller values preserve edges
            more aggressively.

    returns:
        filtered BGR img, same shape as input.
    """
    return cv2.edgePreservingFilter(
        img,
        flags=cv2.RECURS_FILTER,
        sigma_s=sigma_s,
        sigma_r=sigma_r,
    )