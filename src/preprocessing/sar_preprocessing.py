"""
Sentinel-1 SAR preprocessing for the Cross-Modal Retrieval project.

Pipeline:
1. Load VV and VH SAR bands from GeoTIFF.
2. Clip values to [-25, 0] dB.
3. Normalize using training-set mean/std from s1_stats.json.
4. Resize from 120x120 to 224x224.
5. Stack channels as VV, VH.
6. Return a PyTorch tensor of shape (2, 224, 224).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


STATS_PATH = Path(__file__).resolve().parent / "s1_stats.json"

TARGET_SIZE = (224, 224)

CLIP_MIN = -25.0
CLIP_MAX = 0.0


def load_s1_stats() -> tuple[np.ndarray, np.ndarray]:
    """
    Load Sentinel-1 training statistics.

    Returns:
        mean: array containing [VV_mean, VH_mean]
        std:  array containing [VV_std, VH_std]
    """

    if not STATS_PATH.exists():
        raise FileNotFoundError(
            f"Sentinel-1 statistics file not found: {STATS_PATH}"
        )

    with open(STATS_PATH, "r", encoding="utf-8") as file:
        stats = json.load(file)

    if "VV" not in stats or "VH" not in stats:
        raise ValueError(
            "s1_stats.json must contain both 'VV' and 'VH' entries."
        )

    vv_mean = float(stats["VV"]["mean"])
    vv_std = float(stats["VV"]["std"])

    vh_mean = float(stats["VH"]["mean"])
    vh_std = float(stats["VH"]["std"])

    if not np.isfinite(vv_mean) or not np.isfinite(vv_std):
        raise ValueError("Invalid VV statistics in s1_stats.json.")

    if not np.isfinite(vh_mean) or not np.isfinite(vh_std):
        raise ValueError("Invalid VH statistics in s1_stats.json.")

    if vv_std <= 0 or vh_std <= 0:
        raise ValueError(
            "Standard deviation must be greater than zero."
        )

    mean = np.array(
        [vv_mean, vh_mean],
        dtype=np.float32,
    )

    std = np.array(
        [vv_std, vh_std],
        dtype=np.float32,
    )

    return mean, std


def load_sar_band(file_path: Path) -> np.ndarray:
    """
    Load a single SAR GeoTIFF band as float32.

    Args:
        file_path: Path to VV or VH GeoTIFF.

    Returns:
        2D float32 NumPy array.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"SAR band file not found: {file_path}"
        )

    try:
        import rasterio

        with rasterio.open(file_path) as src:
            data = src.read(1)

    except ImportError:
        try:
            with Image.open(file_path) as image:
                data = np.asarray(image)
        except Exception as exc:
            raise IOError(
                f"Could not read SAR band: {file_path}"
            ) from exc

    except Exception as exc:
        raise IOError(
            f"Could not read SAR GeoTIFF: {file_path}"
        ) from exc

    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 2:
        raise ValueError(
            f"Expected a 2D SAR band, got shape {data.shape} "
            f"for {file_path}"
        )

    return data


def clip_sar_band(
    band: np.ndarray,
    min_value: float = CLIP_MIN,
    max_value: float = CLIP_MAX,
) -> np.ndarray:
    """
    Clip SAR backscatter values to the project range [-25, 0] dB.
    """

    if min_value >= max_value:
        raise ValueError(
            "Clip minimum must be smaller than clip maximum."
        )

    return np.clip(
        band,
        min_value,
        max_value,
    ).astype(np.float32)


def normalize_sar_band(
    band: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """
    Apply training-set Z-score normalization.
    """

    if std <= 0:
        raise ValueError(
            f"Standard deviation must be positive, got {std}."
        )

    normalized = (band - mean) / std

    normalized = normalized.astype(np.float32)

    if not np.isfinite(normalized).all():
        raise ValueError(
            "SAR normalization produced NaN or infinite values."
        )

    return normalized


def resize_sar_band(
    band: np.ndarray,
    target_size: tuple[int, int] = TARGET_SIZE,
) -> np.ndarray:
    """
    Resize a single SAR band using bilinear interpolation.
    """

    target_height, target_width = target_size

    image = Image.fromarray(
        band.astype(np.float32),
        mode="F",
    )

    resized = image.resize(
        (target_width, target_height),
        Image.Resampling.BILINEAR,
    )

    return np.asarray(
        resized,
        dtype=np.float32,
    )


def preprocess_sar_patch(
    vv_path: Path,
    vh_path: Path,
    target_size: tuple[int, int] = TARGET_SIZE,
) -> torch.Tensor:
    """
    Complete Sentinel-1 preprocessing pipeline.

    Args:
        vv_path: Path to VV GeoTIFF.
        vh_path: Path to VH GeoTIFF.
        target_size: Output image size.

    Returns:
        Tensor with shape (2, 224, 224).

        Channel 0 = VV
        Channel 1 = VH
    """

    # ---------------------------------------------------------
    # 1. Load raw VV and VH
    # ---------------------------------------------------------

    vv_raw = load_sar_band(vv_path)
    vh_raw = load_sar_band(vh_path)

    if vv_raw.shape != vh_raw.shape:
        raise ValueError(
            "VV and VH dimensions do not match: "
            f"VV={vv_raw.shape}, VH={vh_raw.shape}"
        )

    # ---------------------------------------------------------
    # 2. Clip SAR values
    # ---------------------------------------------------------

    vv_clipped = clip_sar_band(vv_raw)
    vh_clipped = clip_sar_band(vh_raw)

    # ---------------------------------------------------------
    # 3. Load training-set statistics
    # ---------------------------------------------------------

    mean, std = load_s1_stats()

    vv_mean = mean[0]
    vh_mean = mean[1]

    vv_std = std[0]
    vh_std = std[1]

    # ---------------------------------------------------------
    # 4. Z-score normalization
    # ---------------------------------------------------------

    vv_normalized = normalize_sar_band(
        vv_clipped,
        vv_mean,
        vv_std,
    )

    vh_normalized = normalize_sar_band(
        vh_clipped,
        vh_mean,
        vh_std,
    )

    # ---------------------------------------------------------
    # 5. Resize to 224x224
    # ---------------------------------------------------------

    vv_resized = resize_sar_band(
        vv_normalized,
        target_size,
    )

    vh_resized = resize_sar_band(
        vh_normalized,
        target_size,
    )

    # ---------------------------------------------------------
    # 6. Stack channels
    # ---------------------------------------------------------

    output = np.stack(
        [
            vv_resized,
            vh_resized,
        ],
        axis=0,
    ).astype(np.float32)

    # ---------------------------------------------------------
    # 7. Validate final shape
    # ---------------------------------------------------------

    expected_shape = (
        2,
        target_size[0],
        target_size[1],
    )

    if output.shape != expected_shape:
        raise ValueError(
            f"Unexpected SAR output shape: "
            f"{output.shape}, expected {expected_shape}"
        )

    if not np.isfinite(output).all():
        raise ValueError(
            "Final SAR tensor contains NaN or infinite values."
        )

    return torch.from_numpy(output)


if __name__ == "__main__":
    print("Sentinel-1 preprocessing module loaded successfully.")

    mean, std = load_s1_stats()

    print(f"VV mean: {mean[0]:.6f}")
    print(f"VV std:  {std[0]:.6f}")
    print(f"VH mean: {mean[1]:.6f}")
    print(f"VH std:  {std[1]:.6f}")