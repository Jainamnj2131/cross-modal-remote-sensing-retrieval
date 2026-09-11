"""Preprocessing for three-band Sentinel-2 patches."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import json
import numpy as np
import torch
from PIL import Image


# ============================================================
# PATH TO S2 NORMALIZATION STATISTICS
# ============================================================

STATS_PATH = Path(__file__).resolve().parent / "s2_stats.json"


# ============================================================
# LOAD NORMALIZATION STATISTICS
# ============================================================

def load_s2_stats():
    """Load Sentinel-2 mean and standard deviation."""

    if not STATS_PATH.exists():
        raise FileNotFoundError(
            f"Sentinel-2 statistics file not found: {STATS_PATH}"
        )

    with open(STATS_PATH, "r", encoding="utf-8") as file:
        stats = json.load(file)

    if "mean" not in stats or "std" not in stats:
        raise ValueError(
            "s2_stats.json must contain 'mean' and 'std'."
        )

    mean = np.asarray(
        stats["mean"],
        dtype=np.float32
    )

    std = np.asarray(
        stats["std"],
        dtype=np.float32
    )

    if len(mean) != 3 or len(std) != 3:
        raise ValueError(
            "Expected exactly 3 mean and 3 std values "
            "for B04, B03, B02."
        )

    if np.any(std <= 0):
        raise ValueError(
            "S2 standard deviation values must be positive."
        )

    return mean, std


# ============================================================
# FIND SENTINEL-2 BAND
# ============================================================

def _find_band(
    folder: Path,
    band: str
) -> Path:
    """
    Find a Sentinel-2 band file inside a patch folder.
    """

    candidates = sorted(
        path
        for path in folder.rglob("*")
        if (
            path.is_file()
            and band in path.stem
        )
    )

    if not candidates:
        raise FileNotFoundError(
            f"Could not find Sentinel-2 band "
            f"{band} under {folder}"
        )

    return candidates[0]


# ============================================================
# READ BAND
# ============================================================

def _read_band(
    path: Path
) -> np.ndarray:
    """
    Read a single Sentinel-2 band.
    """

    if path.suffix.lower() == ".npy":

        array = np.load(path)

    else:

        with Image.open(path) as image:
            array = np.asarray(image)

    if array.ndim != 2:

        raise ValueError(
            f"Expected a single-band raster at "
            f"{path}, got shape {array.shape}"
        )

    return array.astype(
        np.float32,
        copy=False
    )


# ============================================================
# PREPROCESS SENTINEL-2 PATCH
# ============================================================

def preprocess_s2_patch(
    patch_folder: Path,
    target_size: tuple[int, int] = (224, 224),
    bands: Iterable[str] = ("B04", "B03", "B02"),
) -> torch.Tensor:
    """
    Preprocess a Sentinel-2 patch.

    Channel order:
        Channel 0 -> B04 (Red)
        Channel 1 -> B03 (Green)
        Channel 2 -> B02 (Blue)

    Output:
        Tensor of shape (3, 224, 224)
    """

    patch_folder = Path(patch_folder)

    if not patch_folder.exists():
        raise FileNotFoundError(
            f"Patch folder not found: {patch_folder}"
        )

    bands = tuple(bands)

    if len(bands) != 3:
        raise ValueError(
            "Exactly three Sentinel-2 bands are required."
        )

    # --------------------------------------------------------
    # LOAD B04, B03, B02
    # --------------------------------------------------------

    arrays = [
        _read_band(
            _find_band(
                patch_folder,
                band
            )
        )
        for band in bands
    ]

    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    resized = []

    for array in arrays:

        resized_array = np.asarray(
            Image.fromarray(array).resize(
                (
                    target_size[1],
                    target_size[0]
                ),
                Image.Resampling.BILINEAR
            ),
            dtype=np.float32
        )

        resized.append(
            resized_array
        )

    # --------------------------------------------------------
    # STACK
    # --------------------------------------------------------

    output = np.stack(
        resized,
        axis=0
    )

    # Expected shape:
    # (3, 224, 224)

    if output.shape != (
        3,
        target_size[0],
        target_size[1]
    ):

        raise ValueError(
            f"Unexpected S2 output shape: "
            f"{output.shape}"
        )

    # --------------------------------------------------------
    # DATASET-LEVEL NORMALIZATION
    # --------------------------------------------------------

    mean, std = load_s2_stats()

    mean = mean.reshape(
        3,
        1,
        1
    )

    std = std.reshape(
        3,
        1,
        1
    )

    output = (
        output - mean
    ) / std

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    if not np.isfinite(output).all():

        raise ValueError(
            "S2 preprocessing produced NaN "
            "or infinite values."
        )

    return torch.from_numpy(
        output.astype(
            np.float32,
            copy=False
        )
    )