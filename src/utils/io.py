import os
import numpy as np
import rasterio


def load_sentinel2_patch(patch_path, bands):
    """
    Reads requested Sentinel-2 GeoTIFF image bands from disk and stacks them.

    Args:
        patch_path (str): Path to the Sentinel-2 patch directory.
        bands (list of str): List of band names to load (e.g., ["B04", "B03", "B02"]).

    Returns:
        np.ndarray: Stacked raw NumPy array of shape (Channels, Height, Width).
    """
    if not os.path.exists(patch_path) or not os.path.isdir(patch_path):
        raise FileNotFoundError(
            f"Patch directory does not exist: {patch_path}"
        )

    # Get list of all files in the patch directory
    patch_files = os.listdir(patch_path)

    loaded_bands = []

    for band in bands:
        # Match GeoTIFF file ending with requested band (e.g., '_B04.tif')
        target_file = None
        for filename in patch_files:
            if filename.endswith(f"_{band}.tif") or filename.endswith(f"_{band}.TIF"):
                target_file = filename
                break

        if target_file is None:
            raise FileNotFoundError(
                f"Band '{band}' file missing in patch directory: {patch_path}"
            )

        band_filepath = os.path.join(patch_path, target_file)

        # Read raw band raster data (2D array)
        with rasterio.open(band_filepath) as dataset:
            band_array = dataset.read(1)

        loaded_bands.append(band_array)

    # Stack bands along the channel dimension (Channels, Height, Width)
    raw_band_stack = np.stack(loaded_bands, axis=0)

    return raw_band_stack
