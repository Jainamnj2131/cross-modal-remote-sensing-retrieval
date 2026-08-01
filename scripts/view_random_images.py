import os
import random

import matplotlib.pyplot as plt
import numpy as np
import rasterio

# ==========================
# Dataset Path
# ==========================

DATASET_PATH = r"D:\Dataset"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {DATASET_PATH}"
    )

# ==========================
# Find All Image Patches
# ==========================

patch_folders = []

for scene_folder in os.listdir(DATASET_PATH):

    scene_path = os.path.join(DATASET_PATH, scene_folder)

    if not os.path.isdir(scene_path):
        continue

    for patch_folder in os.listdir(scene_path):

        patch_folder_path = os.path.join(scene_path, patch_folder)

        if not os.path.isdir(patch_folder_path):
            continue

        for patch in os.listdir(patch_folder_path):

            patch_path = os.path.join(patch_folder_path, patch)

            if os.path.isdir(patch_path):
                patch_folders.append(patch_path)

print(f"Total patches found: {len(patch_folders)}")

# ==========================
# Select 10 Random Patches
# ==========================

if len(patch_folders) < 10:
    raise ValueError("Less than 10 patches found.")

random_patches = random.sample(patch_folders, 10)

# ==========================
# Display 10 Random RGB Images
# ==========================

fig, axes = plt.subplots(2, 5, figsize=(16, 7))

for ax, patch in zip(axes.flat, random_patches):

    red = rasterio.open(
        os.path.join(
            patch,
            [f for f in os.listdir(patch) if "_B04.tif" in f][0]
        )
    ).read(1)

    green = rasterio.open(
        os.path.join(
            patch,
            [f for f in os.listdir(patch) if "_B03.tif" in f][0]
        )
    ).read(1)

    blue = rasterio.open(
        os.path.join(
            patch,
            [f for f in os.listdir(patch) if "_B02.tif" in f][0]
        )
    ).read(1)

    rgb = np.dstack((red, green, blue))

    rgb = rgb.astype(np.float32)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    ax.imshow(rgb)
    ax.axis("off")

plt.suptitle("10 Random Sentinel-2 RGB Patches", fontsize=16)

plt.tight_layout()

plt.show()