import os
import sys

# Ensure project root is in Python path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
import numpy as np

from src.data.dataloader import create_dataloader
from src.utils.config import load_config


def verify_dataloader():
    """
    Verification script for BigEarthNet PyTorch Data Pipeline.
    Loads data loaders for train, val, and test splits, inspects tensor statistics,
    and visualizes sample RGB images from a training batch.
    """
    print("=" * 60)
    print("1. Loading Configuration")
    print("=" * 60)
    config = load_config("configs/config.yaml")
    print("Configuration loaded successfully:", config)

    print("\n" + "=" * 60)
    print("2. Instantiating DataLoaders for Train, Val, and Test")
    print("=" * 60)
    train_loader = create_dataloader(split="train", config=config)
    val_loader = create_dataloader(split="val", config=config)
    test_loader = create_dataloader(split="test", config=config)

    print(f"Train Dataset Size: {len(train_loader.dataset)}")
    print(f"Val Dataset Size:   {len(val_loader.dataset)}")
    print(f"Test Dataset Size:  {len(test_loader.dataset)}")

    print("\n" + "=" * 60)
    print("3. Fetching Exactly One Batch from Train DataLoader")
    print("=" * 60)
    batch = next(iter(train_loader))

    images = batch["image"]
    patch_ids = batch["patch_id"]
    labels = batch["labels"]

    print(f"Image Batch Shape:  {images.shape}")
    print(f"Image Batch Dtype:  {images.dtype}")
    print(f"patch_id Type:      {type(patch_ids)} (length: {len(patch_ids)})")
    print(f"labels Type:        {type(labels)} (length: {len(labels)})")

    print("\n" + "=" * 60)
    print("4. Inspecting First 5 Samples in the Batch")
    print("=" * 60)
    for i in range(min(5, len(patch_ids))):
        print(f"\nSample [{i+1}]:")
        print(f"  Patch ID: {patch_ids[i]}")
        print(f"  Labels:   {labels[i]}")

    print("\n" + "=" * 60)
    print("5. Visualizing 6 RGB Images from the Batch")
    print("=" * 60)

    # Ensure output visualization directory exists
    output_dir = os.path.join("outputs", "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, "sample_batch.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    for idx, ax in enumerate(axes.flat):
        if idx >= 6 or idx >= len(images):
            break

        # Convert PyTorch Tensor (C, H, W) to NumPy array for plotting
        img_np = images[idx].numpy()

        # Transpose from (C, H, W) to (H, W, C) for matplotlib
        img_rgb = img_np.transpose(1, 2, 0)

        # Scale pixel values ONLY for matplotlib visualization display [0, 1]
        img_min = img_rgb.min()
        img_max = img_rgb.max()

        if img_max > img_min:
            img_display = (img_rgb - img_min) / (img_max - img_min)
        else:
            img_display = img_rgb.astype(np.float32)

        ax.imshow(img_display)
        # Display first 2 labels as title for readability
        label_text = "\n".join(labels[idx][:2])
        if len(labels[idx]) > 2:
            label_text += f"\n(+{len(labels[idx]) - 2} more)"

        ax.set_title(f"Patch: {patch_ids[idx][-12:]}\n{label_text}", fontsize=9)
        ax.axis("off")

    plt.suptitle("BigEarthNet Sample Batch RGB Patches (Train Split)", fontsize=14)
    plt.tight_layout()

    # Save visualization plot to file
    plt.savefig(output_filepath, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to: {output_filepath}")

    # Display plot interactively
    plt.show()


if __name__ == "__main__":
    verify_dataloader()
