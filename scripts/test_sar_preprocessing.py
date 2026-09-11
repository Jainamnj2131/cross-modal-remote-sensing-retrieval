from pathlib import Path
import sys

import torch


# ---------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.preprocessing.sar_preprocessing import preprocess_sar_patch


# ---------------------------------------------------------
# Dataset location
# ---------------------------------------------------------

DATASET_DIR = Path(r"D:\Dataset")


def find_first_band(pattern: str) -> Path:
    """
    Find the first SAR band matching the given pattern.
    """

    matches = sorted(DATASET_DIR.rglob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"Could not find any SAR band matching: {pattern}"
        )

    return matches[0]


def main():

    print("=" * 60)
    print("SENTINEL-1 PREPROCESSING TEST")
    print("=" * 60)

    # -----------------------------------------------------
    # Find one VV and one VH file
    # -----------------------------------------------------

    vv_path = find_first_band("*VV*.tif*")
    vh_path = find_first_band("*VH*.tif*")

    print("\nVV file:")
    print(vv_path)

    print("\nVH file:")
    print(vh_path)

    # -----------------------------------------------------
    # Run preprocessing
    # -----------------------------------------------------

    print("\nRunning preprocessing...")

    tensor = preprocess_sar_patch(
        vv_path=vv_path,
        vh_path=vh_path,
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\nPreprocessing successful!")

    print(f"Tensor shape : {tensor.shape}")
    print(f"Tensor dtype : {tensor.dtype}")

    print("\nVV statistics after normalization:")

    print(
        f"Mean : {tensor[0].mean().item():.4f}"
    )

    print(
        f"Std  : {tensor[0].std().item():.4f}"
    )

    print(
        f"Min  : {tensor[0].min().item():.4f}"
    )

    print(
        f"Max  : {tensor[0].max().item():.4f}"
    )

    print("\nVH statistics after normalization:")

    print(
        f"Mean : {tensor[1].mean().item():.4f}"
    )

    print(
        f"Std  : {tensor[1].std().item():.4f}"
    )

    print(
        f"Min  : {tensor[1].min().item():.4f}"
    )

    print(
        f"Max  : {tensor[1].max().item():.4f}"
    )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    assert isinstance(tensor, torch.Tensor)

    assert tensor.dtype == torch.float32

    assert tensor.shape == (2, 224, 224)

    assert torch.isfinite(tensor).all()

    # -----------------------------------------------------
    # Success
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("S1 PREPROCESSING TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()