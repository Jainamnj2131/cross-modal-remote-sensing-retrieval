"""
Real-data sanity test for the paired Sentinel-1 + Sentinel-2 pipeline.

This script loads ONE real training batch from:
    - Sentinel-1 required subset
    - Sentinel-2 required subset
    - training split CSV

It verifies:
    SAR    -> (64, 2, 224, 224)
    MS     -> (64, 3, 224, 224)
    Labels -> (64, 19)
    IDs    -> 64 patch IDs

No fake or placeholder data is used.
"""

from pathlib import Path

import torch

from src.data.dataloader import create_paired_dataloaders


# ============================================================
# CONFIG
# ============================================================

TRAIN_CSV = Path("outputs/metadata/train_split.csv")
VAL_CSV = Path("outputs/metadata/val_split.csv")

S1_ROOT = Path(r"C:\BigEarthNet-S1-Required\BigEarthNet-S1-Required")
S2_ROOT = Path(r"C:\BigEarthNet-S2-Required")

BATCH_SIZE = 64
NUM_WORKERS = 0


# ============================================================
# MAIN TEST
# ============================================================

def main() -> None:

    print("=" * 70)
    print("REAL PAIRED S1 + S2 BATCH SANITY TEST")
    print("=" * 70)

    print()
    print("Configuration:")
    print(f"Train CSV : {TRAIN_CSV}")
    print(f"Val CSV   : {VAL_CSV}")
    print(f"S1 Root   : {S1_ROOT}")
    print(f"S2 Root   : {S2_ROOT}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Workers   : {NUM_WORKERS}")
    print()

    # --------------------------------------------------------
    # Check paths before loading
    # --------------------------------------------------------

    paths = {
        "Train CSV": TRAIN_CSV,
        "Val CSV": VAL_CSV,
        "S1 Root": S1_ROOT,
        "S2 Root": S2_ROOT,
    }

    for name, path in paths.items():

        if not path.exists():
            raise FileNotFoundError(
                f"{name} does not exist:\n{path}"
            )

        print(f"[OK] {name}: {path}")

    print()

    # --------------------------------------------------------
    # Create paired DataLoaders
    # --------------------------------------------------------

    print("Creating paired DataLoaders...")
    print()

    train_loader, val_loader = create_paired_dataloaders(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        s1_root=S1_ROOT,
        s2_root=S2_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print()
    print("[OK] Paired DataLoaders created.")

    print(f"Train samples: {len(train_loader.dataset):,}")
    print(f"Val samples  : {len(val_loader.dataset):,}")

    print()

    # --------------------------------------------------------
    # Load ONE REAL TRAINING BATCH
    # --------------------------------------------------------

    print("Loading first REAL training batch...")
    print("(This may take some time because S1/S2 files are read")
    print("and preprocessed on demand.)")
    print()

    sar_batch, ms_batch, label_batch, patch_ids = next(
        iter(train_loader)
    )

    # --------------------------------------------------------
    # Print actual results
    # --------------------------------------------------------

    print("=" * 70)
    print("BATCH RESULTS")
    print("=" * 70)

    print()
    print(f"SAR shape    : {sar_batch.shape}")
    print(f"MS shape     : {ms_batch.shape}")
    print(f"Labels shape : {label_batch.shape}")
    print(f"Patch IDs    : {len(patch_ids)}")

    print()

    print(f"SAR dtype    : {sar_batch.dtype}")
    print(f"MS dtype     : {ms_batch.dtype}")
    print(f"Labels dtype : {label_batch.dtype}")

    print()

    # --------------------------------------------------------
    # Expected shape checks
    # --------------------------------------------------------

    expected_sar_shape = (
        BATCH_SIZE,
        2,
        224,
        224,
    )

    expected_ms_shape = (
        BATCH_SIZE,
        3,
        224,
        224,
    )

    expected_label_shape = (
        BATCH_SIZE,
        19,
    )

    print("=" * 70)
    print("SHAPE VERIFICATION")
    print("=" * 70)

    if tuple(sar_batch.shape) != expected_sar_shape:
        raise RuntimeError(
            f"Incorrect SAR shape.\n"
            f"Expected: {expected_sar_shape}\n"
            f"Got     : {tuple(sar_batch.shape)}"
        )

    print(f"[PASS] SAR shape    = {expected_sar_shape}")

    if tuple(ms_batch.shape) != expected_ms_shape:
        raise RuntimeError(
            f"Incorrect MS shape.\n"
            f"Expected: {expected_ms_shape}\n"
            f"Got     : {tuple(ms_batch.shape)}"
        )

    print(f"[PASS] MS shape     = {expected_ms_shape}")

    if tuple(label_batch.shape) != expected_label_shape:
        raise RuntimeError(
            f"Incorrect label shape.\n"
            f"Expected: {expected_label_shape}\n"
            f"Got     : {tuple(label_batch.shape)}"
        )

    print(f"[PASS] Labels shape = {expected_label_shape}")

    if len(patch_ids) != BATCH_SIZE:
        raise RuntimeError(
            f"Incorrect number of patch IDs.\n"
            f"Expected: {BATCH_SIZE}\n"
            f"Got     : {len(patch_ids)}"
        )

    print(f"[PASS] Patch IDs    = {BATCH_SIZE}")

    # --------------------------------------------------------
    # Numerical sanity checks
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("NUMERICAL VERIFICATION")
    print("=" * 70)

    if not torch.isfinite(sar_batch).all():
        raise RuntimeError(
            "SAR batch contains NaN or Inf values."
        )

    print("[PASS] SAR contains no NaN/Inf")

    if not torch.isfinite(ms_batch).all():
        raise RuntimeError(
            "MS batch contains NaN or Inf values."
        )

    print("[PASS] MS contains no NaN/Inf")

    if not torch.isfinite(label_batch).all():
        raise RuntimeError(
            "Label batch contains NaN or Inf values."
        )

    print("[PASS] Labels contain no NaN/Inf")

    # --------------------------------------------------------
    # Display a few patch IDs
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAMPLE PATCH IDS")
    print("=" * 70)

    for patch_id in patch_ids[:5]:
        print(f"  {patch_id}")

    # --------------------------------------------------------
    # Final success
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)

    print()
    print(
        "The real Sentinel-1 + Sentinel-2 paired pipeline "
        "successfully produced a valid batch."
    )

    print()
    print(
        "We are ready to proceed to the Week 4 training script."
    )


if __name__ == "__main__":
    main()