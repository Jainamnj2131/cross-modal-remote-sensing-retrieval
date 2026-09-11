import os
import pandas as pd


PAIRED_METADATA_PATH = "outputs/metadata/paired_metadata.csv"

TRAIN_OUTPUT_PATH = "outputs/metadata/train_split.csv"
VAL_OUTPUT_PATH = "outputs/metadata/val_split.csv"
TEST_OUTPUT_PATH = "outputs/metadata/test_split.csv"

REPORT_OUTPUT_PATH = "outputs/reports/split_report.txt"

TRAIN_SIZE = 21000
VAL_SIZE = 4500
TEST_SIZE = 4500

RANDOM_SEED = 42


def create_final_splits():

    print("=" * 60)
    print("FINAL DATASET SPLIT GENERATION")
    print("=" * 60)

    # Load paired metadata
    if not os.path.exists(PAIRED_METADATA_PATH):
        raise FileNotFoundError(
            f"Paired metadata not found: {PAIRED_METADATA_PATH}"
        )

    df = pd.read_csv(PAIRED_METADATA_PATH)

    print(f"\nTotal paired samples: {len(df)}")

    print("\nOriginal split distribution:")
    print(df["split"].value_counts())

    # Separate official BigEarthNet partitions
    official_train = df[df["split"] == "train"].copy()

    official_val = df[df["split"] == "validation"].copy()

    official_test = df[df["split"] == "test"].copy()

    # Check sufficient samples exist
    if len(official_train) < TRAIN_SIZE:
        raise ValueError(
            f"Not enough train samples. "
            f"Available: {len(official_train)}, Required: {TRAIN_SIZE}"
        )

    if len(official_val) < VAL_SIZE:
        raise ValueError(
            f"Not enough validation samples. "
            f"Available: {len(official_val)}, Required: {VAL_SIZE}"
        )

    if len(official_test) < TEST_SIZE:
        raise ValueError(
            f"Not enough test samples. "
            f"Available: {len(official_test)}, Required: {TEST_SIZE}"
        )

    # Sample independently from each official split
    train_df = official_train.sample(
        n=TRAIN_SIZE,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    val_df = official_val.sample(
        n=VAL_SIZE,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    test_df = official_test.sample(
        n=TEST_SIZE,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    # -----------------------------
    # VERIFICATION
    # -----------------------------

    train_ids = set(train_df["patch_id"])
    val_ids = set(val_df["patch_id"])
    test_ids = set(test_df["patch_id"])

    train_val_overlap = len(train_ids & val_ids)
    train_test_overlap = len(train_ids & test_ids)
    val_test_overlap = len(val_ids & test_ids)

    total_unique = len(train_ids | val_ids | test_ids)

    expected_total = TRAIN_SIZE + VAL_SIZE + TEST_SIZE

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    print(f"Train samples:      {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples:       {len(test_df)}")

    print(f"\nExpected total: {expected_total}")
    print(f"Unique patch IDs: {total_unique}")

    print("\nSplit overlaps:")
    print(f"Train ↔ Validation: {train_val_overlap}")
    print(f"Train ↔ Test:       {train_test_overlap}")
    print(f"Validation ↔ Test:  {val_test_overlap}")

    # Strict verification
    if len(train_df) != TRAIN_SIZE:
        raise RuntimeError("Train split size verification failed.")

    if len(val_df) != VAL_SIZE:
        raise RuntimeError("Validation split size verification failed.")

    if len(test_df) != TEST_SIZE:
        raise RuntimeError("Test split size verification failed.")

    if total_unique != expected_total:
        raise RuntimeError(
            "Duplicate patch IDs detected across final splits."
        )

    if (
        train_val_overlap != 0
        or train_test_overlap != 0
        or val_test_overlap != 0
    ):
        raise RuntimeError(
            "Split overlap detected."
        )

    print("\n[SUCCESS] All split verification checks passed.")

    # -----------------------------
    # SAVE FILES
    # -----------------------------

    os.makedirs(
        os.path.dirname(TRAIN_OUTPUT_PATH),
        exist_ok=True
    )

    train_df.to_csv(
        TRAIN_OUTPUT_PATH,
        index=False
    )

    val_df.to_csv(
        VAL_OUTPUT_PATH,
        index=False
    )

    test_df.to_csv(
        TEST_OUTPUT_PATH,
        index=False
    )

    # Create report
    report = f"""
============================================================
FINAL DATASET SPLIT REPORT
============================================================

Source:
{PAIRED_METADATA_PATH}

Original Paired Metadata:

Train Available:      {len(official_train)}
Validation Available: {len(official_val)}
Test Available:       {len(official_test)}

------------------------------------------------------------

Final Dataset Splits:

Train:      {len(train_df)}
Validation: {len(val_df)}
Test:       {len(test_df)}

Total: {expected_total}

------------------------------------------------------------

Verification:

Unique Patch IDs: {total_unique}

Train-Val Overlap: {train_val_overlap}
Train-Test Overlap: {train_test_overlap}
Val-Test Overlap: {val_test_overlap}

Status: PASSED
============================================================
"""

    os.makedirs(
        os.path.dirname(REPORT_OUTPUT_PATH),
        exist_ok=True
    )

    with open(
        REPORT_OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    print("\nFiles saved successfully:")

    print(f"  {TRAIN_OUTPUT_PATH}")
    print(f"  {VAL_OUTPUT_PATH}")
    print(f"  {TEST_OUTPUT_PATH}")
    print(f"  {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    create_final_splits()