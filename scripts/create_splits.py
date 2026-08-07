import os
import sys
import numpy as np
import pandas as pd

# Ensure project root is in Python path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

SUBSET_PATH = "outputs/metadata/subset_30k.csv"
TRAIN_OUTPUT_PATH = "outputs/metadata/train_split.csv"
VAL_OUTPUT_PATH = "outputs/metadata/val_split.csv"
TEST_OUTPUT_PATH = "outputs/metadata/test_split.csv"
REPORT_OUTPUT_PATH = "outputs/reports/split_report.txt"

VAL_TARGET_SIZE = 4500
RANDOM_SEED = 42


def create_splits():
    """
    Creates official, non-overlapping train, validation, and test splits from subset_30k.csv.
    Strictly preserves official BigEarthNet test set boundaries.
    """
    print("=" * 60)
    print("Week 1 - Deliverable 3: Dataset Splits Generation")
    print("=" * 60)

    if not os.path.exists(SUBSET_PATH):
        raise FileNotFoundError(f"Subset file not found at: {SUBSET_PATH}")

    print(f"Reading 30k subset from: {SUBSET_PATH}...")
    df = pd.read_csv(SUBSET_PATH)
    total_subset_size = len(df)
    print(f"Total subset samples: {total_subset_size}")

    # Inspect official split column breakdown
    split_counts = df["split"].value_counts().to_dict()
    print("Official split distribution in 30k subset:")
    for s_name, s_count in split_counts.items():
        print(f"  Official '{s_name}': {s_count} samples")

    # 1. Strictly isolate all official test samples into test_split
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    val_candidates_df = df[df["split"] == "validation"].copy().reset_index(drop=True)
    train_candidates_df = df[df["split"] == "train"].copy().reset_index(drop=True)

    # 2. Sample exactly VAL_TARGET_SIZE (4,500) from official validation candidates
    rng = np.random.RandomState(RANDOM_SEED)
    val_perm = rng.permutation(len(val_candidates_df))
    val_selected_idx = val_perm[:VAL_TARGET_SIZE]
    val_remaining_idx = val_perm[VAL_TARGET_SIZE:]

    val_df = val_candidates_df.iloc[val_selected_idx].copy().reset_index(drop=True)
    val_remaining_df = val_candidates_df.iloc[val_remaining_idx].copy()

    # 3. Combine official train candidates + remaining official val candidates into train_split
    train_df = pd.concat([train_candidates_df, val_remaining_df], ignore_index=True)

    # Verification checks
    train_size = len(train_df)
    val_size = len(val_df)
    test_size = len(test_df)
    union_size = train_size + val_size + test_size

    s_train = set(train_df["patch_id"])
    s_val = set(val_df["patch_id"])
    s_test = set(test_df["patch_id"])

    overlap_train_val = len(s_train.intersection(s_val))
    overlap_train_test = len(s_train.intersection(s_test))
    overlap_val_test = len(s_val.intersection(s_test))
    total_duplicates = overlap_train_val + overlap_train_test + overlap_val_test

    unique_total = len(s_train.union(s_val).union(s_test))

    # Check official test boundary compliance: 0 official test samples in train or val
    test_in_train = (train_df["split"] == "test").sum()
    test_in_val = (val_df["split"] == "test").sum()
    official_test_boundary_verified = (test_in_train == 0) and (test_in_val == 0) and (len(test_df) == split_counts.get("test", 0))

    # Format report text
    report_content = (
        "============================================================\n"
        "BigEarthNet Dataset Splits Verification Report (Deliverable 3)\n"
        "============================================================\n"
        f"Input Subset Source:      {SUBSET_PATH}\n"
        f"Total Subset Samples:     {total_subset_size}\n"
        "------------------------------------------------------------\n"
        "Split Sizes:\n"
        f"  train_split.csv:        {train_size} samples ({train_size / total_subset_size * 100:.2f}%)\n"
        f"  val_split.csv:          {val_size} samples ({val_size / total_subset_size * 100:.2f}%)\n"
        f"  test_split.csv:         {test_size} samples ({test_size / total_subset_size * 100:.2f}%)\n"
        f"  Total Union:            {union_size} samples\n"
        "------------------------------------------------------------\n"
        "Verification Checks:\n"
        f"  Duplicate Count:        {total_duplicates}\n"
        f"  Overlap Train <-> Val:  {overlap_train_val}\n"
        f"  Overlap Train <-> Test: {overlap_train_test}\n"
        f"  Overlap Val <-> Test:   {overlap_val_test}\n"
        f"  Total Unique Patches:   {unique_total}\n"
        f"  Official Test Boundary: {'PASSED (0 official test samples in train/val)' if official_test_boundary_verified else 'FAILED'}\n"
        "============================================================\n"
    )

    print("\n" + report_content)

    # Save CSV outputs
    os.makedirs(os.path.dirname(TRAIN_OUTPUT_PATH), exist_ok=True)
    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
    print(f"Saved: {TRAIN_OUTPUT_PATH} ({train_size} rows)")

    val_df.to_csv(VAL_OUTPUT_PATH, index=False)
    print(f"Saved: {VAL_OUTPUT_PATH} ({val_size} rows)")

    test_df.to_csv(TEST_OUTPUT_PATH, index=False)
    print(f"Saved: {TEST_OUTPUT_PATH} ({test_size} rows)")

    # Save report
    os.makedirs(os.path.dirname(REPORT_OUTPUT_PATH), exist_ok=True)
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved report: {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    create_splits()
