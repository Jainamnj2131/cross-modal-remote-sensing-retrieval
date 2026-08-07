import os
import sys
import pandas as pd

# Ensure project root is in Python path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

METADATA_PATH = "D:/Dataset/metadata.parquet"
REPORT_PATH = "outputs/reports/verification_report.txt"
CSV_OUTPUT_PATH = "outputs/metadata/paired_metadata.csv"


def verify_pairs():
    """
    Performs metadata-level verification of Sentinel-1 (SAR) and Sentinel-2 (MSI)
    pair integrity using BigEarthNet metadata.parquet.
    """
    print("=" * 60)
    print("Week 1 - Deliverable 1: Metadata-Level S1 <-> S2 Pair Verification")
    print("=" * 60)

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found at: {METADATA_PATH}")

    print(f"Reading metadata from: {METADATA_PATH}...")
    df = pd.read_parquet(METADATA_PATH)

    total_samples = len(df)
    print(f"Total samples loaded: {total_samples}")

    # Check for missing identifiers
    is_s2_missing = df["patch_id"].isnull() | (df["patch_id"].astype(str).str.strip() == "")
    is_s1_missing = df["s1_name"].isnull() | (df["s1_name"].astype(str).str.strip() == "")

    missing_s2_count = int(is_s2_missing.sum())
    missing_s1_count = int(is_s1_missing.sum())

    # Invalid entries: samples where both S1 and S2 are missing/invalid
    invalid_entries_count = int((is_s2_missing & is_s1_missing).sum())

    # Valid pairs: samples where both patch_id (S2) and s1_name (S1) are valid non-empty strings
    valid_pairs_mask = (~is_s2_missing) & (~is_s1_missing)
    valid_pairs_count = int(valid_pairs_mask.sum())

    # Format verification summary text report
    report_content = (
        "============================================================\n"
        "BigEarthNet Metadata-Level S1 <-> S2 Pair Verification Report\n"
        "============================================================\n"
        f"Metadata Source:        {METADATA_PATH}\n"
        f"Total Samples:          {total_samples}\n"
        f"Valid Pairs:            {valid_pairs_count}\n"
        f"Missing S1 Identifiers: {missing_s1_count}\n"
        f"Missing S2 Identifiers: {missing_s2_count}\n"
        f"Invalid Entries:        {invalid_entries_count}\n"
        "============================================================\n"
        f"Verification Status:    {'SUCCESS (100% Paired)' if valid_pairs_count == total_samples else 'WARNING (Missing/Invalid Pairs Found)'}\n"
        "============================================================\n"
    )

    print("\n" + report_content)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)

    # Export verification report text file
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Verification report saved to: {REPORT_PATH}")

    # Export paired metadata CSV containing valid pairs
    valid_pairs_df = df[valid_pairs_mask].copy()
    valid_pairs_df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"Paired metadata CSV saved to: {CSV_OUTPUT_PATH} ({len(valid_pairs_df)} rows)")


if __name__ == "__main__":
    verify_pairs()
