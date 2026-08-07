import os
import re
import sys
import numpy as np
import pandas as pd
from collections import Counter

# Ensure project root is in Python path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

INPUT_CSV_PATH = "outputs/metadata/paired_metadata.csv"
OUTPUT_CSV_PATH = "outputs/metadata/subset_30k.csv"
RANDOM_SEED = 42
SUBSET_SIZE = 30000

OFFICIAL_TOP_10_LABELS = [
    "Arable land",
    "Mixed forest",
    "Coniferous forest",
    "Transitional woodland, shrub",
    "Broad-leaved forest",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Complex cultivation patterns",
    "Pastures",
    "Urban fabric",
    "Inland waters",
]


def parse_labels(label_str):
    """
    Parses land-cover label string representations into Python list of strings.
    """
    if isinstance(label_str, list):
        return label_str
    return re.findall(r"'([^']+)'", str(label_str))


def create_balanced_subset():
    """
    Generates a class-balanced subset of exactly 30,000 unique paired samples
    from outputs/metadata/paired_metadata.csv using Multi-Label Min-Count Equalization Sampling.
    """
    print("=" * 60)
    print("Week 1 - Deliverable 2: Generating 30,000 Class-Balanced Subset")
    print("=" * 60)

    if not os.path.exists(INPUT_CSV_PATH):
        raise FileNotFoundError(f"Input metadata CSV not found at: {INPUT_CSV_PATH}")

    print(f"Reading input paired metadata from: {INPUT_CSV_PATH}...")
    df = pd.read_csv(INPUT_CSV_PATH)
    original_dataset_size = len(df)
    print(f"Original Dataset Size: {original_dataset_size} samples")

    # Parse labels column for analysis and sampling
    raw_labels = df["labels"].tolist()
    parsed_labels_all = [parse_labels(s) for s in raw_labels]
    
    # Extract only official Top 10 labels for each sample
    top10_labels_per_sample = [
        [lbl for lbl in labels if lbl in OFFICIAL_TOP_10_LABELS]
        for labels in parsed_labels_all
    ]

    # Compute distribution of Top 10 labels before sampling
    before_distribution = Counter()
    for labels in top10_labels_per_sample:
        for lbl in labels:
            before_distribution[lbl] += 1

    print("\n" + "=" * 60)
    print("Top 10 Label Distribution BEFORE Sampling (Original Dataset)")
    print("=" * 60)
    for lbl in OFFICIAL_TOP_10_LABELS:
        count = before_distribution[lbl]
        pct = (count / original_dataset_size) * 100
        print(f"  {lbl:<88}: {count:>7} ({pct:>5.2f}%)")

    # Filter candidates containing at least one of the official Top 10 labels
    candidate_mask = [len(labels) > 0 for labels in top10_labels_per_sample]
    candidate_indices = np.where(candidate_mask)[0]

    # Initialize fixed random state for reproducibility
    rng = np.random.RandomState(RANDOM_SEED)
    shuffled_candidate_indices = rng.permutation(candidate_indices)

    # Pre-build label index mapping for fast greedy min-count retrieval
    label_to_indices = {lbl: [] for lbl in OFFICIAL_TOP_10_LABELS}
    for idx in shuffled_candidate_indices:
        for lbl in top10_labels_per_sample[idx]:
            label_to_indices[lbl].append(idx)

    label_pointers = {lbl: 0 for lbl in OFFICIAL_TOP_10_LABELS}
    selected_indices_set = set()
    after_distribution = Counter()

    print("\nExecuting Multi-Label Min-Count Equalization Sampling...")

    while len(selected_indices_set) < SUBSET_SIZE:
        # Sort labels by current count in selected subset (rarest classes first)
        sorted_labels = sorted(
            OFFICIAL_TOP_10_LABELS,
            key=lambda lbl: (after_distribution[lbl], len(label_to_indices[lbl]))
        )

        found_sample = False
        for lbl in sorted_labels:
            indices_list = label_to_indices[lbl]
            ptr = label_pointers[lbl]

            while ptr < len(indices_list):
                cand_idx = indices_list[ptr]
                label_pointers[lbl] = ptr + 1

                if cand_idx not in selected_indices_set:
                    selected_indices_set.add(cand_idx)
                    for sample_lbl in top10_labels_per_sample[cand_idx]:
                        after_distribution[sample_lbl] += 1
                    found_sample = True
                    break
                ptr += 1

            if found_sample:
                break

        # Fallback: if candidates for Top 10 labels are exhausted, fill remaining
        if not found_sample:
            for idx in shuffled_candidate_indices:
                if len(selected_indices_set) >= SUBSET_SIZE:
                    break
                if idx not in selected_indices_set:
                    selected_indices_set.add(idx)
                    for sample_lbl in top10_labels_per_sample[idx]:
                        after_distribution[sample_lbl] += 1
            break

    selected_indices_list = list(selected_indices_set)
    subset_df = df.iloc[selected_indices_list].copy().reset_index(drop=True)
    final_subset_size = len(subset_df)

    print("\n" + "=" * 60)
    print("Top 10 Label Distribution AFTER Sampling (Subset 30k)")
    print("=" * 60)
    for lbl in OFFICIAL_TOP_10_LABELS:
        count = after_distribution[lbl]
        pct = (count / final_subset_size) * 100
        print(f"  {lbl:<88}: {count:>7} ({pct:>5.2f}%)")

    print("\n" + "=" * 60)
    print("Sampling Summary")
    print("=" * 60)
    print(f"Original Dataset Size: {original_dataset_size}")
    print(f"Final Subset Size:    {final_subset_size}")
    print(f"Unique Patch IDs:     {subset_df['patch_id'].nunique()}")

    # Save subset to outputs/metadata/subset_30k.csv
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    subset_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nSubset successfully saved to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    create_balanced_subset()
