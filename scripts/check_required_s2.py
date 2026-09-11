"""
Check the S2 storage required for the final project splits.

This script:
1. Reads the final train, validation, and test split CSVs.
2. Collects required S2 patch IDs from the `patch_id` column.
3. Searches the actual two-level BigEarthNet-S2 directory structure.
4. Matches the required patch IDs to physical patch folders.
5. Calculates the exact storage required.

IMPORTANT:
This script DOES NOT copy, move, delete, or modify dataset files.
"""

from pathlib import Path
import os

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

S2_ROOT = Path(r"D:\Dataset\BigEarthNet-S2")

TRAIN_CSV = Path(r"outputs\metadata\train_split.csv")
VAL_CSV = Path(r"outputs\metadata\val_split.csv")
TEST_CSV = Path(r"outputs\metadata\test_split.csv")


# ============================================================
# HELPERS
# ============================================================

def format_size(size_bytes: int) -> str:
    """Convert bytes into a human-readable size."""

    units = ["B", "KB", "MB", "GB", "TB"]

    size = float(size_bytes)

    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} PB"


def get_directory_size(directory: Path) -> int:
    """
    Calculate the total size of files inside one required patch.

    This is only called for patches that were actually found.
    """

    total = 0

    try:
        for root, _, files in os.walk(directory):

            for filename in files:

                file_path = os.path.join(root, filename)

                try:
                    total += os.path.getsize(file_path)

                except OSError:
                    pass

    except OSError:
        pass

    return total


# ============================================================
# LOAD REQUIRED PATCH IDs
# ============================================================

def load_required_patch_ids():

    print("=" * 70)
    print("LOADING FINAL PROJECT SPLITS")
    print("=" * 70)

    csv_files = [
        ("Train", TRAIN_CSV),
        ("Validation", VAL_CSV),
        ("Test", TEST_CSV),
    ]

    required_patch_ids = set()

    for split_name, csv_path in csv_files:

        if not csv_path.exists():
            raise FileNotFoundError(
                f"{split_name} CSV not found:\n{csv_path}"
            )

        df = pd.read_csv(csv_path)

        # ----------------------------------------------------
        # We intentionally use patch_id.
        # It matches the physical S2 patch folder name.
        # ----------------------------------------------------

        if "patch_id" not in df.columns:
            raise ValueError(
                f"'patch_id' column missing from:\n{csv_path}"
            )

        names = (
            df["patch_id"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        unique_names = set(names)

        print(
            f"{split_name:12} : "
            f"{len(df):6} rows | "
            f"{len(unique_names):6} unique patch IDs"
        )

        required_patch_ids.update(unique_names)

    print("-" * 70)

    print(
        f"TOTAL UNIQUE S2 PATCHES REQUIRED: "
        f"{len(required_patch_ids):,}"
    )

    return required_patch_ids


# ============================================================
# FIND REQUIRED PATCHES
# ============================================================

def find_required_patches(required_patch_ids):

    print()
    print("=" * 70)
    print("SEARCHING BIGEARTHNET-S2")
    print("=" * 70)

    if not S2_ROOT.exists():
        raise FileNotFoundError(
            f"S2 dataset directory does not exist:\n{S2_ROOT}"
        )

    if not S2_ROOT.is_dir():
        raise NotADirectoryError(
            f"S2 root is not a directory:\n{S2_ROOT}"
        )

    print(f"S2 root: {S2_ROOT}")
    print()
    print("Expected structure:")
    print("  S2 root")
    print("    -> Tile/Granule folder")
    print("       -> Patch folder")
    print()
    print("Searching only these two directory levels.")
    print("Nothing will be copied or modified.")
    print()

    found = {}

    tile_count = 0
    patch_count = 0

    # --------------------------------------------------------
    # Level 1:
    #
    # D:\Dataset\BigEarthNet-S2\
    #     S2A_MSIL2A_...._T33UUP
    #
    # --------------------------------------------------------

    try:

        with os.scandir(S2_ROOT) as tile_entries:

            for tile_entry in tile_entries:

                if not tile_entry.is_dir():
                    continue

                tile_count += 1

                # ------------------------------------------------
                # Level 2:
                #
                # tile folder
                #     -> patch folders
                #
                # ------------------------------------------------

                try:

                    with os.scandir(tile_entry.path) as patch_entries:

                        for patch_entry in patch_entries:

                            if not patch_entry.is_dir():
                                continue

                            patch_count += 1

                            patch_name = patch_entry.name

                            # Only keep patches that are actually
                            # required by our final splits.
                            if patch_name in required_patch_ids:

                                found[patch_name] = Path(
                                    patch_entry.path
                                )

                except PermissionError:

                    print(
                        f"WARNING: Permission denied: "
                        f"{tile_entry.path}"
                    )

                # Progress every 100 tile folders.
                if tile_count % 100 == 0:

                    print(
                        f"Scanned {tile_count:,} tile folders | "
                        f"{patch_count:,} patches checked | "
                        f"{len(found):,} required patches found"
                    )

    except PermissionError as exc:

        raise PermissionError(
            f"Permission denied while reading S2 root:\n"
            f"{S2_ROOT}"
        ) from exc

    print()
    print("-" * 70)

    print(
        f"Tile folders scanned : "
        f"{tile_count:,}"
    )

    print(
        f"Patch folders checked : "
        f"{patch_count:,}"
    )

    print(
        f"Required patches found : "
        f"{len(found):,}"
    )

    # --------------------------------------------------------
    # Determine missing patches
    # --------------------------------------------------------

    missing = sorted(
        required_patch_ids - set(found.keys())
    )

    return found, missing


# ============================================================
# CALCULATE REQUIRED STORAGE
# ============================================================

def calculate_total_size(found_patches):

    print()
    print("=" * 70)
    print("CALCULATING REQUIRED STORAGE")
    print("=" * 70)

    total_size = 0

    total_found = len(found_patches)

    for index, (patch_name, patch_path) in enumerate(
        found_patches.items(),
        start=1,
    ):

        patch_size = get_directory_size(patch_path)

        total_size += patch_size

        if index % 1000 == 0:

            print(
                f"Calculated {index:,} / "
                f"{total_found:,} patches | "
                f"Current size: "
                f"{format_size(total_size)}"
            )

    return total_size


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load required patch IDs
    # --------------------------------------------------------

    required_patch_ids = load_required_patch_ids()

    # --------------------------------------------------------
    # 2. Find physical S2 patch folders
    # --------------------------------------------------------

    found, missing = find_required_patches(
        required_patch_ids
    )

    # --------------------------------------------------------
    # 3. Calculate storage
    # --------------------------------------------------------

    total_size = calculate_total_size(found)

    # --------------------------------------------------------
    # 4. Final result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        f"Required unique S2 patches : "
        f"{len(required_patch_ids):,}"
    )

    print(
        f"Found locally               : "
        f"{len(found):,}"
    )

    print(
        f"Missing locally             : "
        f"{len(missing):,}"
    )

    print(
        f"Required storage            : "
        f"{format_size(total_size)}"
    )

    # --------------------------------------------------------
    # Missing patch information
    # --------------------------------------------------------

    if missing:

        print()
        print("First 20 missing patches:")

        for patch_name in missing[:20]:
            print(f"  {patch_name}")

    else:

        print()
        print(
            "SUCCESS: All required S2 patches "
            "were found locally."
        )

    print()
    print("=" * 70)
    print("NO FILES WERE COPIED OR MODIFIED.")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()