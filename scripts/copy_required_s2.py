"""
Copy only the required BigEarthNet-S2 patches for the project.

Required data:
    Train      : 21,000 patches
    Validation : 4,500 patches
    Test       : 4,500 patches
    Total      : 30,000 unique patches

The script:
1. Reads the final train/validation/test CSV files.
2. Uses patch_id to identify the physical S2 patch folders.
3. Searches the actual BigEarthNet-S2 two-level structure.
4. Copies only the required patch folders.
5. Preserves the original tile/patch directory structure.
6. Skips files that already exist with the same size.
7. Reports copied, skipped, and missing patches.

IMPORTANT:
The original S2 dataset is NEVER modified.
Only files are copied to the destination.
"""

from pathlib import Path
import os
import shutil

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Your complete S2 dataset.
S2_SOURCE = Path(r"D:\Dataset\BigEarthNet-S2")

# ------------------------------------------------------------
# IMPORTANT:
# Change this to the destination where you want the subset.
#
# Example external drive:
# DESTINATION = Path(r"E:\BigEarthNet-S2-Required")
#
# Example another internal drive:
# DESTINATION = Path(r"C:\S2_Required")
# ------------------------------------------------------------

DESTINATION = Path(r"C:\BigEarthNet-S2-Required")

# Final project split files.
TRAIN_CSV = Path(r"outputs\metadata\train_split.csv")
VAL_CSV = Path(r"outputs\metadata\val_split.csv")
TEST_CSV = Path(r"outputs\metadata\test_split.csv")


# ============================================================
# HELPERS
# ============================================================

def format_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable size."""

    units = ["B", "KB", "MB", "GB", "TB"]

    size = float(size_bytes)

    for unit in units:

        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} PB"


def get_directory_size(directory: Path) -> int:
    """Calculate the total size of a directory."""

    total = 0

    for root, _, files in os.walk(directory):

        for filename in files:

            file_path = Path(root) / filename

            try:
                total += file_path.stat().st_size
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
        f"TOTAL UNIQUE PATCHES TO COPY: "
        f"{len(required_patch_ids):,}"
    )

    return required_patch_ids


# ============================================================
# FIND REQUIRED PATCHES
# ============================================================

def find_required_patches(required_patch_ids):

    print()
    print("=" * 70)
    print("LOCATING REQUIRED S2 PATCHES")
    print("=" * 70)

    if not S2_SOURCE.exists():
        raise FileNotFoundError(
            f"S2 source directory does not exist:\n{S2_SOURCE}"
        )

    found = {}

    tile_count = 0
    patch_count = 0

    print(f"Source: {S2_SOURCE}")
    print()
    print("Scanning the BigEarthNet-S2 two-level structure...")
    print()

    with os.scandir(S2_SOURCE) as tile_entries:

        for tile_entry in tile_entries:

            if not tile_entry.is_dir():
                continue

            tile_count += 1

            with os.scandir(tile_entry.path) as patch_entries:

                for patch_entry in patch_entries:

                    if not patch_entry.is_dir():
                        continue

                    patch_count += 1

                    patch_name = patch_entry.name

                    if patch_name in required_patch_ids:

                        found[patch_name] = Path(
                            patch_entry.path
                        )

            if tile_count % 25 == 0:

                print(
                    f"Scanned {tile_count:3} tile folders | "
                    f"{patch_count:,} patches checked | "
                    f"{len(found):,} required patches found"
                )

    print()
    print("-" * 70)

    print(
        f"Tile folders scanned   : "
        f"{tile_count:,}"
    )

    print(
        f"Patch folders checked   : "
        f"{patch_count:,}"
    )

    print(
        f"Required patches found  : "
        f"{len(found):,}"
    )

    missing = sorted(
        required_patch_ids - set(found.keys())
    )

    return found, missing


# ============================================================
# COPY PATCHES
# ============================================================

def copy_patches(found_patches):

    print()
    print("=" * 70)
    print("COPYING REQUIRED S2 PATCHES")
    print("=" * 70)

    print(f"Destination: {DESTINATION}")
    print()

    DESTINATION.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied = 0
    skipped = 0
    failed = 0

    copied_bytes = 0

    total = len(found_patches)

    for index, (patch_name, source_patch) in enumerate(
        found_patches.items(),
        start=1,
    ):

        # ----------------------------------------------------
        # Get tile folder from source path.
        #
        # source:
        # BigEarthNet-S2/
        #     TILE/
        #         PATCH/
        # ----------------------------------------------------

        tile_dir = source_patch.parent

        destination_tile = (
            DESTINATION / tile_dir.name
        )

        destination_patch = (
            destination_tile / patch_name
        )

        try:

            # ------------------------------------------------
            # If the complete patch already exists,
            # compare directory sizes.
            # ------------------------------------------------

            if destination_patch.exists():

                source_size = get_directory_size(
                    source_patch
                )

                destination_size = get_directory_size(
                    destination_patch
                )

                if source_size == destination_size:

                    skipped += 1

                    if index % 1000 == 0:

                        print(
                            f"{index:,}/{total:,} | "
                            f"Skipped existing: {skipped:,} | "
                            f"Copied: {copied:,}"
                        )

                    continue

                # Existing incomplete/different folder.
                # Remove ONLY the destination copy.
                shutil.rmtree(destination_patch)

            destination_tile.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copytree(
                source_patch,
                destination_patch,
            )

            copied += 1

            copied_bytes += get_directory_size(
                source_patch
            )

        except Exception as exc:

            failed += 1

            print()
            print(
                f"ERROR copying patch: {patch_name}"
            )

            print(f"Source: {source_patch}")
            print(f"Error : {exc}")
            print()

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if index % 500 == 0 or index == total:

            print(
                f"{index:,}/{total:,} | "
                f"Copied: {copied:,} | "
                f"Skipped: {skipped:,} | "
                f"Failed: {failed:,} | "
                f"Copied data: {format_size(copied_bytes)}"
            )

    return copied, skipped, failed


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if S2_SOURCE.resolve() == DESTINATION.resolve():

        raise ValueError(
            "SOURCE and DESTINATION cannot be the same directory."
        )

    print()
    print("=" * 70)
    print("BIGEARTHNET-S2 REQUIRED SUBSET COPY")
    print("=" * 70)

    print()
    print(f"Source      : {S2_SOURCE}")
    print(f"Destination : {DESTINATION}")

    print()
    print(
        "The original dataset will NOT be modified."
    )

    # --------------------------------------------------------
    # 1. Load required patches
    # --------------------------------------------------------

    required_patch_ids = load_required_patch_ids()

    # --------------------------------------------------------
    # 2. Find physical patches
    # --------------------------------------------------------

    found, missing = find_required_patches(
        required_patch_ids
    )

    # --------------------------------------------------------
    # 3. Stop if anything is missing
    # --------------------------------------------------------

    if missing:

        print()
        print("=" * 70)
        print("COPY ABORTED")
        print("=" * 70)

        print(
            f"Missing patches: {len(missing):,}"
        )

        print()
        print("First 20 missing patches:")

        for patch_name in missing[:20]:
            print(f"  {patch_name}")

        print()
        print(
            "No copy operation was started."
        )

        return

    # --------------------------------------------------------
    # 4. Copy
    # --------------------------------------------------------

    copied, skipped, failed = copy_patches(
        found
    )

    # --------------------------------------------------------
    # 5. Final result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COPY COMPLETE")
    print("=" * 70)

    print(
        f"Required patches : "
        f"{len(required_patch_ids):,}"
    )

    print(
        f"Copied           : "
        f"{copied:,}"
    )

    print(
        f"Already present  : "
        f"{skipped:,}"
    )

    print(
        f"Failed           : "
        f"{failed:,}"
    )

    print()

    if failed == 0:

        print(
            "SUCCESS: All required S2 patches "
            "were copied successfully."
        )

    else:

        print(
            "WARNING: Some patches failed to copy."
        )

    print()
    print(f"Destination: {DESTINATION}")

    print("=" * 70)


if __name__ == "__main__":
    main()