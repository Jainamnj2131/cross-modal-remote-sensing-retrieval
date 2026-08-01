import os
import pyarrow.parquet as pq

# ==========================
# Metadata Path
# ==========================

METADATA_PATH = r"D:\Dataset\metadata.parquet"

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(
        f"Metadata not found at: {METADATA_PATH}"
    )

# ==========================
# Open Metadata
# ==========================

print("Opening metadata file...")

parquet_file = pq.ParquetFile(METADATA_PATH)

print("Metadata opened successfully!\n")

print("Number of rows:", parquet_file.metadata.num_rows)
print("Number of columns:", parquet_file.metadata.num_columns)
print("Number of row groups:", parquet_file.metadata.num_row_groups)

# ==========================
# Read First Row Group
# ==========================

print("\nReading first row group...")

table = parquet_file.read_row_group(0)

print("Row group loaded successfully!\n")

# ==========================
# Convert to Pandas
# ==========================

df = table.to_pandas()

# ==========================
# Display Information
# ==========================

print("First 5 Rows:\n")
print(df.head())

print("\n============================")
print("Column Names")
print("============================")
print(df.columns)

print("\n============================")
print("Dataset Information")
print("============================")
df.info()

print("\n==============================")
print("Train / Validation / Test Split")
print("==============================")

# Read the complete metadata
metadata = pq.read_table(METADATA_PATH).to_pandas()

print(metadata["split"].value_counts())

print("\n==============================")
print("Countries")
print("==============================")

print(metadata["country"].value_counts())

print("\n==============================")
print("Snow Images")
print("==============================")

print(metadata["contains_seasonal_snow"].value_counts())

print("\n==============================")
print("Cloud Images")
print("==============================")

print(metadata["contains_cloud_or_shadow"].value_counts())

from collections import Counter

label_counter = Counter()

for labels in df["labels"]:
    label_counter.update(labels)

print("\n==============================")
print("Top 20 Most Common Labels")
print("==============================")

from collections import Counter

label_counter = Counter()

for labels in metadata["labels"]:
    label_counter.update(labels)

for label, count in label_counter.most_common(20):
    print(f"{label}: {count}")