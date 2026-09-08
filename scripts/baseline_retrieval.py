import os
import time
import numpy as np
import pandas as pd
import faiss


# ============================================================
# PATHS
# ============================================================

MS_EMBEDDINGS_PATH = "outputs/embeddings/baseline_val_ms_embeddings.npy"
SAR_EMBEDDINGS_PATH = "outputs/embeddings/baseline_val_s1_embeddings.npy"

MS_IDS_PATH = "outputs/embeddings/baseline_val_patch_ids.txt"
SAR_IDS_PATH = "outputs/embeddings/baseline_val_s1_patch_ids.txt"

VAL_CSV_PATH = "outputs/metadata/val_split.csv"

REPORT_PATH = "outputs/reports/baseline_retrieval_results.txt"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_ids(path):
    """Load IDs from a text file."""

    with open(path, "r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def calculate_recall(retrieved_ids, ground_truth_id, k):
    """
    Returns True if the ground-truth ID appears
    within the top-k retrieved results.
    """

    return ground_truth_id in retrieved_ids[:k]


# ============================================================
# MAIN FUNCTION
# ============================================================

def run_baseline_retrieval():

    print("=" * 60)
    print("BASELINE SAR -> MS RETRIEVAL")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD EMBEDDINGS
    # --------------------------------------------------------

    print("\nLoading embeddings...")

    ms_embeddings = np.load(
        MS_EMBEDDINGS_PATH
    ).astype("float32")

    sar_embeddings = np.load(
        SAR_EMBEDDINGS_PATH
    ).astype("float32")

    print("Loading patch IDs...")

    ms_ids = load_ids(MS_IDS_PATH)
    sar_ids = load_ids(SAR_IDS_PATH)

    print("\n" + "=" * 60)
    print("ID FORMAT INSPECTION")
    print("=" * 60)

    print("\nFirst 5 MS IDs from embedding file:")
    for item in ms_ids[:5]:
        print(repr(item))

    print("\nFirst 5 SAR IDs from embedding file:")
    for item in sar_ids[:5]:
        print(repr(item))

    print("\nMS embeddings shape: ", ms_embeddings.shape)
    print("SAR embeddings shape:", sar_embeddings.shape)

    print("MS IDs:", len(ms_ids))
    print("SAR IDs:", len(sar_ids))

    # --------------------------------------------------------
    # BASIC FILE VALIDATION
    # --------------------------------------------------------

    if len(ms_embeddings) != len(ms_ids):
        raise ValueError(
            "MS embedding count does not match MS ID count!"
        )

    if len(sar_embeddings) != len(sar_ids):
        raise ValueError(
            "SAR embedding count does not match SAR ID count!"
        )

    if ms_embeddings.shape[1] != sar_embeddings.shape[1]:
        raise ValueError(
            "MS and SAR embedding dimensions do not match!"
        )

    if len(set(ms_ids)) != len(ms_ids):
        raise ValueError(
            "Duplicate IDs found in MS embedding IDs!"
        )

    if len(set(sar_ids)) != len(sar_ids):
        raise ValueError(
            "Duplicate IDs found in SAR embedding IDs!"
        )

    print("\nBasic validation: PASSED")

    # --------------------------------------------------------
    # LOAD VALIDATION CSV
    # --------------------------------------------------------

    print("\nLoading validation CSV...")

    val_df = pd.read_csv(VAL_CSV_PATH)

    print("\nFirst 5 IDs from validation CSV:")

    print("\npatch_id:")
    for item in val_df["patch_id"].head(5):
        print(repr(str(item)))

    print("\ns1_name:")
    for item in val_df["s1_name"].head(5):
        print(repr(str(item)))

    print("\ns2v1_name:")
    for item in val_df["s2v1_name"].head(5):
        print(repr(str(item)))

    print("Validation CSV rows:", len(val_df))

    required_columns = [
        "patch_id",
        "s1_name",
        "s2v1_name"
    ]

    for column in required_columns:

        if column not in val_df.columns:
            raise ValueError(
                f"Required column '{column}' "
                f"not found in validation CSV!"
            )

    # Convert IDs explicitly to strings
    val_df["patch_id"] = val_df[
        "patch_id"
    ].astype(str)

    val_df["s1_name"] = val_df[
        "s1_name"
    ].astype(str)

    val_df["s2v1_name"] = val_df[
        "s2v1_name"
    ].astype(str)

    # --------------------------------------------------------
    # CHECK CSV DUPLICATES
    # --------------------------------------------------------

    if val_df["patch_id"].duplicated().any():

        raise ValueError(
            "Duplicate patch_id values found "
            "in validation CSV!"
        )

    if val_df["s1_name"].duplicated().any():

        raise ValueError(
            "Duplicate s1_name values found "
            "in validation CSV!"
        )

    if val_df["s2v1_name"].duplicated().any():

        raise ValueError(
            "Duplicate s2v1_name values found "
            "in validation CSV!"
        )

    # --------------------------------------------------------
    # CREATE SAR -> MS GROUND TRUTH MAPPING
    # --------------------------------------------------------

    sar_to_ms = dict(
    zip(
        val_df["s1_name"],
        val_df["patch_id"]
    )
)

    print(
        "SAR -> MS pairs in CSV:",
        len(sar_to_ms)
    )

    # --------------------------------------------------------
    # VERIFY ID ALIGNMENT
    # --------------------------------------------------------

    sar_embedding_id_set = set(sar_ids)
    ms_embedding_id_set = set(ms_ids)

    csv_sar_set = set(val_df["s1_name"])
    csv_ms_set = set(val_df["patch_id"])

    sar_overlap = len(
        sar_embedding_id_set.intersection(
            csv_sar_set
        )
    )

    ms_overlap = len(
        ms_embedding_id_set.intersection(
            csv_ms_set
        )
    )

    print("\n" + "=" * 60)
    print("ID ALIGNMENT CHECK")
    print("=" * 60)

    print("Validation CSV SAR IDs:", len(csv_sar_set))
    print("Validation CSV MS IDs: ", len(csv_ms_set))

    print(
        "SAR embedding IDs matching CSV:",
        sar_overlap
    )

    print(
        "MS embedding IDs matching CSV: ",
        ms_overlap
    )

    # SAR embeddings must correspond exactly
    # to the validation CSV SAR patches
    if sar_embedding_id_set != csv_sar_set:

        missing_sar = (
            sar_embedding_id_set -
            csv_sar_set
        )

        extra_sar = (
            csv_sar_set -
            sar_embedding_id_set
        )

        raise ValueError(
            "SAR embedding IDs and CSV IDs "
            "are not fully aligned!\n"
            f"Missing from CSV: {len(missing_sar)}\n"
            f"Missing from embeddings: {len(extra_sar)}"
        )

    # MS embeddings must correspond exactly
    # to the validation CSV MS patches
    if ms_embedding_id_set != csv_ms_set:

        missing_ms = (
            ms_embedding_id_set -
            csv_ms_set
        )

        extra_ms = (
            csv_ms_set -
            ms_embedding_id_set
        )

        raise ValueError(
            "MS embedding IDs and CSV IDs "
            "are not fully aligned!\n"
            f"Missing from CSV: {len(missing_ms)}\n"
            f"Missing from embeddings: {len(extra_ms)}"
        )

    print("\nID alignment: PASSED")

    # --------------------------------------------------------
    # VERIFY EVERY SAR QUERY HAS GROUND TRUTH
    # --------------------------------------------------------

    missing_ground_truth = [
        sar_id
        for sar_id in sar_ids
        if sar_id not in sar_to_ms
    ]

    if missing_ground_truth:

        raise ValueError(
            f"{len(missing_ground_truth)} SAR queries "
            "have no corresponding MS ground truth!"
        )

    # Verify every ground-truth MS image
    # actually exists in the gallery
    missing_gallery_targets = [
        sar_to_ms[sar_id]
        for sar_id in sar_ids
        if sar_to_ms[sar_id]
        not in ms_embedding_id_set
    ]

    if missing_gallery_targets:

        raise ValueError(
            f"{len(missing_gallery_targets)} "
            "ground-truth MS images are missing "
            "from the embedding gallery!"
        )

    print("Ground-truth mapping: PASSED")

    # --------------------------------------------------------
    # BUILD FAISS INDEX
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BUILDING FAISS INDEX")
    print("=" * 60)

    dimension = ms_embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(ms_embeddings)

    print("Gallery size:", index.ntotal)
    print("Embedding dimension:", dimension)

    # --------------------------------------------------------
    # SEARCH TOP 10
    # --------------------------------------------------------

    print(
        "\nSearching top-10 nearest "
        "MS embeddings..."
    )

    start_time = time.perf_counter()

    distances, indices = index.search(
        sar_embeddings,
        10
    )

    end_time = time.perf_counter()

    total_time = end_time - start_time

    average_time_ms = (
        total_time / len(sar_embeddings)
    ) * 1000

    print(
        f"Total search time: "
        f"{total_time:.4f} seconds"
    )

    print(
        f"Average retrieval time: "
        f"{average_time_ms:.4f} ms/query"
    )

    # --------------------------------------------------------
    # CALCULATE RECALL@K
    # --------------------------------------------------------

    recall_1_correct = 0
    recall_5_correct = 0
    recall_10_correct = 0

    total_queries = len(sar_ids)

    print("\nCalculating Recall@K...")

    for query_index, sar_id in enumerate(sar_ids):

        # Get the correct paired MS image
        ground_truth_ms_id = sar_to_ms[sar_id]

        # Convert FAISS positions
        # into actual MS patch IDs
        retrieved_ms_ids = [
            ms_ids[retrieved_index]
            for retrieved_index
            in indices[query_index]
        ]

        # Recall@1
        if calculate_recall(
            retrieved_ms_ids,
            ground_truth_ms_id,
            1
        ):
            recall_1_correct += 1

        # Recall@5
        if calculate_recall(
            retrieved_ms_ids,
            ground_truth_ms_id,
            5
        ):
            recall_5_correct += 1

        # Recall@10
        if calculate_recall(
            retrieved_ms_ids,
            ground_truth_ms_id,
            10
        ):
            recall_10_correct += 1

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    recall_1 = (
        recall_1_correct /
        total_queries
    )

    recall_5 = (
        recall_5_correct /
        total_queries
    )

    recall_10 = (
        recall_10_correct /
        total_queries
    )

    print("\n" + "=" * 60)
    print("BASELINE CROSS-MODAL RETRIEVAL RESULTS")
    print("SAR -> MS")
    print("=" * 60)

    print(f"\nTotal queries: {total_queries}")

    print(
        f"Gallery size: {len(ms_ids)}"
    )

    print(
        f"Average retrieval time: "
        f"{average_time_ms:.4f} ms/query"
    )

    print("\nRECALL RESULTS")

    print(
        f"Recall@1:  {recall_1:.6f} "
        f"({recall_1 * 100:.4f}%) "
        f"[{recall_1_correct}/{total_queries}]"
    )

    print(
        f"Recall@5:  {recall_5:.6f} "
        f"({recall_5 * 100:.4f}%) "
        f"[{recall_5_correct}/{total_queries}]"
    )

    print(
        f"Recall@10: {recall_10:.6f} "
        f"({recall_10 * 100:.4f}%) "
        f"[{recall_10_correct}/{total_queries}]"
    )

    # --------------------------------------------------------
    # CREATE REPORT
    # --------------------------------------------------------

    report = (
        "============================================================\n"
        "BASELINE CROSS-MODAL RETRIEVAL RESULTS\n"
        "SAR -> MS\n"
        "============================================================\n\n"
        f"Total queries: {total_queries}\n"
        f"Gallery size: {len(ms_ids)}\n"
        f"Embedding dimension: {dimension}\n"
        f"Average retrieval time: "
        f"{average_time_ms:.4f} ms/query\n\n"
        "ID ALIGNMENT\n"
        "------------------------------------------------------------\n"
        f"SAR embedding IDs: {len(sar_ids)}\n"
        f"MS embedding IDs: {len(ms_ids)}\n"
        f"SAR -> CSV overlap: {sar_overlap}\n"
        f"MS -> CSV overlap: {ms_overlap}\n"
        "ID alignment: PASSED\n\n"
        "RECALL RESULTS\n"
        "------------------------------------------------------------\n"
        f"Recall@1:  {recall_1:.6f} "
        f"({recall_1 * 100:.4f}%) "
        f"[{recall_1_correct}/{total_queries}]\n"
        f"Recall@5:  {recall_5:.6f} "
        f"({recall_5 * 100:.4f}%) "
        f"[{recall_5_correct}/{total_queries}]\n"
        f"Recall@10: {recall_10:.6f} "
        f"({recall_10 * 100:.4f}%) "
        f"[{recall_10_correct}/{total_queries}]\n"
        "============================================================\n"
    )

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(REPORT_PATH),
        exist_ok=True
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    print(
        f"\nReport saved to: "
        f"{REPORT_PATH}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run_baseline_retrieval()
