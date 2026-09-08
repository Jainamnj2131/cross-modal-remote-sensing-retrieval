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

REPORT_PATH = (
    "outputs/reports/"
    "baseline_ms_to_sar_retrieval_results.txt"
)


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
    """Check whether the ground truth is in top-k results."""

    return ground_truth_id in retrieved_ids[:k]


# ============================================================
# MAIN FUNCTION
# ============================================================

def run_ms_to_sar_retrieval():

    print("=" * 60)
    print("BASELINE MS -> SAR RETRIEVAL")
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

    print("\nMS embeddings shape:", ms_embeddings.shape)
    print("SAR embeddings shape:", sar_embeddings.shape)

    print("MS IDs:", len(ms_ids))
    print("SAR IDs:", len(sar_ids))

    # --------------------------------------------------------
    # BASIC VALIDATION
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

    print("\nBasic validation: PASSED")

    # --------------------------------------------------------
    # LOAD VALIDATION CSV
    # --------------------------------------------------------

    print("\nLoading validation CSV...")

    val_df = pd.read_csv(VAL_CSV_PATH)

    print("Validation CSV rows:", len(val_df))

    required_columns = [
        "patch_id",
        "s1_name"
    ]

    for column in required_columns:

        if column not in val_df.columns:
            raise ValueError(
                f"Required column '{column}' "
                f"not found in validation CSV!"
            )

    val_df["patch_id"] = (
        val_df["patch_id"]
        .astype(str)
    )

    val_df["s1_name"] = (
        val_df["s1_name"]
        .astype(str)
    )

    # --------------------------------------------------------
    # CREATE MS -> SAR GROUND TRUTH MAPPING
    # --------------------------------------------------------

    ms_to_sar = dict(
        zip(
            val_df["patch_id"],
            val_df["s1_name"]
        )
    )

    print(
        "MS -> SAR pairs in CSV:",
        len(ms_to_sar)
    )

    # --------------------------------------------------------
    # VERIFY ID ALIGNMENT
    # --------------------------------------------------------

    ms_embedding_id_set = set(ms_ids)
    sar_embedding_id_set = set(sar_ids)

    csv_ms_set = set(val_df["patch_id"])
    csv_sar_set = set(val_df["s1_name"])

    ms_overlap = len(
        ms_embedding_id_set.intersection(
            csv_ms_set
        )
    )

    sar_overlap = len(
        sar_embedding_id_set.intersection(
            csv_sar_set
        )
    )

    print("\n" + "=" * 60)
    print("ID ALIGNMENT CHECK")
    print("=" * 60)

    print(
        "MS embedding IDs matching CSV:",
        ms_overlap
    )

    print(
        "SAR embedding IDs matching CSV:",
        sar_overlap
    )

    if ms_embedding_id_set != csv_ms_set:
        raise ValueError(
            "MS embedding IDs and validation CSV "
            "are not fully aligned!"
        )

    if sar_embedding_id_set != csv_sar_set:
        raise ValueError(
            "SAR embedding IDs and validation CSV "
            "are not fully aligned!"
        )

    print("\nID alignment: PASSED")

    # --------------------------------------------------------
    # VERIFY GROUND TRUTH
    # --------------------------------------------------------

    missing_ground_truth = [
        ms_id
        for ms_id in ms_ids
        if ms_id not in ms_to_sar
    ]

    if missing_ground_truth:

        raise ValueError(
            f"{len(missing_ground_truth)} MS queries "
            "have no SAR ground-truth pair!"
        )

    sar_gallery_set = set(sar_ids)

    missing_gallery_targets = [
        ms_to_sar[ms_id]
        for ms_id in ms_ids
        if ms_to_sar[ms_id]
        not in sar_gallery_set
    ]

    if missing_gallery_targets:

        raise ValueError(
            f"{len(missing_gallery_targets)} "
            "ground-truth SAR images are missing "
            "from the gallery!"
        )

    print("Ground-truth mapping: PASSED")

    # --------------------------------------------------------
    # BUILD SAR FAISS INDEX
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BUILDING SAR FAISS INDEX")
    print("=" * 60)

    dimension = sar_embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(sar_embeddings)

    print("Gallery size:", index.ntotal)
    print("Embedding dimension:", dimension)

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    print(
        "\nSearching top-10 nearest "
        "SAR embeddings..."
    )

    start_time = time.perf_counter()

    distances, indices = index.search(
        ms_embeddings,
        10
    )

    end_time = time.perf_counter()

    total_time = end_time - start_time

    average_time_ms = (
        total_time / len(ms_embeddings)
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

    total_queries = len(ms_ids)

    print("\nCalculating Recall@K...")

    for query_index, ms_id in enumerate(ms_ids):

        ground_truth_sar_id = ms_to_sar[ms_id]

        retrieved_sar_ids = [
            sar_ids[retrieved_index]
            for retrieved_index
            in indices[query_index]
        ]

        if calculate_recall(
            retrieved_sar_ids,
            ground_truth_sar_id,
            1
        ):
            recall_1_correct += 1

        if calculate_recall(
            retrieved_sar_ids,
            ground_truth_sar_id,
            5
        ):
            recall_5_correct += 1

        if calculate_recall(
            retrieved_sar_ids,
            ground_truth_sar_id,
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
    print("MS -> SAR")
    print("=" * 60)

    print(f"\nTotal queries: {total_queries}")

    print(
        f"Gallery size: {len(sar_ids)}"
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
    # SAVE REPORT
    # --------------------------------------------------------

    report = (
        "============================================================\n"
        "BASELINE CROSS-MODAL RETRIEVAL RESULTS\n"
        "MS -> SAR\n"
        "============================================================\n\n"
        f"Total queries: {total_queries}\n"
        f"Gallery size: {len(sar_ids)}\n"
        f"Embedding dimension: {dimension}\n"
        f"Average retrieval time: "
        f"{average_time_ms:.4f} ms/query\n\n"
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
    run_ms_to_sar_retrieval()