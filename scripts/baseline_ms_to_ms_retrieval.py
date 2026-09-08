import os
import time
import ast

import numpy as np
import pandas as pd
import faiss


# ============================================================
# PATHS
# ============================================================

MS_EMBEDDINGS_PATH = (
    "outputs/embeddings/baseline_val_ms_embeddings.npy"
)

MS_IDS_PATH = (
    "outputs/embeddings/baseline_val_patch_ids.txt"
)

VAL_CSV_PATH = (
    "outputs/metadata/val_split.csv"
)

REPORT_PATH = (
    "outputs/reports/baseline_ms_to_ms_retrieval_results.txt"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_ids(path):
    """Load patch IDs from a text file."""

    with open(path, "r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def parse_labels(value):
    """
    Convert labels stored in the CSV into a Python set.
    """

    if isinstance(value, list):
        return set(value)

    if pd.isna(value):
        return set()

    try:
        parsed_value = ast.literal_eval(value)

        if isinstance(parsed_value, list):
            return set(parsed_value)

    except (ValueError, SyntaxError):
        pass

    return set()


def has_label_overlap(query_labels, retrieved_labels):
    """
    Returns True if two patches share at least
    one land-cover label.
    """

    return len(
        query_labels.intersection(retrieved_labels)
    ) > 0


# ============================================================
# MAIN FUNCTION
# ============================================================

def run_ms_to_ms_retrieval():

    print("=" * 60)
    print("BASELINE MS -> MS RETRIEVAL")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD EMBEDDINGS
    # --------------------------------------------------------

    print("\nLoading MS embeddings...")

    ms_embeddings = np.load(
        MS_EMBEDDINGS_PATH
    ).astype("float32")

    print("Loading MS patch IDs...")

    ms_ids = load_ids(
        MS_IDS_PATH
    )

    print(
        "\nMS embeddings shape:",
        ms_embeddings.shape
    )

    print(
        "MS IDs:",
        len(ms_ids)
    )

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    if len(ms_embeddings) != len(ms_ids):

        raise ValueError(
            "MS embedding count does not "
            "match MS ID count!"
        )

    print("\nBasic validation: PASSED")

    # --------------------------------------------------------
    # LOAD VALIDATION CSV
    # --------------------------------------------------------

    print("\nLoading validation CSV...")

    val_df = pd.read_csv(
        VAL_CSV_PATH
    )

    print(
        "Validation CSV rows:",
        len(val_df)
    )

    required_columns = [
        "patch_id",
        "labels"
    ]

    for column in required_columns:

        if column not in val_df.columns:

            raise ValueError(
                f"Required column '{column}' "
                "not found in validation CSV!"
            )

    val_df["patch_id"] = (
        val_df["patch_id"]
        .astype(str)
    )

    # --------------------------------------------------------
    # CREATE MS ID -> LABEL MAPPING
    # --------------------------------------------------------

    ms_to_labels = {}

    for _, row in val_df.iterrows():

        ms_id = row["patch_id"]

        labels = parse_labels(
            row["labels"]
        )

        ms_to_labels[ms_id] = labels

    print(
        "MS label mappings:",
        len(ms_to_labels)
    )

    # --------------------------------------------------------
    # VERIFY ID ALIGNMENT
    # --------------------------------------------------------

    ms_embedding_id_set = set(
        ms_ids
    )

    csv_ms_id_set = set(
        val_df["patch_id"]
    )

    ms_overlap = len(
        ms_embedding_id_set.intersection(
            csv_ms_id_set
        )
    )

    print("\n" + "=" * 60)
    print("ID ALIGNMENT CHECK")
    print("=" * 60)

    print(
        "MS embedding IDs:",
        len(ms_embedding_id_set)
    )

    print(
        "CSV MS IDs:",
        len(csv_ms_id_set)
    )

    print(
        "Matching IDs:",
        ms_overlap
    )

    if ms_embedding_id_set != csv_ms_id_set:

        raise ValueError(
            "MS embedding IDs and CSV IDs "
            "are not fully aligned!"
        )

    print("\nID alignment: PASSED")

    # --------------------------------------------------------
    # BUILD FAISS INDEX
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BUILDING MS FAISS INDEX")
    print("=" * 60)

    dimension = ms_embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        ms_embeddings
    )

    print(
        "Gallery size:",
        index.ntotal
    )

    print(
        "Embedding dimension:",
        dimension
    )

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    # Search 11 because one result will be
    # the query image itself, which is excluded.
    search_k = 11

    print(
        f"\nSearching top-{search_k} nearest "
        "MS embeddings..."
    )

    start_time = time.perf_counter()

    distances, indices = index.search(
        ms_embeddings,
        search_k
    )

    end_time = time.perf_counter()

    total_time = (
        end_time - start_time
    )

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

    valid_queries = 0

    print(
        "\nCalculating label-overlap Recall@K..."
    )

    for query_index, query_id in enumerate(ms_ids):

        query_labels = ms_to_labels.get(
            query_id
        )

        if not query_labels:
            continue

        # Remove the query itself from results.
        retrieved_indices = [
            retrieved_index
            for retrieved_index in indices[query_index]
            if retrieved_index != query_index
        ]

        # Keep only the top 10 actual neighbours.
        retrieved_indices = retrieved_indices[:10]

        if not retrieved_indices:
            continue

        valid_queries += 1

        relevant_results = []

        for retrieved_index in retrieved_indices:

            retrieved_id = ms_ids[
                retrieved_index
            ]

            retrieved_labels = (
                ms_to_labels.get(
                    retrieved_id,
                    set()
                )
            )

            is_relevant = has_label_overlap(
                query_labels,
                retrieved_labels
            )

            relevant_results.append(
                is_relevant
            )

        # At least one relevant result within top 1.
        if any(relevant_results[:1]):
            recall_1_correct += 1

        # At least one relevant result within top 5.
        if any(relevant_results[:5]):
            recall_5_correct += 1

        # At least one relevant result within top 10.
        if any(relevant_results[:10]):
            recall_10_correct += 1

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    if valid_queries == 0:

        raise ValueError(
            "No valid queries were available!"
        )

    recall_1 = (
        recall_1_correct /
        valid_queries
    )

    recall_5 = (
        recall_5_correct /
        valid_queries
    )

    recall_10 = (
        recall_10_correct /
        valid_queries
    )

    print("\n" + "=" * 60)
    print("BASELINE SAME-MODAL RETRIEVAL RESULTS")
    print("MS -> MS")
    print("=" * 60)

    print(
        f"\nTotal valid queries: "
        f"{valid_queries}"
    )

    print(
        f"Gallery size: "
        f"{len(ms_ids)}"
    )

    print(
        f"Average retrieval time: "
        f"{average_time_ms:.4f} ms/query"
    )

    print("\nRECALL RESULTS")

    print(
        f"Recall@1:  {recall_1:.6f} "
        f"({recall_1 * 100:.4f}%) "
        f"[{recall_1_correct}/{valid_queries}]"
    )

    print(
        f"Recall@5:  {recall_5:.6f} "
        f"({recall_5 * 100:.4f}%) "
        f"[{recall_5_correct}/{valid_queries}]"
    )

    print(
        f"Recall@10: {recall_10:.6f} "
        f"({recall_10 * 100:.4f}%) "
        f"[{recall_10_correct}/{valid_queries}]"
    )

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    report = (
        "============================================================\n"
        "BASELINE SAME-MODAL RETRIEVAL RESULTS\n"
        "MS -> MS\n"
        "============================================================\n\n"
        f"Total valid queries: {valid_queries}\n"
        f"Gallery size: {len(ms_ids)}\n"
        f"Embedding dimension: {dimension}\n"
        f"Average retrieval time: "
        f"{average_time_ms:.4f} ms/query\n\n"
        "RELEVANCE DEFINITION\n"
        "------------------------------------------------------------\n"
        "A retrieved patch is relevant if it shares at least one\n"
        "land-cover label with the query patch.\n"
        "The query patch itself is excluded from retrieval.\n\n"
        "RECALL RESULTS\n"
        "------------------------------------------------------------\n"
        f"Recall@1:  {recall_1:.6f} "
        f"({recall_1 * 100:.4f}%) "
        f"[{recall_1_correct}/{valid_queries}]\n"
        f"Recall@5:  {recall_5:.6f} "
        f"({recall_5 * 100:.4f}%) "
        f"[{recall_5_correct}/{valid_queries}]\n"
        f"Recall@10: {recall_10:.6f} "
        f"({recall_10 * 100:.4f}%) "
        f"[{recall_10_correct}/{valid_queries}]\n"
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
    run_ms_to_ms_retrieval()