"""
Week 5 - Final FAISS Retrieval and Test Set Evaluation Report.

Combines the existing Week 3 baseline results with the Week 5
trained-model results into the final comparison table.

Important:
- Cross-modal retrieval uses exact paired patch IDs.
- Same-modal retrieval uses shared land-cover labels.
- Baseline mAP values are only reported where they were actually
  calculated using the corresponding evaluation setup.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

REPORT_DIR = Path("outputs/reports")

TRAINED_CROSS_MODAL = (
    REPORT_DIR / "trained_cross_modal_results.csv"
)

TRAINED_SAME_MODAL = (
    REPORT_DIR / "trained_same_modal_results.csv"
)

OUTPUT_CSV = (
    REPORT_DIR / "week5_final_comparison.csv"
)

OUTPUT_TXT = (
    REPORT_DIR / "Week_5_final_results.txt"
)


# ============================================================
# WEEK 3 BASELINE RESULTS
# ============================================================

# Week 3 baseline:
# Cross-modal relevance = exact paired patch ID.
BASELINE_CROSS_MODAL = {
    "SAR -> MS": {
        "Recall@1": 0.2222,
        "Recall@5": 0.4889,
        "Recall@10": 0.8222,
        "mAP": None,
        "Avg Retrieval Time (ms)": 0.0589,
    },
    "MS -> SAR": {
        "Recall@1": 0.4889,
        "Recall@5": 1.5111,
        "Recall@10": 2.1333,
        "mAP": None,
        "Avg Retrieval Time (ms)": 0.0817,
    },

    # Same-modal relevance = shared >=1 land-cover label.
    "SAR -> SAR": {
        "Recall@1": 15.1333,
        "Recall@5": 26.3111,
        "Recall@10": 33.4667,
        "mAP": None,
        "Avg Retrieval Time (ms)": 0.0802,
    },
    "MS -> MS": {
        "Recall@1": 18.6444,
        "Recall@5": 31.5778,
        "Recall@10": 39.6667,
        "mAP": None,
        "Avg Retrieval Time (ms)": 0.0812,
    },
}


# ============================================================
# LOAD TRAINED RESULTS
# ============================================================

def load_trained_results() -> dict[str, dict]:
    """Load Week 5 trained evaluation results."""

    if not TRAINED_CROSS_MODAL.exists():
        raise FileNotFoundError(
            f"Missing trained cross-modal results:\n"
            f"{TRAINED_CROSS_MODAL}"
        )

    if not TRAINED_SAME_MODAL.exists():
        raise FileNotFoundError(
            f"Missing trained same-modal results:\n"
            f"{TRAINED_SAME_MODAL}"
        )

    cross_modal = pd.read_csv(
        TRAINED_CROSS_MODAL
    )

    same_modal = pd.read_csv(
        TRAINED_SAME_MODAL
    )

    results = {}

    for _, row in cross_modal.iterrows():

        results[row["Mode"]] = {
            "Recall@1": float(row["Recall@1"]),
            "Recall@5": float(row["Recall@5"]),
            "Recall@10": float(row["Recall@10"]),
            "mAP": None,
            "Avg Retrieval Time (ms)": float(
                row["Avg Retrieval Time (ms)"]
            ),
        }

    for _, row in same_modal.iterrows():

        results[row["Mode"]] = {
            "Recall@1": float(row["Recall@1"]),
            "Recall@5": float(row["Recall@5"]),
            "Recall@10": float(row["Recall@10"]),
            "mAP": float(row["mAP"]),
            "Avg Retrieval Time (ms)": float(
                row["Avg Retrieval Time (ms)"]
            ),
        }

    return results


# ============================================================
# BUILD FINAL TABLE
# ============================================================

def build_final_table(
    trained_results: dict[str, dict],
) -> pd.DataFrame:

    rows = []

    mode_order = [
        ("SAR -> SAR", "baseline"),
        ("MS -> MS", "baseline"),
        ("SAR -> MS", "baseline"),
        ("MS -> SAR", "baseline"),
        ("SAR -> SAR", "trained"),
        ("MS -> MS", "trained"),
        ("SAR -> MS", "trained"),
        ("MS -> SAR", "trained"),
    ]

    for mode, model_type in mode_order:

        if model_type == "baseline":
            result = BASELINE_CROSS_MODAL[mode]

        else:
            result = trained_results[mode]

        rows.append(
            {
                "Mode": (
                    f"{mode} "
                    f"({model_type})"
                ),
                "Recall@1": result["Recall@1"],
                "Recall@5": result["Recall@5"],
                "Recall@10": result["Recall@10"],
                "mAP": result["mAP"],
                "Avg Retrieval Time (ms)": (
                    result["Avg Retrieval Time (ms)"]
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# WRITE TEXT REPORT
# ============================================================

def write_text_report(
    final_df: pd.DataFrame,
) -> None:

    with OUTPUT_TXT.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "WEEK 5 — FINAL FAISS RETRIEVAL "
            "AND TEST SET EVALUATION\n"
        )

        file.write("=" * 72)
        file.write("\n\n")

        file.write(
            "Dataset / evaluation size: "
            "4,500 test pairs\n\n"
        )

        file.write(
            "Embedding dimension: 512\n\n"
        )

        file.write(
            "FAISS index: IndexFlatIP\n\n"
        )

        file.write(
            "Cross-modal relevance:\n"
            "Exact paired patch ID.\n\n"
        )

        file.write(
            "Same-modal relevance:\n"
            "At least one shared land-cover label; "
            "query item excluded.\n\n"
        )

        file.write(
            "NOTE:\n"
            "Baseline mAP values are not populated because "
            "the Week 3 baseline records available for this "
            "final comparison contain Recall@K and retrieval "
            "time, but do not contain baseline mAP values.\n\n"
        )

        file.write(
            final_df.to_string(
                index=False,
                na_rep="N/A",
            )
        )

        file.write("\n\n")

        file.write(
            "TRAINED CROSS-MODAL RESULTS\n"
        )

        file.write("-" * 72)
        file.write("\n")

        file.write(
            "SAR -> MS:\n"
            "Recall@1 = 38.7778%\n"
            "Recall@5 = 64.8222%\n"
            "Recall@10 = 75.0444%\n"
            "Average retrieval time = 0.0123 ms/query\n\n"
        )

        file.write(
            "MS -> SAR:\n"
            "Recall@1 = 37.0444%\n"
            "Recall@5 = 63.7333%\n"
            "Recall@10 = 74.0000%\n"
            "Average retrieval time = 0.0139 ms/query\n\n"
        )

        file.write(
            "TRAINED SAME-MODAL RESULTS\n"
        )

        file.write("-" * 72)
        file.write("\n")

        file.write(
            "SAR -> SAR:\n"
            "Recall@1 = 53.4000%\n"
            "Recall@5 = 85.9778%\n"
            "Recall@10 = 92.4889%\n"
            "mAP@10 = 61.2397%\n"
            "Average retrieval time = 0.0118 ms/query\n\n"
        )

        file.write(
            "MS -> MS:\n"
            "Recall@1 = 57.8444%\n"
            "Recall@5 = 87.0444%\n"
            "Recall@10 = 93.4667%\n"
            "mAP@10 = 64.0870%\n"
            "Average retrieval time = 0.0113 ms/query\n"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 72)
    print("WEEK 5 — FINAL RESULTS")
    print("=" * 72)
    print()

    trained_results = load_trained_results()

    final_df = build_final_table(
        trained_results
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    write_text_report(
        final_df
    )

    print(
        final_df.to_string(
            index=False,
            na_rep="N/A",
        )
    )

    print()
    print(
        f"[OK] CSV saved: {OUTPUT_CSV}"
    )

    print(
        f"[OK] Text report saved: {OUTPUT_TXT}"
    )

    print()
    print(
        "WEEK 5 FINAL REPORT COMPLETE"
    )


if __name__ == "__main__":
    main()