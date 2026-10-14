# experiments/plane_association/similarity/analyse_similarity_real.py

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_CSV = PROJECT_ROOT / "results" / "similarity" / "real_final.csv"
OUTPUT_CSV = PROJECT_ROOT / "results" / "similarity" / "real_summary.csv"


# ============================================================
# Metric helpers
# ============================================================

def safe_divide(num, den):
    """Safely divide two numbers."""
    return num / den if den != 0 else 0.0


def calculate_metrics(group):
    """
    Calculate classification and runtime metrics for one method.

    gt_label:
        1  = true correspondence
        0  = true non-correspondence
       -1  = ignore / unavailable GT

    predicted_match:
        1  = predicted correspondence
        0  = predicted non-correspondence
    """

    # Only successfully evaluated rows
    successful = group[group["success"] == 1].copy()

    total_pairs = len(successful)

    # Ignore pairs without reliable binary GT
    evaluable = successful[
        successful["gt_label"].isin([0, 1])
    ].copy()

    ignored_pairs = total_pairs - len(evaluable)

    if evaluable.empty:
        return {
            "total_pairs": total_pairs,
            "evaluable_pairs": 0,
            "ignored_pairs": ignored_pairs,

            "positive_gt": 0,
            "negative_gt": 0,

            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,

            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,

            "mean_runtime_ms": successful["runtime_ms"].mean(),
            "median_runtime_ms": successful["runtime_ms"].median(),
            "std_runtime_ms": successful["runtime_ms"].std(),
        }

    gt = evaluable["gt_label"].astype(int)
    pred = evaluable["predicted_match"].astype(int)

    # Confusion matrix
    tp = int(((gt == 1) & (pred == 1)).sum())
    tn = int(((gt == 0) & (pred == 0)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())

    positive_gt = int((gt == 1).sum())
    negative_gt = int((gt == 0).sum())

    # Classification metrics
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(
        2 * precision * recall,
        precision + recall
    )
    accuracy = safe_divide(
        tp + tn,
        tp + tn + fp + fn
    )

    return {
        "total_pairs": total_pairs,
        "evaluable_pairs": len(evaluable),
        "ignored_pairs": ignored_pairs,

        "positive_gt": positive_gt,
        "negative_gt": negative_gt,

        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,

        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,

        "mean_runtime_ms": successful["runtime_ms"].mean(),
        "median_runtime_ms": successful["runtime_ms"].median(),
        "std_runtime_ms": successful["runtime_ms"].std(),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("\n=== Real Pairwise Similarity Analysis ===")

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    print(f"Input file : {INPUT_CSV}")
    print(f"Rows       : {len(df)}")
    print(f"Methods    : {df['method'].nunique()}")

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    required_columns = {
        "method",
        "gt_label",
        "predicted_match",
        "runtime_ms",
        "success",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # Analyse each method
    # --------------------------------------------------------

    rows = []

    for method, group in df.groupby("method", sort=False):

        metrics = calculate_metrics(group)

        rows.append({
            "method": method,
            **metrics,
        })

    summary = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Optional sorting
    # --------------------------------------------------------

    summary = summary.sort_values(
        by="f1",
        ascending=False
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Terminal output
    # --------------------------------------------------------

    print("\n=== Evaluation Summary ===\n")

    for _, row in summary.iterrows():

        print(f"{row['method']}")
        print(
            f"  evaluable={int(row['evaluable_pairs'])}"
            f"  ignored={int(row['ignored_pairs'])}"
        )

        print(
            f"  GT+: {int(row['positive_gt'])}"
            f"  GT-: {int(row['negative_gt'])}"
        )

        print(
            f"  TP={int(row['tp'])}"
            f"  TN={int(row['tn'])}"
            f"  FP={int(row['fp'])}"
            f"  FN={int(row['fn'])}"
        )

        print(
            f"  Precision={row['precision']:.4f}"
            f"  Recall={row['recall']:.4f}"
            f"  F1={row['f1']:.4f}"
            f"  Accuracy={row['accuracy']:.4f}"
        )

        print(
            f"  Runtime:"
            f" mean={row['mean_runtime_ms']:.4f} ms"
            f" median={row['median_runtime_ms']:.4f} ms"
            f" std={row['std_runtime_ms']:.4f} ms"
        )

        print()

    print("Results saved to:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()