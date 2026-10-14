"""
similarity_evaluation.py

Evaluate Pairwise Plane Similarity results.
"""

import csv
from pathlib import Path
from collections import defaultdict

import numpy as np


INPUT_FILE = Path(
    "results/similarity/synthetic_final.csv"
)

OUTPUT_FILE = Path(
    "results/similarity/synthetic_summary.csv"
)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(rows):

    gt = np.array(
        [int(r["gt_match"]) for r in rows],
        dtype=bool,
    )

    pred = np.array(
        [int(r["predicted_match"]) for r in rows],
        dtype=bool,
    )

    runtime = np.array(
        [float(r["runtime_ms"]) for r in rows],
        dtype=float,
    )

    tp = int(np.sum(gt & pred))
    tn = int(np.sum(~gt & ~pred))
    fp = int(np.sum(~gt & pred))
    fn = int(np.sum(gt & ~pred))

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    accuracy = (
        (tp + tn) / len(rows)
        if rows
        else 0.0
    )

    return {
        "samples": len(rows),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "mean_runtime_ms": float(
            np.mean(runtime)
        ),
        "median_runtime_ms": float(
            np.median(runtime)
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    # Ignore failed runs
    rows = [
        r for r in rows
        if int(r["success"]) == 1
    ]

    if not rows:
        raise RuntimeError(
            "No valid similarity results found."
        )

    methods = sorted(
        set(r["method"] for r in rows)
    )

    conditions = sorted(
        set(r["condition"] for r in rows)
    )

    summary_rows = []

    for method in methods:

        method_rows = [
            r for r in rows
            if r["method"] == method
        ]

        # ----------------------------------------------------
        # Overall
        # ----------------------------------------------------

        metrics = calculate_metrics(
            method_rows
        )

        summary_rows.append(
            {
                "method": method,
                "condition": "overall",
                **metrics,
            }
        )

        # ----------------------------------------------------
        # Per condition
        # ----------------------------------------------------

        for condition in conditions:

            subset = [
                r for r in method_rows
                if r["condition"] == condition
            ]

            if not subset:
                continue

            metrics = calculate_metrics(
                subset
            )

            summary_rows.append(
                {
                    "method": method,
                    "condition": condition,
                    **metrics,
                }
            )

    # ========================================================
    # Save
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    # ========================================================
    # Print compact overall results
    # ========================================================

    print(
        "\n=== Synthetic Similarity Results ===\n"
    )

    for row in summary_rows:

        if row["condition"] != "overall":
            continue

        print(
            f"{row['method']:<22} "
            f"P={row['precision']:.3f}  "
            f"R={row['recall']:.3f}  "
            f"F1={row['f1']:.3f}  "
            f"Runtime={row['mean_runtime_ms']:.3f} ms"
        )

    print(
        f"\nSummary saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()