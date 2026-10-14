"""
analyse_similarity.py

Create a compact final analysis table for the
Synthetic Pairwise Plane Similarity experiment.
"""

import csv
from pathlib import Path


INPUT_FILE = Path(
    "results/similarity/synthetic_summary.csv"
)

OUTPUT_FILE = Path(
    "results/similarity/synthetic_analysis.csv"
)


def main():

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(
            "No similarity summary results found."
        )

    methods = sorted(
        set(row["method"] for row in rows)
    )

    final_rows = []

    for method in methods:

        method_rows = {
            row["condition"]: row
            for row in rows
            if row["method"] == method
        }

        if "overall" not in method_rows:
            continue

        overall = method_rows["overall"]

        clean = method_rows.get("clean")
        moderate = method_rows.get("moderate")
        hard = method_rows.get("hard")

        overall_f1 = float(
            overall["f1"]
        )

        clean_f1 = (
            float(clean["f1"])
            if clean
            else float("nan")
        )

        moderate_f1 = (
            float(moderate["f1"])
            if moderate
            else float("nan")
        )

        hard_f1 = (
            float(hard["f1"])
            if hard
            else float("nan")
        )

        robustness_drop = (
            clean_f1 - hard_f1
            if clean and hard
            else float("nan")
        )

        final_rows.append(
            {
                "method": method,

                "precision":
                    float(overall["precision"]),

                "recall":
                    float(overall["recall"]),

                "f1":
                    overall_f1,

                "accuracy":
                    float(overall["accuracy"]),

                "clean_f1":
                    clean_f1,

                "moderate_f1":
                    moderate_f1,

                "hard_f1":
                    hard_f1,

                "f1_drop_clean_to_hard":
                    robustness_drop,

                "mean_runtime_ms":
                    float(
                        overall["mean_runtime_ms"]
                    ),

                "median_runtime_ms":
                    float(
                        overall["median_runtime_ms"]
                    ),
            }
        )

    # Sort by overall F1, highest first
    final_rows.sort(
        key=lambda row: row["f1"],
        reverse=True,
    )

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
            fieldnames=final_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(final_rows)

    print(
        "\n=== Synthetic Similarity Analysis ===\n"
    )

    for row in final_rows:

        print(
            f"{row['method']:<22} "
            f"F1={row['f1']:.3f}  "
            f"Hard={row['hard_f1']:.3f}  "
            f"Drop={row['f1_drop_clean_to_hard']:.3f}  "
            f"Runtime={row['mean_runtime_ms']:.3f} ms"
        )

    print(
        f"\nAnalysis saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()