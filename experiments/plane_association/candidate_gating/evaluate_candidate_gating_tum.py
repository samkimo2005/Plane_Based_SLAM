"""
evaluate_candidate_gating_tum.py

Evaluate the four Candidate Gating methods using:
    candidate_gating_tum_results.csv
    candidate_gating_tum_runtime.csv

GT labels:
     1 = true correspondence
     0 = known non-correspondence
    -1 = ignore / unknown GT

Classification metrics are calculated ONLY on gt_label 0/1.
Candidate reduction is calculated on ALL input pairs because every pair was
actually processed by the gating method.
"""

from pathlib import Path
import csv
import math
from collections import defaultdict


ROOT = Path(r"D:\PlaneBasedSLAM")

RESULT_FILE = (
    ROOT / "results" / "candidate_gating" / "tum"
    / "candidate_gating_tum_results.csv"
)

RUNTIME_FILE = (
    ROOT / "results" / "candidate_gating" / "tum"
    / "candidate_gating_tum_runtime.csv"
)

OUTPUT_DIR = ROOT / "results" / "candidate_gating" / "tum"

SUMMARY_FILE = OUTPUT_DIR / "candidate_gating_tum_summary.csv"
TRANSITION_FILE = OUTPUT_DIR / "candidate_gating_tum_per_transition.csv"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_div(a, b):
    return a / b if b else 0.0


def mean(values):
    return sum(values) / len(values) if values else 0.0


def median(values):
    if not values:
        return 0.0

    values = sorted(values)
    n = len(values)
    mid = n // 2

    if n % 2:
        return values[mid]

    return 0.5 * (values[mid - 1] + values[mid])


def main():
    if not RESULT_FILE.exists():
        raise FileNotFoundError(RESULT_FILE)

    if not RUNTIME_FILE.exists():
        raise FileNotFoundError(RUNTIME_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result_rows = read_csv(RESULT_FILE)
    runtime_rows = read_csv(RUNTIME_FILE)

    # --------------------------------------------------------
    # Group data
    # --------------------------------------------------------

    by_method = defaultdict(list)
    by_transition = defaultdict(list)
    runtime_by_method = defaultdict(list)
    runtime_lookup = {}

    for row in result_rows:
        row["pair_id"] = int(row["pair_id"])
        row["frame_a"] = int(row["frame_a"])
        row["frame_b"] = int(row["frame_b"])
        row["plane_a"] = int(row["plane_a"])
        row["plane_b"] = int(row["plane_b"])
        row["gt_label"] = int(row["gt_label"])
        row["keep"] = int(row["keep"])

        method = row["method"]
        by_method[method].append(row)
        by_transition[
            (method, row["frame_a"], row["frame_b"])
        ].append(row)

    for row in runtime_rows:
        method = row["method"]
        frame_a = int(row["frame_a"])
        frame_b = int(row["frame_b"])
        runtime_ms = float(row["runtime_ms"])

        runtime_by_method[method].append(runtime_ms)
        runtime_lookup[(method, frame_a, frame_b)] = runtime_ms

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    summaries = []

    for method in sorted(by_method):
        rows = by_method[method]

        known = [
            r for r in rows
            if r["gt_label"] in (0, 1)
        ]

        tp = sum(
            r["gt_label"] == 1 and r["keep"] == 1
            for r in known
        )
        fn = sum(
            r["gt_label"] == 1 and r["keep"] == 0
            for r in known
        )
        fp = sum(
            r["gt_label"] == 0 and r["keep"] == 1
            for r in known
        )
        tn = sum(
            r["gt_label"] == 0 and r["keep"] == 0
            for r in known
        )

        ignored = [
            r for r in rows
            if r["gt_label"] == -1
        ]

        kept_all = sum(r["keep"] for r in rows)
        kept_ignored = sum(r["keep"] for r in ignored)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(
            2.0 * precision * recall,
            precision + recall,
        )

        false_acceptance_rate = safe_div(
            fp,
            fp + tn,
        )

        specificity = safe_div(
            tn,
            tn + fp,
        )

        candidate_reduction = (
            1.0 - safe_div(kept_all, len(rows))
        )

        known_candidate_reduction = (
            1.0 - safe_div(
                tp + fp,
                len(known),
            )
        )

        runtimes = runtime_by_method[method]

        summaries.append(
            {
                "method": method,
                "all_input_pairs": len(rows),
                "evaluable_pairs": len(known),
                "ignored_pairs": len(ignored),
                "true_correspondences": tp + fn,
                "known_non_correspondences": fp + tn,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "true_match_retention_recall": recall,
                "f1": f1,
                "false_acceptance_rate": false_acceptance_rate,
                "specificity": specificity,
                "kept_all_pairs": kept_all,
                "candidate_reduction_rate": candidate_reduction,
                "known_pair_reduction_rate": known_candidate_reduction,
                "kept_ignored_pairs": kept_ignored,
                "mean_runtime_ms_per_transition": mean(runtimes),
                "median_runtime_ms_per_transition": median(runtimes),
                "total_runtime_ms": sum(runtimes),
            }
        )

    # --------------------------------------------------------
    # Per-transition metrics
    # --------------------------------------------------------

    transition_rows = []

    for (
        method,
        frame_a,
        frame_b,
    ), rows in sorted(by_transition.items()):

        known = [
            r for r in rows
            if r["gt_label"] in (0, 1)
        ]

        tp = sum(
            r["gt_label"] == 1 and r["keep"] == 1
            for r in known
        )
        fn = sum(
            r["gt_label"] == 1 and r["keep"] == 0
            for r in known
        )
        fp = sum(
            r["gt_label"] == 0 and r["keep"] == 1
            for r in known
        )
        tn = sum(
            r["gt_label"] == 0 and r["keep"] == 0
            for r in known
        )

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(
            2.0 * precision * recall,
            precision + recall,
        )

        kept = sum(r["keep"] for r in rows)
        reduction = 1.0 - safe_div(kept, len(rows))

        transition_rows.append(
            {
                "method": method,
                "frame_a": frame_a,
                "frame_b": frame_b,
                "input_pairs": len(rows),
                "evaluable_pairs": len(known),
                "ignored_pairs": (
                    len(rows) - len(known)
                ),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "true_match_retention_recall": recall,
                "f1": f1,
                "candidate_reduction_rate": reduction,
                "runtime_ms": runtime_lookup.get(
                    (method, frame_a, frame_b),
                    math.nan,
                ),
            }
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with SUMMARY_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summaries[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summaries)

    with TRANSITION_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(transition_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(transition_rows)

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print()
    print("=" * 118)
    print("TUM CANDIDATE GATING EVALUATION")
    print("=" * 118)

    print(
        f"{'Method':<15}"
        f"{'P':>8}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'FAR':>10}"
        f"{'Reduction':>12}"
        f"{'TP':>7}"
        f"{'FP':>7}"
        f"{'FN':>7}"
        f"{'Runtime ms':>14}"
    )

    print("-" * 118)

    for s in summaries:
        print(
            f"{s['method']:<15}"
            f"{s['precision']:>8.3f}"
            f"{s['true_match_retention_recall']:>10.3f}"
            f"{s['f1']:>10.3f}"
            f"{s['false_acceptance_rate']:>10.3f}"
            f"{s['candidate_reduction_rate']:>12.3f}"
            f"{s['tp']:>7d}"
            f"{s['fp']:>7d}"
            f"{s['fn']:>7d}"
            f"{s['mean_runtime_ms_per_transition']:>14.3f}"
        )

    print("=" * 118)
    print()
    print(f"Summary        : {SUMMARY_FILE}")
    print(f"Per-transition : {TRANSITION_FILE}")


if __name__ == "__main__":
    main()
