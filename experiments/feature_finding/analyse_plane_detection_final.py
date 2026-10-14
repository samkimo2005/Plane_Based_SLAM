from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_DIR = (
    PROJECT_ROOT
    / "results"
    / "plane_detection"
    / "final"
)

INPUT_CSV = (
    FINAL_DIR
    / "multiplane_detection_final_results.csv"
)

SUMMARY_CSV = (
    FINAL_DIR
    / "plane_detection_final_summary.csv"
)

BY_PLANE_COUNT_CSV = (
    FINAL_DIR
    / "plane_detection_final_by_plane_count.csv"
)

BY_NOISE_CSV = (
    FINAL_DIR
    / "plane_detection_final_by_noise.csv"
)

BY_OUTLIER_CSV = (
    FINAL_DIR
    / "plane_detection_final_by_outlier.csv"
)

BY_CONDITION_CSV = (
    FINAL_DIR
    / "plane_detection_final_by_condition.csv"
)

ROBUSTNESS_CSV = (
    FINAL_DIR
    / "plane_detection_final_robustness.csv"
)

DECISION_INPUT_CSV = (
    FINAL_DIR
    / "plane_detection_decision_matrix_inputs.csv"
)

FIGURE_DIR = (
    FINAL_DIR
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


EXPECTED_CASES = 60
EXPECTED_METHODS = 5
EXPECTED_ROWS = EXPECTED_CASES * EXPECTED_METHODS


# ============================================================
# 2. CSV loading
# ============================================================

NUMERIC_FIELDS = {
    "plane_count_gt",
    "noise_sigma_m",
    "outlier_ratio",
    "repeat_id",
    "dataset_seed",
    "algorithm_seed",
    "num_points",
    "detected_plane_count",
    "matched_plane_count",
    "false_plane_count",
    "missed_plane_count",
    "correct_plane_count",
    "plane_precision",
    "plane_recall",
    "plane_f1",
    "matched_mean_iou",
    "mean_normal_error_deg",
    "mean_offset_error_m",
    "runtime_ms",
    "success",
}


def parse_number(value):
    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    try:
        return float(text)

    except ValueError:
        return None


def load_rows():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            "FINAL result CSV not found:\n"
            f"{INPUT_CSV}\n\n"
            "Run run_plane_detection_final.py first."
        )

    rows = []

    with open(
        INPUT_CSV,
        "r",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        for raw in reader:
            row = dict(raw)

            for field in NUMERIC_FIELDS:
                if field in row:
                    row[field] = parse_number(
                        row[field]
                    )

            rows.append(row)

    return rows


# ============================================================
# 3. Validation
# ============================================================

def validate_rows(rows):
    if len(rows) != EXPECTED_ROWS:
        print(
            f"WARNING: expected {EXPECTED_ROWS} rows "
            f"but found {len(rows)}."
        )

    methods = sorted(
        {
            row["method"]
            for row in rows
        }
    )

    print()
    print("Detected methods:")

    for method in methods:
        count = sum(
            row["method"] == method
            for row in rows
        )

        print(
            f"  {method}: {count} rows"
        )

    if len(methods) != EXPECTED_METHODS:
        print(
            f"WARNING: expected {EXPECTED_METHODS} methods "
            f"but found {len(methods)}."
        )

    for row in rows:
        if row.get("dataset_split") != "FINAL":
            raise RuntimeError(
                "Input CSV contains a row that is not "
                "marked as FINAL."
            )

    return methods


# ============================================================
# 4. Statistical helpers
# ============================================================

def valid_values(rows, key):
    values = []

    for row in rows:
        value = row.get(key)

        if value is None:
            continue

        value = float(value)

        if math.isfinite(value):
            values.append(value)

    return values


def mean_or_blank(rows, key):
    values = valid_values(rows, key)

    if not values:
        return ""

    return float(
        np.mean(values)
    )


def median_or_blank(rows, key):
    values = valid_values(rows, key)

    if not values:
        return ""

    return float(
        np.median(values)
    )


def std_or_blank(rows, key):
    values = valid_values(rows, key)

    if not values:
        return ""

    if len(values) < 2:
        return 0.0

    return float(
        np.std(
            values,
            ddof=1,
        )
    )


def sum_or_zero(rows, key):
    return float(
        sum(
            row.get(key) or 0.0
            for row in rows
        )
    )


def micro_plane_metrics(rows):
    """
    Aggregate TP/FP/FN over all cases before calculating
    precision, recall and F1.

    Here:
        TP = matched planes
        FP = false detected planes
        FN = missed GT planes
    """

    tp = int(
        round(
            sum_or_zero(
                rows,
                "matched_plane_count",
            )
        )
    )

    fp = int(
        round(
            sum_or_zero(
                rows,
                "false_plane_count",
            )
        )
    )

    fn = int(
        round(
            sum_or_zero(
                rows,
                "missed_plane_count",
            )
        )
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "total_tp": tp,
        "total_fp": fp,
        "total_fn": fn,
        "micro_precision": float(precision),
        "micro_recall": float(recall),
        "micro_f1": float(f1),
    }


# ============================================================
# 5. Aggregate one group
# ============================================================

def aggregate_group(rows):
    micro = micro_plane_metrics(rows)

    return {
        "num_cases":
            len(rows),

        "mean_precision":
            mean_or_blank(
                rows,
                "plane_precision",
            ),

        "mean_recall":
            mean_or_blank(
                rows,
                "plane_recall",
            ),

        "mean_f1":
            mean_or_blank(
                rows,
                "plane_f1",
            ),

        **micro,

        "correct_plane_count_rate":
            mean_or_blank(
                rows,
                "correct_plane_count",
            ),

        "mean_matched_iou":
            mean_or_blank(
                rows,
                "matched_mean_iou",
            ),

        "mean_normal_error_deg":
            mean_or_blank(
                rows,
                "mean_normal_error_deg",
            ),

        "mean_offset_error_m":
            mean_or_blank(
                rows,
                "mean_offset_error_m",
            ),

        "mean_false_plane_count":
            mean_or_blank(
                rows,
                "false_plane_count",
            ),

        "mean_missed_plane_count":
            mean_or_blank(
                rows,
                "missed_plane_count",
            ),

        "success_rate":
            mean_or_blank(
                rows,
                "success",
            ),

        "mean_runtime_ms":
            mean_or_blank(
                rows,
                "runtime_ms",
            ),

        "median_runtime_ms":
            median_or_blank(
                rows,
                "runtime_ms",
            ),

        "std_runtime_ms":
            std_or_blank(
                rows,
                "runtime_ms",
            ),
    }


# ============================================================
# 6. Generic grouping / CSV output
# ============================================================

def group_rows(rows, keys):
    groups = defaultdict(list)

    for row in rows:
        key = tuple(
            row[field]
            for field in keys
        )

        groups[key].append(row)

    return groups


def write_aggregate_csv(
    rows,
    group_keys,
    output_path,
):
    groups = group_rows(
        rows,
        group_keys,
    )

    output_rows = []

    for key, group in sorted(
        groups.items(),
        key=lambda item: item[0],
    ):
        output_row = {
            field: value
            for field, value in zip(
                group_keys,
                key,
            )
        }

        output_row.update(
            aggregate_group(group)
        )

        output_rows.append(
            output_row
        )

    if not output_rows:
        return

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                output_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )


# ============================================================
# 7. Robustness summary
# ============================================================

def subset_rows(
    rows,
    method,
    noise=None,
    outlier=None,
):
    selected = []

    for row in rows:
        if row["method"] != method:
            continue

        if (
            noise is not None
            and not np.isclose(
                row["noise_sigma_m"],
                noise,
            )
        ):
            continue

        if (
            outlier is not None
            and not np.isclose(
                row["outlier_ratio"],
                outlier,
            )
        ):
            continue

        selected.append(row)

    return selected


def write_robustness_csv(
    rows,
    methods,
):
    output_rows = []

    for method in methods:
        clean = subset_rows(
            rows,
            method,
            noise=0.0,
            outlier=0.0,
        )

        noisy = subset_rows(
            rows,
            method,
            noise=0.020,
            outlier=0.0,
        )

        outlier_only = subset_rows(
            rows,
            method,
            noise=0.0,
            outlier=0.20,
        )

        hardest = subset_rows(
            rows,
            method,
            noise=0.020,
            outlier=0.20,
        )

        clean_f1 = mean_or_blank(
            clean,
            "plane_f1",
        )

        noisy_f1 = mean_or_blank(
            noisy,
            "plane_f1",
        )

        outlier_f1 = mean_or_blank(
            outlier_only,
            "plane_f1",
        )

        hardest_f1 = mean_or_blank(
            hardest,
            "plane_f1",
        )

        clean_iou = mean_or_blank(
            clean,
            "matched_mean_iou",
        )

        hardest_iou = mean_or_blank(
            hardest,
            "matched_mean_iou",
        )

        output_rows.append(
            {
                "method":
                    method,

                "clean_mean_f1":
                    clean_f1,

                "noise_only_mean_f1":
                    noisy_f1,

                "outlier_only_mean_f1":
                    outlier_f1,

                "noise_plus_outlier_mean_f1":
                    hardest_f1,

                "f1_drop_clean_to_noise":
                    (
                        clean_f1 - noisy_f1
                        if clean_f1 != ""
                        and noisy_f1 != ""
                        else ""
                    ),

                "f1_drop_clean_to_outlier":
                    (
                        clean_f1 - outlier_f1
                        if clean_f1 != ""
                        and outlier_f1 != ""
                        else ""
                    ),

                "f1_drop_clean_to_hardest":
                    (
                        clean_f1 - hardest_f1
                        if clean_f1 != ""
                        and hardest_f1 != ""
                        else ""
                    ),

                "clean_mean_iou":
                    clean_iou,

                "hardest_mean_iou":
                    hardest_iou,

                "iou_drop_clean_to_hardest":
                    (
                        clean_iou - hardest_iou
                        if clean_iou != ""
                        and hardest_iou != ""
                        else ""
                    ),
            }
        )

    with open(
        ROBUSTNESS_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                output_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )


# ============================================================
# 8. Decision-matrix input table
#
# This file deliberately contains measured FINAL evidence only.
# It does NOT automatically assign subjective implementation
# or resource scores.
# ============================================================

def write_decision_inputs(
    rows,
    methods,
):
    output_rows = []

    for method in methods:
        subset = [
            row
            for row in rows
            if row["method"] == method
        ]

        overall = aggregate_group(
            subset
        )

        hardest = subset_rows(
            rows,
            method,
            noise=0.020,
            outlier=0.20,
        )

        hardest_f1 = mean_or_blank(
            hardest,
            "plane_f1",
        )

        output_rows.append(
            {
                "method":
                    method,

                "final_mean_f1":
                    overall["mean_f1"],

                "final_micro_f1":
                    overall["micro_f1"],

                "final_correct_plane_count_rate":
                    overall[
                        "correct_plane_count_rate"
                    ],

                "final_mean_matched_iou":
                    overall[
                        "mean_matched_iou"
                    ],

                "final_mean_normal_error_deg":
                    overall[
                        "mean_normal_error_deg"
                    ],

                "final_mean_offset_error_m":
                    overall[
                        "mean_offset_error_m"
                    ],

                "final_mean_runtime_ms":
                    overall[
                        "mean_runtime_ms"
                    ],

                "hardest_condition_mean_f1":
                    hardest_f1,

                "mean_false_plane_count":
                    overall[
                        "mean_false_plane_count"
                    ],

                "mean_missed_plane_count":
                    overall[
                        "mean_missed_plane_count"
                    ],
            }
        )

    with open(
        DECISION_INPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                output_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )


# ============================================================
# 9. Figure helpers
# ============================================================

def mean_metric_for(
    rows,
    method,
    metric,
    *,
    plane_count=None,
    noise=None,
    outlier=None,
):
    subset = []

    for row in rows:
        if row["method"] != method:
            continue

        if (
            plane_count is not None
            and int(
                row["plane_count_gt"]
            )
            != int(
                plane_count
            )
        ):
            continue

        if (
            noise is not None
            and not np.isclose(
                row["noise_sigma_m"],
                noise,
            )
        ):
            continue

        if (
            outlier is not None
            and not np.isclose(
                row["outlier_ratio"],
                outlier,
            )
        ):
            continue

        subset.append(row)

    values = valid_values(
        subset,
        metric,
    )

    if not values:
        return np.nan

    return float(
        np.mean(values)
    )


def plot_overall_metric(
    rows,
    methods,
    metric,
    ylabel,
    title,
    filename,
    ylim=None,
):
    values = [
        mean_metric_for(
            rows,
            method,
            metric,
        )
        for method in methods
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        methods,
        values,
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    if ylim is not None:
        ax.set_ylim(
            *ylim
        )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / filename,
        dpi=200,
    )

    plt.close(fig)


def plot_grouped_metric(
    rows,
    methods,
    category_values,
    category_label,
    metric,
    ylabel,
    title,
    filename,
    filter_name,
    ylim=None,
):
    x = np.arange(
        len(category_values)
    )

    width = (
        0.8
        / max(
            len(methods),
            1,
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    centre = (
        len(methods) - 1
    ) / 2.0

    for index, method in enumerate(
        methods
    ):
        values = []

        for category in category_values:
            kwargs = {
                "plane_count": None,
                "noise": None,
                "outlier": None,
            }

            kwargs[
                filter_name
            ] = category

            values.append(
                mean_metric_for(
                    rows,
                    method,
                    metric,
                    **kwargs,
                )
            )

        ax.bar(
            x
            + (
                index
                - centre
            )
            * width,
            values,
            width=width,
            label=method,
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            str(value)
            for value in category_values
        ]
    )

    ax.set_xlabel(
        category_label
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    if ylim is not None:
        ax.set_ylim(
            *ylim
        )

    ax.legend(
        fontsize=8
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / filename,
        dpi=200,
    )

    plt.close(fig)


def generate_figures(
    rows,
    methods,
):
    plot_overall_metric(
        rows,
        methods,
        "plane_f1",
        "Mean plane-level F1",
        "Overall Plane Detection F1",
        "overall_f1.png",
        ylim=(0.0, 1.05),
    )

    plot_overall_metric(
        rows,
        methods,
        "matched_mean_iou",
        "Mean matched-plane IoU",
        "Overall Matched-Plane IoU",
        "overall_iou.png",
        ylim=(0.0, 1.05),
    )

    plot_overall_metric(
        rows,
        methods,
        "correct_plane_count",
        "Correct plane-count rate",
        "Correct Plane-Count Rate",
        "overall_plane_count_accuracy.png",
        ylim=(0.0, 1.05),
    )

    plot_overall_metric(
        rows,
        methods,
        "runtime_ms",
        "Mean runtime (ms)",
        "Plane Detection Runtime",
        "overall_runtime.png",
    )

    plot_overall_metric(
        rows,
        methods,
        "mean_normal_error_deg",
        "Mean normal error (deg)",
        "Matched-Plane Normal Error",
        "overall_normal_error.png",
    )

    plot_overall_metric(
        rows,
        methods,
        "mean_offset_error_m",
        "Mean offset error (m)",
        "Matched-Plane Offset Error",
        "overall_offset_error.png",
    )

    plot_grouped_metric(
        rows,
        methods,
        [2, 3, 4],
        "Ground-truth plane count",
        "plane_f1",
        "Mean plane-level F1",
        "F1 vs Plane Count",
        "f1_by_plane_count.png",
        "plane_count",
        ylim=(0.0, 1.05),
    )

    plot_grouped_metric(
        rows,
        methods,
        [0.0, 0.020],
        "Noise sigma (m)",
        "plane_f1",
        "Mean plane-level F1",
        "F1 vs Noise",
        "f1_by_noise.png",
        "noise",
        ylim=(0.0, 1.05),
    )

    plot_grouped_metric(
        rows,
        methods,
        [0.0, 0.20],
        "Outlier ratio",
        "plane_f1",
        "Mean plane-level F1",
        "F1 vs Outlier Ratio",
        "f1_by_outlier.png",
        "outlier",
        ylim=(0.0, 1.05),
    )


# ============================================================
# 10. Console summary
# ============================================================

def print_summary(
    rows,
    methods,
):
    print()
    print(
        "=" * 86
    )

    print(
        "PLANE DETECTION FINAL SUMMARY"
    )

    print(
        "=" * 86
    )

    for method in methods:
        subset = [
            row
            for row in rows
            if row["method"] == method
        ]

        result = aggregate_group(
            subset
        )

        print()
        print(method)

        print(
            f"  Mean F1              : "
            f"{result['mean_f1']:.4f}"
        )

        print(
            f"  Micro F1             : "
            f"{result['micro_f1']:.4f}"
        )

        print(
            f"  Correct count rate   : "
            f"{result['correct_plane_count_rate']:.4f}"
        )

        print(
            f"  Mean matched IoU     : "
            f"{result['mean_matched_iou']:.4f}"
        )

        print(
            f"  Mean normal error    : "
            f"{result['mean_normal_error_deg']:.4f} deg"
        )

        print(
            f"  Mean offset error    : "
            f"{result['mean_offset_error_m']:.6f} m"
        )

        print(
            f"  Mean false planes    : "
            f"{result['mean_false_plane_count']:.4f}"
        )

        print(
            f"  Mean missed planes   : "
            f"{result['mean_missed_plane_count']:.4f}"
        )

        print(
            f"  Mean runtime         : "
            f"{result['mean_runtime_ms']:.3f} ms"
        )


# ============================================================
# 11. Main
# ============================================================

def main():
    print(
        "=" * 86
    )

    print(
        "Analyse Plane Detection FINAL Benchmark"
    )

    print(
        "=" * 86
    )

    print()
    print(
        "Input:"
    )

    print(
        INPUT_CSV
    )

    rows = load_rows()

    methods = validate_rows(
        rows
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    write_aggregate_csv(
        rows,
        [
            "method",
        ],
        SUMMARY_CSV,
    )

    # --------------------------------------------------------
    # Controlled robustness breakdowns
    # --------------------------------------------------------

    write_aggregate_csv(
        rows,
        [
            "method",
            "plane_count_gt",
        ],
        BY_PLANE_COUNT_CSV,
    )

    write_aggregate_csv(
        rows,
        [
            "method",
            "noise_sigma_m",
        ],
        BY_NOISE_CSV,
    )

    write_aggregate_csv(
        rows,
        [
            "method",
            "outlier_ratio",
        ],
        BY_OUTLIER_CSV,
    )

    write_aggregate_csv(
        rows,
        [
            "method",
            "plane_count_gt",
            "noise_sigma_m",
            "outlier_ratio",
        ],
        BY_CONDITION_CSV,
    )

    # --------------------------------------------------------
    # Robustness and decision-matrix evidence
    # --------------------------------------------------------

    write_robustness_csv(
        rows,
        methods,
    )

    write_decision_inputs(
        rows,
        methods,
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    generate_figures(
        rows,
        methods,
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print_summary(
        rows,
        methods,
    )

    print()
    print(
        "=" * 86
    )

    print(
        "ANALYSIS OUTPUTS"
    )

    print(
        "=" * 86
    )

    for path in [
        SUMMARY_CSV,
        BY_PLANE_COUNT_CSV,
        BY_NOISE_CSV,
        BY_OUTLIER_CSV,
        BY_CONDITION_CSV,
        ROBUSTNESS_CSV,
        DECISION_INPUT_CSV,
    ]:
        print(path)

    print(
        FIGURE_DIR
    )

    print()
    print(
        "The decision-matrix input CSV contains measured "
        "FINAL evidence only. Subjective implementation/resource "
        "scores are intentionally not assigned by this script."
    )


if __name__ == "__main__":
    main()
