import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "feature_finding"

INPUT_FILE = RESULTS_DIR / "multiplane_plane_detection_results.csv"

CONDITION_OUTPUT = (
    RESULTS_DIR
    / "multiplane_comparison_by_condition.csv"
)

OVERALL_OUTPUT = (
    RESULTS_DIR
    / "multiplane_overall_comparison.csv"
)


# ==========================================================
# Helpers
# ==========================================================

def mean(values):
    values = list(values)

    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


def std(values):
    values = list(values)

    if len(values) <= 1:
        return 0.0 if len(values) == 1 else np.nan

    return float(
        np.std(
            values,
            ddof=1,
        )
    )


def median(values):
    values = list(values)

    if len(values) == 0:
        return np.nan

    return float(np.median(values))


def p95(values):
    values = list(values)

    if len(values) == 0:
        return np.nan

    return float(
        np.percentile(
            values,
            95,
        )
    )


def float_values(rows, column):
    values = []

    for row in rows:
        value = str(
            row[column]
        ).strip()

        if value == "":
            continue

        values.append(float(value))

    return values


# ==========================================================
# Load
# ==========================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find: {INPUT_FILE}"
    )

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8",
) as f:
    rows = list(
        csv.DictReader(f)
    )

if len(rows) != 300:
    raise RuntimeError(
        f"Expected 300 runs "
        f"(60 samples x 5 methods), "
        f"found {len(rows)}"
    )

print(
    "=== Multi-Plane Benchmark Analysis ==="
)

print(
    "Rows:",
    len(rows)
)


# ==========================================================
# Condition-level summary
#
# Condition =
#   method + GT plane count + noise + outlier ratio
#
# 5 repetitions per condition.
# ==========================================================

groups = defaultdict(list)

for row in rows:
    key = (
        row["method"],
        int(row["plane_count_gt"]),
        float(row["noise_sigma_m"]),
        float(row["outlier_ratio"]),
    )

    groups[key].append(row)


condition_rows = []

for key in sorted(
    groups,
    key=lambda item: (
        item[0],
        item[1],
        item[2],
        item[3],
    ),
):
    method, plane_count, noise, outlier_ratio = key
    group = groups[key]

    if len(group) != 5:
        raise RuntimeError(
            f"{key} has {len(group)} repeats, expected 5."
        )

    runtime = float_values(
        group,
        "runtime_ms",
    )

    plane_precision = float_values(
        group,
        "plane_precision",
    )

    plane_recall = float_values(
        group,
        "plane_recall",
    )

    plane_f1 = float_values(
        group,
        "plane_f1",
    )

    iou = float_values(
        group,
        "matched_mean_iou",
    )

    normal_error = float_values(
        group,
        "mean_normal_error_deg",
    )

    offset_error = float_values(
        group,
        "mean_offset_error_m",
    )

    correct_count = [
        int(row["correct_plane_count"])
        for row in group
    ]

    false_planes = [
        int(row["false_plane_count"])
        for row in group
    ]

    missed_planes = [
        int(row["missed_plane_count"])
        for row in group
    ]

    detected_count = [
        int(row["detected_plane_count"])
        for row in group
    ]

    condition_rows.append(
        {
            "method": method,
            "plane_count_gt": plane_count,
            "noise_sigma_m": noise,
            "outlier_ratio": outlier_ratio,

            "plane_precision_mean": mean(plane_precision),
            "plane_precision_std": std(plane_precision),

            "plane_recall_mean": mean(plane_recall),
            "plane_recall_std": std(plane_recall),

            "plane_f1_mean": mean(plane_f1),
            "plane_f1_std": std(plane_f1),

            "matched_mean_iou": mean(iou),

            "normal_error_mean_deg": mean(normal_error),
            "offset_error_mean_m": mean(offset_error),

            "correct_plane_count_rate": mean(correct_count),

            "detected_plane_count_mean": mean(detected_count),
            "false_plane_count_mean": mean(false_planes),
            "missed_plane_count_mean": mean(missed_planes),

            "runtime_mean_ms": mean(runtime),
            "runtime_std_ms": std(runtime),
            "runtime_median_ms": median(runtime),
            "runtime_p95_ms": p95(runtime),
        }
    )


condition_fieldnames = list(
    condition_rows[0].keys()
)

with open(
    CONDITION_OUTPUT,
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=condition_fieldnames,
    )

    writer.writeheader()
    writer.writerows(condition_rows)


# ==========================================================
# Overall summary by method
# ==========================================================

method_groups = defaultdict(list)

for row in rows:
    method_groups[
        row["method"]
    ].append(row)


overall_rows = []

for method in sorted(method_groups):
    group = method_groups[method]

    runtime = float_values(
        group,
        "runtime_ms",
    )

    plane_precision = float_values(
        group,
        "plane_precision",
    )

    plane_recall = float_values(
        group,
        "plane_recall",
    )

    plane_f1 = float_values(
        group,
        "plane_f1",
    )

    iou = float_values(
        group,
        "matched_mean_iou",
    )

    normal_error = float_values(
        group,
        "mean_normal_error_deg",
    )

    offset_error = float_values(
        group,
        "mean_offset_error_m",
    )

    correct_count = [
        int(row["correct_plane_count"])
        for row in group
    ]

    false_planes = [
        int(row["false_plane_count"])
        for row in group
    ]

    missed_planes = [
        int(row["missed_plane_count"])
        for row in group
    ]

    # Runtime scaling with plane count.
    runtime_by_plane_count = {}

    f1_by_plane_count = {}

    for plane_count in [2, 3, 4]:
        subset = [
            row
            for row in group
            if int(row["plane_count_gt"])
            == plane_count
        ]

        runtime_by_plane_count[
            plane_count
        ] = mean(
            float_values(
                subset,
                "runtime_ms",
            )
        )

        f1_by_plane_count[
            plane_count
        ] = mean(
            float_values(
                subset,
                "plane_f1",
            )
        )

    overall_rows.append(
        {
            "method": method,

            "plane_precision_mean": mean(plane_precision),
            "plane_recall_mean": mean(plane_recall),
            "plane_f1_mean": mean(plane_f1),
            "plane_f1_std": std(plane_f1),

            "matched_mean_iou": mean(iou),

            "normal_error_mean_deg": mean(normal_error),
            "offset_error_mean_m": mean(offset_error),

            "correct_plane_count_rate": mean(correct_count),

            "false_plane_count_mean": mean(false_planes),
            "missed_plane_count_mean": mean(missed_planes),

            "runtime_mean_ms": mean(runtime),
            "runtime_std_ms": std(runtime),
            "runtime_median_ms": median(runtime),
            "runtime_p95_ms": p95(runtime),

            "plane_f1_2_planes": f1_by_plane_count[2],
            "plane_f1_3_planes": f1_by_plane_count[3],
            "plane_f1_4_planes": f1_by_plane_count[4],

            "runtime_2_planes_ms": runtime_by_plane_count[2],
            "runtime_3_planes_ms": runtime_by_plane_count[3],
            "runtime_4_planes_ms": runtime_by_plane_count[4],
        }
    )


overall_fieldnames = list(
    overall_rows[0].keys()
)

with open(
    OVERALL_OUTPUT,
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=overall_fieldnames,
    )

    writer.writeheader()
    writer.writerows(overall_rows)


# ==========================================================
# Console summary
# ==========================================================

print(
    "\n=============================================="
)

print(
    "MULTI-PLANE OVERALL COMPARISON"
)

print(
    "=============================================="
)

print(
    f"{'Method':<26}"
    f"{'Prec':>8}"
    f"{'Recall':>8}"
    f"{'F1':>8}"
    f"{'IoU':>8}"
    f"{'Count':>8}"
    f"{'Runtime':>11}"
)

print(
    "-" * 77
)

for row in overall_rows:
    print(
        f"{row['method']:<26}"
        f"{row['plane_precision_mean']:>8.3f}"
        f"{row['plane_recall_mean']:>8.3f}"
        f"{row['plane_f1_mean']:>8.3f}"
        f"{row['matched_mean_iou']:>8.3f}"
        f"{row['correct_plane_count_rate']:>8.3f}"
        f"{row['runtime_mean_ms']:>9.1f}ms"
    )

print(
    "\nCondition-level results:"
)
print(CONDITION_OUTPUT)

print(
    "\nOverall results:"
)
print(OVERALL_OUTPUT)
