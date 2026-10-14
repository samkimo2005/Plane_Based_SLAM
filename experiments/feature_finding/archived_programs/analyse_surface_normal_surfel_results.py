import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


# ==========================================================
# Project paths
# ==========================================================

project_root = Path(
    __file__
).resolve().parents[2]


results_dir = (
    project_root
    / "results"
    / "feature_finding"
)


input_file = (
    results_dir
    / "surface_normal_surfel_results.csv"
)


summary_file = (
    results_dir
    / "surface_normal_surfel_summary_by_condition.csv"
)


EXPECTED_ROWS = 200
EXPECTED_CONDITIONS = 20
EXPECTED_REPEATS = 10


print(
    "=== Surface-Normal / Surfel Results Verification ==="
)


# ==========================================================
# Helper functions
# ==========================================================

def mean_value(values):

    if len(values) == 0:
        return np.nan

    return float(
        np.mean(values)
    )


def std_value(values):

    if len(values) == 0:
        return np.nan

    if len(values) == 1:
        return 0.0

    return float(
        np.std(
            values,
            ddof=1
        )
    )


def median_value(values):

    if len(values) == 0:
        return np.nan

    return float(
        np.median(values)
    )


def percentile_95(values):

    if len(values) == 0:
        return np.nan

    return float(
        np.percentile(
            values,
            95
        )
    )


# ==========================================================
# Load CSV
# ==========================================================

if not input_file.exists():

    raise FileNotFoundError(
        f"Cannot find result file: {input_file}"
    )


with open(
    input_file,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    columns = reader.fieldnames

    rows = list(reader)


# ==========================================================
# Test 1: required columns
# ==========================================================

print(
    "\nTest 1: Checking CSV columns"
)


required_columns = [
    "method",

    "sample_id",

    "noise_sigma_m",
    "outlier_ratio",
    "repeat_id",

    "num_points",
    "num_true_inliers",
    "num_true_outliers",

    "detected_inlier_count",

    "normal_error_deg",
    "offset_error_m",

    "tp",
    "fp",
    "fn",
    "tn",

    "precision",
    "recall",
    "f1",

    "runtime_ms",

    "num_planar_surfels",
    "num_orientation_support",
    "offset_peak_count",
    "initial_support",

    "failure_reason",
    "success"
]


for column in required_columns:

    assert column in columns, (
        f"Missing column: {column}"
    )


print(
    "Test 1 PASSED"
)


# ==========================================================
# Test 2: number of rows
# ==========================================================

print(
    "\nTest 2: Checking number of results"
)


assert len(rows) == EXPECTED_ROWS, (
    f"Expected {EXPECTED_ROWS} rows, "
    f"but found {len(rows)}"
)


print(
    "Rows:",
    len(rows)
)


print(
    "Test 2 PASSED"
)


# ==========================================================
# Test 3: unique sample IDs
# ==========================================================

print(
    "\nTest 3: Checking sample IDs"
)


sample_ids = [
    row["sample_id"]
    for row in rows
]


assert len(set(sample_ids)) == EXPECTED_ROWS, (
    "Duplicate sample IDs found"
)


print(
    "Test 3 PASSED"
)


# ==========================================================
# Test 4: success / failure status
# ==========================================================

print(
    "\nTest 4: Checking success / failure status"
)


successful_rows = []
failed_rows = []


for row in rows:

    success = int(
        row["success"]
    )


    assert success in (
        0,
        1
    )


    if success == 1:

        successful_rows.append(
            row
        )

    else:

        failed_rows.append(
            row
        )


overall_success_rate = (
    len(successful_rows)
    / len(rows)
)


print(
    "Successful runs:",
    len(successful_rows)
)


print(
    "Failed runs:",
    len(failed_rows)
)


print(
    "Success rate:",
    f"{overall_success_rate:.4f}"
)


print(
    "Test 4 PASSED"
)


# ==========================================================
# Test 5: numerical validity
# ==========================================================

print(
    "\nTest 5: Checking numerical values"
)


for row in rows:

    runtime = float(
        row["runtime_ms"]
    )


    assert np.isfinite(runtime), (
        f"{row['sample_id']} has invalid runtime"
    )


    assert runtime >= 0


    diagnostic_columns = [
        "num_planar_surfels",
        "num_orientation_support",
        "offset_peak_count",
        "initial_support"
    ]


    for column in diagnostic_columns:

        value = int(
            row[column]
        )

        assert value >= 0


    if int(
        row["success"]
    ) == 1:

        common_metrics = [
            "normal_error_deg",
            "offset_error_m",
            "precision",
            "recall",
            "f1"
        ]


        for column in common_metrics:

            value = float(
                row[column]
            )


            assert np.isfinite(value), (
                f"{row['sample_id']} has invalid "
                f"value in {column}"
            )


print(
    "Test 5 PASSED"
)


# ==========================================================
# Test 6: metric ranges
# ==========================================================

print(
    "\nTest 6: Checking metric ranges"
)


for row in successful_rows:

    precision = float(
        row["precision"]
    )

    recall = float(
        row["recall"]
    )

    f1 = float(
        row["f1"]
    )

    normal_error = float(
        row["normal_error_deg"]
    )

    offset_error = float(
        row["offset_error_m"]
    )


    assert (
        0.0
        <= precision
        <= 1.0
    )


    assert (
        0.0
        <= recall
        <= 1.0
    )


    assert (
        0.0
        <= f1
        <= 1.0
    )


    assert normal_error >= 0

    assert offset_error >= 0


print(
    "Test 6 PASSED"
)


# ==========================================================
# Test 7: confusion matrix consistency
# ==========================================================

print(
    "\nTest 7: Checking confusion matrix"
)


for row in successful_rows:

    tp = int(
        row["tp"]
    )

    fp = int(
        row["fp"]
    )

    fn = int(
        row["fn"]
    )

    tn = int(
        row["tn"]
    )


    num_points = int(
        row["num_points"]
    )


    num_true_inliers = int(
        row["num_true_inliers"]
    )


    num_true_outliers = int(
        row["num_true_outliers"]
    )


    detected_inliers = int(
        row["detected_inlier_count"]
    )


    assert (
        tp
        + fp
        + fn
        + tn
    ) == num_points


    assert (
        tp
        + fn
    ) == num_true_inliers


    assert (
        fp
        + tn
    ) == num_true_outliers


    assert (
        tp
        + fp
    ) == detected_inliers


print(
    "Test 7 PASSED"
)


# ==========================================================
# Test 8: surfel diagnostics consistency
# ==========================================================

print(
    "\nTest 8: Checking surfel diagnostics"
)


for row in successful_rows:

    num_points = int(
        row["num_points"]
    )


    planar_surfels = int(
        row["num_planar_surfels"]
    )


    orientation_support = int(
        row["num_orientation_support"]
    )


    offset_peak_count = int(
        row["offset_peak_count"]
    )


    initial_support = int(
        row["initial_support"]
    )


    assert (
        planar_surfels
        <= num_points
    )


    assert (
        orientation_support
        <= planar_surfels
    )


    assert (
        offset_peak_count
        <= orientation_support
    )


    assert (
        initial_support
        <= orientation_support
    )


print(
    "Test 8 PASSED"
)


# ==========================================================
# Test 9: condition structure
# ==========================================================

print(
    "\nTest 9: Checking experimental conditions"
)


condition_groups = defaultdict(
    list
)


for row in rows:

    condition = (
        float(
            row["noise_sigma_m"]
        ),

        float(
            row["outlier_ratio"]
        )
    )


    condition_groups[
        condition
    ].append(
        row
    )


assert len(
    condition_groups
) == EXPECTED_CONDITIONS, (
    f"Expected {EXPECTED_CONDITIONS} conditions, "
    f"but found {len(condition_groups)}"
)


for condition, condition_rows in condition_groups.items():

    assert len(
        condition_rows
    ) == EXPECTED_REPEATS, (
        f"Condition {condition} has "
        f"{len(condition_rows)} repeats"
    )


print(
    "Conditions:",
    len(condition_groups)
)


print(
    "Repeats per condition:",
    EXPECTED_REPEATS
)


print(
    "Test 9 PASSED"
)


print(
    "\n=== Verification PASSED ==="
)


# ==========================================================
# Condition-level analysis
# ==========================================================

print(
    "\n=== Surface-Normal / Surfel Analysis ==="
)


summary_rows = []


for condition in sorted(
    condition_groups
):

    noise = condition[0]

    outlier_ratio = condition[1]


    condition_rows = (
        condition_groups[
            condition
        ]
    )


    condition_success = [
        row
        for row in condition_rows
        if int(
            row["success"]
        ) == 1
    ]


    condition_fail = [
        row
        for row in condition_rows
        if int(
            row["success"]
        ) == 0
    ]


    num_success = len(
        condition_success
    )


    num_fail = len(
        condition_fail
    )


    success_rate = (
        num_success
        / EXPECTED_REPEATS
    )


    failure_rate = (
        num_fail
        / EXPECTED_REPEATS
    )


    # ------------------------------------------------------
    # Accuracy metrics
    # ------------------------------------------------------

    normal_errors = [
        float(
            row["normal_error_deg"]
        )
        for row in condition_success
    ]


    offset_errors = [
        float(
            row["offset_error_m"]
        )
        for row in condition_success
    ]


    precisions = [
        float(
            row["precision"]
        )
        for row in condition_success
    ]


    recalls = [
        float(
            row["recall"]
        )
        for row in condition_success
    ]


    f1_values = [
        float(
            row["f1"]
        )
        for row in condition_success
    ]


    # ------------------------------------------------------
    # Runtime
    # ------------------------------------------------------

    runtimes = [
        float(
            row["runtime_ms"]
        )
        for row in condition_rows
    ]


    # ------------------------------------------------------
    # Surfel diagnostics
    # ------------------------------------------------------

    planar_surfels = [
        int(
            row["num_planar_surfels"]
        )
        for row in condition_success
    ]


    orientation_support = [
        int(
            row["num_orientation_support"]
        )
        for row in condition_success
    ]


    offset_peak_counts = [
        int(
            row["offset_peak_count"]
        )
        for row in condition_success
    ]


    initial_support = [
        int(
            row["initial_support"]
        )
        for row in condition_success
    ]


    # ------------------------------------------------------
    # Effective F1
    # ------------------------------------------------------

    if num_success > 0:

        f1_mean = mean_value(
            f1_values
        )


        effective_f1 = (
            f1_mean
            * success_rate
        )

    else:

        f1_mean = np.nan

        effective_f1 = 0.0


    # ------------------------------------------------------
    # Print condition result
    # ------------------------------------------------------

    print(
        "\nNoise:",
        noise,
        "m"
    )


    print(
        "Outlier ratio:",
        outlier_ratio
    )


    print(
        "Success:",
        num_success,
        "/",
        EXPECTED_REPEATS
    )


    if num_success > 0:

        print(
            "Normal error:",
            f"{mean_value(normal_errors):.6f}",
            "±",
            f"{std_value(normal_errors):.6f}",
            "deg"
        )


        print(
            "Offset error:",
            f"{mean_value(offset_errors):.6f}",
            "±",
            f"{std_value(offset_errors):.6f}",
            "m"
        )


        print(
            "Precision:",
            f"{mean_value(precisions):.6f}",
            "±",
            f"{std_value(precisions):.6f}"
        )


        print(
            "Recall:",
            f"{mean_value(recalls):.6f}",
            "±",
            f"{std_value(recalls):.6f}"
        )


        print(
            "F1:",
            f"{mean_value(f1_values):.6f}",
            "±",
            f"{std_value(f1_values):.6f}"
        )


        print(
            "Effective F1:",
            f"{effective_f1:.6f}"
        )


        print(
            "Planar surfels:",
            f"{mean_value(planar_surfels):.1f}"
        )


        print(
            "Orientation support:",
            f"{mean_value(orientation_support):.1f}"
        )


        print(
            "Offset peak count:",
            f"{mean_value(offset_peak_counts):.1f}"
        )


        print(
            "Initial support:",
            f"{mean_value(initial_support):.1f}"
        )


    print(
        "Runtime:",
        f"{mean_value(runtimes):.3f}",
        "±",
        f"{std_value(runtimes):.3f}",
        "ms"
    )


    print(
        "Runtime median:",
        f"{median_value(runtimes):.3f}",
        "ms"
    )


    print(
        "Runtime P95:",
        f"{percentile_95(runtimes):.3f}",
        "ms"
    )


    # ------------------------------------------------------
    # Save condition summary
    # ------------------------------------------------------

    summary_rows.append(
        {
            "method": (
                "Surface-Normal / Surfel"
            ),

            "noise_sigma_m": (
                noise
            ),

            "outlier_ratio": (
                outlier_ratio
            ),

            "num_repeats": (
                EXPECTED_REPEATS
            ),

            "num_success": (
                num_success
            ),

            "num_fail": (
                num_fail
            ),

            "success_rate": (
                success_rate
            ),

            "failure_rate": (
                failure_rate
            ),

            "normal_error_mean_deg": (
                mean_value(
                    normal_errors
                )
            ),

            "normal_error_std_deg": (
                std_value(
                    normal_errors
                )
            ),

            "offset_error_mean_m": (
                mean_value(
                    offset_errors
                )
            ),

            "offset_error_std_m": (
                std_value(
                    offset_errors
                )
            ),

            "precision_mean": (
                mean_value(
                    precisions
                )
            ),

            "precision_std": (
                std_value(
                    precisions
                )
            ),

            "recall_mean": (
                mean_value(
                    recalls
                )
            ),

            "recall_std": (
                std_value(
                    recalls
                )
            ),

            "f1_mean": (
                mean_value(
                    f1_values
                )
            ),

            "f1_std": (
                std_value(
                    f1_values
                )
            ),

            "effective_f1": (
                effective_f1
            ),

            "runtime_mean_ms": (
                mean_value(
                    runtimes
                )
            ),

            "runtime_std_ms": (
                std_value(
                    runtimes
                )
            ),

            "runtime_median_ms": (
                median_value(
                    runtimes
                )
            ),

            "runtime_p95_ms": (
                percentile_95(
                    runtimes
                )
            ),

            "planar_surfels_mean": (
                mean_value(
                    planar_surfels
                )
            ),

            "orientation_support_mean": (
                mean_value(
                    orientation_support
                )
            ),

            "offset_peak_count_mean": (
                mean_value(
                    offset_peak_counts
                )
            ),

            "initial_support_mean": (
                mean_value(
                    initial_support
                )
            )
        }
    )


# ==========================================================
# Save condition summary
# ==========================================================

summary_columns = [
    "method",

    "noise_sigma_m",
    "outlier_ratio",

    "num_repeats",
    "num_success",
    "num_fail",

    "success_rate",
    "failure_rate",

    "normal_error_mean_deg",
    "normal_error_std_deg",

    "offset_error_mean_m",
    "offset_error_std_m",

    "precision_mean",
    "precision_std",

    "recall_mean",
    "recall_std",

    "f1_mean",
    "f1_std",

    "effective_f1",

    "runtime_mean_ms",
    "runtime_std_ms",
    "runtime_median_ms",
    "runtime_p95_ms",

    "planar_surfels_mean",
    "orientation_support_mean",
    "offset_peak_count_mean",
    "initial_support_mean"
]


with open(
    summary_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary_columns
    )

    writer.writeheader()

    writer.writerows(
        summary_rows
    )


# ==========================================================
# Overall analysis
# ==========================================================

print(
    "\n=== Overall Surface-Normal / Surfel Results ==="
)


print(
    "Successful:",
    len(successful_rows),
    "/",
    len(rows)
)


print(
    "Failed:",
    len(failed_rows)
)


print(
    "Success rate:",
    f"{overall_success_rate:.4f}"
)


# ----------------------------------------------------------
# Accuracy
# ----------------------------------------------------------

if len(
    successful_rows
) > 0:

    all_normal_errors = [
        float(
            row["normal_error_deg"]
        )
        for row in successful_rows
    ]


    all_offset_errors = [
        float(
            row["offset_error_m"]
        )
        for row in successful_rows
    ]


    all_precisions = [
        float(
            row["precision"]
        )
        for row in successful_rows
    ]


    all_recalls = [
        float(
            row["recall"]
        )
        for row in successful_rows
    ]


    all_f1 = [
        float(
            row["f1"]
        )
        for row in successful_rows
    ]


    print(
        "\nNormal error:",
        f"{mean_value(all_normal_errors):.6f}",
        "±",
        f"{std_value(all_normal_errors):.6f}",
        "deg"
    )


    print(
        "Offset error:",
        f"{mean_value(all_offset_errors):.6f}",
        "±",
        f"{std_value(all_offset_errors):.6f}",
        "m"
    )


    print(
        "Precision:",
        f"{mean_value(all_precisions):.6f}",
        "±",
        f"{std_value(all_precisions):.6f}"
    )


    print(
        "Recall:",
        f"{mean_value(all_recalls):.6f}",
        "±",
        f"{std_value(all_recalls):.6f}"
    )


    print(
        "F1:",
        f"{mean_value(all_f1):.6f}",
        "±",
        f"{std_value(all_f1):.6f}"
    )


    overall_effective_f1 = (
        mean_value(
            all_f1
        )
        * overall_success_rate
    )


    print(
        "Effective F1:",
        f"{overall_effective_f1:.6f}"
    )


# ----------------------------------------------------------
# Runtime
# ----------------------------------------------------------

all_runtimes = [
    float(
        row["runtime_ms"]
    )
    for row in rows
]


print(
    "\nRuntime mean:",
    f"{mean_value(all_runtimes):.3f}",
    "ms"
)


print(
    "Runtime median:",
    f"{median_value(all_runtimes):.3f}",
    "ms"
)


print(
    "Runtime std:",
    f"{std_value(all_runtimes):.3f}",
    "ms"
)


print(
    "Runtime minimum:",
    f"{min(all_runtimes):.3f}",
    "ms"
)


print(
    "Runtime maximum:",
    f"{max(all_runtimes):.3f}",
    "ms"
)


print(
    "Runtime P95:",
    f"{percentile_95(all_runtimes):.3f}",
    "ms"
)


# ----------------------------------------------------------
# Overall surfel diagnostics
# ----------------------------------------------------------

if len(
    successful_rows
) > 0:

    all_planar_surfels = [
        int(
            row["num_planar_surfels"]
        )
        for row in successful_rows
    ]


    all_orientation_support = [
        int(
            row["num_orientation_support"]
        )
        for row in successful_rows
    ]


    all_offset_peak_count = [
        int(
            row["offset_peak_count"]
        )
        for row in successful_rows
    ]


    all_initial_support = [
        int(
            row["initial_support"]
        )
        for row in successful_rows
    ]


    print(
        "\nSurfel diagnostics:"
    )


    print(
        "Mean planar surfels:",
        f"{mean_value(all_planar_surfels):.2f}"
    )


    print(
        "Mean orientation support:",
        f"{mean_value(all_orientation_support):.2f}"
    )


    print(
        "Mean offset peak count:",
        f"{mean_value(all_offset_peak_count):.2f}"
    )


    print(
        "Mean initial support:",
        f"{mean_value(all_initial_support):.2f}"
    )


# ==========================================================
# Failure reasons
# ==========================================================

if len(
    failed_rows
) > 0:

    failure_reasons = defaultdict(
        int
    )


    for row in failed_rows:

        failure_reasons[
            row["failure_reason"]
        ] += 1


    print(
        "\nFailure reasons:"
    )


    for reason, count in failure_reasons.items():

        print(
            reason,
            ":",
            count
        )


# ==========================================================
# Robustness analysis
# ==========================================================

print(
    "\n=== Robustness Summary ==="
)


effective_f1_values = np.array(
    [
        row[
            "effective_f1"
        ]
        for row in summary_rows
    ],
    dtype=np.float64
)


print(
    "Mean condition effective F1:",
    f"{np.mean(effective_f1_values):.6f}"
)


print(
    "Effective F1 std across conditions:",
    f"{np.std(effective_f1_values, ddof=1):.6f}"
)


print(
    "Best condition effective F1:",
    f"{np.max(effective_f1_values):.6f}"
)


print(
    "Worst condition effective F1:",
    f"{np.min(effective_f1_values):.6f}"
)


# ----------------------------------------------------------
# Average performance for each noise level
# ----------------------------------------------------------

noise_groups = defaultdict(
    list
)


for row in summary_rows:

    noise_groups[
        float(
            row["noise_sigma_m"]
        )
    ].append(
        float(
            row["effective_f1"]
        )
    )


print(
    "\nEffective F1 by noise level:"
)


noise_level_means = {}


for noise in sorted(
    noise_groups
):

    value = mean_value(
        noise_groups[
            noise
        ]
    )


    noise_level_means[
        noise
    ] = value


    print(
        f"Noise {noise:.3f} m:",
        f"{value:.6f}"
    )


# ----------------------------------------------------------
# Noise degradation
#
# Defined as:
#
# effective F1 at 0 mm noise
# minus
# effective F1 at 20 mm noise
# ----------------------------------------------------------

if (
    0.0 in noise_level_means
    and 0.02 in noise_level_means
):

    noise_degradation = (
        noise_level_means[
            0.0
        ]
        - noise_level_means[
            0.02
        ]
    )


    print(
        "\nNoise degradation "
        "(0 mm -> 20 mm):",
        f"{noise_degradation:.6f}"
    )


# ----------------------------------------------------------
# Average performance for each outlier level
# ----------------------------------------------------------

outlier_groups = defaultdict(
    list
)


for row in summary_rows:

    outlier_groups[
        float(
            row["outlier_ratio"]
        )
    ].append(
        float(
            row["effective_f1"]
        )
    )


print(
    "\nEffective F1 by outlier ratio:"
)


for outlier_ratio in sorted(
    outlier_groups
):

    value = mean_value(
        outlier_groups[
            outlier_ratio
        ]
    )


    print(
        f"Outlier ratio {outlier_ratio:.1f}:",
        f"{value:.6f}"
    )


# ==========================================================
# Diagnostic correlations
# ==========================================================

if len(
    successful_rows
) > 1:

    runtime_array = np.asarray(
        [
            float(
                row["runtime_ms"]
            )
            for row in successful_rows
        ],
        dtype=np.float64
    )


    f1_array = np.asarray(
        [
            float(
                row["f1"]
            )
            for row in successful_rows
        ],
        dtype=np.float64
    )


    planar_array = np.asarray(
        [
            int(
                row["num_planar_surfels"]
            )
            for row in successful_rows
        ],
        dtype=np.float64
    )


    orientation_array = np.asarray(
        [
            int(
                row["num_orientation_support"]
            )
            for row in successful_rows
        ],
        dtype=np.float64
    )


    initial_support_array = np.asarray(
        [
            int(
                row["initial_support"]
            )
            for row in successful_rows
        ],
        dtype=np.float64
    )


    print(
        "\n=== Diagnostic Correlations ==="
    )


    if np.std(
        planar_array
    ) > 0:

        correlation = np.corrcoef(
            f1_array,
            planar_array
        )[0, 1]


        print(
            "F1 vs planar surfel count:",
            f"{correlation:.4f}"
        )


    if np.std(
        orientation_array
    ) > 0:

        correlation = np.corrcoef(
            f1_array,
            orientation_array
        )[0, 1]


        print(
            "F1 vs orientation support:",
            f"{correlation:.4f}"
        )


    if np.std(
        initial_support_array
    ) > 0:

        correlation = np.corrcoef(
            f1_array,
            initial_support_array
        )[0, 1]


        print(
            "F1 vs initial support:",
            f"{correlation:.4f}"
        )


    if np.std(
        planar_array
    ) > 0:

        correlation = np.corrcoef(
            runtime_array,
            planar_array
        )[0, 1]


        print(
            "Runtime vs planar surfel count:",
            f"{correlation:.4f}"
        )


# ==========================================================
# Finish
# ==========================================================

print(
    "\nSummary saved to:"
)


print(
    summary_file
)