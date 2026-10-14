import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


# ==================================================
# Project paths
# ==================================================

project_root = Path(__file__).resolve().parents[2]

results_dir = (
    project_root
    / "results"
    / "feature_finding"
)

input_file = (
    results_dir
    / "direct_plane_model_results.csv"
)

summary_file = (
    results_dir
    / "direct_plane_model_summary_by_condition.csv"
)


EXPECTED_ROWS = 200
EXPECTED_CONDITIONS = 20
EXPECTED_REPEATS = 10


print(
    "=== Direct Plane Model Results Verification ==="
)


# ==================================================
# Read CSV
# ==================================================

if not input_file.exists():

    raise FileNotFoundError(
        f"Cannot find result file: {input_file}"
    )


with open(
    input_file,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(
        f
    )

    columns = reader.fieldnames

    rows = list(
        reader
    )


# ==================================================
# Test 1: Check required columns
# ==================================================

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

    "local_support_count",
    "initial_inlier_count",
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


# ==================================================
# Test 2: Check number of rows
# ==================================================

print(
    "\nTest 2: Checking number of results"
)


assert len(
    rows
) == EXPECTED_ROWS, (
    f"Expected {EXPECTED_ROWS} rows, "
    f"but found {len(rows)}"
)


print(
    "Number of result rows:",
    len(rows)
)


print(
    "Test 2 PASSED"
)


# ==================================================
# Test 3: Check sample IDs
# ==================================================

print(
    "\nTest 3: Checking sample IDs"
)


sample_ids = []


for row in rows:

    sample_ids.append(
        row[
            "sample_id"
        ]
    )


assert len(
    set(
        sample_ids
    )
) == EXPECTED_ROWS, (
    "Some sample IDs are repeated"
)


print(
    "Test 3 PASSED"
)


# ==================================================
# Test 4: Count success and failure
# ==================================================

print(
    "\nTest 4: Checking detection success"
)


success_count = 0
failure_count = 0


for row in rows:

    success = int(
        row[
            "success"
        ]
    )


    if success == 1:

        success_count += 1

    else:

        failure_count += 1


print(
    "Successful runs:",
    success_count
)


print(
    "Failed runs:",
    failure_count
)


print(
    "Success rate:",
    f"{success_count / len(rows):.4f}"
)


print(
    "Test 4 PASSED"
)


# ==================================================
# Test 5: Check numerical values
# ==================================================

print(
    "\nTest 5: Checking numerical values"
)


for row in rows:

    runtime = float(
        row[
            "runtime_ms"
        ]
    )


    assert np.isfinite(
        runtime
    )


    assert runtime >= 0


    success = int(
        row[
            "success"
        ]
    )


    if success == 1:

        numerical_columns = [
            "normal_error_deg",
            "offset_error_m",
            "precision",
            "recall",
            "f1"
        ]


        for column in numerical_columns:

            value = float(
                row[
                    column
                ]
            )


            assert np.isfinite(
                value
            ), (
                f"{row['sample_id']} has "
                f"invalid value in {column}"
            )


print(
    "Test 5 PASSED"
)


# ==================================================
# Test 6: Check value ranges
# ==================================================

print(
    "\nTest 6: Checking result ranges"
)


for row in rows:

    success = int(
        row[
            "success"
        ]
    )


    if success == 0:

        continue


    precision = float(
        row[
            "precision"
        ]
    )


    recall = float(
        row[
            "recall"
        ]
    )


    f1 = float(
        row[
            "f1"
        ]
    )


    normal_error = float(
        row[
            "normal_error_deg"
        ]
    )


    offset_error = float(
        row[
            "offset_error_m"
        ]
    )


    assert 0 <= precision <= 1

    assert 0 <= recall <= 1

    assert 0 <= f1 <= 1

    assert normal_error >= 0

    assert offset_error >= 0


print(
    "Test 6 PASSED"
)


# ==================================================
# Test 7: Check classification counts
# ==================================================

print(
    "\nTest 7: Checking classification counts"
)


for row in rows:

    success = int(
        row[
            "success"
        ]
    )


    if success == 0:

        continue


    tp = int(
        row[
            "tp"
        ]
    )

    fp = int(
        row[
            "fp"
        ]
    )

    fn = int(
        row[
            "fn"
        ]
    )

    tn = int(
        row[
            "tn"
        ]
    )


    total_points = int(
        row[
            "num_points"
        ]
    )


    true_inliers = int(
        row[
            "num_true_inliers"
        ]
    )


    true_outliers = int(
        row[
            "num_true_outliers"
        ]
    )


    detected_inliers = int(
        row[
            "detected_inlier_count"
        ]
    )


    assert (
        tp + fp + fn + tn
    ) == total_points


    assert (
        tp + fn
    ) == true_inliers


    assert (
        fp + tn
    ) == true_outliers


    assert (
        tp + fp
    ) == detected_inliers


print(
    "Test 7 PASSED"
)


# ==================================================
# Test 8: Group results by condition
# ==================================================

print(
    "\nTest 8: Checking experimental conditions"
)


condition_groups = defaultdict(
    list
)


for row in rows:

    noise = float(
        row[
            "noise_sigma_m"
        ]
    )


    outlier_ratio = float(
        row[
            "outlier_ratio"
        ]
    )


    condition = (
        noise,
        outlier_ratio
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


for condition in condition_groups:

    assert len(
        condition_groups[
            condition
        ]
    ) == EXPECTED_REPEATS, (
        f"Condition {condition} does not "
        f"have {EXPECTED_REPEATS} repeats"
    )


print(
    "Number of conditions:",
    len(
        condition_groups
    )
)


print(
    "Repeats per condition:",
    EXPECTED_REPEATS
)


print(
    "Test 8 PASSED"
)


print(
    "\n=== Direct Plane Model Results Verification PASSED ==="
)


# ==================================================
# Analysis
# ==================================================

print(
    "\n=== Direct Plane Model Results Analysis ==="
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


    successful_rows = []
    failed_rows = []


    for row in condition_rows:

        if int(
            row[
                "success"
            ]
        ) == 1:

            successful_rows.append(
                row
            )

        else:

            failed_rows.append(
                row
            )


    num_success = len(
        successful_rows
    )

    num_fail = len(
        failed_rows
    )


    success_rate = (
        num_success
        / len(
            condition_rows
        )
    )


    failure_rate = (
        num_fail
        / len(
            condition_rows
        )
    )


    normal_errors = []
    offset_errors = []
    precisions = []
    recalls = []
    f1_values = []

    runtimes = []

    local_support_counts = []
    initial_inlier_counts = []


    # --------------------------------------------------
    # Accuracy values from successful runs
    # --------------------------------------------------

    for row in successful_rows:

        normal_errors.append(
            float(
                row[
                    "normal_error_deg"
                ]
            )
        )


        offset_errors.append(
            float(
                row[
                    "offset_error_m"
                ]
            )
        )


        precisions.append(
            float(
                row[
                    "precision"
                ]
            )
        )


        recalls.append(
            float(
                row[
                    "recall"
                ]
            )
        )


        f1_values.append(
            float(
                row[
                    "f1"
                ]
            )
        )


        local_support_counts.append(
            int(
                row[
                    "local_support_count"
                ]
            )
        )


        initial_inlier_counts.append(
            int(
                row[
                    "initial_inlier_count"
                ]
            )
        )


    # Runtime includes both successful and failed runs
    for row in condition_rows:

        runtimes.append(
            float(
                row[
                    "runtime_ms"
                ]
            )
        )


    # --------------------------------------------------
    # Mean and standard deviation
    # --------------------------------------------------

    if num_success > 0:

        normal_mean = np.mean(
            normal_errors
        )

        offset_mean = np.mean(
            offset_errors
        )

        precision_mean = np.mean(
            precisions
        )

        recall_mean = np.mean(
            recalls
        )

        f1_mean = np.mean(
            f1_values
        )

        local_support_mean = np.mean(
            local_support_counts
        )

        initial_inlier_mean = np.mean(
            initial_inlier_counts
        )


        if num_success > 1:

            normal_std = np.std(
                normal_errors,
                ddof=1
            )

            offset_std = np.std(
                offset_errors,
                ddof=1
            )

            precision_std = np.std(
                precisions,
                ddof=1
            )

            recall_std = np.std(
                recalls,
                ddof=1
            )

            f1_std = np.std(
                f1_values,
                ddof=1
            )

            local_support_std = np.std(
                local_support_counts,
                ddof=1
            )

            initial_inlier_std = np.std(
                initial_inlier_counts,
                ddof=1
            )

        else:

            normal_std = 0.0
            offset_std = 0.0

            precision_std = 0.0
            recall_std = 0.0
            f1_std = 0.0

            local_support_std = 0.0
            initial_inlier_std = 0.0


    else:

        normal_mean = np.nan
        normal_std = np.nan

        offset_mean = np.nan
        offset_std = np.nan

        precision_mean = np.nan
        precision_std = np.nan

        recall_mean = np.nan
        recall_std = np.nan

        f1_mean = np.nan
        f1_std = np.nan

        local_support_mean = np.nan
        local_support_std = np.nan

        initial_inlier_mean = np.nan
        initial_inlier_std = np.nan


    runtime_mean = np.mean(
        runtimes
    )


    runtime_std = np.std(
        runtimes,
        ddof=1
    )


    # --------------------------------------------------
    # Print condition summary
    # --------------------------------------------------

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
        "Successful runs:",
        num_success,
        "/",
        EXPECTED_REPEATS
    )


    print(
        "Success rate:",
        f"{success_rate:.3f}"
    )


    if num_success > 0:

        print(
            "Normal error:",
            f"{normal_mean:.6f}",
            "±",
            f"{normal_std:.6f}",
            "degrees"
        )


        print(
            "Offset error:",
            f"{offset_mean:.6f}",
            "±",
            f"{offset_std:.6f}",
            "m"
        )


        print(
            "Precision:",
            f"{precision_mean:.6f}",
            "±",
            f"{precision_std:.6f}"
        )


        print(
            "Recall:",
            f"{recall_mean:.6f}",
            "±",
            f"{recall_std:.6f}"
        )


        print(
            "F1:",
            f"{f1_mean:.6f}",
            "±",
            f"{f1_std:.6f}"
        )


        print(
            "Local support:",
            f"{local_support_mean:.2f}",
            "±",
            f"{local_support_std:.2f}"
        )


        print(
            "Initial inliers:",
            f"{initial_inlier_mean:.2f}",
            "±",
            f"{initial_inlier_std:.2f}"
        )

    else:

        print(
            "No successful detections"
        )


    print(
        "Runtime:",
        f"{runtime_mean:.3f}",
        "±",
        f"{runtime_std:.3f}",
        "ms"
    )


    # --------------------------------------------------
    # Store summary
    # --------------------------------------------------

    summary_rows.append(
        {
            "method": (
                "CPA-style Direct Plane Model"
            ),

            "noise_sigma_m": noise,

            "outlier_ratio": (
                outlier_ratio
            ),

            "num_repeats": (
                len(
                    condition_rows
                )
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
                normal_mean
            ),

            "normal_error_std_deg": (
                normal_std
            ),

            "offset_error_mean_m": (
                offset_mean
            ),

            "offset_error_std_m": (
                offset_std
            ),

            "precision_mean": (
                precision_mean
            ),

            "precision_std": (
                precision_std
            ),

            "recall_mean": (
                recall_mean
            ),

            "recall_std": (
                recall_std
            ),

            "f1_mean": (
                f1_mean
            ),

            "f1_std": (
                f1_std
            ),

            "local_support_mean": (
                local_support_mean
            ),

            "local_support_std": (
                local_support_std
            ),

            "initial_inlier_mean": (
                initial_inlier_mean
            ),

            "initial_inlier_std": (
                initial_inlier_std
            ),

            "runtime_mean_ms": (
                runtime_mean
            ),

            "runtime_std_ms": (
                runtime_std
            )
        }
    )


# ==================================================
# Save condition summary
# ==================================================

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

    "local_support_mean",
    "local_support_std",

    "initial_inlier_mean",
    "initial_inlier_std",

    "runtime_mean_ms",
    "runtime_std_ms"
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


# ==================================================
# Overall results
# ==================================================

print(
    "\n=== Overall Direct Plane Model Results ==="
)


all_successful_rows = []

all_runtimes = []


for row in rows:

    all_runtimes.append(
        float(
            row[
                "runtime_ms"
            ]
        )
    )


    if int(
        row[
            "success"
        ]
    ) == 1:

        all_successful_rows.append(
            row
        )


overall_success_count = len(
    all_successful_rows
)


overall_failure_count = (
    len(rows)
    - overall_success_count
)


overall_success_rate = (
    overall_success_count
    / len(rows)
)


print(
    "Successful runs:",
    overall_success_count,
    "/",
    len(rows)
)


print(
    "Failed runs:",
    overall_failure_count
)


print(
    "Success rate:",
    f"{overall_success_rate:.4f}"
)


if overall_success_count > 0:

    all_normal_errors = []
    all_offset_errors = []

    all_precisions = []
    all_recalls = []
    all_f1_values = []


    for row in all_successful_rows:

        all_normal_errors.append(
            float(
                row[
                    "normal_error_deg"
                ]
            )
        )


        all_offset_errors.append(
            float(
                row[
                    "offset_error_m"
                ]
            )
        )


        all_precisions.append(
            float(
                row[
                    "precision"
                ]
            )
        )


        all_recalls.append(
            float(
                row[
                    "recall"
                ]
            )
        )


        all_f1_values.append(
            float(
                row[
                    "f1"
                ]
            )
        )


    print(
        "Normal error:",
        f"{np.mean(all_normal_errors):.6f}",
        "±",
        f"{np.std(all_normal_errors, ddof=1):.6f}",
        "degrees"
    )


    print(
        "Offset error:",
        f"{np.mean(all_offset_errors):.6f}",
        "±",
        f"{np.std(all_offset_errors, ddof=1):.6f}",
        "m"
    )


    print(
        "Precision:",
        f"{np.mean(all_precisions):.6f}",
        "±",
        f"{np.std(all_precisions, ddof=1):.6f}"
    )


    print(
        "Recall:",
        f"{np.mean(all_recalls):.6f}",
        "±",
        f"{np.std(all_recalls, ddof=1):.6f}"
    )


    print(
        "F1:",
        f"{np.mean(all_f1_values):.6f}",
        "±",
        f"{np.std(all_f1_values, ddof=1):.6f}"
    )


print(
    "Runtime:",
    f"{np.mean(all_runtimes):.3f}",
    "±",
    f"{np.std(all_runtimes, ddof=1):.3f}",
    "ms"
)


# ==================================================
# Failure reasons
# ==================================================

failure_reasons = defaultdict(
    int
)


for row in rows:

    if int(
        row[
            "success"
        ]
    ) == 0:

        reason = row[
            "failure_reason"
        ]


        failure_reasons[
            reason
        ] += 1


if overall_failure_count > 0:

    print(
        "\nFailure reasons:"
    )


    for reason in failure_reasons:

        print(
            reason,
            ":",
            failure_reasons[
                reason
            ]
        )


print(
    "\nSummary saved to:"
)


print(
    summary_file
)
