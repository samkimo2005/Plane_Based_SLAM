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
    / "multi_object_ransac_results_v2.csv"
)


summary_file = (
    results_dir
    / "multi_object_ransac_summary_by_condition_v2.csv"
)


EXPECTED_ROWS = 200
EXPECTED_CONDITIONS = 20
EXPECTED_REPEATS = 10


print(
    "=== Multi-Object RANSAC V2 Results Verification ==="
)


# ==========================================================
# Helper functions
# ==========================================================

def mean_value(
    values
):
    if len(
        values
    ) == 0:

        return np.nan

    return float(
        np.mean(
            values
        )
    )


def std_value(
    values
):
    if len(
        values
    ) == 0:

        return np.nan

    if len(
        values
    ) == 1:

        return 0.0

    return float(
        np.std(
            values,
            ddof=1
        )
    )


def median_value(
    values
):
    if len(
        values
    ) == 0:

        return np.nan

    return float(
        np.median(
            values
        )
    )


def percentile_95(
    values
):
    if len(
        values
    ) == 0:

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

    reader = csv.DictReader(
        f
    )

    columns = reader.fieldnames

    rows = list(
        reader
    )


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

    "num_detected_models",
    "winning_cluster_hypotheses",

    "num_valid_hypotheses",
    "num_raw_clusters",
    "num_accepted_clusters",

    "hypothesis_runtime_ms",
    "clustering_runtime_ms",
    "refinement_runtime_ms",

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


assert len(
    rows
) == EXPECTED_ROWS, (
    f"Expected {EXPECTED_ROWS} rows, "
    f"but found {len(rows)}"
)


print(
    "Rows:",
    len(
        rows
    )
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
    row[
        "sample_id"
    ]
    for row in rows
]


assert len(
    set(
        sample_ids
    )
) == EXPECTED_ROWS, (
    "Duplicate sample IDs found"
)


print(
    "Test 3 PASSED"
)


# ==========================================================
# Test 4: success and failure count
# ==========================================================

print(
    "\nTest 4: Checking success / failure status"
)


successful_rows = []
failed_rows = []


for row in rows:

    success = int(
        row[
            "success"
        ]
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


print(
    "Successful runs:",
    len(
        successful_rows
    )
)


print(
    "Failed runs:",
    len(
        failed_rows
    )
)


print(
    "Success rate:",
    f"{len(successful_rows) / len(rows):.4f}"
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

    # Runtime metrics exist for both success and failure.

    runtime_columns = [
        "runtime_ms",
        "hypothesis_runtime_ms",
        "clustering_runtime_ms",
        "refinement_runtime_ms"
    ]


    for column in runtime_columns:

        value = float(
            row[
                column
            ]
        )


        assert np.isfinite(
            value
        ), (
            f"{row['sample_id']} has invalid "
            f"value in {column}"
        )


        assert value >= 0


    diagnostic_columns = [
        "num_valid_hypotheses",
        "num_raw_clusters",
        "num_accepted_clusters",
        "num_detected_models",
        "winning_cluster_hypotheses"
    ]


    for column in diagnostic_columns:

        value = int(
            row[
                column
            ]
        )


        assert value >= 0


    success = int(
        row[
            "success"
        ]
    )


    if success == 1:

        common_metrics = [
            "normal_error_deg",
            "offset_error_m",
            "precision",
            "recall",
            "f1"
        ]


        for column in common_metrics:

            value = float(
                row[
                    column
                ]
            )


            assert np.isfinite(
                value
            ), (
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
        tp
        + fp
        + fn
        + tn
    ) == total_points


    assert (
        tp
        + fn
    ) == true_inliers


    assert (
        fp
        + tn
    ) == true_outliers


    assert (
        tp
        + fp
    ) == detected_inliers


print(
    "Test 7 PASSED"
)


# ==========================================================
# Test 8: Multi-Object diagnostic consistency
# ==========================================================

print(
    "\nTest 8: Checking Multi-Object diagnostics"
)


for row in rows:

    valid_hypotheses = int(
        row[
            "num_valid_hypotheses"
        ]
    )


    raw_clusters = int(
        row[
            "num_raw_clusters"
        ]
    )


    accepted_clusters = int(
        row[
            "num_accepted_clusters"
        ]
    )


    detected_models = int(
        row[
            "num_detected_models"
        ]
    )


    winning_votes = int(
        row[
            "winning_cluster_hypotheses"
        ]
    )


    # Cannot have more clusters than retained hypotheses.

    assert (
        raw_clusters
        <= valid_hypotheses
    )


    # Cannot accept more clusters than exist.

    assert (
        accepted_clusters
        <= raw_clusters
    )


    # Some accepted clusters may still fail refinement,
    # therefore detected models can be <= accepted clusters.

    assert (
        detected_models
        <= accepted_clusters
    )


    if int(
        row[
            "success"
        ]
    ) == 1:

        assert detected_models >= 1

        assert winning_votes >= 1


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
            row[
                "noise_sigma_m"
            ]
        ),

        float(
            row[
                "outlier_ratio"
            ]
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
    len(
        condition_groups
    )
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
    "\n=== Multi-Object RANSAC V2 Analysis ==="
)


summary_rows = []


for condition in sorted(
    condition_groups
):

    noise = condition[
        0
    ]

    outlier_ratio = condition[
        1
    ]


    condition_rows = (
        condition_groups[
            condition
        ]
    )


    condition_success = [
        row
        for row in condition_rows
        if int(
            row[
                "success"
            ]
        ) == 1
    ]


    condition_fail = [
        row
        for row in condition_rows
        if int(
            row[
                "success"
            ]
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
    # Successful-run accuracy metrics
    # ------------------------------------------------------

    normal_errors = [
        float(
            row[
                "normal_error_deg"
            ]
        )
        for row in condition_success
    ]


    offset_errors = [
        float(
            row[
                "offset_error_m"
            ]
        )
        for row in condition_success
    ]


    precisions = [
        float(
            row[
                "precision"
            ]
        )
        for row in condition_success
    ]


    recalls = [
        float(
            row[
                "recall"
            ]
        )
        for row in condition_success
    ]


    f1_values = [
        float(
            row[
                "f1"
            ]
        )
        for row in condition_success
    ]


    # ------------------------------------------------------
    # Runtime includes successes and failures
    # ------------------------------------------------------

    runtimes = [
        float(
            row[
                "runtime_ms"
            ]
        )
        for row in condition_rows
    ]


    hypothesis_runtimes = [
        float(
            row[
                "hypothesis_runtime_ms"
            ]
        )
        for row in condition_rows
    ]


    clustering_runtimes = [
        float(
            row[
                "clustering_runtime_ms"
            ]
        )
        for row in condition_rows
    ]


    refinement_runtimes = [
        float(
            row[
                "refinement_runtime_ms"
            ]
        )
        for row in condition_rows
    ]


    # ------------------------------------------------------
    # Algorithm diagnostics
    # ------------------------------------------------------

    valid_hypotheses = [
        int(
            row[
                "num_valid_hypotheses"
            ]
        )
        for row in condition_rows
    ]


    raw_clusters = [
        int(
            row[
                "num_raw_clusters"
            ]
        )
        for row in condition_rows
    ]


    accepted_clusters = [
        int(
            row[
                "num_accepted_clusters"
            ]
        )
        for row in condition_rows
    ]


    detected_models = [
        int(
            row[
                "num_detected_models"
            ]
        )
        for row in condition_rows
    ]


    winning_votes = [
        int(
            row[
                "winning_cluster_hypotheses"
            ]
        )
        for row in condition_success
    ]


    # ------------------------------------------------------
    # Effective F1
    #
    # If a condition has failures, this prevents
    # successful-run F1 from hiding those failures.
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


    print(
        "Valid hypotheses:",
        f"{mean_value(valid_hypotheses):.1f}"
    )


    print(
        "Raw clusters:",
        f"{mean_value(raw_clusters):.1f}"
    )


    print(
        "Accepted clusters:",
        f"{mean_value(accepted_clusters):.1f}"
    )


    print(
        "Detected models:",
        f"{mean_value(detected_models):.1f}"
    )


    # ------------------------------------------------------
    # Save summary row
    # ------------------------------------------------------

    summary_rows.append(
        {
            "method": (
                "Multi-Object RANSAC V2"
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

            "hypothesis_runtime_mean_ms": (
                mean_value(
                    hypothesis_runtimes
                )
            ),

            "clustering_runtime_mean_ms": (
                mean_value(
                    clustering_runtimes
                )
            ),

            "refinement_runtime_mean_ms": (
                mean_value(
                    refinement_runtimes
                )
            ),

            "valid_hypotheses_mean": (
                mean_value(
                    valid_hypotheses
                )
            ),

            "raw_clusters_mean": (
                mean_value(
                    raw_clusters
                )
            ),

            "accepted_clusters_mean": (
                mean_value(
                    accepted_clusters
                )
            ),

            "detected_models_mean": (
                mean_value(
                    detected_models
                )
            ),

            "winning_cluster_hypotheses_mean": (
                mean_value(
                    winning_votes
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

    "hypothesis_runtime_mean_ms",
    "clustering_runtime_mean_ms",
    "refinement_runtime_mean_ms",

    "valid_hypotheses_mean",
    "raw_clusters_mean",
    "accepted_clusters_mean",
    "detected_models_mean",

    "winning_cluster_hypotheses_mean"
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
    "\n=== Overall Multi-Object RANSAC V2 Results ==="
)


overall_success_rate = (
    len(
        successful_rows
    )
    / len(
        rows
    )
)


print(
    "Successful:",
    len(
        successful_rows
    ),
    "/",
    len(
        rows
    )
)


print(
    "Failed:",
    len(
        failed_rows
    )
)


print(
    "Success rate:",
    f"{overall_success_rate:.4f}"
)


# ----------------------------------------------------------
# Overall successful-run accuracy
# ----------------------------------------------------------

if len(
    successful_rows
) > 0:

    all_normal_errors = [
        float(
            row[
                "normal_error_deg"
            ]
        )
        for row in successful_rows
    ]


    all_offset_errors = [
        float(
            row[
                "offset_error_m"
            ]
        )
        for row in successful_rows
    ]


    all_precisions = [
        float(
            row[
                "precision"
            ]
        )
        for row in successful_rows
    ]


    all_recalls = [
        float(
            row[
                "recall"
            ]
        )
        for row in successful_rows
    ]


    all_f1 = [
        float(
            row[
                "f1"
            ]
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
# Overall runtime distribution
# ----------------------------------------------------------

all_runtimes = [
    float(
        row[
            "runtime_ms"
        ]
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
# Runtime stage breakdown
# ----------------------------------------------------------

all_hypothesis_runtime = [
    float(
        row[
            "hypothesis_runtime_ms"
        ]
    )
    for row in rows
]


all_clustering_runtime = [
    float(
        row[
            "clustering_runtime_ms"
        ]
    )
    for row in rows
]


all_refinement_runtime = [
    float(
        row[
            "refinement_runtime_ms"
        ]
    )
    for row in rows
]


print(
    "\nMean runtime breakdown:"
)


print(
    "Hypothesis generation:",
    f"{mean_value(all_hypothesis_runtime):.3f}",
    "ms"
)


print(
    "Clustering:",
    f"{mean_value(all_clustering_runtime):.3f}",
    "ms"
)


print(
    "Refinement:",
    f"{mean_value(all_refinement_runtime):.3f}",
    "ms"
)


# ----------------------------------------------------------
# Overall Multi-Object diagnostics
# ----------------------------------------------------------

all_valid_hypotheses = [
    int(
        row[
            "num_valid_hypotheses"
        ]
    )
    for row in rows
]


all_raw_clusters = [
    int(
        row[
            "num_raw_clusters"
        ]
    )
    for row in rows
]


all_accepted_clusters = [
    int(
        row[
            "num_accepted_clusters"
        ]
    )
    for row in rows
]


all_detected_models = [
    int(
        row[
            "num_detected_models"
        ]
    )
    for row in rows
]


print(
    "\nMulti-Object diagnostics:"
)


print(
    "Mean valid hypotheses:",
    f"{mean_value(all_valid_hypotheses):.2f}"
)


print(
    "Mean raw clusters:",
    f"{mean_value(all_raw_clusters):.2f}"
)


print(
    "Mean accepted clusters:",
    f"{mean_value(all_accepted_clusters):.2f}"
)


print(
    "Mean detected models:",
    f"{mean_value(all_detected_models):.2f}"
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
            row[
                "failure_reason"
            ]
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
# Runtime correlation diagnostics
# ==========================================================

runtime_array = np.asarray(
    all_runtimes,
    dtype=np.float64
)


raw_cluster_array = np.asarray(
    all_raw_clusters,
    dtype=np.float64
)


valid_hypothesis_array = np.asarray(
    all_valid_hypotheses,
    dtype=np.float64
)


accepted_cluster_array = np.asarray(
    all_accepted_clusters,
    dtype=np.float64
)


if np.std(
    raw_cluster_array
) > 0:

    correlation = np.corrcoef(
        runtime_array,
        raw_cluster_array
    )[0, 1]


    print(
        "\nCorrelation between runtime "
        "and raw cluster count:",
        f"{correlation:.4f}"
    )


if np.std(
    valid_hypothesis_array
) > 0:

    correlation = np.corrcoef(
        runtime_array,
        valid_hypothesis_array
    )[0, 1]


    print(
        "Correlation between runtime "
        "and valid hypotheses:",
        f"{correlation:.4f}"
    )


if np.std(
    accepted_cluster_array
) > 0:

    correlation = np.corrcoef(
        runtime_array,
        accepted_cluster_array
    )[0, 1]


    print(
        "Correlation between runtime "
        "and accepted clusters:",
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