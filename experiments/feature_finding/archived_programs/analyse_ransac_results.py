import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


# Find project directory
project_root = Path(__file__).resolve().parents[2]

results_dir = (
    project_root
    / "results"
    / "feature_finding"
)

input_file = (
    results_dir
    / "ransac_results.csv"
)

summary_file = (
    results_dir
    / "ransac_summary_by_condition.csv"
)


EXPECTED_ROWS = 200
EXPECTED_CONDITIONS = 20
EXPECTED_REPEATS = 10


print("=== RANSAC Results Verification ===")


# --------------------------------------------------
# Read CSV file
# --------------------------------------------------

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


# --------------------------------------------------
# Test 1: Check required columns
# --------------------------------------------------

print("\nTest 1: Checking CSV columns")


required_columns = [
    "method",
    "sample_id",
    "noise_sigma_m",
    "outlier_ratio",
    "repeat_id",
    "num_points",
    "num_true_inliers",
    "num_true_outliers",
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
    "success"
]


for column in required_columns:

    assert column in columns, (
        f"Missing column: {column}"
    )


print("Test 1 PASSED")


# --------------------------------------------------
# Test 2: Check number of rows
# --------------------------------------------------

print("\nTest 2: Checking number of results")


assert len(rows) == EXPECTED_ROWS, (
    f"Expected {EXPECTED_ROWS} rows, "
    f"but found {len(rows)}"
)


print(
    "Number of result rows:",
    len(rows)
)

print("Test 2 PASSED")


# --------------------------------------------------
# Test 3: Check sample IDs
# --------------------------------------------------

print("\nTest 3: Checking sample IDs")


sample_ids = []

for row in rows:

    sample_ids.append(
        row["sample_id"]
    )


assert len(
    set(sample_ids)
) == EXPECTED_ROWS, (
    "Some sample IDs are repeated"
)


print("Test 3 PASSED")


# --------------------------------------------------
# Test 4: Check all runs succeeded
# --------------------------------------------------

print("\nTest 4: Checking RANSAC success")


success_count = 0


for row in rows:

    success = int(
        row["success"]
    )

    if success == 1:

        success_count += 1


failure_count = (
    len(rows)
    - success_count
)


print(
    "Successful runs:",
    success_count
)

print(
    "Failed runs:",
    failure_count
)


assert failure_count == 0, (
    "Some RANSAC runs failed"
)


print("Test 4 PASSED")


# --------------------------------------------------
# Test 5: Check numerical values
# --------------------------------------------------

print("\nTest 5: Checking numerical values")


numerical_columns = [
    "normal_error_deg",
    "offset_error_m",
    "precision",
    "recall",
    "f1",
    "runtime_ms"
]


for row in rows:

    for column in numerical_columns:

        value = float(
            row[column]
        )

        assert np.isfinite(
            value
        ), (
            f"{row['sample_id']} has "
            f"an invalid value in {column}"
        )


print("Test 5 PASSED")


# --------------------------------------------------
# Test 6: Check value ranges
# --------------------------------------------------

print("\nTest 6: Checking result ranges")


for row in rows:

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

    runtime = float(
        row["runtime_ms"]
    )


    assert 0 <= precision <= 1

    assert 0 <= recall <= 1

    assert 0 <= f1 <= 1

    assert normal_error >= 0

    assert offset_error >= 0

    assert runtime >= 0


print("Test 6 PASSED")


# --------------------------------------------------
# Test 7: Check TP, FP, FN and TN
# --------------------------------------------------

print(
    "\nTest 7: Checking classification counts"
)


for row in rows:

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

    total_points = int(
        row["num_points"]
    )

    true_inliers = int(
        row["num_true_inliers"]
    )

    true_outliers = int(
        row["num_true_outliers"]
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


print("Test 7 PASSED")


# --------------------------------------------------
# Test 8: Group results by condition
# --------------------------------------------------

print(
    "\nTest 8: Checking experimental conditions"
)


condition_groups = defaultdict(
    list
)


for row in rows:

    noise = float(
        row["noise_sigma_m"]
    )

    outlier_ratio = float(
        row["outlier_ratio"]
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
    len(condition_groups)
)

print(
    "Repeats per condition:",
    EXPECTED_REPEATS
)

print("Test 8 PASSED")


print(
    "\n=== RANSAC Results Verification PASSED ==="
)


# ==================================================
# Analysis
# ==================================================

print(
    "\n=== RANSAC Results Analysis ==="
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


    normal_errors = []
    offset_errors = []
    precisions = []
    recalls = []
    f1_values = []
    runtimes = []


    for row in condition_rows:

        normal_errors.append(
            float(
                row["normal_error_deg"]
            )
        )

        offset_errors.append(
            float(
                row["offset_error_m"]
            )
        )

        precisions.append(
            float(
                row["precision"]
            )
        )

        recalls.append(
            float(
                row["recall"]
            )
        )

        f1_values.append(
            float(
                row["f1"]
            )
        )

        runtimes.append(
            float(
                row["runtime_ms"]
            )
        )


    normal_mean = np.mean(
        normal_errors
    )

    normal_std = np.std(
        normal_errors,
        ddof=1
    )


    offset_mean = np.mean(
        offset_errors
    )

    offset_std = np.std(
        offset_errors,
        ddof=1
    )


    precision_mean = np.mean(
        precisions
    )

    precision_std = np.std(
        precisions,
        ddof=1
    )


    recall_mean = np.mean(
        recalls
    )

    recall_std = np.std(
        recalls,
        ddof=1
    )


    f1_mean = np.mean(
        f1_values
    )

    f1_std = np.std(
        f1_values,
        ddof=1
    )


    runtime_mean = np.mean(
        runtimes
    )

    runtime_std = np.std(
        runtimes,
        ddof=1
    )


    # Print result for each condition

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
        "Runtime:",
        f"{runtime_mean:.3f}",
        "±",
        f"{runtime_std:.3f}",
        "ms"
    )


    summary_rows.append(
        {
            "method": "RANSAC",

            "noise_sigma_m": noise,

            "outlier_ratio": (
                outlier_ratio
            ),

            "num_repeats": (
                len(
                    condition_rows
                )
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

            "runtime_mean_ms": (
                runtime_mean
            ),

            "runtime_std_ms": (
                runtime_std
            )
        }
    )


# --------------------------------------------------
# Save condition summary
# --------------------------------------------------

summary_columns = [
    "method",
    "noise_sigma_m",
    "outlier_ratio",
    "num_repeats",

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


# --------------------------------------------------
# Overall results
# --------------------------------------------------

all_normal_errors = []
all_offset_errors = []
all_precisions = []
all_recalls = []
all_f1_values = []
all_runtimes = []


for row in rows:

    all_normal_errors.append(
        float(
            row["normal_error_deg"]
        )
    )

    all_offset_errors.append(
        float(
            row["offset_error_m"]
        )
    )

    all_precisions.append(
        float(
            row["precision"]
        )
    )

    all_recalls.append(
        float(
            row["recall"]
        )
    )

    all_f1_values.append(
        float(
            row["f1"]
        )
    )

    all_runtimes.append(
        float(
            row["runtime_ms"]
        )
    )


normal_mean_all = np.mean(
    all_normal_errors
)

normal_std_all = np.std(
    all_normal_errors,
    ddof=1
)


offset_mean_all = np.mean(
    all_offset_errors
)

offset_std_all = np.std(
    all_offset_errors,
    ddof=1
)


precision_mean_all = np.mean(
    all_precisions
)

precision_std_all = np.std(
    all_precisions,
    ddof=1
)


recall_mean_all = np.mean(
    all_recalls
)

recall_std_all = np.std(
    all_recalls,
    ddof=1
)


f1_mean_all = np.mean(
    all_f1_values
)

f1_std_all = np.std(
    all_f1_values,
    ddof=1
)


runtime_mean_all = np.mean(
    all_runtimes
)

runtime_std_all = np.std(
    all_runtimes,
    ddof=1
)


print(
    "\n=== Overall RANSAC Results ==="
)


print(
    "Normal error:",
    f"{normal_mean_all:.6f}",
    "±",
    f"{normal_std_all:.6f}",
    "degrees"
)


print(
    "Offset error:",
    f"{offset_mean_all:.6f}",
    "±",
    f"{offset_std_all:.6f}",
    "m"
)


print(
    "Precision:",
    f"{precision_mean_all:.6f}",
    "±",
    f"{precision_std_all:.6f}"
)


print(
    "Recall:",
    f"{recall_mean_all:.6f}",
    "±",
    f"{recall_std_all:.6f}"
)


print(
    "F1:",
    f"{f1_mean_all:.6f}",
    "±",
    f"{f1_std_all:.6f}"
)


print(
    "Runtime:",
    f"{runtime_mean_all:.3f}",
    "±",
    f"{runtime_std_all:.3f}",
    "ms"
)


print(
    "\nSummary saved to:"
)

print(
    summary_file
)