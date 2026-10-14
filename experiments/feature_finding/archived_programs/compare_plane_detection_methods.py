import csv
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


output_condition_file = (
    results_dir
    / "plane_detection_comparison_by_condition.csv"
)


output_overall_file = (
    results_dir
    / "plane_detection_overall_comparison.csv"
)


# ==========================================================
# Formal plane-detection methods
# ==========================================================

METHODS = {
    "RANSAC": {
        "raw_file": (
            results_dir
            / "ransac_results.csv"
        ),

        "summary_file": (
            results_dir
            / "ransac_summary_by_condition.csv"
        )
    },


    "Multi-Object RANSAC V2": {
        "raw_file": (
            results_dir
            / "multi_object_ransac_results_v2.csv"
        ),

        "summary_file": (
            results_dir
            / "multi_object_ransac_summary_by_condition_v2.csv"
        )
    },


    "Surface-Normal / Surfel": {
        "raw_file": (
            results_dir
            / "surface_normal_surfel_results.csv"
        ),

        "summary_file": (
            results_dir
            / "surface_normal_surfel_summary_by_condition.csv"
        )
    },


    "Region Growing": {
        "raw_file": (
            results_dir
            / "region_growing_results.csv"
        ),

        "summary_file": (
            results_dir
            / "region_growing_summary_by_condition.csv"
        )
    },


    "3D Hough V2": {
        "raw_file": (
            results_dir
            / "hough_results_v2.csv"
        ),

        "summary_file": (
            results_dir
            / "hough_summary_by_condition_v2.csv"
        )
    }
}


EXPECTED_RAW_ROWS = 200
EXPECTED_CONDITIONS = 20


# ==========================================================
# Helper functions
# ==========================================================

def read_csv(
    file_path
):
    """
    Read a CSV file and return a list of dictionaries.
    """

    if not file_path.exists():

        raise FileNotFoundError(
            f"Cannot find file:\n{file_path}"
        )


    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(
            f
        )

        rows = list(
            reader
        )


    return rows


def mean_value(
    values
):
    """
    Arithmetic mean.
    """

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
    """
    Sample standard deviation.
    """

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


def safe_float(
    value
):
    """
    Convert CSV text to float.

    Blank values are returned as NaN.
    """

    if value is None:

        return np.nan


    value = str(
        value
    ).strip()


    if value == "":

        return np.nan


    return float(
        value
    )


def safe_int(
    value,
    default=0
):
    if value is None:

        return default


    value = str(
        value
    ).strip()


    if value == "":

        return default


    return int(
        float(
            value
        )
    )


def get_value(
    row,
    possible_names,
    default=np.nan
):
    """
    Different analysis scripts may use slightly different
    column names.

    This helper searches possible aliases.
    """

    for name in possible_names:

        if (
            name in row
            and str(
                row[
                    name
                ]
            ).strip()
            != ""
        ):

            return safe_float(
                row[
                    name
                ]
            )


    return default


# ==========================================================
# Verification
# ==========================================================

print(
    "=== Plane Detection Method Comparison ==="
)


print(
    "\nFormal methods:"
)


for method_name in METHODS:

    print(
        " -",
        method_name
    )


# ==========================================================
# Load all data
# ==========================================================

raw_data = {}

summary_data = {}


for method_name, configuration in METHODS.items():

    print(
        f"\nLoading {method_name}..."
    )


    raw_rows = read_csv(
        configuration[
            "raw_file"
        ]
    )


    summary_rows = read_csv(
        configuration[
            "summary_file"
        ]
    )


    # ------------------------------------------------------
    # Check raw benchmark size
    # ------------------------------------------------------

    if len(
        raw_rows
    ) != EXPECTED_RAW_ROWS:

        raise RuntimeError(
            f"{method_name}: expected "
            f"{EXPECTED_RAW_ROWS} raw results, "
            f"but found {len(raw_rows)}"
        )


    # ------------------------------------------------------
    # Check condition summary size
    # ------------------------------------------------------

    if len(
        summary_rows
    ) != EXPECTED_CONDITIONS:

        raise RuntimeError(
            f"{method_name}: expected "
            f"{EXPECTED_CONDITIONS} conditions, "
            f"but found {len(summary_rows)}"
        )


    raw_data[
        method_name
    ] = raw_rows


    summary_data[
        method_name
    ] = summary_rows


    print(
        "  raw results:",
        len(
            raw_rows
        )
    )


    print(
        "  conditions:",
        len(
            summary_rows
        )
    )


print(
    "\nAll result files loaded successfully."
)


# ==========================================================
# Verify that all methods use the same 20 conditions
# ==========================================================

print(
    "\n=== Checking common benchmark conditions ==="
)


reference_method = next(
    iter(
        METHODS
    )
)


reference_conditions = set()


for row in summary_data[
    reference_method
]:

    condition = (
        safe_float(
            row[
                "noise_sigma_m"
            ]
        ),

        safe_float(
            row[
                "outlier_ratio"
            ]
        )
    )


    reference_conditions.add(
        condition
    )


for method_name in METHODS:

    method_conditions = set()


    for row in summary_data[
        method_name
    ]:

        condition = (
            safe_float(
                row[
                    "noise_sigma_m"
                ]
            ),

            safe_float(
                row[
                    "outlier_ratio"
                ]
            )
        )


        method_conditions.add(
            condition
        )


    if (
        method_conditions
        != reference_conditions
    ):

        raise RuntimeError(
            f"{method_name} does not contain "
            "the same benchmark conditions."
        )


print(
    "All five methods use the same",
    len(
        reference_conditions
    ),
    "conditions."
)


# ==========================================================
# Build condition-level comparison table
# ==========================================================

print(
    "\n=== Building condition-level comparison ==="
)


condition_comparison_rows = []


for method_name in METHODS:

    for row in summary_data[
        method_name
    ]:

        noise = safe_float(
            row[
                "noise_sigma_m"
            ]
        )


        outlier_ratio = safe_float(
            row[
                "outlier_ratio"
            ]
        )


        success_rate = get_value(
            row,
            [
                "success_rate"
            ],
            default=1.0
        )


        failure_rate = get_value(
            row,
            [
                "failure_rate"
            ],
            default=(
                1.0
                - success_rate
            )
        )


        normal_error = get_value(
            row,
            [
                "normal_error_mean_deg",
                "normal_error_deg_mean",
                "mean_normal_error_deg"
            ]
        )


        normal_error_std = get_value(
            row,
            [
                "normal_error_std_deg",
                "normal_error_deg_std",
                "std_normal_error_deg"
            ]
        )


        offset_error = get_value(
            row,
            [
                "offset_error_mean_m",
                "offset_error_m_mean",
                "mean_offset_error_m"
            ]
        )


        offset_error_std = get_value(
            row,
            [
                "offset_error_std_m",
                "offset_error_m_std",
                "std_offset_error_m"
            ]
        )


        precision = get_value(
            row,
            [
                "precision_mean",
                "mean_precision"
            ]
        )


        recall = get_value(
            row,
            [
                "recall_mean",
                "mean_recall"
            ]
        )


        f1 = get_value(
            row,
            [
                "f1_mean",
                "mean_f1"
            ]
        )


        f1_std = get_value(
            row,
            [
                "f1_std",
                "std_f1"
            ]
        )


        effective_f1 = get_value(
            row,
            [
                "effective_f1"
            ]
        )


        # If an older summary file did not explicitly
        # calculate effective F1, calculate it here.

        if np.isnan(
            effective_f1
        ):

            if not np.isnan(
                f1
            ):

                effective_f1 = (
                    f1
                    * success_rate
                )

            else:

                effective_f1 = 0.0


        runtime_mean = get_value(
            row,
            [
                "runtime_mean_ms",
                "mean_runtime_ms"
            ]
        )


        runtime_std = get_value(
            row,
            [
                "runtime_std_ms",
                "std_runtime_ms"
            ]
        )


        runtime_median = get_value(
            row,
            [
                "runtime_median_ms",
                "median_runtime_ms"
            ]
        )


        runtime_p95 = get_value(
            row,
            [
                "runtime_p95_ms",
                "p95_runtime_ms"
            ]
        )


        condition_comparison_rows.append(
            {
                "method": (
                    method_name
                ),

                "noise_sigma_m": (
                    noise
                ),

                "outlier_ratio": (
                    outlier_ratio
                ),

                "success_rate": (
                    success_rate
                ),

                "failure_rate": (
                    failure_rate
                ),

                "normal_error_mean_deg": (
                    normal_error
                ),

                "normal_error_std_deg": (
                    normal_error_std
                ),

                "offset_error_mean_m": (
                    offset_error
                ),

                "offset_error_std_m": (
                    offset_error_std
                ),

                "precision_mean": (
                    precision
                ),

                "recall_mean": (
                    recall
                ),

                "f1_mean": (
                    f1
                ),

                "f1_std": (
                    f1_std
                ),

                "effective_f1": (
                    effective_f1
                ),

                "runtime_mean_ms": (
                    runtime_mean
                ),

                "runtime_std_ms": (
                    runtime_std
                ),

                "runtime_median_ms": (
                    runtime_median
                ),

                "runtime_p95_ms": (
                    runtime_p95
                )
            }
        )


# ==========================================================
# Save condition comparison
# ==========================================================

condition_columns = [
    "method",

    "noise_sigma_m",
    "outlier_ratio",

    "success_rate",
    "failure_rate",

    "normal_error_mean_deg",
    "normal_error_std_deg",

    "offset_error_mean_m",
    "offset_error_std_m",

    "precision_mean",
    "recall_mean",

    "f1_mean",
    "f1_std",

    "effective_f1",

    "runtime_mean_ms",
    "runtime_std_ms",
    "runtime_median_ms",
    "runtime_p95_ms"
]


with open(
    output_condition_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=condition_columns
    )


    writer.writeheader()


    writer.writerows(
        condition_comparison_rows
    )


print(
    "Condition comparison saved to:"
)


print(
    output_condition_file
)


# ==========================================================
# Overall comparison
# ==========================================================

print(
    "\n=== Calculating overall performance ==="
)


overall_rows = []


for method_name in METHODS:

    raw_rows = raw_data[
        method_name
    ]


    condition_rows = [
        row
        for row in condition_comparison_rows
        if row[
            "method"
        ] == method_name
    ]


    # ------------------------------------------------------
    # Success / failure
    # ------------------------------------------------------

    successful_rows = [
        row
        for row in raw_rows
        if safe_int(
            row.get(
                "success",
                1
            )
        ) == 1
    ]


    failed_rows = [
        row
        for row in raw_rows
        if safe_int(
            row.get(
                "success",
                1
            )
        ) == 0
    ]


    success_rate = (
        len(
            successful_rows
        )
        / len(
            raw_rows
        )
    )


    failure_rate = (
        len(
            failed_rows
        )
        / len(
            raw_rows
        )
    )


    # ------------------------------------------------------
    # Common accuracy metrics
    #
    # These are calculated directly from the 200 raw
    # benchmark results rather than averages of summaries.
    # ------------------------------------------------------

    normal_errors = [
        safe_float(
            row[
                "normal_error_deg"
            ]
        )
        for row in successful_rows
        if str(
            row[
                "normal_error_deg"
            ]
        ).strip()
        != ""
    ]


    offset_errors = [
        safe_float(
            row[
                "offset_error_m"
            ]
        )
        for row in successful_rows
        if str(
            row[
                "offset_error_m"
            ]
        ).strip()
        != ""
    ]


    precisions = [
        safe_float(
            row[
                "precision"
            ]
        )
        for row in successful_rows
        if str(
            row[
                "precision"
            ]
        ).strip()
        != ""
    ]


    recalls = [
        safe_float(
            row[
                "recall"
            ]
        )
        for row in successful_rows
        if str(
            row[
                "recall"
            ]
        ).strip()
        != ""
    ]


    f1_values = [
        safe_float(
            row[
                "f1"
            ]
        )
        for row in successful_rows
        if str(
            row[
                "f1"
            ]
        ).strip()
        != ""
    ]


    # ------------------------------------------------------
    # Runtime uses every run, including failures
    # ------------------------------------------------------

    runtimes = [
        safe_float(
            row[
                "runtime_ms"
            ]
        )
        for row in raw_rows
    ]


    # ------------------------------------------------------
    # Effective F1
    # ------------------------------------------------------

    mean_f1 = mean_value(
        f1_values
    )


    if np.isnan(
        mean_f1
    ):

        overall_effective_f1 = 0.0

    else:

        overall_effective_f1 = (
            mean_f1
            * success_rate
        )


    # ------------------------------------------------------
    # Condition-level robustness
    # ------------------------------------------------------

    condition_effective_f1 = np.asarray(
        [
            row[
                "effective_f1"
            ]
            for row in condition_rows
        ],
        dtype=np.float64
    )


    robustness_std = std_value(
        condition_effective_f1
    )


    worst_condition_f1 = float(
        np.min(
            condition_effective_f1
        )
    )


    best_condition_f1 = float(
        np.max(
            condition_effective_f1
        )
    )


    # ------------------------------------------------------
    # Noise robustness
    # ------------------------------------------------------

    noise_groups = {}


    for row in condition_rows:

        noise = float(
            row[
                "noise_sigma_m"
            ]
        )


        if noise not in noise_groups:

            noise_groups[
                noise
            ] = []


        noise_groups[
            noise
        ].append(
            float(
                row[
                    "effective_f1"
                ]
            )
        )


    noise_means = {}


    for noise, values in noise_groups.items():

        noise_means[
            noise
        ] = mean_value(
            values
        )


    f1_noise_0mm = noise_means.get(
        0.0,
        np.nan
    )


    f1_noise_5mm = noise_means.get(
        0.005,
        np.nan
    )


    f1_noise_10mm = noise_means.get(
        0.01,
        np.nan
    )


    f1_noise_20mm = noise_means.get(
        0.02,
        np.nan
    )


    if (
        not np.isnan(
            f1_noise_0mm
        )
        and not np.isnan(
            f1_noise_20mm
        )
    ):

        noise_degradation = (
            f1_noise_0mm
            - f1_noise_20mm
        )

    else:

        noise_degradation = np.nan


    # ------------------------------------------------------
    # Outlier robustness
    # ------------------------------------------------------

    outlier_groups = {}


    for row in condition_rows:

        outlier_ratio = float(
            row[
                "outlier_ratio"
            ]
        )


        if outlier_ratio not in outlier_groups:

            outlier_groups[
                outlier_ratio
            ] = []


        outlier_groups[
            outlier_ratio
        ].append(
            float(
                row[
                    "effective_f1"
                ]
            )
        )


    outlier_means = {}


    for ratio, values in outlier_groups.items():

        outlier_means[
            ratio
        ] = mean_value(
            values
        )


    f1_outlier_0 = outlier_means.get(
        0.0,
        np.nan
    )


    f1_outlier_10 = outlier_means.get(
        0.1,
        np.nan
    )


    f1_outlier_20 = outlier_means.get(
        0.2,
        np.nan
    )


    f1_outlier_30 = outlier_means.get(
        0.3,
        np.nan
    )


    f1_outlier_40 = outlier_means.get(
        0.4,
        np.nan
    )


    # ------------------------------------------------------
    # Assemble row
    # ------------------------------------------------------

    overall_rows.append(
        {
            "method": (
                method_name
            ),

            # ----------------------------------------------
            # Accuracy
            # ----------------------------------------------

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

            "recall_mean": (
                mean_value(
                    recalls
                )
            ),

            "f1_mean": (
                mean_f1
            ),

            "f1_std": (
                std_value(
                    f1_values
                )
            ),

            # ----------------------------------------------
            # Detection success
            # ----------------------------------------------

            "success_rate": (
                success_rate
            ),

            "failure_rate": (
                failure_rate
            ),

            "num_success": (
                len(
                    successful_rows
                )
            ),

            "num_fail": (
                len(
                    failed_rows
                )
            ),

            "effective_f1": (
                overall_effective_f1
            ),

            # ----------------------------------------------
            # Runtime
            # ----------------------------------------------

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

            "runtime_min_ms": (
                float(
                    np.min(
                        runtimes
                    )
                )
            ),

            "runtime_max_ms": (
                float(
                    np.max(
                        runtimes
                    )
                )
            ),

            "runtime_p95_ms": (
                percentile_95(
                    runtimes
                )
            ),

            # ----------------------------------------------
            # Robustness
            # ----------------------------------------------

            "condition_effective_f1_std": (
                robustness_std
            ),

            "worst_condition_effective_f1": (
                worst_condition_f1
            ),

            "best_condition_effective_f1": (
                best_condition_f1
            ),

            # ----------------------------------------------
            # Noise robustness
            # ----------------------------------------------

            "effective_f1_noise_0mm": (
                f1_noise_0mm
            ),

            "effective_f1_noise_5mm": (
                f1_noise_5mm
            ),

            "effective_f1_noise_10mm": (
                f1_noise_10mm
            ),

            "effective_f1_noise_20mm": (
                f1_noise_20mm
            ),

            "noise_degradation_0_to_20mm": (
                noise_degradation
            ),

            # ----------------------------------------------
            # Outlier robustness
            # ----------------------------------------------

            "effective_f1_outlier_0": (
                f1_outlier_0
            ),

            "effective_f1_outlier_10": (
                f1_outlier_10
            ),

            "effective_f1_outlier_20": (
                f1_outlier_20
            ),

            "effective_f1_outlier_30": (
                f1_outlier_30
            ),

            "effective_f1_outlier_40": (
                f1_outlier_40
            )
        }
    )


# ==========================================================
# Save overall comparison
# ==========================================================

overall_columns = [
    "method",

    # Accuracy
    "normal_error_mean_deg",
    "normal_error_std_deg",

    "offset_error_mean_m",
    "offset_error_std_m",

    "precision_mean",
    "recall_mean",

    "f1_mean",
    "f1_std",

    # Detection success
    "success_rate",
    "failure_rate",
    "num_success",
    "num_fail",

    "effective_f1",

    # Runtime
    "runtime_mean_ms",
    "runtime_std_ms",
    "runtime_median_ms",
    "runtime_min_ms",
    "runtime_max_ms",
    "runtime_p95_ms",

    # Robustness
    "condition_effective_f1_std",
    "worst_condition_effective_f1",
    "best_condition_effective_f1",

    # Noise
    "effective_f1_noise_0mm",
    "effective_f1_noise_5mm",
    "effective_f1_noise_10mm",
    "effective_f1_noise_20mm",

    "noise_degradation_0_to_20mm",

    # Outliers
    "effective_f1_outlier_0",
    "effective_f1_outlier_10",
    "effective_f1_outlier_20",
    "effective_f1_outlier_30",
    "effective_f1_outlier_40"
]


with open(
    output_overall_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=overall_columns
    )


    writer.writeheader()


    writer.writerows(
        overall_rows
    )


# ==========================================================
# Console summary
# ==========================================================

print(
    "\n=============================================="
)


print(
    "OVERALL PLANE-DETECTION COMPARISON"
)


print(
    "=============================================="
)


for row in overall_rows:

    print(
        f"\n{row['method']}"
    )


    print(
        "  Normal error:",
        f"{row['normal_error_mean_deg']:.6f}",
        "±",
        f"{row['normal_error_std_deg']:.6f}",
        "deg"
    )


    print(
        "  Offset error:",
        f"{row['offset_error_mean_m']:.6f}",
        "±",
        f"{row['offset_error_std_m']:.6f}",
        "m"
    )


    print(
        "  Precision:",
        f"{row['precision_mean']:.6f}"
    )


    print(
        "  Recall:",
        f"{row['recall_mean']:.6f}"
    )


    print(
        "  F1:",
        f"{row['f1_mean']:.6f}"
    )


    print(
        "  Success rate:",
        f"{row['success_rate']:.4f}"
    )


    print(
        "  Effective F1:",
        f"{row['effective_f1']:.6f}"
    )


    print(
        "  Runtime mean:",
        f"{row['runtime_mean_ms']:.3f}",
        "ms"
    )


    print(
        "  Runtime median:",
        f"{row['runtime_median_ms']:.3f}",
        "ms"
    )


    print(
        "  Runtime std:",
        f"{row['runtime_std_ms']:.3f}",
        "ms"
    )


    print(
        "  Runtime P95:",
        f"{row['runtime_p95_ms']:.3f}",
        "ms"
    )


    print(
        "  Worst condition effective F1:",
        f"{row['worst_condition_effective_f1']:.6f}"
    )


    print(
        "  20 mm noise effective F1:",
        f"{row['effective_f1_noise_20mm']:.6f}"
    )


    print(
        "  40% outlier effective F1:",
        f"{row['effective_f1_outlier_40']:.6f}"
    )


    print(
        "  Noise degradation:",
        f"{row['noise_degradation_0_to_20mm']:.6f}"
    )


# ==========================================================
# Compact comparison tables
# ==========================================================

print(
    "\n=============================================="
)


print(
    "COMPACT ACCURACY / SPEED TABLE"
)


print(
    "=============================================="
)


print(
    f"{'Method':<28}"
    f"{'F1':>10}"
    f"{'Eff.F1':>10}"
    f"{'Success':>10}"
    f"{'Runtime':>12}"
)


print(
    "-" * 70
)


for row in overall_rows:

    print(
        f"{row['method']:<28}"
        f"{row['f1_mean']:>10.4f}"
        f"{row['effective_f1']:>10.4f}"
        f"{row['success_rate']:>10.4f}"
        f"{row['runtime_mean_ms']:>10.2f}ms"
    )


print(
    "\n=============================================="
)


print(
    "ROBUSTNESS TABLE"
)


print(
    "=============================================="
)


print(
    f"{'Method':<28}"
    f"{'Worst F1':>12}"
    f"{'20mm F1':>12}"
    f"{'40% Out':>12}"
    f"{'Degrade':>12}"
)


print(
    "-" * 76
)


for row in overall_rows:

    print(
        f"{row['method']:<28}"
        f"{row['worst_condition_effective_f1']:>12.4f}"
        f"{row['effective_f1_noise_20mm']:>12.4f}"
        f"{row['effective_f1_outlier_40']:>12.4f}"
        f"{row['noise_degradation_0_to_20mm']:>12.4f}"
    )


# ==========================================================
# Finish
# ==========================================================

print(
    "\nComparison complete."
)


print(
    "\nCondition-level comparison:"
)


print(
    output_condition_file
)


print(
    "\nOverall comparison:"
)


print(
    output_overall_file
)