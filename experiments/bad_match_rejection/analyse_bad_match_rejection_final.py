from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent

RESULT_DIR = (
    EXPERIMENTS_DIR
    / "results"
    / "bad_match_rejection"
    / "final"
)

INPUT_FILE = (
    RESULT_DIR
    / "final_results.csv"
)

ANALYSIS_DIR = (
    RESULT_DIR
    / "analysis"
)

ANALYSIS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Utility functions
# ============================================================

def safe_divide(
    numerator,
    denominator,
    empty_value=1.0,
):
    if denominator == 0:
        return empty_value

    return numerator / denominator


def f1_from_counts(
    tp,
    fp,
    fn,
):
    denominator = (
        2 * tp
        + fp
        + fn
    )

    if denominator == 0:
        return 1.0

    return (
        2.0 * tp
        / denominator
    )


def numeric_values(series):
    """
    Convert a pandas Series to finite numeric values.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
    ]

    return values


def mean_or_nan(series):
    values = numeric_values(series)

    if len(values) == 0:
        return np.nan

    return float(
        np.mean(values)
    )


def median_or_nan(series):
    values = numeric_values(series)

    if len(values) == 0:
        return np.nan

    return float(
        np.median(values)
    )


def percentile_or_nan(
    series,
    percentile,
):
    values = numeric_values(series)

    if len(values) == 0:
        return np.nan

    return float(
        np.percentile(
            values,
            percentile,
        )
    )


def bool_mean(series):
    """
    Robust mean for columns containing bools or True/False strings.
    """

    if series.dtype == bool:
        values = series

    else:
        values = (
            series
            .astype(str)
            .str.lower()
            .map({
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            })
        )

    values = values.dropna()

    if len(values) == 0:
        return np.nan

    return float(
        values.astype(bool).mean()
    )


# ============================================================
# Core group summary
# ============================================================

def summarise_group(group):
    """
    Summarise one method under one experimental condition.

    Positive class:
        bad correspondence

    TP:
        bad correspondence correctly rejected

    FP:
        correct correspondence incorrectly rejected

    FN:
        bad correspondence incorrectly retained

    TN:
        correct correspondence correctly retained
    """

    # --------------------------------------------------------
    # Aggregate confusion matrix
    # --------------------------------------------------------

    tp = int(
        pd.to_numeric(
            group["tp"],
            errors="coerce",
        ).fillna(0).sum()
    )

    fp = int(
        pd.to_numeric(
            group["fp"],
            errors="coerce",
        ).fillna(0).sum()
    )

    fn = int(
        pd.to_numeric(
            group["fn"],
            errors="coerce",
        ).fillna(0).sum()
    )

    tn = int(
        pd.to_numeric(
            group["tn"],
            errors="coerce",
        ).fillna(0).sum()
    )

    # --------------------------------------------------------
    # Bad-match class
    # --------------------------------------------------------

    bad_precision = safe_divide(
        tp,
        tp + fp,
    )

    bad_recall = safe_divide(
        tp,
        tp + fn,
    )

    bad_f1 = f1_from_counts(
        tp,
        fp,
        fn,
    )

    # --------------------------------------------------------
    # Correct-match class
    #
    # Correct correspondences become the positive class:
    #
    # TP_correct = TN
    # FP_correct = FN
    # FN_correct = FP
    # --------------------------------------------------------

    correct_precision = safe_divide(
        tn,
        tn + fn,
    )

    correct_recall = safe_divide(
        tn,
        tn + fp,
    )

    correct_f1 = f1_from_counts(
        tn,
        fn,
        fp,
    )

    # Correct-match retention is identical to
    # correct-class recall.
    correct_retention = (
        correct_recall
    )

    false_rejection_rate = (
        1.0
        - correct_retention
    )

    # --------------------------------------------------------
    # Balanced classification metric
    # --------------------------------------------------------

    macro_f1 = (
        bad_f1
        + correct_f1
    ) / 2.0

    overall_accuracy = safe_divide(
        tp + tn,
        tp + fp + fn + tn,
    )

    # --------------------------------------------------------
    # Algorithm / pose availability
    # --------------------------------------------------------

    rejection_completed_rate = bool_mean(
        group[
            "rejection_completed"
        ]
    )

    pose_usable_rate = bool_mean(
        group[
            "pose_usable"
        ]
    )

    # Input column "pose_success" means that a pose estimator
    # returned a valid numerical pose. It does NOT guarantee
    # that the estimated pose is accurate.
    pose_output_rate = bool_mean(
        group[
            "pose_success"
        ]
    )

    # --------------------------------------------------------
    # Pose accuracy
    # --------------------------------------------------------

    rotation_mean = mean_or_nan(
        group[
            "rotation_error_deg"
        ]
    )

    rotation_median = median_or_nan(
        group[
            "rotation_error_deg"
        ]
    )

    rotation_p95 = percentile_or_nan(
        group[
            "rotation_error_deg"
        ],
        95,
    )

    translation_mean = mean_or_nan(
        group[
            "translation_error_m"
        ]
    )

    translation_median = median_or_nan(
        group[
            "translation_error_m"
        ]
    )

    translation_p95 = percentile_or_nan(
        group[
            "translation_error_m"
        ],
        95,
    )

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    runtime_mean = mean_or_nan(
        group[
            "runtime_ms"
        ]
    )

    runtime_median = median_or_nan(
        group[
            "runtime_ms"
        ]
    )

    runtime_p95 = percentile_or_nan(
        group[
            "runtime_ms"
        ],
        95,
    )

    # --------------------------------------------------------
    # Correspondence counts
    # --------------------------------------------------------

    mean_kept = mean_or_nan(
        group["kept"]
    )

    mean_rejected = mean_or_nan(
        group["rejected"]
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "samples":
            int(len(group)),

        "tp":
            tp,

        "fp":
            fp,

        "fn":
            fn,

        "tn":
            tn,

        "bad_precision":
            bad_precision,

        "bad_recall":
            bad_recall,

        "bad_f1":
            bad_f1,

        "correct_precision":
            correct_precision,

        "correct_recall":
            correct_recall,

        "correct_f1":
            correct_f1,

        "correct_retention":
            correct_retention,

        "false_rejection_rate":
            false_rejection_rate,

        "macro_f1":
            macro_f1,

        "overall_accuracy":
            overall_accuracy,

        "rejection_completed_rate":
            rejection_completed_rate,

        "pose_usable_rate":
            pose_usable_rate,

        "pose_output_rate":
            pose_output_rate,

        "rotation_error_mean_deg":
            rotation_mean,

        "rotation_error_median_deg":
            rotation_median,

        "rotation_error_p95_deg":
            rotation_p95,

        "translation_error_mean_m":
            translation_mean,

        "translation_error_median_m":
            translation_median,

        "translation_error_p95_m":
            translation_p95,

        "runtime_mean_ms":
            runtime_mean,

        "runtime_median_ms":
            runtime_median,

        "runtime_p95_ms":
            runtime_p95,

        "mean_kept":
            mean_kept,

        "mean_rejected":
            mean_rejected,
    }


# ============================================================
# Generic grouped analysis
# ============================================================

def grouped_summary(
    df,
    group_columns,
):
    rows = []

    grouped = df.groupby(
        group_columns,
        dropna=False,
        sort=False,
    )

    for keys, group in grouped:

        if not isinstance(
            keys,
            tuple,
        ):
            keys = (keys,)

        row = {
            column: value
            for column, value
            in zip(
                group_columns,
                keys,
            )
        }

        row.update(
            summarise_group(
                group
            )
        )

        rows.append(row)

    return pd.DataFrame(rows)


def analyse_by(
    df,
    condition_column,
    filename,
):
    result = grouped_summary(
        df,
        [
            condition_column,
            "method",
        ],
    )

    result.to_csv(
        ANALYSIS_DIR
        / filename,
        index=False,
    )

    return result


# ============================================================
# Overall analysis
# ============================================================

def analyse_overall(df):

    result = grouped_summary(
        df,
        ["method"],
    )

    # Sorting is only for easier inspection.
    result = result.sort_values(
        "macro_f1",
        ascending=False,
    )

    result.to_csv(
        ANALYSIS_DIR
        / "overall_summary.csv",
        index=False,
    )

    return result


# ============================================================
# 0% bad-match clean case
# ============================================================

def analyse_clean_case(df):

    clean = df[
        np.isclose(
            df[
                "requested_bad_ratio"
            ],
            0.0,
        )
    ].copy()

    result = grouped_summary(
        clean,
        ["method"],
    )

    # For the 0%-bad condition the key measure is
    # preservation of valid correspondences.
    output_columns = [
        "method",
        "samples",
        "correct_retention",
        "false_rejection_rate",
        "pose_usable_rate",
        "pose_output_rate",
        "rotation_error_median_deg",
        "translation_error_median_m",
        "runtime_median_ms",
        "mean_kept",
        "mean_rejected",
    ]

    result = result[
        output_columns
    ]

    result.to_csv(
        ANALYSIS_DIR
        / "clean_case_summary.csv",
        index=False,
    )

    return result


# ============================================================
# Non-zero bad-ratio analysis
# ============================================================

def analyse_nonzero_bad_ratios(df):

    nonzero = df[
        df[
            "requested_bad_ratio"
        ] > 0.0
    ].copy()

    result = analyse_by(
        nonzero,
        "requested_bad_ratio",
        "by_nonzero_bad_ratio.csv",
    )

    return result


# ============================================================
# Extreme stress case
# ============================================================

def analyse_extreme_stress(df):

    stress = df[
        np.isclose(
            df[
                "requested_bad_ratio"
            ],
            0.8,
        )
        &
        np.isclose(
            df[
                "normal_noise_deg"
            ],
            5.0,
        )
        &
        (
            df["geometry"]
            == "mostly_parallel"
        )
    ].copy()

    if len(stress) == 0:

        print(
            "Warning: no extreme "
            "stress-case samples found."
        )

        return pd.DataFrame()

    result = grouped_summary(
        stress,
        ["method"],
    )

    result.to_csv(
        ANALYSIS_DIR
        / "extreme_stress_summary.csv",
        index=False,
    )

    return result


# ============================================================
# Robustness summary
# ============================================================

def build_robustness_summary(
    by_noise,
    by_nonzero_bad_ratio,
    by_geometry,
    clean_case,
    extreme_stress,
):

    rows = []

    methods = sorted(
        by_noise[
            "method"
        ].unique()
    )

    for method in methods:

        # ----------------------------------------------------
        # Normal-noise robustness
        # ----------------------------------------------------

        noise_data = by_noise[
            by_noise["method"]
            == method
        ]

        worst_noise_macro_f1 = float(
            noise_data[
                "macro_f1"
            ].min()
        )

        best_noise_macro_f1 = float(
            noise_data[
                "macro_f1"
            ].max()
        )

        noise_macro_f1_range = (
            best_noise_macro_f1
            - worst_noise_macro_f1
        )

        # ----------------------------------------------------
        # Bad-ratio robustness
        #
        # IMPORTANT:
        # 0% bad is excluded here.
        # ----------------------------------------------------

        bad_data = (
            by_nonzero_bad_ratio[
                by_nonzero_bad_ratio[
                    "method"
                ]
                == method
            ]
        )

        if len(bad_data) > 0:

            worst_nonzero_bad_ratio_macro_f1 = float(
                bad_data[
                    "macro_f1"
                ].min()
            )

        else:

            worst_nonzero_bad_ratio_macro_f1 = (
                np.nan
            )

        # ----------------------------------------------------
        # Geometry robustness
        # ----------------------------------------------------

        geometry_data = (
            by_geometry[
                by_geometry["method"]
                == method
            ]
        )

        worst_geometry_macro_f1 = float(
            geometry_data[
                "macro_f1"
            ].min()
        )

        best_geometry_macro_f1 = float(
            geometry_data[
                "macro_f1"
            ].max()
        )

        geometry_macro_f1_range = (
            best_geometry_macro_f1
            - worst_geometry_macro_f1
        )

        # ----------------------------------------------------
        # Clean-case preservation
        # ----------------------------------------------------

        clean_data = (
            clean_case[
                clean_case["method"]
                == method
            ]
        )

        if len(clean_data) > 0:

            clean_correct_retention = float(
                clean_data.iloc[0][
                    "correct_retention"
                ]
            )

            clean_false_rejection_rate = float(
                clean_data.iloc[0][
                    "false_rejection_rate"
                ]
            )

        else:

            clean_correct_retention = np.nan
            clean_false_rejection_rate = np.nan

        # ----------------------------------------------------
        # Extreme stress
        # ----------------------------------------------------

        if len(extreme_stress) > 0:

            stress_data = (
                extreme_stress[
                    extreme_stress[
                        "method"
                    ]
                    == method
                ]
            )

        else:

            stress_data = (
                pd.DataFrame()
            )

        if len(stress_data) > 0:

            extreme_stress_macro_f1 = float(
                stress_data.iloc[0][
                    "macro_f1"
                ]
            )

            extreme_stress_pose_output_rate = float(
                stress_data.iloc[0][
                    "pose_output_rate"
                ]
            )

        else:

            extreme_stress_macro_f1 = np.nan
            extreme_stress_pose_output_rate = np.nan

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        rows.append({
            "method":
                method,

            "worst_noise_macro_f1":
                worst_noise_macro_f1,

            "noise_macro_f1_range":
                noise_macro_f1_range,

            "worst_nonzero_bad_ratio_macro_f1":
                worst_nonzero_bad_ratio_macro_f1,

            "clean_correct_retention":
                clean_correct_retention,

            "clean_false_rejection_rate":
                clean_false_rejection_rate,

            "worst_geometry_macro_f1":
                worst_geometry_macro_f1,

            "geometry_macro_f1_range":
                geometry_macro_f1_range,

            "extreme_stress_macro_f1":
                extreme_stress_macro_f1,

            "extreme_stress_pose_output_rate":
                extreme_stress_pose_output_rate,
        })

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        ANALYSIS_DIR
        / "robustness_summary.csv",
        index=False,
    )

    return result


# ============================================================
# Decision-matrix evidence
# ============================================================

def build_decision_evidence(
    overall,
    robustness,
):

    merged = overall.merge(
        robustness,
        on="method",
        how="left",
    )

    evidence = merged[
        [
            # Method
            "method",

            # -----------------------------------------------
            # Criterion 1:
            # Correspondence Rejection Accuracy
            # -----------------------------------------------
            "bad_precision",
            "bad_recall",
            "bad_f1",
            "correct_retention",
            "false_rejection_rate",
            "correct_f1",
            "macro_f1",

            # -----------------------------------------------
            # Criterion 2:
            # Downstream Pose Accuracy
            # -----------------------------------------------
            "rotation_error_median_deg",
            "rotation_error_p95_deg",
            "translation_error_median_m",
            "translation_error_p95_m",

            # -----------------------------------------------
            # Criterion 3:
            # Robustness
            # -----------------------------------------------
            "worst_noise_macro_f1",
            "noise_macro_f1_range",
            "worst_nonzero_bad_ratio_macro_f1",
            "clean_correct_retention",
            "clean_false_rejection_rate",
            "worst_geometry_macro_f1",
            "geometry_macro_f1_range",
            "extreme_stress_macro_f1",
            "extreme_stress_pose_output_rate",

            # -----------------------------------------------
            # Criterion 4:
            # Runtime Efficiency
            # -----------------------------------------------
            "runtime_median_ms",
            "runtime_p95_ms",

            # -----------------------------------------------
            # Criterion 5:
            # Operational Reliability
            # -----------------------------------------------
            "rejection_completed_rate",
            "pose_usable_rate",
            "pose_output_rate",

            # -----------------------------------------------
            # Additional context
            # -----------------------------------------------
            "mean_kept",
            "mean_rejected",
        ]
    ].copy()

    evidence.to_csv(
        ANALYSIS_DIR
        / "decision_evidence_summary.csv",
        index=False,
    )

    return evidence


# ============================================================
# Input validation
# ============================================================

def validate_input(df):

    required_columns = {
        "sample",
        "method",
        "match_count",
        "requested_bad_ratio",
        "actual_bad_ratio",
        "normal_noise_deg",
        "geometry",
        "rejection_completed",
        "pose_usable",
        "runtime_ms",
        "kept",
        "rejected",
        "tp",
        "fp",
        "fn",
        "tn",
        "pose_success",
        "rotation_error_deg",
        "translation_error_m",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "final_results.csv is missing "
            "the following columns:\n"
            f"{sorted(missing)}"
        )

    unique_samples = (
        df[
            "sample"
        ].nunique()
    )

    unique_methods = (
        df[
            "method"
        ].nunique()
    )

    expected_rows = (
        unique_samples
        * unique_methods
    )

    print(
        f"Rows:               {len(df)}"
    )

    print(
        f"Unique samples:     {unique_samples}"
    )

    print(
        f"Methods:            {unique_methods}"
    )

    print(
        f"Expected rows:      {expected_rows}"
    )

    if len(df) != expected_rows:

        print(
            "WARNING: row count does not "
            "equal samples × methods."
        )

    completed = bool_mean(
        df[
            "rejection_completed"
        ]
    )

    if np.isnan(completed):

        failures = np.nan

    else:

        failures = int(
            round(
                len(df)
                * (1.0 - completed)
            )
        )

    print(
        f"Execution failures: {failures}"
    )


# ============================================================
# Terminal summary
# ============================================================

def print_overall_summary(
    overall,
):

    columns = [
        "method",
        "macro_f1",
        "bad_precision",
        "bad_recall",
        "correct_retention",
        "pose_output_rate",
        "rotation_error_median_deg",
        "translation_error_median_m",
        "runtime_median_ms",
    ]

    display = overall[
        columns
    ].copy()

    numeric_columns = (
        display
        .select_dtypes(
            include=[np.number]
        )
        .columns
    )

    display[
        numeric_columns
    ] = (
        display[
            numeric_columns
        ]
        .round(4)
    )

    print()
    print("=" * 110)
    print(
        "OVERALL SYNTHETIC FINAL SUMMARY"
    )
    print("=" * 110)

    print(
        display.to_string(
            index=False
        )
    )


def print_clean_summary(
    clean_case,
):

    display = clean_case[
        [
            "method",
            "correct_retention",
            "false_rejection_rate",
        ]
    ].copy()

    display[
        [
            "correct_retention",
            "false_rejection_rate",
        ]
    ] = (
        display[
            [
                "correct_retention",
                "false_rejection_rate",
            ]
        ]
        .round(4)
    )

    print()
    print("=" * 80)
    print(
        "0% BAD-MATCH CLEAN CASE"
    )
    print("=" * 80)

    print(
        display.to_string(
            index=False
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Could not find:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    # Ensure numeric experimental columns.
    numeric_columns = [
        "match_count",
        "requested_bad_ratio",
        "actual_bad_ratio",
        "normal_noise_deg",
        "runtime_ms",
        "kept",
        "rejected",
        "tp",
        "fp",
        "fn",
        "tn",
        "rotation_error_deg",
        "translation_error_m",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    print()
    print("=" * 75)
    print(
        "BAD-MATCH REJECTION "
        "SYNTHETIC FINAL ANALYSIS"
    )
    print("=" * 75)

    validate_input(
        df
    )

    # ========================================================
    # Overall
    # ========================================================

    overall = analyse_overall(
        df
    )

    # ========================================================
    # Main controlled factors
    # ========================================================

    by_normal_noise = analyse_by(
        df,
        "normal_noise_deg",
        "by_normal_noise.csv",
    )

    by_bad_ratio = analyse_by(
        df,
        "requested_bad_ratio",
        "by_bad_ratio.csv",
    )

    by_geometry = analyse_by(
        df,
        "geometry",
        "by_geometry.csv",
    )

    by_match_count = analyse_by(
        df,
        "match_count",
        "by_match_count.csv",
    )

    # ========================================================
    # Supplementary:
    # actual realised bad ratio
    # ========================================================

    by_actual_bad_ratio = analyse_by(
        df,
        "actual_bad_ratio",
        "by_actual_bad_ratio.csv",
    )

    # ========================================================
    # Clean 0%-bad case
    # ========================================================

    clean_case = analyse_clean_case(
        df
    )

    # ========================================================
    # Non-zero bad-ratio analysis
    # ========================================================

    by_nonzero_bad_ratio = (
        analyse_nonzero_bad_ratios(
            df
        )
    )

    # ========================================================
    # Extreme stress case
    # ========================================================

    extreme_stress = (
        analyse_extreme_stress(
            df
        )
    )

    # ========================================================
    # Robustness summary
    # ========================================================

    robustness = (
        build_robustness_summary(
            by_normal_noise,
            by_nonzero_bad_ratio,
            by_geometry,
            clean_case,
            extreme_stress,
        )
    )

    # ========================================================
    # Decision evidence
    # ========================================================

    decision_evidence = (
        build_decision_evidence(
            overall,
            robustness,
        )
    )

    # ========================================================
    # Terminal output
    # ========================================================

    print_overall_summary(
        overall
    )

    print_clean_summary(
        clean_case
    )

    print()
    print("=" * 75)
    print(
        "ANALYSIS COMPLETE"
    )
    print("=" * 75)

    print(
        f"Analysis directory:\n"
        f"{ANALYSIS_DIR}"
    )

    print()
    print(
        "Generated files:"
    )

    for file in sorted(
        ANALYSIS_DIR.glob("*.csv")
    ):

        print(
            f"  {file.name}"
        )


if __name__ == "__main__":
    main()