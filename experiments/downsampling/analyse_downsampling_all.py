# analyse_downsampling_real.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = THIS_DIR.parent

REAL_DIR = (
    EXPERIMENTS_DIR
    / "results"
    / "downsampling"
    / "real"
)

RESULT_FILE = (
    REAL_DIR
    / "real_results.csv"
)

OUTPUT_DIR = (
    EXPERIMENTS_DIR
    / "results"
    / "downsampling"
    / "real_analysis"
)

OVERALL_FILE = (
    OUTPUT_DIR
    / "real_overall_summary.csv"
)

RETENTION_FILE = (
    OUTPUT_DIR
    / "real_retention_summary.csv"
)

DEGRADATION_FILE = (
    OUTPUT_DIR
    / "real_retention_degradation.csv"
)

EVIDENCE_FILE = (
    OUTPUT_DIR
    / "real_decision_evidence.csv"
)


# ============================================================
# Configuration
# ============================================================

METHODS = [
    "Voxel",
    "FPS",
    "Uniform",
    "Random",
]

RETENTION_LEVELS = [
    0.50,
    0.25,
    0.10,
]


# ============================================================
# Helpers
# ============================================================

def p95(
    values,
) -> float:

    values = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.percentile(
            values,
            95,
        )
    )


def aggregate_group(
    group: pd.DataFrame,
) -> pd.Series:

    return pd.Series(
        {
            # --------------------------------------------
            # Number of runs
            # --------------------------------------------

            "runs":
                len(group),

            # --------------------------------------------
            # Retention fairness
            # --------------------------------------------

            "mean_actual_retention":
                group[
                    "actual_retention"
                ].mean(),

            "mean_retention_error":
                group[
                    "retention_error"
                ].mean(),

            "max_retention_error":
                group[
                    "retention_error"
                ].max(),

            # --------------------------------------------
            # Spatial density regularity
            # --------------------------------------------

            "mean_density_cv":
                group[
                    "density_cv"
                ].mean(),

            "median_density_cv":
                group[
                    "density_cv"
                ].median(),

            # --------------------------------------------
            # Geometric coverage
            # --------------------------------------------

            "mean_coverage_mm":
                (
                    group[
                        "coverage_mean_m"
                    ].mean()
                    * 1000.0
                ),

            "median_coverage_mm":
                (
                    group[
                        "coverage_mean_m"
                    ].median()
                    * 1000.0
                ),

            "mean_p95_coverage_mm":
                (
                    group[
                        "coverage_p95_m"
                    ].mean()
                    * 1000.0
                ),

            # --------------------------------------------
            # Processing speed
            # --------------------------------------------

            "median_runtime_ms":
                group[
                    "runtime_ms"
                ].median(),

            "mean_runtime_ms":
                group[
                    "runtime_ms"
                ].mean(),

            "p95_runtime_ms":
                p95(
                    group[
                        "runtime_ms"
                    ]
                ),

            # --------------------------------------------
            # Voxel calibration
            # --------------------------------------------

            "median_calibration_ms":
                group[
                    "calibration_ms"
                ].median(),
        }
    )


# ============================================================
# Validation
# ============================================================

def validate_results(
    results: pd.DataFrame,
):

    print(
        "=" * 78
    )

    print(
        "VALIDATING REAL-DATA RESULTS"
    )

    print(
        "=" * 78
    )

    required_columns = [
        "frame_index",
        "method",
        "target_retention",
        "actual_retention",
        "retention_error",
        "density_cv",
        "coverage_mean_m",
        "coverage_p95_m",
        "runtime_ms",
        "calibration_ms",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in results.columns
    ]

    if missing_columns:

        raise RuntimeError(
            "Missing columns:\n"
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Methods
    # --------------------------------------------------------

    found_methods = sorted(
        results[
            "method"
        ].unique()
    )

    if found_methods != sorted(
        METHODS
    ):

        raise RuntimeError(
            f"Method mismatch.\n"
            f"Found: {found_methods}"
        )

    # --------------------------------------------------------
    # Retention levels
    # --------------------------------------------------------

    found_retention = sorted(
        results[
            "target_retention"
        ].unique()
    )

    if not np.allclose(
        found_retention,
        sorted(
            RETENTION_LEVELS
        ),
    ):

        raise RuntimeError(
            "Unexpected retention levels."
        )

    # --------------------------------------------------------
    # Number of frames / runs
    # --------------------------------------------------------

    n_frames = (
        results[
            "frame_index"
        ]
        .nunique()
    )

    expected_runs = (
        n_frames
        * len(METHODS)
        * len(RETENTION_LEVELS)
    )

    print(
        f"Frames             : "
        f"{n_frames}"
    )

    print(
        f"Rows               : "
        f"{len(results)}"
    )

    print(
        f"Expected rows      : "
        f"{expected_runs}"
    )

    if len(results) != expected_runs:

        raise RuntimeError(
            "Missing or extra real-data runs."
        )

    # --------------------------------------------------------
    # Duplicate conditions
    # --------------------------------------------------------

    duplicate_mask = (
        results.duplicated(
            subset=[
                "frame_index",
                "method",
                "target_retention",
            ],
            keep=False,
        )
    )

    if duplicate_mask.any():

        raise RuntimeError(
            "Duplicate experimental "
            "conditions found."
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    core_columns = [
        "actual_retention",
        "density_cv",
        "coverage_mean_m",
        "coverage_p95_m",
        "runtime_ms",
    ]

    for column in core_columns:

        missing = (
            results[
                column
            ]
            .isna()
            .sum()
        )

        print(
            f"{column:25s}: "
            f"missing={missing}"
        )

        if missing > 0:

            raise RuntimeError(
                f"Missing values in "
                f"{column}."
            )

    print()


# ============================================================
# Overall summary
# ============================================================

def build_overall_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:

    return (
        results
        .groupby(
            "method",
            sort=False,
        )
        .apply(
            aggregate_group
        )
        .reset_index()
    )


# ============================================================
# Per-retention summary
# ============================================================

def build_retention_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:

    return (
        results
        .groupby(
            [
                "method",
                "target_retention",
            ],
            sort=False,
        )
        .apply(
            aggregate_group
        )
        .reset_index()
    )


# ============================================================
# 50% -> 10% degradation
# ============================================================

def build_degradation_summary(
    retention_summary: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for method in METHODS:

        method_data = (
            retention_summary[
                retention_summary[
                    "method"
                ]
                == method
            ]
        )

        row_50 = method_data[
            np.isclose(
                method_data[
                    "target_retention"
                ],
                0.50,
            )
        ]

        row_10 = method_data[
            np.isclose(
                method_data[
                    "target_retention"
                ],
                0.10,
            )
        ]

        if (
            len(row_50) != 1
            or len(row_10) != 1
        ):

            continue

        row_50 = row_50.iloc[0]
        row_10 = row_10.iloc[0]

        rows.append(
            {
                "method":
                    method,

                "density_cv_50":
                    row_50[
                        "mean_density_cv"
                    ],

                "density_cv_10":
                    row_10[
                        "mean_density_cv"
                    ],

                "delta_density_cv":
                    (
                        row_10[
                            "mean_density_cv"
                        ]
                        -
                        row_50[
                            "mean_density_cv"
                        ]
                    ),

                "coverage_50_mm":
                    row_50[
                        "mean_coverage_mm"
                    ],

                "coverage_10_mm":
                    row_10[
                        "mean_coverage_mm"
                    ],

                "delta_coverage_mm":
                    (
                        row_10[
                            "mean_coverage_mm"
                        ]
                        -
                        row_50[
                            "mean_coverage_mm"
                        ]
                    ),

                "runtime_50_ms":
                    row_50[
                        "median_runtime_ms"
                    ],

                "runtime_10_ms":
                    row_10[
                        "median_runtime_ms"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Real-data decision evidence
# ============================================================

def build_decision_evidence(
    overall: pd.DataFrame,
) -> pd.DataFrame:

    """
    Evidence only.

    No final 1--5 scores are assigned here.
    """

    return overall[
        [
            "method",

            # Spatial / geometry preservation
            "mean_density_cv",
            "mean_coverage_mm",
            "mean_p95_coverage_mm",

            # Speed
            "median_runtime_ms",
            "p95_runtime_ms",

            # Fairness
            "mean_retention_error",
            "max_retention_error",
        ]
    ].copy()


# ============================================================
# Console output
# ============================================================

def print_overall(
    summary: pd.DataFrame,
):

    print(
        "=" * 100
    )

    print(
        "OVERALL REAL-DATA SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Method':<9}"
        f"{'Density CV':>13}"
        f"{'Coverage':>14}"
        f"{'P95 cov.':>13}"
        f"{'Median ms':>13}"
        f"{'P95 ms':>12}"
    )

    print(
        "-" * 100
    )

    for _, row in summary.iterrows():

        print(
            f"{row['method']:<9}"
            f"{row['mean_density_cv']:>13.4f}"
            f"{row['mean_coverage_mm']:>14.3f}"
            f"{row['mean_p95_coverage_mm']:>13.3f}"
            f"{row['median_runtime_ms']:>13.3f}"
            f"{row['p95_runtime_ms']:>12.3f}"
        )

    print()


def print_retention(
    summary: pd.DataFrame,
):

    print(
        "=" * 100
    )

    print(
        "REAL-DATA RESULTS BY RETENTION LEVEL"
    )

    print(
        "=" * 100
    )

    for retention in (
        RETENTION_LEVELS
    ):

        subset = summary[
            np.isclose(
                summary[
                    "target_retention"
                ],
                retention,
            )
        ]

        print(
            f"\nTarget retention = "
            f"{retention:.2f}"
        )

        print(
            "-" * 100
        )

        print(
            f"{'Method':<9}"
            f"{'Actual':>10}"
            f"{'Density CV':>13}"
            f"{'Coverage':>14}"
            f"{'P95 cov.':>13}"
            f"{'Median ms':>13}"
        )

        for _, row in subset.iterrows():

            print(
                f"{row['method']:<9}"
                f"{row['mean_actual_retention']:>10.4f}"
                f"{row['mean_density_cv']:>13.4f}"
                f"{row['mean_coverage_mm']:>14.3f}"
                f"{row['mean_p95_coverage_mm']:>13.3f}"
                f"{row['median_runtime_ms']:>13.3f}"
            )

    print()


def print_degradation(
    summary: pd.DataFrame,
):

    print(
        "=" * 90
    )

    print(
        "RETENTION DEGRADATION: 50% -> 10%"
    )

    print(
        "=" * 90
    )

    print(
        f"{'Method':<9}"
        f"{'Delta CV':>14}"
        f"{'Delta coverage':>20}"
    )

    print(
        "-" * 90
    )

    for _, row in summary.iterrows():

        print(
            f"{row['method']:<9}"
            f"{row['delta_density_cv']:>14.4f}"
            f"{row['delta_coverage_mm']:>17.3f} mm"
        )

    print()


# ============================================================
# Main
# ============================================================

def main():

    if not RESULT_FILE.exists():

        raise FileNotFoundError(
            f"real_results.csv not found:\n"
            f"{RESULT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    results = pd.read_csv(
        RESULT_FILE
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_results(
        results
    )

    # --------------------------------------------------------
    # Analyse
    # --------------------------------------------------------

    overall = (
        build_overall_summary(
            results
        )
    )

    retention = (
        build_retention_summary(
            results
        )
    )

    degradation = (
        build_degradation_summary(
            retention
        )
    )

    evidence = (
        build_decision_evidence(
            overall
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    overall.to_csv(
        OVERALL_FILE,
        index=False,
    )

    retention.to_csv(
        RETENTION_FILE,
        index=False,
    )

    degradation.to_csv(
        DEGRADATION_FILE,
        index=False,
    )

    evidence.to_csv(
        EVIDENCE_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print_overall(
        overall
    )

    print_retention(
        retention
    )

    print_degradation(
        degradation
    )

    print(
        "=" * 78
    )

    print(
        "REAL-DATA ANALYSIS COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        f"Overall summary       : "
        f"{OVERALL_FILE}"
    )

    print(
        f"Retention summary     : "
        f"{RETENTION_FILE}"
    )

    print(
        f"Retention degradation : "
        f"{DEGRADATION_FILE}"
    )

    print(
        f"Decision evidence     : "
        f"{EVIDENCE_FILE}"
    )


if __name__ == "__main__":
    main()