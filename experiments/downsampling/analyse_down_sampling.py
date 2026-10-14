# analyse_downsampling.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = THIS_DIR.parent

SYNTHETIC_DIR = (
    EXPERIMENTS_DIR
    / "results"
    / "downsampling"
    / "synthetic"
)

REPEATABILITY_DIR = (
    EXPERIMENTS_DIR
    / "results"
    / "downsampling"
    / "repeatability"
)

RESULT_FILE = (
    SYNTHETIC_DIR
    / "results.csv"
)

PER_PLANE_FILE = (
    SYNTHETIC_DIR
    / "per_plane_results.csv"
)

REPEATABILITY_FILE = (
    REPEATABILITY_DIR
    / "repeatability_summary.csv"
)

OUTPUT_DIR = (
    EXPERIMENTS_DIR
    / "results"
    / "downsampling"
    / "analysis"
)

OVERALL_FILE = (
    OUTPUT_DIR
    / "overall_summary.csv"
)

RETENTION_FILE = (
    OUTPUT_DIR
    / "retention_summary.csv"
)

NOISE_FILE = (
    OUTPUT_DIR
    / "noise_summary.csv"
)

GRID_FILE = (
    OUTPUT_DIR
    / "grid_size_summary.csv"
)

EVIDENCE_FILE = (
    OUTPUT_DIR
    / "synthetic_decision_evidence.csv"
)


# ============================================================
# Expected experiment structure
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

EXPECTED_SAMPLES = 54
EXPECTED_RUNS = (
    EXPECTED_SAMPLES
    * len(METHODS)
    * len(RETENTION_LEVELS)
)


# ============================================================
# Utility functions
# ============================================================

def require_file(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{path}"
        )


def p95(
    values,
) -> float:

    values = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
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
            # Retention fairness
            # --------------------------------------------

            "runs":
                len(group),

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
            # Planar geometry
            # --------------------------------------------

            "mean_normal_error_deg":
                group[
                    "normal_error_deg"
                ].mean(),

            "median_normal_error_deg":
                group[
                    "normal_error_deg"
                ].median(),

            "mean_offset_error_mm":
                (
                    group[
                        "offset_error_m"
                    ].mean()
                    * 1000.0
                ),

            "median_offset_error_mm":
                (
                    group[
                        "offset_error_m"
                    ].median()
                    * 1000.0
                ),

            "mean_fit_rmse_mm":
                (
                    group[
                        "fit_rmse_m"
                    ].mean()
                    * 1000.0
                ),

            "mean_delta_normal_error_deg":
                group[
                    "delta_normal_error_deg"
                ].mean(),

            "mean_delta_offset_error_mm":
                (
                    group[
                        "delta_offset_error_m"
                    ].mean()
                    * 1000.0
                ),

            "mean_delta_fit_rmse_mm":
                (
                    group[
                        "delta_fit_rmse_m"
                    ].mean()
                    * 1000.0
                ),

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
            # Reliability
            # --------------------------------------------

            "mean_valid_plane_fraction":
                group[
                    "valid_plane_fraction"
                ].mean(),

            "min_valid_plane_fraction":
                group[
                    "valid_plane_fraction"
                ].min(),

            # --------------------------------------------
            # Processing speed
            # --------------------------------------------

            "mean_runtime_ms":
                group[
                    "runtime_ms"
                ].mean(),

            "median_runtime_ms":
                group[
                    "runtime_ms"
                ].median(),

            "p95_runtime_ms":
                p95(
                    group[
                        "runtime_ms"
                    ]
                ),

            # Voxel calibration is deliberately
            # not included in runtime_ms.
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
    per_plane: pd.DataFrame,
):

    print(
        "=" * 78
    )

    print(
        "VALIDATING SYNTHETIC RESULTS"
    )

    print(
        "=" * 78
    )

    # --------------------------------------------------------
    # Run count
    # --------------------------------------------------------

    print(
        f"Main rows          : "
        f"{len(results)}"
    )

    print(
        f"Expected rows      : "
        f"{EXPECTED_RUNS}"
    )

    if len(results) != EXPECTED_RUNS:

        raise RuntimeError(
            "Unexpected number of synthetic runs."
        )

    # --------------------------------------------------------
    # Methods
    # --------------------------------------------------------

    found_methods = sorted(
        results[
            "method"
        ].unique()
    )

    expected_methods = sorted(
        METHODS
    )

    if found_methods != expected_methods:

        raise RuntimeError(
            f"Method mismatch.\n"
            f"Found: {found_methods}\n"
            f"Expected: {expected_methods}"
        )

    # --------------------------------------------------------
    # Retention levels
    # --------------------------------------------------------

    found_retentions = sorted(
        results[
            "target_retention"
        ].unique()
    )

    expected_retentions = sorted(
        RETENTION_LEVELS
    )

    if not np.allclose(
        found_retentions,
        expected_retentions,
    ):

        raise RuntimeError(
            "Retention levels do not match "
            "the experiment design."
        )

    # --------------------------------------------------------
    # Duplicate condition check
    # --------------------------------------------------------

    duplicate_mask = (
        results.duplicated(
            subset=[
                "sample",
                "method",
                "target_retention",
            ],
            keep=False,
        )
    )

    if duplicate_mask.any():

        raise RuntimeError(
            "Duplicate synthetic conditions found."
        )

    # --------------------------------------------------------
    # Missing values in core fields
    # --------------------------------------------------------

    core_columns = [
        "normal_error_deg",
        "offset_error_m",
        "fit_rmse_m",
        "density_cv",
        "valid_plane_fraction",
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
            f"{column:28s}: "
            f"missing={missing}"
        )

        if missing > 0:

            raise RuntimeError(
                f"Missing values found in "
                f"{column}."
            )

    # --------------------------------------------------------
    # Plane-fit success
    # --------------------------------------------------------

    failed_planes = (
        per_plane[
            "fit_success"
        ]
        == 0
    ).sum()

    print(
        f"Per-plane rows     : "
        f"{len(per_plane)}"
    )

    print(
        f"Plane-fit failures : "
        f"{failed_planes}"
    )

    print()


# ============================================================
# Overall summary
# ============================================================

def build_overall_summary(
    results: pd.DataFrame,
    repeatability: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
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

    repeatability = (
        repeatability[
            [
                "method",
                "mean_overlap",
                "worst_overlap",
                "mean_output_count_cv",
            ]
        ]
        .rename(
            columns={
                "mean_overlap":
                    "repeatability_mean_overlap",

                "worst_overlap":
                    "repeatability_worst_overlap",

                "mean_output_count_cv":
                    "repeatability_count_cv",
            }
        )
    )

    summary = summary.merge(
        repeatability,
        on="method",
        how="left",
        validate="one_to_one",
    )

    return summary


# ============================================================
# Retention-level summary
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
# Noise summary
# ============================================================

def build_noise_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:

    return (
        results
        .groupby(
            [
                "method",
                "noise_sigma_m",
            ],
            sort=False,
        )
        .apply(
            aggregate_group
        )
        .reset_index()
    )


# ============================================================
# Grid-size / scaling summary
# ============================================================

def build_grid_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:

    return (
        results
        .groupby(
            [
                "method",
                "grid_size",
            ],
            sort=False,
        )
        .apply(
            aggregate_group
        )
        .reset_index()
    )


# ============================================================
# Decision-matrix evidence
# ============================================================

def build_decision_evidence(
    overall: pd.DataFrame,
) -> pd.DataFrame:

    """
    This is evidence only.

    No 1--5 final scores are assigned here because
    real-data validation has not yet been included.
    """

    columns = [
        "method",

        # Planar-geometry preservation
        "mean_normal_error_deg",
        "mean_offset_error_mm",
        "mean_delta_fit_rmse_mm",
        "mean_valid_plane_fraction",

        # Processing speed
        "median_runtime_ms",
        "p95_runtime_ms",

        # Spatial density regularity
        "mean_density_cv",

        # Repeatability
        "repeatability_mean_overlap",
        "repeatability_worst_overlap",

        # Experimental fairness
        "mean_retention_error",
        "max_retention_error",
    ]

    evidence = (
        overall[
            columns
        ]
        .copy()
    )

    return evidence


# ============================================================
# Console output
# ============================================================

def print_overall_summary(
    summary: pd.DataFrame,
):

    print(
        "=" * 110
    )

    print(
        "OVERALL SYNTHETIC DOWNSAMPLING SUMMARY"
    )

    print(
        "=" * 110
    )

    header = (
        f"{'Method':<9}"
        f"{'Normal(deg)':>13}"
        f"{'Offset(mm)':>13}"
        f"{'dRMSE(mm)':>12}"
        f"{'Density CV':>13}"
        f"{'Runtime':>12}"
        f"{'P95':>10}"
        f"{'Repeat':>11}"
    )

    print(
        header
    )

    print(
        "-" * 110
    )

    for _, row in (
        summary.iterrows()
    ):

        print(
            f"{row['method']:<9}"
            f"{row['mean_normal_error_deg']:>13.4f}"
            f"{row['mean_offset_error_mm']:>13.3f}"
            f"{row['mean_delta_fit_rmse_mm']:>12.3f}"
            f"{row['mean_density_cv']:>13.4f}"
            f"{row['median_runtime_ms']:>12.3f}"
            f"{row['p95_runtime_ms']:>10.3f}"
            f"{row['repeatability_mean_overlap']:>11.4f}"
        )

    print()


def print_retention_summary(
    summary: pd.DataFrame,
):

    print(
        "=" * 105
    )

    print(
        "PER-RETENTION SUMMARY"
    )

    print(
        "=" * 105
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
            "-" * 105
        )

        print(
            f"{'Method':<9}"
            f"{'Normal':>11}"
            f"{'Offset(mm)':>13}"
            f"{'dRMSE(mm)':>12}"
            f"{'Density CV':>13}"
            f"{'Median ms':>12}"
        )

        for _, row in (
            subset.iterrows()
        ):

            print(
                f"{row['method']:<9}"
                f"{row['mean_normal_error_deg']:>11.4f}"
                f"{row['mean_offset_error_mm']:>13.3f}"
                f"{row['mean_delta_fit_rmse_mm']:>12.3f}"
                f"{row['mean_density_cv']:>13.4f}"
                f"{row['median_runtime_ms']:>12.3f}"
            )

    print()


def print_noise_summary(
    summary: pd.DataFrame,
):

    print(
        "=" * 95
    )

    print(
        "NOISE SENSITIVITY"
    )

    print(
        "=" * 95
    )

    for noise in sorted(
        summary[
            "noise_sigma_m"
        ].unique()
    ):

        subset = summary[
            np.isclose(
                summary[
                    "noise_sigma_m"
                ],
                noise,
            )
        ]

        print(
            f"\nNoise sigma = "
            f"{noise * 1000.0:.1f} mm"
        )

        print(
            "-" * 95
        )

        print(
            f"{'Method':<9}"
            f"{'Normal(deg)':>14}"
            f"{'Offset(mm)':>14}"
            f"{'dRMSE(mm)':>13}"
            f"{'Density CV':>13}"
        )

        for _, row in (
            subset.iterrows()
        ):

            print(
                f"{row['method']:<9}"
                f"{row['mean_normal_error_deg']:>14.4f}"
                f"{row['mean_offset_error_mm']:>14.3f}"
                f"{row['mean_delta_fit_rmse_mm']:>13.3f}"
                f"{row['mean_density_cv']:>13.4f}"
            )

    print()


def print_runtime_scaling(
    summary: pd.DataFrame,
):

    print(
        "=" * 80
    )

    print(
        "RUNTIME SCALING BY INPUT GRID SIZE"
    )

    print(
        "=" * 80
    )

    print(
        f"{'Method':<9}"
        f"{'Grid':>8}"
        f"{'Median ms':>14}"
        f"{'P95 ms':>14}"
        f"{'Mean ms':>14}"
    )

    print(
        "-" * 80
    )

    for _, row in (
        summary.iterrows()
    ):

        print(
            f"{row['method']:<9}"
            f"{int(row['grid_size']):>8d}"
            f"{row['median_runtime_ms']:>14.3f}"
            f"{row['p95_runtime_ms']:>14.3f}"
            f"{row['mean_runtime_ms']:>14.3f}"
        )

    print()


# ============================================================
# Main
# ============================================================

def main():

    require_file(
        RESULT_FILE
    )

    require_file(
        PER_PLANE_FILE
    )

    require_file(
        REPEATABILITY_FILE
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

    per_plane = pd.read_csv(
        PER_PLANE_FILE
    )

    repeatability = pd.read_csv(
        REPEATABILITY_FILE
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_results(
        results,
        per_plane,
    )

    # --------------------------------------------------------
    # Analyse
    # --------------------------------------------------------

    overall = (
        build_overall_summary(
            results,
            repeatability,
        )
    )

    retention = (
        build_retention_summary(
            results
        )
    )

    noise = (
        build_noise_summary(
            results
        )
    )

    grid = (
        build_grid_summary(
            results
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

    noise.to_csv(
        NOISE_FILE,
        index=False,
    )

    grid.to_csv(
        GRID_FILE,
        index=False,
    )

    evidence.to_csv(
        EVIDENCE_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print_overall_summary(
        overall
    )

    print_retention_summary(
        retention
    )

    print_noise_summary(
        noise
    )

    print_runtime_scaling(
        grid
    )

    print(
        "=" * 78
    )

    print(
        "ANALYSIS COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        f"Overall summary    : "
        f"{OVERALL_FILE}"
    )

    print(
        f"Retention summary  : "
        f"{RETENTION_FILE}"
    )

    print(
        f"Noise summary      : "
        f"{NOISE_FILE}"
    )

    print(
        f"Grid-size summary  : "
        f"{GRID_FILE}"
    )

    print(
        f"Decision evidence  : "
        f"{EVIDENCE_FILE}"
    )


if __name__ == "__main__":
    main()