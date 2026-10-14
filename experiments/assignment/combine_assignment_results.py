"""
combine_assignment_results.py

Combine Synthetic FINAL and TUM real-data Assignment evidence.

This program DOES NOT:
- rerun Assignment algorithms;
- retune parameters;
- assign 1-5 WDM scores;
- rank the candidate algorithms.

It only combines the quantitative experimental evidence that will
later be used by the Assignment Weighted Decision Matrix.

Combination policy
------------------
For metrics that can be measured meaningfully on both synthetic
and real data:

    combined = 0.30 * synthetic + 0.70 * real

The real-data benchmark receives the larger weight because the final
SLAM system operates on real RGB-D observations.

Criteria
--------
1. Assignment Accuracy
       70% Real F1 + 30% Synthetic overall F1

2. Processing Speed
       70% Real mean runtime + 30% Synthetic mean runtime

3. Robustness
       Synthetic hard/ambiguous F1
   Real data do not provide a paired controlled clean/degraded test.

4. Handling Unmatched / Unequal Plane Sets
       Synthetic unmatched-plane F1
       Synthetic FFMR
   Synthetic data provide complete GT for deliberately constructed
   unmatched cases.

5. Implementation Complexity
       NOT calculated here.
       This remains a qualitative WDM criterion.

Inputs
------
results/assignment/synthetic_final/analysis/
    controlled_experiment_summary.csv

results/assignment/tum/analysis/
    real_analysis_summary.csv

Outputs
-------
results/assignment/combined/
    combined_evidence.csv
    decision_matrix_quantitative_evidence.csv
    combination_policy.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

ALGORITHM_ORDER = [
    "hungarian",
    "auction",
    "jonker_volgenant",
    "cost_scaling",
]


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_SYNTHETIC_FILE = (
    PROJECT_ROOT
    / "results"
    / "assignment"
    / "synthetic_final"
    / "analysis"
    / "controlled_experiment_summary.csv"
)


DEFAULT_REAL_FILE = (
    PROJECT_ROOT
    / "results"
    / "assignment"
    / "tum"
    / "analysis"
    / "real_analysis_summary.csv"
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "assignment"
    / "combined"
)


DEFAULT_REAL_WEIGHT = 0.70
DEFAULT_SYNTHETIC_WEIGHT = 0.30


# ============================================================
# CSV helpers
# ============================================================

def load_csv(
    path: Path,
) -> list[dict]:

    if not path.exists():

        raise FileNotFoundError(
            f"Input CSV not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    if not rows:

        raise ValueError(
            f"No rows found in:\n{path}"
        )

    return rows


def save_csv(
    path: Path,
    rows: list[dict],
):

    if not rows:

        raise ValueError(
            f"No rows to save:\n{path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = []

    seen = set()

    for row in rows:

        for key in row.keys():

            if key not in seen:

                seen.add(key)

                fieldnames.append(key)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Numeric helper
# ============================================================

def value(
    row: dict,
    key: str,
) -> float:

    if key not in row:

        raise KeyError(
            f"Required column '{key}' "
            f"not found."
        )

    raw = row[key]

    if raw in (
        "",
        None,
    ):

        raise ValueError(
            f"Empty value for column "
            f"'{key}'."
        )

    return float(raw)


# ============================================================
# Validation
# ============================================================

def validate_weights(
    real_weight: float,
    synthetic_weight: float,
):

    if real_weight < 0:

        raise ValueError(
            "Real-data weight cannot "
            "be negative."
        )

    if synthetic_weight < 0:

        raise ValueError(
            "Synthetic-data weight cannot "
            "be negative."
        )

    total = (
        real_weight
        + synthetic_weight
    )

    if abs(
        total - 1.0
    ) > 1e-9:

        raise ValueError(
            "Real + synthetic weights "
            f"must equal 1.0, got {total}."
        )


def validate_synthetic(
    rows: list[dict],
):

    required = {
        "algorithm",
        "F1_overall",
        "F1_hard",
        "F1_unmatched_plane",
        "FFMR_unmatched",
        "runtime_mean_ms",
        "runtime_median_ms",
        "runtime_p95_ms",
    }

    missing = (
        required
        - set(
            rows[0].keys()
        )
    )

    if missing:

        raise ValueError(
            "Synthetic controlled summary "
            "missing columns:\n"
            f"{sorted(missing)}"
        )


def validate_real(
    rows: list[dict],
):

    required = {
        "algorithm",
        "F1_real",
        "precision_real",
        "recall_real",
        "exact_assignment_rate",
        "unknown_selection_rate",
        "runtime_mean_ms",
        "runtime_median_ms",
        "runtime_p95_ms",
        "objective_agreement_with_hungarian",
    }

    missing = (
        required
        - set(
            rows[0].keys()
        )
    )

    if missing:

        raise ValueError(
            "Real analysis summary "
            "missing columns:\n"
            f"{sorted(missing)}"
        )


# ============================================================
# Index rows by algorithm
# ============================================================

def index_by_algorithm(
    rows: list[dict],
) -> dict[str, dict]:

    output = {}

    for row in rows:

        algorithm = row[
            "algorithm"
        ]

        if algorithm in output:

            raise ValueError(
                f"Duplicate algorithm "
                f"'{algorithm}'."
            )

        output[
            algorithm
        ] = row

    return output


# ============================================================
# Combination
# ============================================================

def combine_results(
    synthetic_rows: list[dict],
    real_rows: list[dict],
    real_weight: float,
    synthetic_weight: float,
) -> list[dict]:

    synthetic = (
        index_by_algorithm(
            synthetic_rows
        )
    )

    real = (
        index_by_algorithm(
            real_rows
        )
    )

    output = []

    for algorithm in ALGORITHM_ORDER:

        if algorithm not in synthetic:

            raise KeyError(
                f"Algorithm '{algorithm}' "
                f"missing from synthetic results."
            )

        if algorithm not in real:

            raise KeyError(
                f"Algorithm '{algorithm}' "
                f"missing from real results."
            )

        s = synthetic[
            algorithm
        ]

        r = real[
            algorithm
        ]

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        f1_synthetic = value(
            s,
            "F1_overall",
        )

        f1_real = value(
            r,
            "F1_real",
        )

        f1_combined = (
            synthetic_weight
            * f1_synthetic
            +
            real_weight
            * f1_real
        )

        # Supporting accuracy evidence
        precision_real = value(
            r,
            "precision_real",
        )

        recall_real = value(
            r,
            "recall_real",
        )

        exact_synthetic = value(
            s,
            "exact_assignment_rate",
        )

        exact_real = value(
            r,
            "exact_assignment_rate",
        )

        exact_combined = (
            synthetic_weight
            * exact_synthetic
            +
            real_weight
            * exact_real
        )

        # ----------------------------------------------------
        # Runtime
        # ----------------------------------------------------

        runtime_synthetic_mean = value(
            s,
            "runtime_mean_ms",
        )

        runtime_real_mean = value(
            r,
            "runtime_mean_ms",
        )

        runtime_combined_mean = (
            synthetic_weight
            * runtime_synthetic_mean
            +
            real_weight
            * runtime_real_mean
        )

        runtime_synthetic_median = value(
            s,
            "runtime_median_ms",
        )

        runtime_real_median = value(
            r,
            "runtime_median_ms",
        )

        runtime_combined_median = (
            synthetic_weight
            * runtime_synthetic_median
            +
            real_weight
            * runtime_real_median
        )

        runtime_synthetic_p95 = value(
            s,
            "runtime_p95_ms",
        )

        runtime_real_p95 = value(
            r,
            "runtime_p95_ms",
        )

        runtime_combined_p95 = (
            synthetic_weight
            * runtime_synthetic_p95
            +
            real_weight
            * runtime_real_p95
        )

        # ----------------------------------------------------
        # Robustness
        # ----------------------------------------------------
        #
        # This remains a synthetic-only controlled metric.
        # ----------------------------------------------------

        f1_hard = value(
            s,
            "F1_hard",
        )

        # ----------------------------------------------------
        # Unmatched / unequal-plane handling
        # ----------------------------------------------------
        #
        # Also synthetic-only because these experiments have
        # complete analytical GT and deliberately created
        # unmatched plane sets.
        # ----------------------------------------------------

        f1_unmatched = value(
            s,
            "F1_unmatched_plane",
        )

        ffmr_unmatched = value(
            s,
            "FFMR_unmatched",
        )

        # ----------------------------------------------------
        # Real-reference diagnostics
        # ----------------------------------------------------

        unknown_selection_rate = value(
            r,
            "unknown_selection_rate",
        )

        objective_agreement = value(
            r,
            "objective_agreement_with_hungarian",
        )

        # ----------------------------------------------------
        # Final evidence row
        # ----------------------------------------------------

        output.append(
            {
                "algorithm":
                    algorithm,

                # ============================================
                # Dataset weighting
                # ============================================

                "synthetic_weight":
                    synthetic_weight,

                "real_weight":
                    real_weight,

                # ============================================
                # Assignment Accuracy
                # ============================================

                "F1_synthetic":
                    f1_synthetic,

                "F1_real":
                    f1_real,

                "F1_combined":
                    f1_combined,

                "precision_real":
                    precision_real,

                "recall_real":
                    recall_real,

                "exact_assignment_synthetic":
                    exact_synthetic,

                "exact_assignment_real":
                    exact_real,

                "exact_assignment_combined":
                    exact_combined,

                # ============================================
                # Processing Speed
                # ============================================

                "runtime_synthetic_mean_ms":
                    runtime_synthetic_mean,

                "runtime_real_mean_ms":
                    runtime_real_mean,

                "runtime_combined_mean_ms":
                    runtime_combined_mean,

                "runtime_synthetic_median_ms":
                    runtime_synthetic_median,

                "runtime_real_median_ms":
                    runtime_real_median,

                "runtime_combined_median_ms":
                    runtime_combined_median,

                "runtime_synthetic_p95_ms":
                    runtime_synthetic_p95,

                "runtime_real_p95_ms":
                    runtime_real_p95,

                "runtime_combined_p95_ms":
                    runtime_combined_p95,

                # ============================================
                # Robustness
                # ============================================

                "F1_hard":
                    f1_hard,

                # ============================================
                # Unmatched / Unequal Plane Sets
                # ============================================

                "F1_unmatched":
                    f1_unmatched,

                "FFMR_unmatched":
                    ffmr_unmatched,

                # ============================================
                # Real-data diagnostics
                # ============================================

                "unknown_selection_rate_real":
                    unknown_selection_rate,

                "objective_agreement_real":
                    objective_agreement,
            }
        )

    return output


# ============================================================
# WDM-ready quantitative subset
# ============================================================

def build_wdm_quantitative_evidence(
    rows: list[dict],
) -> list[dict]:

    """
    Extract only the quantitative metrics required by the
    previously defined Assignment Weighted Decision Matrix.

    No 1-5 scoring is performed here.
    """

    output = []

    for row in rows:

        output.append(
            {
                "algorithm":
                    row[
                        "algorithm"
                    ],

                # Criterion 1
                # Assignment Accuracy — weight 0.30
                "accuracy_primary_F1_combined":
                    row[
                        "F1_combined"
                    ],

                "accuracy_support_exact_combined":
                    row[
                        "exact_assignment_combined"
                    ],

                # Criterion 2
                # Processing Speed — weight 0.20
                "speed_primary_mean_runtime_ms":
                    row[
                        "runtime_combined_mean_ms"
                    ],

                "speed_support_median_runtime_ms":
                    row[
                        "runtime_combined_median_ms"
                    ],

                "speed_support_p95_runtime_ms":
                    row[
                        "runtime_combined_p95_ms"
                    ],

                # Criterion 3
                # Robustness — weight 0.20
                "robustness_primary_hard_F1":
                    row[
                        "F1_hard"
                    ],

                # Criterion 4
                # Handling Unmatched / Unequal Sets — 0.15
                "unmatched_primary_F1":
                    row[
                        "F1_unmatched"
                    ],

                "unmatched_support_FFMR":
                    row[
                        "FFMR_unmatched"
                    ],

                # Criterion 5
                # Implementation Complexity — 0.15
                # deliberately not numerically assigned here.
                "implementation_complexity":
                    "TO_BE_SCORED_IN_WDM",
            }
        )

    return output


# ============================================================
# Console summary
# ============================================================

def print_summary(
    rows: list[dict],
):

    print()
    print("=" * 112)

    print(
        "COMBINED ASSIGNMENT EXPERIMENTAL EVIDENCE"
    )

    print("=" * 112)

    print(
        f"{'Algorithm':<20}"
        f"{'F1 Syn':>10}"
        f"{'F1 Real':>10}"
        f"{'F1 Comb':>10}"
        f"{'Hard F1':>10}"
        f"{'Unmatch':>10}"
        f"{'Mean ms':>12}"
        f"{'Obj Agr':>10}"
    )

    print(
        "-" * 112
    )

    for row in rows:

        print(
            f"{row['algorithm']:<20}"
            f"{row['F1_synthetic']:>10.4f}"
            f"{row['F1_real']:>10.4f}"
            f"{row['F1_combined']:>10.4f}"
            f"{row['F1_hard']:>10.4f}"
            f"{row['F1_unmatched']:>10.4f}"
            f"{row['runtime_combined_mean_ms']:>12.4f}"
            f"{row['objective_agreement_real']:>10.4f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Combine Synthetic FINAL and "
            "TUM real Assignment evidence."
        )
    )

    parser.add_argument(
        "--synthetic-file",
        type=Path,
        default=
            DEFAULT_SYNTHETIC_FILE,
    )

    parser.add_argument(
        "--real-file",
        type=Path,
        default=
            DEFAULT_REAL_FILE,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=
            DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--real-weight",
        type=float,
        default=
            DEFAULT_REAL_WEIGHT,
    )

    parser.add_argument(
        "--synthetic-weight",
        type=float,
        default=
            DEFAULT_SYNTHETIC_WEIGHT,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate weighting policy
    # --------------------------------------------------------

    validate_weights(
        real_weight=
            args.real_weight,
        synthetic_weight=
            args.synthetic_weight,
    )

    # --------------------------------------------------------
    # Load experimental evidence
    # --------------------------------------------------------

    synthetic_rows = load_csv(
        args.synthetic_file
    )

    real_rows = load_csv(
        args.real_file
    )

    validate_synthetic(
        synthetic_rows
    )

    validate_real(
        real_rows
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = combine_results(
        synthetic_rows=
            synthetic_rows,

        real_rows=
            real_rows,

        real_weight=
            args.real_weight,

        synthetic_weight=
            args.synthetic_weight,
    )

    wdm_evidence = (
        build_wdm_quantitative_evidence(
            combined
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_file = (
        args.output_dir
        / "combined_evidence.csv"
    )

    wdm_file = (
        args.output_dir
        / "decision_matrix_quantitative_evidence.csv"
    )

    policy_file = (
        args.output_dir
        / "combination_policy.json"
    )

    save_csv(
        combined_file,
        combined,
    )

    save_csv(
        wdm_file,
        wdm_evidence,
    )

    policy = {
        "real_weight":
            args.real_weight,

        "synthetic_weight":
            args.synthetic_weight,

        "combination_formula":
            (
                "combined = "
                "real_weight * real "
                "+ synthetic_weight * synthetic"
            ),

        "criteria": {

            "assignment_accuracy": {
                "weight_in_wdm":
                    0.30,

                "primary_metric":
                    "F1_combined",

                "combination":
                    "70% TUM real + 30% Synthetic FINAL",
            },

            "processing_speed": {
                "weight_in_wdm":
                    0.20,

                "primary_metric":
                    "runtime_combined_mean_ms",

                "supporting_metrics": [
                    "runtime_combined_median_ms",
                    "runtime_combined_p95_ms",
                ],

                "combination":
                    "70% TUM real + 30% Synthetic FINAL",
            },

            "robustness": {
                "weight_in_wdm":
                    0.20,

                "primary_metric":
                    "F1_hard",

                "source":
                    "Synthetic hard/ambiguous cases only",

                "reason":
                    (
                        "The real sequence does not provide "
                        "a controlled paired clean/degraded test."
                    ),
            },

            "unmatched_unequal_handling": {
                "weight_in_wdm":
                    0.15,

                "primary_metric":
                    "F1_unmatched",

                "supporting_metric":
                    "FFMR_unmatched",

                "source":
                    (
                        "Synthetic unmatched/unequal-plane "
                        "cases with complete GT"
                    ),
            },

            "implementation_complexity": {
                "weight_in_wdm":
                    0.15,

                "source":
                    "qualitative implementation evidence",

                "status":
                    "not scored by this script",
            },
        },

        "real_reference_note":
            (
                "Unknown real-data correspondences are "
                "ignored during accuracy evaluation. "
                "unknown_selection_rate is diagnostic only "
                "and is not used as a WDM penalty."
            ),
    }

    with policy_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            policy,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print_summary(
        combined
    )

    print()
    print(
        "Combination policy:"
    )

    print(
        f"  Real      = "
        f"{args.real_weight:.2f}"
    )

    print(
        f"  Synthetic = "
        f"{args.synthetic_weight:.2f}"
    )

    print()

    print(
        "Files saved:"
    )

    print(
        f"  {combined_file}"
    )

    print(
        f"  {wdm_file}"
    )

    print(
        f"  {policy_file}"
    )


if __name__ == "__main__":
    main()