"""
analyse_assignment_real.py

Analyse the frozen real-data Assignment experiment on
TUM fr1_xyz + ETH3D-derived persistent plane identities.

This script DOES NOT rerun or retune any Assignment algorithm.

Inputs
------
results/assignment/tum/per_case_results.csv
results/assignment/tum/summary.csv

Outputs
-------
results/assignment/tum/analysis/
    overall_comparison.csv
    runtime_comparison.csv
    solver_agreement.csv
    case_level_summary.csv
    real_analysis_summary.csv

Interpretation
--------------
Real-data GT is incomplete:

    gt_label =  1 : known true correspondence
    gt_label =  0 : known false correspondence
    gt_label = -1 : unknown / ignore

Therefore:
- precision / recall / F1 are calculated only from evaluable GT;
- unknown_selection_rate is a reference-coverage diagnostic,
  NOT an error metric.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


# ============================================================
# Configuration
# ============================================================

ALGORITHMS = [
    "hungarian",
    "auction",
    "jonker_volgenant",
    "cost_scaling",
]


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "assignment"
    / "tum"
)

DEFAULT_OUTPUT_DIR = (
    DEFAULT_RESULT_DIR
    / "analysis"
)


# ============================================================
# CSV loading
# ============================================================

def load_csv(path: Path) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found:\n{path}"
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


# ============================================================
# Helpers
# ============================================================

def f(
    row: dict,
    key: str,
    default: float = 0.0,
) -> float:

    value = row.get(
        key,
        "",
    )

    if value in (
        "",
        None,
    ):
        return default

    return float(value)


def i(
    row: dict,
    key: str,
    default: int = 0,
) -> int:

    value = row.get(
        key,
        "",
    )

    if value in (
        "",
        None,
    ):
        return default

    return int(
        float(value)
    )


def mean(values):

    values = list(values)

    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


def median(values):

    values = list(values)

    if not values:
        return 0.0

    return float(
        np.median(values)
    )


def std(values):

    values = list(values)

    if not values:
        return 0.0

    return float(
        np.std(values)
    )


def p95(values):

    values = list(values)

    if not values:
        return 0.0

    return float(
        np.percentile(
            values,
            95,
        )
    )


# ============================================================
# Validation
# ============================================================

def validate_per_case_rows(
    rows: list[dict],
):

    required = {
        "algorithm",
        "case_id",
        "frame_current",
        "frame_reference",
        "candidate_edges",
        "reachable_gt_matches",
        "selected_matches",
        "selected_known_positive",
        "selected_known_negative",
        "selected_unknown",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "exact_evaluable_assignment",
        "runtime_ms",
        "total_augmented_cost",
    }

    missing = (
        required
        - set(rows[0].keys())
    )

    if missing:

        raise ValueError(
            "per_case_results.csv "
            "missing columns:\n"
            f"{sorted(missing)}"
        )

    algorithms_present = {
        row["algorithm"]
        for row in rows
    }

    missing_algorithms = [
        algorithm
        for algorithm in ALGORITHMS
        if algorithm
        not in algorithms_present
    ]

    if missing_algorithms:

        raise ValueError(
            "Missing Assignment algorithms: "
            f"{missing_algorithms}"
        )


# ============================================================
# Overall comparison
# ============================================================

def build_overall_comparison(
    rows: list[dict],
) -> list[dict]:

    output = []

    for algorithm in ALGORITHMS:

        selected = [
            row
            for row in rows
            if row["algorithm"]
            == algorithm
        ]

        if not selected:
            continue

        tp = sum(
            i(row, "tp")
            for row in selected
        )

        fp = sum(
            i(row, "fp")
            for row in selected
        )

        fn = sum(
            i(row, "fn")
            for row in selected
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        f1_value = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
            if precision + recall > 0
            else 0.0
        )

        runtime = [
            f(row, "runtime_ms")
            for row in selected
        ]

        case_f1 = [
            f(row, "f1")
            for row in selected
        ]

        exact = [
            f(
                row,
                "exact_evaluable_assignment",
            )
            for row in selected
        ]

        selected_matches = sum(
            i(
                row,
                "selected_matches",
            )
            for row in selected
        )

        selected_positive = sum(
            i(
                row,
                "selected_known_positive",
            )
            for row in selected
        )

        selected_negative = sum(
            i(
                row,
                "selected_known_negative",
            )
            for row in selected
        )

        selected_unknown = sum(
            i(
                row,
                "selected_unknown",
            )
            for row in selected
        )

        unknown_rate = (
            selected_unknown
            / selected_matches
            if selected_matches > 0
            else 0.0
        )

        evaluable_rate = (
            (
                selected_positive
                + selected_negative
            )
            / selected_matches
            if selected_matches > 0
            else 0.0
        )

        output.append(
            {
                "algorithm":
                    algorithm,

                "n_cases":
                    len(selected),

                "tp":
                    tp,

                "fp":
                    fp,

                "fn":
                    fn,

                "precision":
                    precision,

                "recall":
                    recall,

                "f1":
                    f1_value,

                "mean_case_f1":
                    mean(case_f1),

                "case_f1_std":
                    std(case_f1),

                "exact_evaluable_assignment_rate":
                    mean(exact),

                "selected_matches":
                    selected_matches,

                "selected_known_positive":
                    selected_positive,

                "selected_known_negative":
                    selected_negative,

                "selected_unknown":
                    selected_unknown,

                "unknown_selection_rate":
                    unknown_rate,

                "evaluable_selection_rate":
                    evaluable_rate,

                "runtime_mean_ms":
                    mean(runtime),

                "runtime_median_ms":
                    median(runtime),

                "runtime_p95_ms":
                    p95(runtime),

                "runtime_std_ms":
                    std(runtime),
            }
        )

    return output


# ============================================================
# Runtime comparison
# ============================================================

def build_runtime_comparison(
    rows: list[dict],
) -> list[dict]:

    output = []

    for algorithm in ALGORITHMS:

        selected = [
            row
            for row in rows
            if row["algorithm"]
            == algorithm
        ]

        if not selected:
            continue

        runtime = np.asarray(
            [
                f(
                    row,
                    "runtime_ms",
                )
                for row in selected
            ],
            dtype=float,
        )

        output.append(
            {
                "algorithm":
                    algorithm,

                "n_cases":
                    len(runtime),

                "runtime_mean_ms":
                    float(
                        np.mean(runtime)
                    ),

                "runtime_median_ms":
                    float(
                        np.median(runtime)
                    ),

                "runtime_p95_ms":
                    float(
                        np.percentile(
                            runtime,
                            95,
                        )
                    ),

                "runtime_std_ms":
                    float(
                        np.std(runtime)
                    ),

                "runtime_min_ms":
                    float(
                        np.min(runtime)
                    ),

                "runtime_max_ms":
                    float(
                        np.max(runtime)
                    ),
            }
        )

    return output


# ============================================================
# Solver agreement
# ============================================================

def build_solver_agreement(
    rows: list[dict],
    tolerance: float,
) -> list[dict]:

    by_case = defaultdict(dict)

    for row in rows:

        by_case[
            row["case_id"]
        ][
            row["algorithm"]
        ] = row

    output = []

    reference_algorithm = (
        "hungarian"
    )

    for case_id in sorted(
        by_case.keys()
    ):

        algorithms = (
            by_case[case_id]
        )

        if reference_algorithm not in algorithms:
            continue

        ref = algorithms[
            reference_algorithm
        ]

        ref_cost = f(
            ref,
            "total_augmented_cost",
        )

        ref_f1 = f(
            ref,
            "f1",
        )

        ref_tp = i(
            ref,
            "tp",
        )

        ref_fp = i(
            ref,
            "fp",
        )

        ref_fn = i(
            ref,
            "fn",
        )

        for algorithm in ALGORITHMS:

            if algorithm == (
                reference_algorithm
            ):
                continue

            if algorithm not in algorithms:
                continue

            row = algorithms[
                algorithm
            ]

            cost = f(
                row,
                "total_augmented_cost",
            )

            cost_gap = (
                cost
                - ref_cost
            )

            f1_gap = (
                f(row, "f1")
                - ref_f1
            )

            metric_agreement = int(
                (
                    i(row, "tp")
                    == ref_tp
                )
                and
                (
                    i(row, "fp")
                    == ref_fp
                )
                and
                (
                    i(row, "fn")
                    == ref_fn
                )
            )

            objective_agreement = int(
                abs(cost_gap)
                <= tolerance
            )

            output.append(
                {
                    "case_id":
                        case_id,

                    "frame_current":
                        ref[
                            "frame_current"
                        ],

                    "frame_reference":
                        ref[
                            "frame_reference"
                        ],

                    "reference_algorithm":
                        reference_algorithm,

                    "compared_algorithm":
                        algorithm,

                    "reference_cost":
                        ref_cost,

                    "compared_cost":
                        cost,

                    "absolute_cost_gap":
                        abs(
                            cost_gap
                        ),

                    "objective_agreement":
                        objective_agreement,

                    "reference_f1":
                        ref_f1,

                    "compared_f1":
                        f(
                            row,
                            "f1",
                        ),

                    "absolute_f1_gap":
                        abs(
                            f1_gap
                        ),

                    "tp_fp_fn_agreement":
                        metric_agreement,
                }
            )

    return output


# ============================================================
# Case-level analysis
# ============================================================

def build_case_level_summary(
    rows: list[dict],
) -> list[dict]:

    by_case = defaultdict(list)

    for row in rows:

        by_case[
            row["case_id"]
        ].append(row)

    output = []

    for case_id in sorted(
        by_case.keys()
    ):

        group = by_case[
            case_id
        ]

        reference = group[0]

        f1_values = [
            f(row, "f1")
            for row in group
        ]

        exact_values = [
            f(
                row,
                "exact_evaluable_assignment",
            )
            for row in group
        ]

        unknown_rates = [
            f(
                row,
                "unknown_selection_rate",
            )
            for row in group
        ]

        costs = [
            f(
                row,
                "total_augmented_cost",
            )
            for row in group
        ]

        output.append(
            {
                "case_id":
                    case_id,

                "frame_current":
                    reference[
                        "frame_current"
                    ],

                "frame_reference":
                    reference[
                        "frame_reference"
                    ],

                "n_current":
                    reference[
                        "n_current"
                    ],

                "n_reference":
                    reference[
                        "n_reference"
                    ],

                "candidate_edges":
                    reference[
                        "candidate_edges"
                    ],

                "reachable_gt_matches":
                    reference[
                        "reachable_gt_matches"
                    ],

                "mean_f1":
                    mean(
                        f1_values
                    ),

                "min_f1":
                    float(
                        np.min(
                            f1_values
                        )
                    ),

                "max_f1":
                    float(
                        np.max(
                            f1_values
                        )
                    ),

                "f1_solver_spread":
                    float(
                        np.max(
                            f1_values
                        )
                        -
                        np.min(
                            f1_values
                        )
                    ),

                "exact_all_solvers":
                    int(
                        all(
                            value == 1.0
                            for value
                            in exact_values
                        )
                    ),

                "mean_unknown_selection_rate":
                    mean(
                        unknown_rates
                    ),

                "objective_solver_spread":
                    float(
                        np.max(
                            costs
                        )
                        -
                        np.min(
                            costs
                        )
                    ),
            }
        )

    return output


# ============================================================
# Compact report-ready summary
# ============================================================

def build_real_analysis_summary(
    overall_rows: list[dict],
    agreement_rows: list[dict],
) -> list[dict]:

    output = []

    for row in overall_rows:

        algorithm = row[
            "algorithm"
        ]

        relevant_agreements = [
            agreement
            for agreement
            in agreement_rows
            if (
                agreement[
                    "compared_algorithm"
                ]
                == algorithm
            )
        ]

        if algorithm == "hungarian":

            objective_agreement_rate = 1.0

            metric_agreement_rate = 1.0

        elif relevant_agreements:

            objective_agreement_rate = mean(
                int(
                    agreement[
                        "objective_agreement"
                    ]
                )
                for agreement
                in relevant_agreements
            )

            metric_agreement_rate = mean(
                int(
                    agreement[
                        "tp_fp_fn_agreement"
                    ]
                )
                for agreement
                in relevant_agreements
            )

        else:

            objective_agreement_rate = 0.0
            metric_agreement_rate = 0.0

        output.append(
            {
                "algorithm":
                    algorithm,

                "F1_real":
                    row["f1"],

                "precision_real":
                    row["precision"],

                "recall_real":
                    row["recall"],

                "exact_assignment_rate":
                    row[
                        "exact_evaluable_assignment_rate"
                    ],

                "unknown_selection_rate":
                    row[
                        "unknown_selection_rate"
                    ],

                "runtime_mean_ms":
                    row[
                        "runtime_mean_ms"
                    ],

                "runtime_median_ms":
                    row[
                        "runtime_median_ms"
                    ],

                "runtime_p95_ms":
                    row[
                        "runtime_p95_ms"
                    ],

                "objective_agreement_with_hungarian":
                    objective_agreement_rate,

                "metric_agreement_with_hungarian":
                    metric_agreement_rate,
            }
        )

    return output


# ============================================================
# CSV saving
# ============================================================

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

        for key in row:

            if key not in seen:

                seen.add(key)

                fieldnames.append(
                    key
                )

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
# Console summary
# ============================================================

def print_summary(
    rows: list[dict],
):

    print()
    print("=" * 100)

    print(
        "REAL ASSIGNMENT ANALYSIS SUMMARY"
    )

    print("=" * 100)

    print(
        f"{'Algorithm':<20}"
        f"{'P':>9}"
        f"{'R':>9}"
        f"{'F1':>9}"
        f"{'Exact':>9}"
        f"{'Unknown':>10}"
        f"{'Mean ms':>12}"
        f"{'Obj agree':>12}"
    )

    print("-" * 100)

    for row in rows:

        print(
            f"{row['algorithm']:<20}"
            f"{row['precision_real']:>9.4f}"
            f"{row['recall_real']:>9.4f}"
            f"{row['F1_real']:>9.4f}"
            f"{row['exact_assignment_rate']:>9.4f}"
            f"{row['unknown_selection_rate']:>10.4f}"
            f"{row['runtime_mean_ms']:>12.4f}"
            f"{row['objective_agreement_with_hungarian']:>12.4f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyse frozen real-data "
            "Assignment experiment."
        )
    )

    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--cost-tolerance",
        type=float,
        default=1e-6,
    )

    args = parser.parse_args()

    per_case_file = (
        args.result_dir
        / "per_case_results.csv"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    rows = load_csv(
        per_case_file
    )

    validate_per_case_rows(
        rows
    )

    print(
        f"Loaded {len(rows)} "
        f"algorithm-case results."
    )

    # --------------------------------------------------------
    # Analysis
    # --------------------------------------------------------

    overall = (
        build_overall_comparison(
            rows
        )
    )

    runtime = (
        build_runtime_comparison(
            rows
        )
    )

    agreement = (
        build_solver_agreement(
            rows,
            tolerance=
                args.cost_tolerance,
        )
    )

    case_summary = (
        build_case_level_summary(
            rows
        )
    )

    report_summary = (
        build_real_analysis_summary(
            overall,
            agreement,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_csv(
        args.output_dir
        / "overall_comparison.csv",
        overall,
    )

    save_csv(
        args.output_dir
        / "runtime_comparison.csv",
        runtime,
    )

    save_csv(
        args.output_dir
        / "solver_agreement.csv",
        agreement,
    )

    save_csv(
        args.output_dir
        / "case_level_summary.csv",
        case_summary,
    )

    save_csv(
        args.output_dir
        / "real_analysis_summary.csv",
        report_summary,
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print_summary(
        report_summary
    )

    print()

    print(
        "Important:"
    )

    print(
        "  unknown_selection_rate is "
        "NOT an error rate."
    )

    print(
        "  It represents matches whose "
        "GT identity is unavailable."
    )

    print()

    print(
        "Analysis saved to:"
    )

    print(
        f"  {args.output_dir}"
    )


if __name__ == "__main__":
    main()