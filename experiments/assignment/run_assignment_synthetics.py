"""
run_assignment_synthetic.py

Run the four one-to-one plane correspondence Assignment algorithms on
the frozen Synthetic FINAL dataset.

Algorithms
----------
1. Hungarian
2. Auction
3. Jonker-Volgenant
4. Cost-Scaling

Experimental policy
-------------------
- Parameters are loaded from frozen_parameters.json.
- Synthetic FINAL is never used for parameter tuning.
- All algorithms receive exactly the same AssignmentCase inputs.
- All algorithms use the same evaluation functions.
- Results are reported overall and separately for:
    normal
    hard
    unmatched

Outputs
-------
results/assignment/synthetic_final/
    per_case_results.csv
    summary.csv
    decision_matrix_evidence.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

from assignment_common import (
    AssignmentCase,
    load_assignment_case,
)

from assignment_evaluation import (
    evaluate_assignment_result,
    aggregate_evaluation_rows,
)

from algorithms.hungarian_assignment import (
    run_hungarian_assignment,
)

from algorithms.auction_assignment import (
    run_auction_assignment,
)

from algorithms.jonker_volgenant_assignment import (
    run_jonker_volgenant_assignment,
)

from algorithms.cost_scaling_assignment import (
    run_cost_scaling_assignment,
)


# ============================================================
# Constants
# ============================================================

ALGORITHM_ORDER = [
    "hungarian",
    "auction",
    "jonker_volgenant",
    "cost_scaling",
]

SCENARIOS = [
    "normal",
    "hard",
    "unmatched",
]


# ============================================================
# Dataset loading
# ============================================================

def load_final_cases(
    data_dir: Path,
) -> list[AssignmentCase]:
    """
    Load all Synthetic FINAL Assignment cases.
    """

    files = sorted(
        data_dir.glob("*.npz")
    )

    if not files:
        raise FileNotFoundError(
            f"No .npz Assignment cases found in:\n"
            f"{data_dir}"
        )

    cases = [
        load_assignment_case(path)
        for path in files
    ]

    # --------------------------------------------------------
    # Safety check:
    # this runner is intended for FINAL only.
    # --------------------------------------------------------

    for case in cases:

        split = str(
            case.metadata.get(
                "split",
                "",
            )
        ).lower()

        if (
            split
            and split != "synthetic_final"
        ):
            raise ValueError(
                f"Case {case.case_id} belongs to split "
                f"'{split}', not synthetic_final."
            )

    print(
        f"Loaded {len(cases)} "
        f"Synthetic FINAL cases."
    )

    scenario_counts = {
        scenario: sum(
            1
            for case in cases
            if case.metadata.get(
                "scenario"
            ) == scenario
        )
        for scenario in SCENARIOS
    }

    for scenario in SCENARIOS:

        print(
            f"  {scenario:<10}: "
            f"{scenario_counts[scenario]}"
        )

    return cases


# ============================================================
# Frozen parameters
# ============================================================

def load_frozen_parameters(
    path: Path,
) -> dict:
    """
    Load parameters previously selected using Synthetic DEV.
    """

    if not path.exists():
        raise FileNotFoundError(
            "Frozen parameter file not found:\n"
            f"{path}\n\n"
            "Run tune_assignment.py first."
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        parameters = json.load(f)

    required = [
        "shared_unmatched_cost",
        "auction_epsilon",
        "invalid_cost",
    ]

    missing = [
        key
        for key in required
        if key not in parameters
    ]

    if missing:
        raise KeyError(
            "Missing frozen parameters: "
            + ", ".join(missing)
        )

    print(
        "\nFrozen Assignment parameters:"
    )

    print(
        "  unmatched_cost = "
        f"{parameters['shared_unmatched_cost']}"
    )

    print(
        "  auction_epsilon = "
        f"{parameters['auction_epsilon']}"
    )

    print(
        "  invalid_cost = "
        f"{parameters['invalid_cost']}"
    )

    return parameters


# ============================================================
# Algorithm wrappers
# ============================================================

def build_algorithm_runners(
    parameters: dict,
) -> dict[str, Callable]:
    """
    Construct frozen algorithm runners.

    Every algorithm uses the same unmatched_cost and invalid_cost.
    """

    unmatched_cost = float(
        parameters[
            "shared_unmatched_cost"
        ]
    )

    invalid_cost = float(
        parameters[
            "invalid_cost"
        ]
    )

    auction_epsilon = float(
        parameters[
            "auction_epsilon"
        ]
    )

    cost_scale = int(
        parameters.get(
            "cost_scaling_cost_scale",
            1_000_000,
        )
    )

    epsilon_factor = float(
        parameters.get(
            "cost_scaling_epsilon_factor",
            4.0,
        )
    )

    def run_hungarian(case):

        return run_hungarian_assignment(
            case=case,
            unmatched_cost=unmatched_cost,
            invalid_cost=invalid_cost,
        )

    def run_auction(case):

        return run_auction_assignment(
            case=case,
            unmatched_cost=unmatched_cost,
            epsilon=auction_epsilon,
            invalid_cost=invalid_cost,
        )

    def run_jv(case):

        return (
            run_jonker_volgenant_assignment(
                case=case,
                unmatched_cost=unmatched_cost,
                invalid_cost=invalid_cost,
            )
        )

    def run_cost_scaling(case):

        return run_cost_scaling_assignment(
            case=case,
            unmatched_cost=unmatched_cost,
            invalid_cost=invalid_cost,
            cost_scale=cost_scale,
            epsilon_factor=epsilon_factor,
        )

    return {
        "hungarian":
            run_hungarian,

        "auction":
            run_auction,

        "jonker_volgenant":
            run_jv,

        "cost_scaling":
            run_cost_scaling,
    }


# ============================================================
# Run experiment
# ============================================================

def run_final_experiment(
    cases: list[AssignmentCase],
    runners: dict[str, Callable],
) -> list[dict]:
    """
    Run all algorithms on all Synthetic FINAL cases.

    Any algorithm failure raises immediately rather than silently
    removing the failed case from the statistics.
    """

    all_rows: list[dict] = []

    total_runs = (
        len(cases)
        * len(ALGORITHM_ORDER)
    )

    completed = 0

    for algorithm_name in ALGORITHM_ORDER:

        print(
            "\n========================================"
        )

        print(
            f"Running {algorithm_name}"
        )

        print(
            "========================================"
        )

        runner = runners[
            algorithm_name
        ]

        for index, case in enumerate(
            cases,
            start=1,
        ):

            try:

                result = runner(
                    case
                )

            except Exception as exc:

                raise RuntimeError(
                    f"\nAssignment algorithm failed.\n"
                    f"Algorithm: {algorithm_name}\n"
                    f"Case:      {case.case_id}\n"
                    f"Scenario:  "
                    f"{case.metadata.get('scenario')}\n"
                    f"Original error: {exc}"
                ) from exc

            metrics = (
                evaluate_assignment_result(
                    case,
                    result,
                )
            )

            # ------------------------------------------------
            # Add solver metadata useful for debugging.
            # ------------------------------------------------

            metrics[
                "total_augmented_cost"
            ] = result.metadata.get(
                "total_augmented_cost",
                "",
            )

            metrics[
                "real_match_cost"
            ] = result.metadata.get(
                "real_match_cost",
                "",
            )

            all_rows.append(
                metrics
            )

            completed += 1

            # Progress output without printing every case.
            if (
                index % 50 == 0
                or index == len(cases)
            ):

                print(
                    f"  {index:>4}/"
                    f"{len(cases)} cases"
                )

    print(
        "\nCompleted "
        f"{completed}/{total_runs} "
        "algorithm-case runs."
    )

    return all_rows


# ============================================================
# Summaries
# ============================================================

def build_summary(
    all_rows: list[dict],
) -> list[dict]:
    """
    Produce overall and scenario-specific summaries.
    """

    summary_rows: list[dict] = []

    for algorithm in ALGORITHM_ORDER:

        algorithm_rows = [
            row
            for row in all_rows
            if row["algorithm"]
            == algorithm
        ]

        if not algorithm_rows:
            continue

        # ----------------------------------------------------
        # Overall
        # ----------------------------------------------------

        overall = (
            aggregate_evaluation_rows(
                algorithm_rows
            )
        )

        summary_rows.append(
            {
                "algorithm":
                    algorithm,

                "scope":
                    "overall",

                **overall,
            }
        )

        # ----------------------------------------------------
        # Per scenario
        # ----------------------------------------------------

        for scenario in SCENARIOS:

            scenario_rows = [
                row
                for row in algorithm_rows
                if row["scenario"]
                == scenario
            ]

            if not scenario_rows:
                continue

            summary = (
                aggregate_evaluation_rows(
                    scenario_rows
                )
            )

            summary_rows.append(
                {
                    "algorithm":
                        algorithm,

                    "scope":
                        scenario,

                    **summary,
                }
            )

    return summary_rows


# ============================================================
# Decision-matrix evidence
# ============================================================

def _find_summary(
    summary_rows: list[dict],
    algorithm: str,
    scope: str,
) -> dict:
    """
    Find one summary row.
    """

    for row in summary_rows:

        if (
            row["algorithm"]
            == algorithm

            and row["scope"]
            == scope
        ):
            return row

    raise KeyError(
        f"Summary not found: "
        f"{algorithm}, {scope}"
    )


def build_decision_matrix_evidence(
    summary_rows: list[dict],
) -> list[dict]:
    """
    Extract the principal Synthetic FINAL evidence required later by
    the Assignment weighted decision matrix.

    Metrics
    -------
    F1:
        overall Assignment accuracy.

    F1_hard:
        correspondence F1 on hard / ambiguous cases.

    F1_unmatched:
        unmatched-plane F1 on unmatched / unequal cases.

    t_mean:
        overall mean solver runtime.

    Supporting:
        exact assignment rate and FFMR.
    """

    evidence_rows: list[dict] = []

    for algorithm in ALGORITHM_ORDER:

        overall = _find_summary(
            summary_rows,
            algorithm,
            "overall",
        )

        hard = _find_summary(
            summary_rows,
            algorithm,
            "hard",
        )

        unmatched = _find_summary(
            summary_rows,
            algorithm,
            "unmatched",
        )

        evidence_rows.append(
            {
                "algorithm":
                    algorithm,

                "F1_overall":
                    overall[
                        "f1_mean"
                    ],

                "F1_hard":
                    hard[
                        "f1_mean"
                    ],

                "F1_unmatched":
                    unmatched[
                        "unmatched_f1_mean"
                    ],

                "runtime_mean_ms":
                    overall[
                        "runtime_mean_ms"
                    ],

                "runtime_median_ms":
                    overall[
                        "runtime_median_ms"
                    ],

                "runtime_p95_ms":
                    overall[
                        "runtime_p95_ms"
                    ],

                "exact_assignment_rate":
                    overall[
                        "exact_assignment_rate"
                    ],

                "FFMR_unmatched":
                    unmatched[
                        "ffmr_mean"
                    ],
            }
        )

    return evidence_rows


# ============================================================
# CSV output
# ============================================================

def save_csv(
    path: Path,
    rows: list[dict],
) -> None:
    """
    Save dictionaries to CSV.
    """

    if not rows:
        raise ValueError(
            f"No rows to save to {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Use the union of all keys to avoid problems if rows contain
    # slightly different metadata.
    fieldnames: list[str] = []

    seen = set()

    for row in rows:

        for key in row.keys():

            if key not in seen:

                fieldnames.append(key)

                seen.add(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Console report
# ============================================================

def print_evidence_table(
    evidence_rows: list[dict],
) -> None:
    """
    Print compact final results.
    """

    print(
        "\n"
        "=============================================================="
    )

    print(
        "SYNTHETIC FINAL — ASSIGNMENT EVIDENCE"
    )

    print(
        "=============================================================="
    )

    print(
        f"{'Algorithm':<20}"
        f"{'F1':>8}"
        f"{'Hard':>8}"
        f"{'Unmatch':>10}"
        f"{'Mean ms':>10}"
        f"{'Exact':>9}"
        f"{'FFMR':>8}"
    )

    print(
        "-" * 73
    )

    for row in evidence_rows:

        print(
            f"{row['algorithm']:<20}"
            f"{row['F1_overall']:>8.4f}"
            f"{row['F1_hard']:>8.4f}"
            f"{row['F1_unmatched']:>10.4f}"
            f"{row['runtime_mean_ms']:>10.4f}"
            f"{row['exact_assignment_rate']:>9.4f}"
            f"{row['FFMR_unmatched']:>8.4f}"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run frozen Assignment algorithms "
            "on Synthetic FINAL."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "data/assignment/"
            "synthetic_final"
        ),
    )

    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path(
            "results/assignment/"
            "tuning/"
            "frozen_parameters.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/assignment/"
            "synthetic_final"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load frozen experiment configuration
    # --------------------------------------------------------

    parameters = (
        load_frozen_parameters(
            args.parameters
        )
    )

    cases = load_final_cases(
        args.data_dir
    )

    runners = (
        build_algorithm_runners(
            parameters
        )
    )

    # --------------------------------------------------------
    # Run FINAL experiments
    # --------------------------------------------------------

    per_case_rows = (
        run_final_experiment(
            cases=cases,
            runners=runners,
        )
    )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    summary_rows = (
        build_summary(
            per_case_rows
        )
    )

    evidence_rows = (
        build_decision_matrix_evidence(
            summary_rows
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
        / "per_case_results.csv",

        per_case_rows,
    )

    save_csv(
        args.output_dir
        / "summary.csv",

        summary_rows,
    )

    save_csv(
        args.output_dir
        / "decision_matrix_evidence.csv",

        evidence_rows,
    )

    # Copy the frozen configuration used for this experiment
    # into the output folder for reproducibility.

    with (
        args.output_dir
        / "parameters_used.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            parameters,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print_evidence_table(
        evidence_rows
    )

    print(
        "\nSaved:"
    )

    print(
        "  "
        f"{args.output_dir / 'per_case_results.csv'}"
    )

    print(
        "  "
        f"{args.output_dir / 'summary.csv'}"
    )

    print(
        "  "
        f"{args.output_dir / 'decision_matrix_evidence.csv'}"
    )

    print(
        "  "
        f"{args.output_dir / 'parameters_used.json'}"
    )

    print(
        "\nSynthetic FINAL evaluation complete."
    )


if __name__ == "__main__":
    main()