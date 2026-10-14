"""
tune_assignment.py

Parameter tuning for the one-to-one plane correspondence Assignment
stage.

Tuning policy
-------------
1. A single shared unmatched_cost is selected using Synthetic DEV.
   Hungarian is used as the exact reference solver for this stage.

2. After unmatched_cost is frozen, Auction epsilon is tuned separately.

3. Synthetic FINAL must NOT be used during tuning.

Outputs
-------
results/assignment/tuning/
    unmatched_cost_sweep.csv
    auction_epsilon_sweep.csv
    frozen_parameters.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from assignment_common import (
    load_assignment_case,
)

from assignment_evaluation import (
    evaluate_assignment_result,
)

from algorithms.hungarian_assignment import (
    run_hungarian_assignment,
)

from algorithms.auction_assignment import (
    run_auction_assignment,
)


# ============================================================
# Default tuning grids
# ============================================================

DEFAULT_UNMATCHED_COSTS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
]

DEFAULT_AUCTION_EPSILONS = [
    1e-2,
    5e-3,
    1e-3,
    5e-4,
    1e-4,
    5e-5,
    1e-5,
]


# ============================================================
# Dataset loading
# ============================================================

def load_dev_cases(
    data_dir: Path,
) -> list:
    """
    Load all Synthetic DEV AssignmentCase files.
    """

    files = sorted(
        data_dir.glob("*.npz")
    )

    if not files:
        raise FileNotFoundError(
            f"No NPZ cases found in: {data_dir}"
        )

    cases = [
        load_assignment_case(path)
        for path in files
    ]

    print(
        f"Loaded {len(cases)} Synthetic DEV cases "
        f"from {data_dir}"
    )

    return cases


# ============================================================
# Metric aggregation
# ============================================================

def _mean(
    values: list[float],
) -> float:
    """
    Safe arithmetic mean.
    """

    if not values:
        return 0.0

    return float(
        np.mean(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def summarise_rows(
    rows: list[dict],
) -> dict[str, float]:
    """
    Summarise per-case evaluation results.

    The main quantities used for parameter selection are:

        overall F1
        hard-case F1
        unmatched-case unmatched F1

    Runtime is recorded as supporting evidence.
    """

    if not rows:
        raise ValueError(
            "Cannot summarise empty evaluation rows"
        )

    all_f1 = [
        float(row["f1"])
        for row in rows
    ]

    hard_f1 = [
        float(row["f1"])
        for row in rows
        if row["scenario"] == "hard"
    ]

    unmatched_correspondence_f1 = [
        float(row["f1"])
        for row in rows
        if row["scenario"] == "unmatched"
    ]

    unmatched_plane_f1 = [
        float(row["unmatched_f1"])
        for row in rows
        if row["scenario"] == "unmatched"
    ]

    normal_f1 = [
        float(row["f1"])
        for row in rows
        if row["scenario"] == "normal"
    ]

    runtime = [
        float(row["runtime_ms"])
        for row in rows
    ]

    exact = [
        float(row["exact_assignment"])
        for row in rows
    ]

    ffmr_unmatched = [
        float(row["ffmr"])
        for row in rows
        if row["scenario"] == "unmatched"
    ]

    return {
        "n_cases":
            len(rows),

        "f1_mean":
            _mean(all_f1),

        "normal_f1_mean":
            _mean(normal_f1),

        "hard_f1_mean":
            _mean(hard_f1),

        "unmatched_case_f1_mean":
            _mean(
                unmatched_correspondence_f1
            ),

        "unmatched_f1_mean":
            _mean(
                unmatched_plane_f1
            ),

        "exact_assignment_rate":
            _mean(exact),

        "ffmr_mean":
            _mean(
                ffmr_unmatched
            ),

        "runtime_mean_ms":
            _mean(runtime),

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
    }


# ============================================================
# Tuning score
# ============================================================

def unmatched_cost_selection_score(
    summary: dict[str, float],
) -> float:
    """
    Score used only to select the shared unmatched_cost.

    The relative weights follow the experimentally relevant parts of
    the Assignment decision criteria:

        Assignment Accuracy       0.30
        Robustness                0.20
        Unmatched Handling        0.15

    These are normalised to sum to one for parameter tuning.

    Runtime is intentionally excluded because unmatched_cost controls
    matching behaviour rather than solver efficiency.
    """

    w_accuracy = 0.30
    w_hard = 0.20
    w_unmatched = 0.15

    total = (
        w_accuracy
        + w_hard
        + w_unmatched
    )

    score = (
        w_accuracy
        * summary["f1_mean"]

        + w_hard
        * summary["hard_f1_mean"]

        + w_unmatched
        * summary["unmatched_f1_mean"]
    ) / total

    return float(score)


# ============================================================
# Shared unmatched-cost sweep
# ============================================================

def tune_unmatched_cost(
    cases: list,
    unmatched_costs: list[float],
    invalid_cost: float,
) -> tuple[
    float,
    list[dict],
]:
    """
    Tune the common per-plane unmatched cost.

    Hungarian is used as the exact reference solver.

    Important
    ---------
    In the augmented formulation, leaving a current plane and a
    reference plane unmatched generally incurs TWO unmatched penalties.

    Therefore unmatched_cost is a per-plane penalty, not a direct
    similarity threshold.
    """

    sweep_rows: list[dict] = []

    print(
        "\n========================================"
    )
    print(
        "Tuning shared unmatched_cost"
    )
    print(
        "========================================"
    )

    for unmatched_cost in unmatched_costs:

        evaluation_rows = []

        for case in cases:

            result = (
                run_hungarian_assignment(
                    case=case,
                    unmatched_cost=unmatched_cost,
                    invalid_cost=invalid_cost,
                )
            )

            metrics = (
                evaluate_assignment_result(
                    case,
                    result,
                )
            )

            evaluation_rows.append(
                metrics
            )

        summary = summarise_rows(
            evaluation_rows
        )

        selection_score = (
            unmatched_cost_selection_score(
                summary
            )
        )

        row = {
            "unmatched_cost":
                float(unmatched_cost),

            "selection_score":
                selection_score,

            **summary,
        }

        sweep_rows.append(row)

        print(
            f"u={unmatched_cost:>5.2f} | "
            f"Score={selection_score:.4f} | "
            f"F1={summary['f1_mean']:.4f} | "
            f"Hard={summary['hard_f1_mean']:.4f} | "
            f"Unmatched={summary['unmatched_f1_mean']:.4f} | "
            f"FFMR={summary['ffmr_mean']:.4f}"
        )

    # --------------------------------------------------------
    # Select best unmatched cost
    #
    # Tie-breaking:
    #   1. highest tuning score
    #   2. highest unmatched F1
    #   3. highest overall F1
    #   4. lower FFMR
    #   5. smaller unmatched cost
    # --------------------------------------------------------

    best_row = max(
        sweep_rows,
        key=lambda row: (
            row["selection_score"],
            row["unmatched_f1_mean"],
            row["f1_mean"],
            -row["ffmr_mean"],
            -row["unmatched_cost"],
        ),
    )

    best_cost = float(
        best_row["unmatched_cost"]
    )

    print(
        "\nSelected shared unmatched_cost:"
    )

    print(
        f"  {best_cost:.6f}"
    )

    return (
        best_cost,
        sweep_rows,
    )


# ============================================================
# Auction epsilon sweep
# ============================================================

def tune_auction_epsilon(
    cases: list,
    unmatched_cost: float,
    epsilons: list[float],
    invalid_cost: float,
) -> tuple[
    float,
    list[dict],
]:
    """
    Tune Auction epsilon after unmatched_cost has been frozen.

    Selection strategy
    ------------------
    1. Find the best DEV correspondence F1 achieved by any epsilon.
    2. Retain epsilon values within 0.001 absolute F1 of that best value.
    3. Among those, prefer the lowest mean runtime.
    4. Use unmatched F1 and exact assignment rate as tie-breakers.

    This avoids selecting an unnecessarily small epsilon when it gives
    negligible accuracy improvement but substantially higher runtime.
    """

    sweep_rows: list[dict] = []

    print(
        "\n========================================"
    )
    print(
        "Tuning Auction epsilon"
    )
    print(
        "========================================"
    )

    for epsilon in epsilons:

        evaluation_rows = []

        failures = 0

        for case in cases:

            try:

                result = (
                    run_auction_assignment(
                        case=case,
                        unmatched_cost=
                            unmatched_cost,

                        epsilon=
                            epsilon,

                        invalid_cost=
                            invalid_cost,
                    )
                )

                metrics = (
                    evaluate_assignment_result(
                        case,
                        result,
                    )
                )

                evaluation_rows.append(
                    metrics
                )

            except RuntimeError as exc:

                failures += 1

                print(
                    f"WARNING: Auction failed on "
                    f"{case.case_id} "
                    f"with epsilon={epsilon}: {exc}"
                )

        if not evaluation_rows:

            summary = {
                "n_cases": 0,
                "f1_mean": 0.0,
                "normal_f1_mean": 0.0,
                "hard_f1_mean": 0.0,
                "unmatched_case_f1_mean": 0.0,
                "unmatched_f1_mean": 0.0,
                "exact_assignment_rate": 0.0,
                "ffmr_mean": 1.0,
                "runtime_mean_ms": float("inf"),
                "runtime_median_ms": float("inf"),
                "runtime_p95_ms": float("inf"),
            }

        else:

            summary = summarise_rows(
                evaluation_rows
            )

        row = {
            "epsilon":
                float(epsilon),

            "failures":
                int(failures),

            **summary,
        }

        sweep_rows.append(
            row
        )

        print(
            f"eps={epsilon:.1e} | "
            f"F1={summary['f1_mean']:.4f} | "
            f"Hard={summary['hard_f1_mean']:.4f} | "
            f"Unmatched={summary['unmatched_f1_mean']:.4f} | "
            f"Runtime={summary['runtime_mean_ms']:.4f} ms | "
            f"Failures={failures}"
        )

    # --------------------------------------------------------
    # Ignore settings that failed on any DEV case.
    # --------------------------------------------------------

    valid_rows = [
        row
        for row in sweep_rows
        if (
            row["failures"] == 0
            and np.isfinite(
                row["runtime_mean_ms"]
            )
        )
    ]

    if not valid_rows:
        raise RuntimeError(
            "No Auction epsilon completed all DEV cases"
        )

    best_f1 = max(
        row["f1_mean"]
        for row in valid_rows
    )

    # 0.001 absolute F1 tolerance.
    accuracy_tolerance = 0.001

    near_optimal = [
        row
        for row in valid_rows
        if (
            best_f1
            - row["f1_mean"]
            <= accuracy_tolerance
        )
    ]

    best_row = min(
        near_optimal,
        key=lambda row: (
            row["runtime_mean_ms"],
            -row["unmatched_f1_mean"],
            -row["exact_assignment_rate"],
            row["epsilon"],
        ),
    )

    best_epsilon = float(
        best_row["epsilon"]
    )

    print(
        "\nSelected Auction epsilon:"
    )

    print(
        f"  {best_epsilon:.8g}"
    )

    return (
        best_epsilon,
        sweep_rows,
    )


# ============================================================
# CSV output
# ============================================================

def save_csv(
    path: Path,
    rows: list[dict],
) -> None:
    """
    Save a list of flat dictionaries to CSV.
    """

    if not rows:
        raise ValueError(
            "Cannot save empty CSV"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
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
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Tune Assignment parameters using "
            "Synthetic DEV only."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "data/assignment/synthetic_dev"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/assignment/tuning"
        ),
    )

    parser.add_argument(
        "--invalid-cost",
        type=float,
        default=1e6,
    )

    parser.add_argument(
        "--unmatched-costs",
        type=float,
        nargs="*",
        default=DEFAULT_UNMATCHED_COSTS,
    )

    parser.add_argument(
        "--auction-epsilons",
        type=float,
        nargs="*",
        default=DEFAULT_AUCTION_EPSILONS,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if "final" in str(
        args.data_dir
    ).lower():

        raise ValueError(
            "Synthetic FINAL must not be used for tuning."
        )

    if not args.unmatched_costs:
        raise ValueError(
            "No unmatched-cost values supplied"
        )

    if not args.auction_epsilons:
        raise ValueError(
            "No Auction epsilon values supplied"
        )

    if any(
        value < 0
        for value in args.unmatched_costs
    ):
        raise ValueError(
            "unmatched_cost values must be non-negative"
        )

    if any(
        value <= 0
        for value in args.auction_epsilons
    ):
        raise ValueError(
            "Auction epsilon values must be positive"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load DEV data
    # --------------------------------------------------------

    cases = load_dev_cases(
        args.data_dir
    )

    # --------------------------------------------------------
    # Stage 1:
    # shared unmatched-cost tuning
    # --------------------------------------------------------

    (
        best_unmatched_cost,
        unmatched_rows,
    ) = tune_unmatched_cost(
        cases=cases,
        unmatched_costs=
            args.unmatched_costs,

        invalid_cost=
            args.invalid_cost,
    )

    save_csv(
        args.output_dir
        / "unmatched_cost_sweep.csv",

        unmatched_rows,
    )

    # --------------------------------------------------------
    # Stage 2:
    # Auction epsilon tuning
    # --------------------------------------------------------

    (
        best_epsilon,
        auction_rows,
    ) = tune_auction_epsilon(
        cases=cases,
        unmatched_cost=
            best_unmatched_cost,

        epsilons=
            args.auction_epsilons,

        invalid_cost=
            args.invalid_cost,
    )

    save_csv(
        args.output_dir
        / "auction_epsilon_sweep.csv",

        auction_rows,
    )

    # --------------------------------------------------------
    # Freeze parameters
    # --------------------------------------------------------

    frozen_parameters = {
        "dataset":
            str(args.data_dir),

        "tuning_split":
            "synthetic_dev",

        "shared_unmatched_cost":
            best_unmatched_cost,

        "auction_epsilon":
            best_epsilon,

        "invalid_cost":
            float(
                args.invalid_cost
            ),

        "cost_scaling_cost_scale":
            1_000_000,

        "cost_scaling_epsilon_factor":
            4.0,

        "policy": {
            "unmatched_cost":
                (
                    "Selected using Hungarian as the "
                    "exact reference solver on Synthetic DEV"
                ),

            "auction_epsilon":
                (
                    "Fastest setting within 0.001 absolute "
                    "F1 of the best Auction DEV result"
                ),

            "synthetic_final":
                "Not used during tuning",
        },
    }

    frozen_path = (
        args.output_dir
        / "frozen_parameters.json"
    )

    with frozen_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            frozen_parameters,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Final console summary
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "ASSIGNMENT PARAMETER FREEZE"
    )

    print(
        "========================================"
    )

    print(
        f"shared_unmatched_cost = "
        f"{best_unmatched_cost}"
    )

    print(
        f"auction_epsilon       = "
        f"{best_epsilon}"
    )

    print(
        f"\nSaved to:\n  {frozen_path}"
    )

    print(
        "\nDo NOT retune these parameters "
        "using Synthetic FINAL."
    )


if __name__ == "__main__":
    main()