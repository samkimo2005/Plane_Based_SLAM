"""
run_assignment_tum.py

Run the four frozen Assignment methods on the prepared
TUM + ETH3D-derived real-data cases.

Algorithms
----------
1. Hungarian
2. Auction
3. Jonker-Volgenant
4. Cost-Scaling

Inputs
------
data/assignment/tum/*.npz
data/assignment/tum_reference/*_gt.npz
results/assignment/tuning/frozen_parameters.json

Outputs
-------
results/assignment/tum/
    per_case_results.csv
    summary.csv
    decision_matrix_evidence.csv
    parameters_used.json

Real-data GT policy
-------------------
The ETH3D-derived reference is incomplete.

gt_label_matrix:
     1 = known true correspondence
     0 = known non-correspondence
    -1 = unknown / ignore

Predicted matches with label -1 are NOT automatically treated as
false positives.

Assignment-stage recall is measured against reachable GT matches,
i.e. true correspondences that survived the upstream Candidate Gating
stage and were therefore available to Assignment.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from assignment_common import (
    AssignmentCase,
    AssignmentResult,
    load_assignment_case,
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
# Configuration
# ============================================================

ALGORITHM_ORDER = [
    "hungarian",
    "auction",
    "jonker_volgenant",
    "cost_scaling",
]


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "assignment"
    / "tum"
)

DEFAULT_REFERENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "assignment"
    / "tum_reference"
)

DEFAULT_PARAMETER_FILE = (
    PROJECT_ROOT
    / "results"
    / "assignment"
    / "tuning"
    / "frozen_parameters.json"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "assignment"
    / "tum"
)


# ============================================================
# Helpers
# ============================================================

def safe_divide(
    numerator: float,
    denominator: float,
) -> float:

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


def precision_recall_f1(
    tp: int,
    fp: int,
    fn: int,
):

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    f1 = safe_divide(
        2.0 * precision * recall,
        precision + recall,
    )

    return (
        precision,
        recall,
        f1,
    )


# ============================================================
# Load frozen Assignment parameters
# ============================================================

def load_frozen_parameters(
    path: Path,
) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            "Frozen Assignment parameter file "
            f"not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        params = json.load(f)

    required = {
        "shared_unmatched_cost",
        "auction_epsilon",
        "invalid_cost",
    }

    missing = required - set(
        params.keys()
    )

    if missing:
        raise KeyError(
            "Frozen parameter file missing keys: "
            f"{sorted(missing)}"
        )

    # These were fixed during the synthetic experiments.
    params.setdefault(
        "cost_scaling_cost_scale",
        1_000_000,
    )

    params.setdefault(
        "cost_scaling_epsilon_factor",
        4.0,
    )

    return params


# ============================================================
# Load Assignment cases
# ============================================================

def load_cases(
    data_dir: Path,
) -> list[AssignmentCase]:

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Assignment data directory not found:\n"
            f"{data_dir}"
        )

    paths = sorted(
        data_dir.glob("tum_*.npz")
    )

    if not paths:
        raise RuntimeError(
            f"No TUM Assignment cases found in:\n"
            f"{data_dir}"
        )

    cases = []

    for path in paths:

        case = load_assignment_case(
            path
        )

        cases.append(case)

    print(
        f"Loaded Assignment cases : "
        f"{len(cases)}"
    )

    return cases


# ============================================================
# Load real-data GT sidecar
# ============================================================

def load_reference(
    reference_dir: Path,
    case_id: str,
) -> dict:

    path = (
        reference_dir
        / f"{case_id}_gt.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"GT sidecar not found:\n{path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        stored_case_id = str(
            data["case_id"].item()
        )

        if stored_case_id != case_id:
            raise RuntimeError(
                f"Case-ID mismatch: expected "
                f"{case_id}, found "
                f"{stored_case_id}"
            )

        reference = {
            "gt_label_matrix":
                np.asarray(
                    data[
                        "gt_label_matrix"
                    ],
                    dtype=np.int8,
                ),

            "pair_id_matrix":
                np.asarray(
                    data[
                        "pair_id_matrix"
                    ],
                    dtype=np.int64,
                ),

            "current_gt_plane_ids":
                np.asarray(
                    data[
                        "current_gt_plane_ids"
                    ],
                    dtype=np.int32,
                ),

            "reference_gt_plane_ids":
                np.asarray(
                    data[
                        "reference_gt_plane_ids"
                    ],
                    dtype=np.int32,
                ),

            "current_gt_known":
                np.asarray(
                    data[
                        "current_gt_known"
                    ],
                    dtype=bool,
                ),

            "reference_gt_known":
                np.asarray(
                    data[
                        "reference_gt_known"
                    ],
                    dtype=bool,
                ),

            "full_gt_matches":
                np.asarray(
                    data[
                        "full_gt_matches"
                    ],
                    dtype=np.int32,
                ).reshape(-1, 2),

            "assignment_gt_matches":
                np.asarray(
                    data[
                        "assignment_gt_matches"
                    ],
                    dtype=np.int32,
                ).reshape(-1, 2),
        }

    return reference


# ============================================================
# Run one solver
# ============================================================

def run_algorithm(
    algorithm: str,
    case: AssignmentCase,
    params: dict,
) -> AssignmentResult:

    unmatched_cost = float(
        params[
            "shared_unmatched_cost"
        ]
    )

    invalid_cost = float(
        params[
            "invalid_cost"
        ]
    )

    if algorithm == "hungarian":

        return run_hungarian_assignment(
            case,
            unmatched_cost=
                unmatched_cost,
            invalid_cost=
                invalid_cost,
        )

    if algorithm == "auction":

        return run_auction_assignment(
            case,
            unmatched_cost=
                unmatched_cost,
            epsilon=float(
                params[
                    "auction_epsilon"
                ]
            ),
            invalid_cost=
                invalid_cost,
        )

    if algorithm == "jonker_volgenant":

        return (
            run_jonker_volgenant_assignment(
                case,
                unmatched_cost=
                    unmatched_cost,
                invalid_cost=
                    invalid_cost,
            )
        )

    if algorithm == "cost_scaling":

        return run_cost_scaling_assignment(
            case,
            unmatched_cost=
                unmatched_cost,
            invalid_cost=
                invalid_cost,
            cost_scale=int(
                params[
                    "cost_scaling_cost_scale"
                ]
            ),
            epsilon_factor=float(
                params[
                    "cost_scaling_epsilon_factor"
                ]
            ),
        )

    raise ValueError(
        f"Unknown algorithm: {algorithm}"
    )


# ============================================================
# Validate solver output
# ============================================================

def validate_result(
    case: AssignmentCase,
    result: AssignmentResult,
):

    matches = {
        (int(i), int(j))
        for i, j in result.matches
    }

    current_seen = set()
    reference_seen = set()

    for i, j in matches:

        if (
            i < 0
            or i >= len(case.current_ids)
        ):
            raise RuntimeError(
                f"{result.algorithm}: invalid "
                f"current index {i}"
            )

        if (
            j < 0
            or j >= len(case.reference_ids)
        ):
            raise RuntimeError(
                f"{result.algorithm}: invalid "
                f"reference index {j}"
            )

        if not case.candidate_mask[i, j]:
            raise RuntimeError(
                f"{result.algorithm}: selected "
                f"non-candidate edge ({i},{j})"
            )

        if i in current_seen:
            raise RuntimeError(
                f"{result.algorithm}: current "
                f"plane {i} assigned twice"
            )

        if j in reference_seen:
            raise RuntimeError(
                f"{result.algorithm}: reference "
                f"plane {j} assigned twice"
            )

        current_seen.add(i)
        reference_seen.add(j)


# ============================================================
# Objective calculation
# ============================================================

def calculate_objective(
    case: AssignmentCase,
    result: AssignmentResult,
    unmatched_cost: float,
):

    matches = {
        (int(i), int(j))
        for i, j in result.matches
    }

    real_match_cost = 0.0

    for i, j in matches:

        real_match_cost += (
            1.0
            - float(
                case.similarity[i, j]
            )
        )

    unmatched_endpoint_cost = (
        unmatched_cost
        * (
            len(
                result.unmatched_current
            )
            +
            len(
                result.unmatched_reference
            )
        )
    )

    total_cost = (
        real_match_cost
        + unmatched_endpoint_cost
    )

    return {
        "real_match_cost":
            float(real_match_cost),

        "unmatched_cost_total":
            float(
                unmatched_endpoint_cost
            ),

        "total_augmented_cost":
            float(total_cost),
    }


# ============================================================
# Real-data evaluation
# ============================================================

def evaluate_real_result(
    case: AssignmentCase,
    result: AssignmentResult,
    reference: dict,
    unmatched_cost: float,
) -> dict:

    validate_result(
        case,
        result,
    )

    gt_labels = reference[
        "gt_label_matrix"
    ]

    if (
        gt_labels.shape
        != case.similarity.shape
    ):
        raise RuntimeError(
            f"{case.case_id}: GT matrix "
            f"shape {gt_labels.shape} does not "
            f"match case shape "
            f"{case.similarity.shape}"
        )

    selected = {
        (int(i), int(j))
        for i, j in result.matches
    }

    reachable_gt = {
        (int(i), int(j))
        for i, j in
        reference[
            "assignment_gt_matches"
        ]
    }

    full_gt = {
        (int(i), int(j))
        for i, j in
        reference[
            "full_gt_matches"
        ]
    }

    # --------------------------------------------------------
    # Selected correspondence classes
    # --------------------------------------------------------

    selected_positive = set()

    selected_negative = set()

    selected_unknown = set()

    for i, j in selected:

        label = int(
            gt_labels[i, j]
        )

        if label == 1:
            selected_positive.add(
                (i, j)
            )

        elif label == 0:
            selected_negative.add(
                (i, j)
            )

        elif label == -1:
            selected_unknown.add(
                (i, j)
            )

        else:
            raise RuntimeError(
                f"Unexpected GT label "
                f"{label} at ({i},{j})"
            )

    # --------------------------------------------------------
    # Stage-level TP / FP / FN
    # --------------------------------------------------------

    tp = len(
        selected_positive
        & reachable_gt
    )

    fp = len(
        selected_negative
    )

    fn = len(
        reachable_gt
        - selected_positive
    )

    (
        precision,
        recall,
        f1,
    ) = precision_recall_f1(
        tp,
        fp,
        fn,
    )

    # Unknown selected edges are deliberately excluded
    # from TP/FP/FN.
    selected_known = (
        len(selected_positive)
        + len(selected_negative)
    )

    selected_total = len(
        selected
    )

    unknown_selected = len(
        selected_unknown
    )

    unknown_selection_rate = (
        safe_divide(
            unknown_selected,
            selected_total,
        )
    )

    evaluable_selection_rate = (
        safe_divide(
            selected_known,
            selected_total,
        )
    )

    # --------------------------------------------------------
    # Exact evaluable assignment
    # --------------------------------------------------------
    #
    # Unknown selected matches cannot be judged, so they
    # do not invalidate the case.
    #
    # A case is exact on the evaluable reference if:
    #   - all reachable GT positives are selected;
    #   - no known GT-negative edge is selected.
    # --------------------------------------------------------

    exact_evaluable_assignment = int(
        (
            selected_positive
            == reachable_gt
        )
        and
        (
            len(
                selected_negative
            )
            == 0
        )
    )

    # --------------------------------------------------------
    # Upstream availability
    # --------------------------------------------------------

    full_gt_count = len(
        full_gt
    )

    reachable_gt_count = len(
        reachable_gt
    )

    upstream_gt_retention = (
        safe_divide(
            reachable_gt_count,
            full_gt_count,
        )
        if full_gt_count > 0
        else 1.0
    )

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    objective = calculate_objective(
        case,
        result,
        unmatched_cost,
    )

    # --------------------------------------------------------
    # Similarity statistics of selected edges
    # --------------------------------------------------------

    selected_scores = [
        float(
            case.similarity[i, j]
        )
        for i, j in selected
    ]

    if selected_scores:

        selected_similarity_mean = float(
            np.mean(
                selected_scores
            )
        )

        selected_similarity_min = float(
            np.min(
                selected_scores
            )
        )

    else:

        selected_similarity_mean = 0.0

        selected_similarity_min = 0.0

    return {
        "algorithm":
            result.algorithm,

        "case_id":
            case.case_id,

        "frame_current":
            case.metadata.get(
                "frame_current",
                "",
            ),

        "frame_reference":
            case.metadata.get(
                "frame_reference",
                "",
            ),

        "n_current":
            len(case.current_ids),

        "n_reference":
            len(case.reference_ids),

        "candidate_edges":
            int(
                np.sum(
                    case.candidate_mask
                )
            ),

        "full_gt_matches":
            full_gt_count,

        "reachable_gt_matches":
            reachable_gt_count,

        "upstream_gt_retention":
            upstream_gt_retention,

        "selected_matches":
            selected_total,

        "selected_known_positive":
            len(
                selected_positive
            ),

        "selected_known_negative":
            len(
                selected_negative
            ),

        "selected_unknown":
            unknown_selected,

        "evaluable_selection_rate":
            evaluable_selection_rate,

        "unknown_selection_rate":
            unknown_selection_rate,

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
            f1,

        "exact_evaluable_assignment":
            exact_evaluable_assignment,

        "unmatched_current":
            len(
                result.unmatched_current
            ),

        "unmatched_reference":
            len(
                result.unmatched_reference
            ),

        "selected_similarity_mean":
            selected_similarity_mean,

        "selected_similarity_min":
            selected_similarity_min,

        "runtime_ms":
            float(
                result.runtime_ms
            ),

        **objective,
    }


# ============================================================
# Aggregate real-data results
# ============================================================

def aggregate_algorithm(
    rows: list[dict],
) -> dict:

    if not rows:
        raise ValueError(
            "Cannot aggregate empty result set."
        )

    tp = sum(
        int(row["tp"])
        for row in rows
    )

    fp = sum(
        int(row["fp"])
        for row in rows
    )

    fn = sum(
        int(row["fn"])
        for row in rows
    )

    (
        precision,
        recall,
        f1,
    ) = precision_recall_f1(
        tp,
        fp,
        fn,
    )

    runtimes = np.asarray(
        [
            float(
                row["runtime_ms"]
            )
            for row in rows
        ],
        dtype=float,
    )

    objective_costs = np.asarray(
        [
            float(
                row[
                    "total_augmented_cost"
                ]
            )
            for row in rows
        ],
        dtype=float,
    )

    selected_total = sum(
        int(
            row["selected_matches"]
        )
        for row in rows
    )

    selected_unknown = sum(
        int(
            row["selected_unknown"]
        )
        for row in rows
    )

    selected_known_positive = sum(
        int(
            row[
                "selected_known_positive"
            ]
        )
        for row in rows
    )

    selected_known_negative = sum(
        int(
            row[
                "selected_known_negative"
            ]
        )
        for row in rows
    )

    exact_rate = float(
        np.mean(
            [
                float(
                    row[
                        "exact_evaluable_assignment"
                    ]
                )
                for row in rows
            ]
        )
    )

    case_f1 = np.asarray(
        [
            float(row["f1"])
            for row in rows
        ],
        dtype=float,
    )

    return {
        "algorithm":
            rows[0]["algorithm"],

        "n_cases":
            len(rows),

        "tp":
            tp,

        "fp":
            fp,

        "fn":
            fn,

        # Primary real-data accuracy metrics:
        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        # Supporting case-level metric:
        "mean_case_f1":
            float(
                np.mean(case_f1)
            ),

        "exact_evaluable_assignment_rate":
            exact_rate,

        "selected_matches":
            selected_total,

        "selected_known_positive":
            selected_known_positive,

        "selected_known_negative":
            selected_known_negative,

        "selected_unknown":
            selected_unknown,

        "unknown_selection_rate":
            safe_divide(
                selected_unknown,
                selected_total,
            ),

        "evaluable_selection_rate":
            safe_divide(
                selected_known_positive
                + selected_known_negative,
                selected_total,
            ),

        "runtime_mean_ms":
            float(
                np.mean(runtimes)
            ),

        "runtime_median_ms":
            float(
                np.median(runtimes)
            ),

        "runtime_p95_ms":
            float(
                np.percentile(
                    runtimes,
                    95,
                )
            ),

        "runtime_std_ms":
            float(
                np.std(runtimes)
            ),

        "objective_mean":
            float(
                np.mean(
                    objective_costs
                )
            ),

        "objective_median":
            float(
                np.median(
                    objective_costs
                )
            ),
    }


# ============================================================
# CSV utilities
# ============================================================

def save_csv(
    path: Path,
    rows: list[dict],
):

    if not rows:
        raise ValueError(
            f"No rows to save: {path}"
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

        writer.writerows(rows)


# ============================================================
# Console summary
# ============================================================

def print_summary(
    summaries: list[dict],
):

    print()
    print("=" * 92)
    print("TUM REAL ASSIGNMENT SUMMARY")
    print("=" * 92)

    print(
        f"{'Algorithm':<20}"
        f"{'P':>9}"
        f"{'R':>9}"
        f"{'F1':>9}"
        f"{'Exact':>9}"
        f"{'Unknown':>10}"
        f"{'Mean ms':>12}"
    )

    print(
        "-" * 92
    )

    for row in summaries:

        print(
            f"{row['algorithm']:<20}"
            f"{row['precision']:>9.4f}"
            f"{row['recall']:>9.4f}"
            f"{row['f1']:>9.4f}"
            f"{row['exact_evaluable_assignment_rate']:>9.4f}"
            f"{row['unknown_selection_rate']:>10.4f}"
            f"{row['runtime_mean_ms']:>12.4f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run frozen Assignment algorithms "
            "on TUM real-data cases."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )

    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
    )

    parser.add_argument(
        "--parameter-file",
        type=Path,
        default=DEFAULT_PARAMETER_FILE,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    params = load_frozen_parameters(
        args.parameter_file
    )

    cases = load_cases(
        args.data_dir
    )

    print()
    print("=" * 78)
    print("RUN TUM REAL ASSIGNMENT")
    print("=" * 78)

    print(
        f"Cases              : "
        f"{len(cases)}"
    )

    print(
        f"Unmatched cost     : "
        f"{params['shared_unmatched_cost']}"
    )

    print(
        f"Auction epsilon    : "
        f"{params['auction_epsilon']}"
    )

    print(
        "Parameters         : FROZEN"
    )

    print(
        "Real GT policy     : "
        "unknown edges ignored"
    )

    print()

    # --------------------------------------------------------
    # Run all algorithms
    # --------------------------------------------------------

    all_rows = []

    total_runs = (
        len(cases)
        * len(ALGORITHM_ORDER)
    )

    run_index = 0

    unmatched_cost = float(
        params[
            "shared_unmatched_cost"
        ]
    )

    for case_index, case in enumerate(
        cases,
        start=1,
    ):

        reference = load_reference(
            args.reference_dir,
            case.case_id,
        )

        print(
            f"[Case {case_index:02d}/"
            f"{len(cases):02d}] "
            f"{case.case_id}"
        )

        for algorithm in ALGORITHM_ORDER:

            run_index += 1

            result = run_algorithm(
                algorithm,
                case,
                params,
            )

            row = evaluate_real_result(
                case=case,
                result=result,
                reference=reference,
                unmatched_cost=
                    unmatched_cost,
            )

            all_rows.append(
                row
            )

            print(
                f"    "
                f"{algorithm:<20} "
                f"F1={row['f1']:.4f} | "
                f"TP={row['tp']:2d} "
                f"FP={row['fp']:2d} "
                f"FN={row['fn']:2d} | "
                f"unknown="
                f"{row['selected_unknown']:2d} | "
                f"{row['runtime_ms']:.4f} ms"
            )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    summary_rows = []

    for algorithm in ALGORITHM_ORDER:

        selected = [
            row
            for row in all_rows
            if row["algorithm"]
            == algorithm
        ]

        summary_rows.append(
            aggregate_algorithm(
                selected
            )
        )

    # --------------------------------------------------------
    # Decision-matrix evidence
    # --------------------------------------------------------

    evidence_rows = []

    for row in summary_rows:

        evidence_rows.append(
            {
                "algorithm":
                    row["algorithm"],

                # Primary real-data accuracy
                "F1_real":
                    row["f1"],

                "precision_real":
                    row["precision"],

                "recall_real":
                    row["recall"],

                # Supporting assignment measure
                "exact_evaluable_assignment_rate":
                    row[
                        "exact_evaluable_assignment_rate"
                    ],

                # Reference-coverage diagnostic.
                # This is NOT treated as an error metric.
                "unknown_selection_rate":
                    row[
                        "unknown_selection_rate"
                    ],

                # Processing speed
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
            }
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
        all_rows,
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

    parameters_used = {
        "dataset":
            "TUM fr1_xyz + ETH3D-derived persistent plane identities",

        "n_cases":
            len(cases),

        "algorithms":
            ALGORITHM_ORDER,

        "parameter_policy":
            "Frozen from Synthetic DEV before real-data evaluation",

        "gt_policy":
            {
                "positive":
                    "gt_label == 1",

                "negative":
                    "gt_label == 0",

                "unknown":
                    "gt_label == -1; ignored in TP/FP/FN",

                "assignment_recall_reference":
                    "reachable GT matches only",
            },

        "frozen_parameters":
            params,
    }

    with (
        args.output_dir
        / "parameters_used.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            parameters_used,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print_summary(
        summary_rows
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        f"  {args.output_dir}"
    )

    print()
    print(
        "TUM real Assignment experiment complete."
    )


if __name__ == "__main__":
    main()