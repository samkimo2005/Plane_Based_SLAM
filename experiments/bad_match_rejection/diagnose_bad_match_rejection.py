"""
diagnose_bad_match_tum_distribution.py

Diagnose why the TUM real-data Bad-Match Rejection experiment produced
near-identical all-keep classification results.

The script focuses on the 13 known-bad correspondences and answers:

1. In how many assigned-pair cases do the known bad matches occur?
2. How many occur in cases with 1, 2, 3, ... assigned correspondences?
3. Which methods actually had enough information to make a raw decision?
4. For each method, how many known-bad correspondences were rejected
   BEFORE the TUM runner's structural-failure fallback?
5. Were good or ambiguous correspondences rejected instead?

No tuning is performed.

Outputs
-------
experiments/results/bad_match_rejection/tum/
    bad_match_distribution.csv
    raw_method_diagnostics.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_bad_match_rejection_tum import (
    DEFAULT_DATA_DIR,
    DEFAULT_PARAMETER_FILE,
    DEFAULT_OUTPUT_DIR,
    METHOD_ORDER,
    load_case,
    load_frozen_parameters,
    run_method,
    extract_failure_reason,
)


# ============================================================
# Helpers
# ============================================================

def save_csv(
    path: Path,
    rows: list[dict],
) -> None:

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = []
    seen = set()

    for row in rows:

        for key in row.keys():

            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

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


def raw_counts(
    keep_mask: np.ndarray,
    gt_labels: np.ndarray,
) -> dict:
    """
    Count raw correspondence decisions before any fallback.

    GT:
         1 = known good
         0 = known bad
        -1 = ambiguous
    """

    keep_mask = np.asarray(
        keep_mask,
        dtype=bool,
    ).reshape(-1)

    gt_labels = np.asarray(
        gt_labels,
        dtype=np.int8,
    ).reshape(-1)

    if len(keep_mask) != len(gt_labels):

        raise ValueError(
            "keep_mask / gt_labels length mismatch."
        )

    rejected = ~keep_mask

    return {
        "bad_total":
            int(np.sum(gt_labels == 0)),

        "bad_rejected":
            int(
                np.sum(
                    rejected
                    & (gt_labels == 0)
                )
            ),

        "bad_kept":
            int(
                np.sum(
                    keep_mask
                    & (gt_labels == 0)
                )
            ),

        "good_total":
            int(np.sum(gt_labels == 1)),

        "good_rejected":
            int(
                np.sum(
                    rejected
                    & (gt_labels == 1)
                )
            ),

        "good_kept":
            int(
                np.sum(
                    keep_mask
                    & (gt_labels == 1)
                )
            ),

        "ambiguous_total":
            int(np.sum(gt_labels == -1)),

        "ambiguous_rejected":
            int(
                np.sum(
                    rejected
                    & (gt_labels == -1)
                )
            ),

        "ambiguous_kept":
            int(
                np.sum(
                    keep_mask
                    & (gt_labels == -1)
                )
            ),
    }


def minimum_evidence_class(
    method: str,
    n_assigned: int,
) -> str:
    """
    Describe only the obvious minimum-input requirement from the
    current implementations.

    This does NOT guarantee geometric observability.
    """

    if method == "SelMaP-inspired":

        return (
            "enough_for_pairwise_test"
            if n_assigned >= 2
            else "too_few_for_pairwise_test"
        )

    return (
        "enough_for_pose_or_subset_test"
        if n_assigned >= 3
        else "too_few_for_pose_or_subset_test"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Diagnose TUM known-bad correspondence "
            "distribution and raw rejection behaviour."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
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

    best = load_frozen_parameters(
        args.parameter_file
    )

    files = sorted(
        args.data_dir.glob(
            "tum_*.npz"
        )
    )

    if not files:

        raise FileNotFoundError(
            "No prepared TUM bad-match cases found in:\n"
            f"{args.data_dir}"
        )

    case_rows = []
    raw_rows = []

    bad_by_assigned_count = Counter()
    bad_cases_by_assigned_count = Counter()

    total_known_bad = 0
    cases_with_bad = 0

    # Aggregate raw method behaviour over ALL cases containing
    # at least one known bad match.
    method_totals = {
        method: defaultdict(int)
        for method in METHOD_ORDER
    }

    for sample_index, path in enumerate(
        files
    ):

        case = load_case(
            path
        )

        assigned_pairs = case[
            "assigned_pairs"
        ]

        gt_labels = case[
            "gt_labels"
        ]

        n_assigned = len(
            assigned_pairs
        )

        n_good = int(
            np.sum(
                gt_labels == 1
            )
        )

        n_bad = int(
            np.sum(
                gt_labels == 0
            )
        )

        n_ambiguous = int(
            np.sum(
                gt_labels == -1
            )
        )

        if n_bad == 0:
            continue

        cases_with_bad += 1
        total_known_bad += n_bad

        bad_by_assigned_count[
            n_assigned
        ] += n_bad

        bad_cases_by_assigned_count[
            n_assigned
        ] += 1

        case_row = {
            "case_id":
                case["case_id"],

            "frame_a":
                case["frame_a"],

            "frame_b":
                case["frame_b"],

            "n_assigned":
                n_assigned,

            "n_good":
                n_good,

            "n_bad":
                n_bad,

            "n_ambiguous":
                n_ambiguous,

            "assigned_ge_2":
                int(
                    n_assigned >= 2
                ),

            "assigned_ge_3":
                int(
                    n_assigned >= 3
                ),
        }

        # ----------------------------------------------------
        # Raw decisions from each method, BEFORE fallback.
        # ----------------------------------------------------

        for method in METHOD_ORDER:

            exception_text = ""

            try:

                result = run_method(
                    method_name=
                        method,

                    planes_a=
                        case["planes_a"],

                    planes_b=
                        case["planes_b"],

                    assigned_pairs=
                        assigned_pairs,

                    best=
                        best,

                    sample_index=
                        sample_index,
                )

                keep_mask = np.asarray(
                    result[
                        "keep_mask"
                    ],
                    dtype=bool,
                ).reshape(-1)

                failure_reason = (
                    extract_failure_reason(
                        result
                    )
                )

                success_flag = bool(
                    result.get(
                        "success",
                        False,
                    )
                )

                if len(keep_mask) != n_assigned:

                    raise RuntimeError(
                        "keep_mask length mismatch"
                    )

                counts = raw_counts(
                    keep_mask,
                    gt_labels,
                )

            except Exception as error:

                keep_mask = None

                failure_reason = "exception"

                success_flag = False

                exception_text = (
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                counts = {
                    "bad_total":
                        n_bad,

                    "bad_rejected":
                        0,

                    "bad_kept":
                        0,

                    "good_total":
                        n_good,

                    "good_rejected":
                        0,

                    "good_kept":
                        0,

                    "ambiguous_total":
                        n_ambiguous,

                    "ambiguous_rejected":
                        0,

                    "ambiguous_kept":
                        0,
                }

            evidence_class = (
                minimum_evidence_class(
                    method,
                    n_assigned,
                )
            )

            # "Raw decision usable" here is intentionally stricter
            # than merely having a keep_mask:
            # - explicit failure reason => unusable;
            # - obvious below-minimum input => unusable.
            below_minimum = (
                (
                    method
                    == "SelMaP-inspired"
                    and n_assigned < 2
                )
                or
                (
                    method
                    != "SelMaP-inspired"
                    and n_assigned < 3
                )
            )

            raw_decision_usable = (
                keep_mask is not None
                and not failure_reason
                and not below_minimum
            )

            raw_rows.append(
                {
                    "case_id":
                        case["case_id"],

                    "method":
                        method,

                    "n_assigned":
                        n_assigned,

                    "n_good":
                        n_good,

                    "n_bad":
                        n_bad,

                    "n_ambiguous":
                        n_ambiguous,

                    "minimum_evidence":
                        evidence_class,

                    "algorithm_success_flag":
                        success_flag,

                    "failure_reason":
                        failure_reason,

                    "raw_decision_usable":
                        int(
                            raw_decision_usable
                        ),

                    "raw_kept":
                        (
                            int(
                                np.sum(
                                    keep_mask
                                )
                            )
                            if keep_mask
                            is not None
                            else ""
                        ),

                    "raw_rejected":
                        (
                            int(
                                np.sum(
                                    ~keep_mask
                                )
                            )
                            if keep_mask
                            is not None
                            else ""
                        ),

                    **counts,

                    "exception":
                        exception_text,
                }
            )

            prefix = (
                method
                .lower()
                .replace(
                    "-",
                    "_",
                )
                .replace(
                    " ",
                    "_",
                )
            )

            case_row[
                f"{prefix}_usable"
            ] = int(
                raw_decision_usable
            )

            case_row[
                f"{prefix}_bad_rejected"
            ] = counts[
                "bad_rejected"
            ]

            case_row[
                f"{prefix}_good_rejected"
            ] = counts[
                "good_rejected"
            ]

            case_row[
                f"{prefix}_amb_rejected"
            ] = counts[
                "ambiguous_rejected"
            ]

            case_row[
                f"{prefix}_failure"
            ] = failure_reason

            totals = method_totals[
                method
            ]

            totals[
                "bad_total"
            ] += counts[
                "bad_total"
            ]

            totals[
                "bad_rejected"
            ] += counts[
                "bad_rejected"
            ]

            totals[
                "good_total"
            ] += counts[
                "good_total"
            ]

            totals[
                "good_rejected"
            ] += counts[
                "good_rejected"
            ]

            totals[
                "ambiguous_total"
            ] += counts[
                "ambiguous_total"
            ]

            totals[
                "ambiguous_rejected"
            ] += counts[
                "ambiguous_rejected"
            ]

            totals[
                "usable_cases"
            ] += int(
                raw_decision_usable
            )

            totals[
                "failure_cases"
            ] += int(
                bool(
                    failure_reason
                )
            )

        case_rows.append(
            case_row
        )

    # ========================================================
    # Save detailed diagnosis
    # ========================================================

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    distribution_file = (
        args.output_dir
        / "bad_match_distribution.csv"
    )

    raw_file = (
        args.output_dir
        / "raw_method_diagnostics.csv"
    )

    save_csv(
        distribution_file,
        case_rows,
    )

    save_csv(
        raw_file,
        raw_rows,
    )

    # ========================================================
    # Terminal report
    # ========================================================

    print()
    print("=" * 92)
    print("TUM KNOWN-BAD MATCH DISTRIBUTION")
    print("=" * 92)

    print(
        f"Cases containing known bad : "
        f"{cases_with_bad}"
    )

    print(
        f"Known bad correspondences  : "
        f"{total_known_bad}"
    )

    print()
    print(
        "Known bad matches by number of assigned correspondences:"
    )

    for n_assigned in sorted(
        bad_by_assigned_count
    ):

        print(
            f"  N={n_assigned}: "
            f"{bad_by_assigned_count[n_assigned]} "
            f"bad match(es) in "
            f"{bad_cases_by_assigned_count[n_assigned]} "
            f"case(s)"
        )

    bad_under_3 = sum(
        count
        for n_assigned, count
        in bad_by_assigned_count.items()
        if n_assigned < 3
    )

    bad_ge_3 = (
        total_known_bad
        - bad_under_3
    )

    print()

    print(
        f"Bad matches in N<3 cases   : "
        f"{bad_under_3}/"
        f"{total_known_bad}"
    )

    print(
        f"Bad matches in N>=3 cases  : "
        f"{bad_ge_3}/"
        f"{total_known_bad}"
    )

    print()
    print("=" * 92)
    print("RAW METHOD BEHAVIOUR ON BAD-CONTAINING CASES")
    print("=" * 92)

    print(
        f"{'Method':<24}"
        f"{'Usable':>8}"
        f"{'Bad rej':>10}"
        f"{'Good rej':>11}"
        f"{'Amb rej':>10}"
        f"{'Failures':>10}"
    )

    print("-" * 73)

    for method in METHOD_ORDER:

        totals = method_totals[
            method
        ]

        print(
            f"{method:<24}"
            f"{totals['usable_cases']:>8}"
            f"{totals['bad_rejected']:>10}"
            f"{totals['good_rejected']:>11}"
            f"{totals['ambiguous_rejected']:>10}"
            f"{totals['failure_cases']:>10}"
        )

    print()
    print(
        "Per-case bad-match distribution:"
    )

    print(
        f"{'Case':<20}"
        f"{'N':>4}"
        f"{'Good':>7}"
        f"{'Bad':>6}"
        f"{'Amb':>6}"
    )

    print("-" * 43)

    for row in case_rows:

        print(
            f"{row['case_id']:<20}"
            f"{row['n_assigned']:>4}"
            f"{row['n_good']:>7}"
            f"{row['n_bad']:>6}"
            f"{row['n_ambiguous']:>6}"
        )

    print()
    print(
        f"Detailed distribution:\n"
        f"  {distribution_file}"
    )

    print(
        f"Raw method diagnostics:\n"
        f"  {raw_file}"
    )


if __name__ == "__main__":
    main()
