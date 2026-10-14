"""
generate_assignment_synthetic.py

Generate synthetic datasets for the one-to-one plane correspondence
Assignment experiments.

Three scenarios are generated:

1. normal
   Clear distinction between correct and incorrect correspondences.

2. hard
   Ambiguous high-similarity distractors and many-to-one competition.

3. unmatched
   Unequal plane sets and planes with no valid correspondence.

The same generated cases are used by all Assignment algorithms to
ensure a fair comparison.

Ground-truth correspondence edges are always retained in the candidate
mask so that these experiments isolate the Assignment stage rather than
re-testing Candidate Gating.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from assignment_common import (
    AssignmentCase,
    save_assignment_case,
    summarise_assignment_case,
)


# ============================================================
# Utility functions
# ============================================================

def _clip_scores(scores: np.ndarray) -> np.ndarray:
    """
    Keep all similarity values inside the valid [0, 1] range.
    """
    return np.clip(scores, 0.0, 1.0)


def _make_plane_ids(
    prefix: str,
    n: int,
) -> list[str]:
    """
    Generate simple plane identifiers.
    """
    return [
        f"{prefix}{i:02d}"
        for i in range(n)
    ]


def _balanced_scenario_counts(
    total_cases: int,
) -> dict[str, int]:
    """
    Divide cases approximately equally between the three scenarios.
    """

    scenarios = [
        "normal",
        "hard",
        "unmatched",
    ]

    base = total_cases // len(scenarios)
    remainder = total_cases % len(scenarios)

    counts = {
        scenario: base
        for scenario in scenarios
    }

    for i in range(remainder):
        counts[scenarios[i]] += 1

    return counts


# ============================================================
# Ground-truth generation
# ============================================================

def _generate_complete_gt(
    n: int,
    rng: np.random.Generator,
) -> set[tuple[int, int]]:
    """
    Generate a complete one-to-one GT assignment.

    All current planes and all reference planes are matched.
    """

    permutation = rng.permutation(n)

    return {
        (i, int(permutation[i]))
        for i in range(n)
    }


def _generate_partial_gt(
    n_current: int,
    n_reference: int,
    rng: np.random.Generator,
) -> set[tuple[int, int]]:
    """
    Generate a partial one-to-one assignment.

    At least one plane on each side is deliberately unmatched whenever
    possible.
    """

    max_matches = min(
        n_current,
        n_reference,
    )

    # Leave approximately 1--3 planes unmatched.
    max_remove = min(
        3,
        max_matches - 2,
    )

    if max_remove < 1:
        n_matches = max_matches
    else:
        n_remove = int(
            rng.integers(
                1,
                max_remove + 1,
            )
        )

        n_matches = max_matches - n_remove

    current_indices = rng.choice(
        n_current,
        size=n_matches,
        replace=False,
    )

    reference_indices = rng.choice(
        n_reference,
        size=n_matches,
        replace=False,
    )

    rng.shuffle(reference_indices)

    return {
        (int(i), int(j))
        for i, j in zip(
            current_indices,
            reference_indices,
        )
    }


# ============================================================
# Candidate-mask helpers
# ============================================================

def _random_candidate_mask(
    n_current: int,
    n_reference: int,
    density: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate an initial random Candidate Gating mask.
    """

    return (
        rng.random(
            (n_current, n_reference)
        )
        < density
    )


def _retain_gt_edges(
    candidate_mask: np.ndarray,
    gt_matches: set[tuple[int, int]],
) -> None:
    """
    Force all GT correspondences to remain valid candidates.

    This is essential because the experiment is intended to evaluate
    Assignment rather than Candidate Gating.
    """

    for i, j in gt_matches:
        candidate_mask[i, j] = True


# ============================================================
# Normal scenario
# ============================================================

def generate_normal_case(
    case_id: str,
    rng: np.random.Generator,
) -> AssignmentCase:
    """
    Generate an ordinary one-to-one assignment problem.

    Expected behaviour:
        Correct pairs have clearly higher similarity than distractors.
    """

    n = int(
        rng.integers(4, 11)
    )

    gt_matches = _generate_complete_gt(
        n=n,
        rng=rng,
    )

    candidate_mask = _random_candidate_mask(
        n_current=n,
        n_reference=n,
        density=0.35,
        rng=rng,
    )

    # Incorrect correspondences:
    # generally low similarity.
    similarity = rng.normal(
        loc=0.20,
        scale=0.08,
        size=(n, n),
    )

    # Correct correspondences:
    # strong and clearly separated.
    for i, j in gt_matches:
        similarity[i, j] = rng.normal(
            loc=0.90,
            scale=0.04,
        )

    _retain_gt_edges(
        candidate_mask,
        gt_matches,
    )

    similarity = _clip_scores(similarity)

    return AssignmentCase(
        case_id=case_id,
        current_ids=_make_plane_ids(
            "c",
            n,
        ),
        reference_ids=_make_plane_ids(
            "r",
            n,
        ),
        similarity=similarity,
        candidate_mask=candidate_mask,
        gt_matches=gt_matches,
        metadata={
            "scenario": "normal",
            "difficulty": "normal",
        },
    )


# ============================================================
# Hard / ambiguous scenario
# ============================================================

def generate_hard_case(
    case_id: str,
    rng: np.random.Generator,
) -> AssignmentCase:
    """
    Generate an ambiguous assignment problem.

    The correct correspondence often has one or more competing
    high-similarity alternatives.

    Some distractors deliberately compete for the same reference plane,
    producing many-to-one ambiguity that requires global Assignment.
    """

    n = int(
        rng.integers(5, 12)
    )

    gt_matches = _generate_complete_gt(
        n=n,
        rng=rng,
    )

    gt_dict = {
        i: j
        for i, j in gt_matches
    }

    candidate_mask = _random_candidate_mask(
        n_current=n,
        n_reference=n,
        density=0.60,
        rng=rng,
    )

    # Background incorrect pairs.
    similarity = rng.normal(
        loc=0.30,
        scale=0.10,
        size=(n, n),
    )

    # GT scores are still generally strong, but separation is smaller.
    for i, j in gt_matches:
        similarity[i, j] = rng.normal(
            loc=0.74,
            scale=0.07,
        )

    # --------------------------------------------------------
    # Structured ambiguous distractors
    # --------------------------------------------------------

    for i in range(n):

        true_j = gt_dict[i]

        possible_refs = [
            j
            for j in range(n)
            if j != true_j
        ]

        # Usually add 1--2 strong competing candidates.
        n_distractors = int(
            rng.integers(1, 3)
        )

        distractors = rng.choice(
            possible_refs,
            size=min(
                n_distractors,
                len(possible_refs),
            ),
            replace=False,
        )

        for j in distractors:

            j = int(j)

            candidate_mask[i, j] = True

            similarity[i, j] = rng.normal(
                loc=0.68,
                scale=0.08,
            )

    # --------------------------------------------------------
    # Add explicit many-to-one competition
    # --------------------------------------------------------

    if n >= 3:

        shared_reference = int(
            rng.integers(0, n)
        )

        competing_current = rng.choice(
            n,
            size=min(3, n),
            replace=False,
        )

        for i in competing_current:

            i = int(i)

            if gt_dict[i] == shared_reference:
                continue

            candidate_mask[
                i,
                shared_reference,
            ] = True

            similarity[
                i,
                shared_reference,
            ] = rng.normal(
                loc=0.70,
                scale=0.06,
            )

    _retain_gt_edges(
        candidate_mask,
        gt_matches,
    )

    similarity = _clip_scores(similarity)

    return AssignmentCase(
        case_id=case_id,
        current_ids=_make_plane_ids(
            "c",
            n,
        ),
        reference_ids=_make_plane_ids(
            "r",
            n,
        ),
        similarity=similarity,
        candidate_mask=candidate_mask,
        gt_matches=gt_matches,
        metadata={
            "scenario": "hard",
            "difficulty": "ambiguous",
        },
    )


# ============================================================
# Unmatched / unequal scenario
# ============================================================

def generate_unmatched_case(
    case_id: str,
    rng: np.random.Generator,
) -> AssignmentCase:
    """
    Generate a problem containing unmatched planes.

    Current and reference sets may have different sizes and the ground
    truth deliberately contains only a partial matching.
    """

    n_current = int(
        rng.integers(5, 11)
    )

    n_reference = int(
        rng.integers(5, 11)
    )

    # Occasionally force unequal set sizes.
    if n_reference == n_current:
        if rng.random() < 0.75:
            if n_reference < 10:
                n_reference += 1
            else:
                n_reference -= 1

    gt_matches = _generate_partial_gt(
        n_current=n_current,
        n_reference=n_reference,
        rng=rng,
    )

    candidate_mask = _random_candidate_mask(
        n_current=n_current,
        n_reference=n_reference,
        density=0.40,
        rng=rng,
    )

    similarity = rng.normal(
        loc=0.22,
        scale=0.10,
        size=(
            n_current,
            n_reference,
        ),
    )

    # True matches remain strong.
    for i, j in gt_matches:
        similarity[i, j] = rng.normal(
            loc=0.87,
            scale=0.05,
        )

    # --------------------------------------------------------
    # Make unmatched planes non-trivial:
    # give some of them moderate but incorrect candidates.
    # --------------------------------------------------------

    matched_current = {
        i
        for i, _ in gt_matches
    }

    matched_reference = {
        j
        for _, j in gt_matches
    }

    unmatched_current = [
        i
        for i in range(n_current)
        if i not in matched_current
    ]

    unmatched_reference = [
        j
        for j in range(n_reference)
        if j not in matched_reference
    ]

    for i in unmatched_current:

        number = min(
            int(rng.integers(1, 4)),
            n_reference,
        )

        refs = rng.choice(
            n_reference,
            size=number,
            replace=False,
        )

        for j in refs:

            j = int(j)

            candidate_mask[i, j] = True

            similarity[i, j] = rng.normal(
                loc=0.40,
                scale=0.08,
            )

    for j in unmatched_reference:

        number = min(
            int(rng.integers(1, 4)),
            n_current,
        )

        currents = rng.choice(
            n_current,
            size=number,
            replace=False,
        )

        for i in currents:

            i = int(i)

            candidate_mask[i, j] = True

            similarity[i, j] = max(
                similarity[i, j],
                rng.normal(
                    loc=0.40,
                    scale=0.08,
                ),
            )

    _retain_gt_edges(
        candidate_mask,
        gt_matches,
    )

    similarity = _clip_scores(similarity)

    return AssignmentCase(
        case_id=case_id,
        current_ids=_make_plane_ids(
            "c",
            n_current,
        ),
        reference_ids=_make_plane_ids(
            "r",
            n_reference,
        ),
        similarity=similarity,
        candidate_mask=candidate_mask,
        gt_matches=gt_matches,
        metadata={
            "scenario": "unmatched",
            "difficulty": "unmatched_unequal",
        },
    )


# ============================================================
# Dataset generation
# ============================================================

GENERATORS = {
    "normal": generate_normal_case,
    "hard": generate_hard_case,
    "unmatched": generate_unmatched_case,
}


def generate_split(
    split_name: str,
    output_dir: Path,
    total_cases: int,
    seed: int,
) -> None:
    """
    Generate one complete synthetic dataset split.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(seed)

    scenario_counts = _balanced_scenario_counts(
        total_cases
    )

    manifest_rows: list[dict] = []

    case_number = 0

    for scenario, count in scenario_counts.items():

        generator = GENERATORS[scenario]

        for _ in range(count):

            case_id = (
                f"{split_name}_"
                f"{scenario}_"
                f"{case_number:04d}"
            )

            case = generator(
                case_id=case_id,
                rng=rng,
            )

            case.metadata.update(
                {
                    "split": split_name,
                    "dataset_seed": seed,
                    "case_number": case_number,
                }
            )

            filename = f"{case_id}.npz"

            save_assignment_case(
                case,
                output_dir / filename,
            )

            summary = summarise_assignment_case(
                case
            )

            manifest_rows.append(
                {
                    "case_id":
                        case.case_id,

                    "split":
                        split_name,

                    "scenario":
                        scenario,

                    "n_current":
                        summary["n_current"],

                    "n_reference":
                        summary["n_reference"],

                    "n_candidates":
                        summary["n_candidates"],

                    "candidate_density":
                        summary["candidate_density"],

                    "n_gt_matches":
                        summary["n_gt_matches"],

                    "n_gt_unmatched_current":
                        summary[
                            "n_gt_unmatched_current"
                        ],

                    "n_gt_unmatched_reference":
                        summary[
                            "n_gt_unmatched_reference"
                        ],

                    "file":
                        filename,
                }
            )

            case_number += 1

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    manifest_path = (
        output_dir
        / "manifest.csv"
    )

    fieldnames = [
        "case_id",
        "split",
        "scenario",
        "n_current",
        "n_reference",
        "n_candidates",
        "candidate_density",
        "n_gt_matches",
        "n_gt_unmatched_current",
        "n_gt_unmatched_reference",
        "file",
    ]

    with manifest_path.open(
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
            manifest_rows
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print(
        f"\nGenerated {split_name}:"
    )

    print(
        f"  Output:      {output_dir}"
    )

    print(
        f"  Total cases: {total_cases}"
    )

    for scenario in (
        "normal",
        "hard",
        "unmatched",
    ):
        print(
            f"  {scenario:<10}: "
            f"{scenario_counts[scenario]}"
        )

    print(
        f"  Manifest:    {manifest_path}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic datasets for "
            "plane correspondence Assignment experiments."
        )
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/assignment"
        ),
        help=(
            "Root output directory. "
            "Default: data/assignment"
        ),
    )

    parser.add_argument(
        "--n-dev",
        type=int,
        default=300,
        help=(
            "Number of Synthetic DEV cases. "
            "Default: 300"
        ),
    )

    parser.add_argument(
        "--n-final",
        type=int,
        default=600,
        help=(
            "Number of Synthetic FINAL cases. "
            "Default: 600"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260927,
        help=(
            "Base random seed. "
            "Default: 20260927"
        ),
    )

    args = parser.parse_args()

    if args.n_dev <= 0:
        raise ValueError(
            "--n-dev must be positive"
        )

    if args.n_final <= 0:
        raise ValueError(
            "--n-final must be positive"
        )

    # DEV and FINAL deliberately use independent random streams.
    dev_seed = args.seed
    final_seed = args.seed + 10000

    generate_split(
        split_name="synthetic_dev",
        output_dir=(
            args.output_root
            / "synthetic_dev"
        ),
        total_cases=args.n_dev,
        seed=dev_seed,
    )

    generate_split(
        split_name="synthetic_final",
        output_dir=(
            args.output_root
            / "synthetic_final"
        ),
        total_cases=args.n_final,
        seed=final_seed,
    )

    print(
        "\nSynthetic Assignment dataset generation complete."
    )


if __name__ == "__main__":
    main()