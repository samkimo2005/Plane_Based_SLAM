"""
candidate_gating_evaluation.py

Common evaluation module for the Candidate Gating experiments.

This file evaluates the four candidate-gating algorithms using
exactly the same ground truth and metrics.

Algorithms:

    [29] Motion-model-based distance gating
    [28] Contour-point gating
    [71] Normal-based statistical gating
    [72] PCPM-inspired gating

Main metrics:

    - TP
    - FP
    - FN
    - TN

    - Precision
    - Recall / True Match Retention Rate
    - F1 score

    - Candidate Reduction Rate
    - Runtime

Candidate Gating objective
--------------------------

The most important metric is True Match Retention Rate because
a correct plane pair removed during Candidate Gating cannot be
recovered by later Similarity, Assignment or Bad-Match Rejection.

Candidate Reduction Rate measures how effectively the gate
reduces the number of pairs passed to downstream modules.

This file performs evaluation only.

It does NOT tune algorithm parameters.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from candidate_gating_common import (
    PROJECT_ROOT,
    CandidateSample,
    load_candidate_sample,
)

from motion_model_distance_gating import (
    run_on_sample as run_motion_model,
)

from contour_point_gating import (
    run_on_sample as run_contour_point,
)

from normal_based_gating import (
    run_on_sample as run_normal_based,
)

from pcpm_gating import (
    run_on_sample as run_pcpm,
)


# ============================================================
# USER SETTINGS
# ============================================================

# Evaluation is currently performed on DEV.
#
# Later, run_candidate_gating.py will handle the independent
# FINAL dataset using frozen parameters.
SPLIT = "dev"

# Save the baseline evaluation table.
SAVE_RESULTS = True


# ============================================================
# Paths
# ============================================================

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "candidate_gating"
    / SPLIT
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "candidate_gating"
    / "evaluation"
)

RESULTS_FILE = (
    RESULTS_DIR
    / f"{SPLIT}_baseline_evaluation.csv"
)


# ============================================================
# Evaluation data structure
# ============================================================

@dataclass
class GatingMetrics:
    """
    Evaluation result for one algorithm on one sample.
    """

    sample_id: str
    condition: str
    algorithm: str

    source_planes: int
    target_planes: int

    possible_pairs: int
    gt_pairs: int
    surviving_pairs: int

    tp: int
    fp: int
    fn: int
    tn: int

    precision: float
    recall: float
    f1_score: float

    true_match_retention_rate: float
    candidate_reduction_rate: float

    runtime_ms: float


# ============================================================
# Candidate pair helpers
# ============================================================

def candidates_to_pair_set(
    candidates: List[Dict],
) -> set[tuple[int, int]]:
    """
    Convert algorithm output into a set of:

        (source_plane_id, target_plane_id)

    Duplicate candidate pairs are automatically removed.
    """

    pair_set = set()

    for candidate in candidates:

        if "source_plane_id" not in candidate:
            raise KeyError(
                "Candidate output is missing "
                "'source_plane_id'."
            )

        if "target_plane_id" not in candidate:
            raise KeyError(
                "Candidate output is missing "
                "'target_plane_id'."
            )

        source_idx = int(
            candidate[
                "source_plane_id"
            ]
        )

        target_idx = int(
            candidate[
                "target_plane_id"
            ]
        )

        pair_set.add(
            (
                source_idx,
                target_idx,
            )
        )

    return pair_set


def ground_truth_pair_set(
    sample: CandidateSample,
) -> set[tuple[int, int]]:
    """
    Convert sample GT correspondence list into a set.
    """

    return {

        (
            int(source_idx),
            int(target_idx),
        )

        for (
            source_idx,
            target_idx
        ) in sample.gt_pairs
    }


# ============================================================
# Validation of algorithm output
# ============================================================

def validate_candidate_pairs(
    sample: CandidateSample,
    candidate_pairs: set[tuple[int, int]],
) -> None:
    """
    Check that every returned pair contains valid plane indices.
    """

    number_source = len(
        sample.source
    )

    number_target = len(
        sample.target
    )

    for (
        source_idx,
        target_idx,
    ) in candidate_pairs:

        if not (
            0
            <= source_idx
            < number_source
        ):
            raise ValueError(
                "Algorithm returned invalid source-plane index: "
                f"{source_idx}"
            )

        if not (
            0
            <= target_idx
            < number_target
        ):
            raise ValueError(
                "Algorithm returned invalid target-plane index: "
                f"{target_idx}"
            )


# ============================================================
# Metric calculation
# ============================================================

def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """
    Divide safely.

    Returns 0 when denominator is zero.
    """

    if denominator == 0:
        return 0.0

    return float(
        numerator
        / denominator
    )


def calculate_metrics(
    sample: CandidateSample,
    candidates: List[Dict],
    algorithm_name: str,
    runtime_ms: float,
    sample_id: str,
) -> GatingMetrics:
    """
    Calculate all Candidate Gating metrics for one sample.
    """

    gt_pairs = ground_truth_pair_set(
        sample
    )

    predicted_pairs = (
        candidates_to_pair_set(
            candidates
        )
    )

    validate_candidate_pairs(
        sample=sample,
        candidate_pairs=predicted_pairs,
    )

    # --------------------------------------------------------
    # Total pair space
    # --------------------------------------------------------

    number_source = len(
        sample.source
    )

    number_target = len(
        sample.target
    )

    possible_pairs = (
        number_source
        * number_target
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    true_positive_pairs = (
        gt_pairs
        & predicted_pairs
    )

    false_positive_pairs = (
        predicted_pairs
        - gt_pairs
    )

    false_negative_pairs = (
        gt_pairs
        - predicted_pairs
    )

    tp = len(
        true_positive_pairs
    )

    fp = len(
        false_positive_pairs
    )

    fn = len(
        false_negative_pairs
    )

    tn = (
        possible_pairs
        - tp
        - fp
        - fn
    )

    if tn < 0:
        raise ValueError(
            "Calculated TN is negative. "
            "Check candidate-pair output."
        )

    # --------------------------------------------------------
    # Standard classification metrics
    # --------------------------------------------------------

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    f1_score = safe_divide(
        2.0
        * precision
        * recall,

        precision
        + recall,
    )

    # --------------------------------------------------------
    # Candidate-Gating-specific metrics
    # --------------------------------------------------------

    true_match_retention_rate = recall

    candidate_reduction_rate = (
        1.0
        - safe_divide(
            len(predicted_pairs),
            possible_pairs,
        )
    )

    # --------------------------------------------------------
    # Experimental condition
    # --------------------------------------------------------

    condition_name = "unknown"

    if sample.metadata is not None:

        condition_data = (
            sample.metadata.get(
                "condition",
                {}
            )
        )

        condition_name = (
            condition_data.get(
                "name",
                "unknown",
            )
        )

    return GatingMetrics(

        sample_id=sample_id,

        condition=condition_name,

        algorithm=algorithm_name,


        source_planes=number_source,

        target_planes=number_target,


        possible_pairs=possible_pairs,

        gt_pairs=len(
            gt_pairs
        ),

        surviving_pairs=len(
            predicted_pairs
        ),


        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,


        precision=precision,

        recall=recall,

        f1_score=f1_score,


        true_match_retention_rate=(
            true_match_retention_rate
        ),

        candidate_reduction_rate=(
            candidate_reduction_rate
        ),

        runtime_ms=float(
            runtime_ms
        ),
    )


# ============================================================
# Generic algorithm evaluator
# ============================================================

def evaluate_algorithm(
    sample: CandidateSample,
    algorithm_function: Callable,
    algorithm_name: str,
    parameters: Optional[Dict] = None,
    sample_id: str = "unknown",
) -> GatingMetrics:
    """
    Run one algorithm on one sample and evaluate it.

    algorithm_function must have the form:

        algorithm_function(
            sample,
            parameters
        )

    and return a list of candidate dictionaries.
    """

    start_time = (
        time.perf_counter()
    )

    candidates = (
        algorithm_function(
            sample,
            parameters,
        )
    )

    end_time = (
        time.perf_counter()
    )

    runtime_ms = (
        end_time
        - start_time
    ) * 1000.0

    metrics = (
        calculate_metrics(

            sample=sample,

            candidates=candidates,

            algorithm_name=(
                algorithm_name
            ),

            runtime_ms=runtime_ms,

            sample_id=sample_id,
        )
    )

    return metrics


# ============================================================
# Evaluate all four algorithms on one sample
# ============================================================

def evaluate_all_algorithms(
    sample: CandidateSample,
    sample_id: str,
    parameter_sets: Optional[
        Dict[str, Dict]
    ] = None,
) -> List[GatingMetrics]:
    """
    Evaluate all four Candidate Gating algorithms.

    parameter_sets can later be supplied by the tuning program.

    Example:

        parameter_sets = {

            "motion_model":
                {...},

            "contour_point":
                {...},

            "normal_based":
                {...},

            "pcpm":
                {...},
        }

    If None, each algorithm uses its own default parameters.
    """

    if parameter_sets is None:
        parameter_sets = {}

    results = []

    # --------------------------------------------------------
    # [29] Motion-model-based distance constraint
    # --------------------------------------------------------

    results.append(

        evaluate_algorithm(

            sample=sample,

            algorithm_function=(
                run_motion_model
            ),

            algorithm_name=(
                "motion_model_distance"
            ),

            parameters=(
                parameter_sets.get(
                    "motion_model"
                )
            ),

            sample_id=sample_id,
        )
    )

    # --------------------------------------------------------
    # [28] Contour-point gating
    # --------------------------------------------------------

    results.append(

        evaluate_algorithm(

            sample=sample,

            algorithm_function=(
                run_contour_point
            ),

            algorithm_name=(
                "contour_point"
            ),

            parameters=(
                parameter_sets.get(
                    "contour_point"
                )
            ),

            sample_id=sample_id,
        )
    )

    # --------------------------------------------------------
    # [71] Normal-based statistical gating
    # --------------------------------------------------------

    results.append(

        evaluate_algorithm(

            sample=sample,

            algorithm_function=(
                run_normal_based
            ),

            algorithm_name=(
                "normal_based"
            ),

            parameters=(
                parameter_sets.get(
                    "normal_based"
                )
            ),

            sample_id=sample_id,
        )
    )

    # --------------------------------------------------------
    # [72] PCPM-inspired gating
    # --------------------------------------------------------

    results.append(

        evaluate_algorithm(

            sample=sample,

            algorithm_function=(
                run_pcpm
            ),

            algorithm_name=(
                "pcpm"
            ),

            parameters=(
                parameter_sets.get(
                    "pcpm"
                )
            ),

            sample_id=sample_id,
        )
    )

    return results


# ============================================================
# Dataset-level evaluation
# ============================================================

def find_sample_files(
    data_dir: Path,
) -> List[Path]:
    """
    Find generated NPZ samples in deterministic order.
    """

    sample_files = sorted(
        data_dir.glob(
            "sample_*.npz"
        )
    )

    if len(sample_files) == 0:

        raise FileNotFoundError(
            "\nNo Candidate Gating samples were found in:\n"
            f"{data_dir}\n\n"
            "Run generate_candidate_gating.py first."
        )

    return sample_files


def evaluate_dataset(
    data_dir: Path,
    parameter_sets: Optional[
        Dict[str, Dict]
    ] = None,
) -> List[GatingMetrics]:
    """
    Evaluate all algorithms over all samples in one dataset.
    """

    sample_files = (
        find_sample_files(
            data_dir
        )
    )

    all_results = []

    total_samples = len(
        sample_files
    )

    print()

    print(
        f"Evaluating {total_samples} samples..."
    )

    print()

    for sample_number, sample_file in enumerate(
        sample_files,
        start=1,
    ):

        sample_id = (
            sample_file.stem
        )

        sample = (
            load_candidate_sample(
                sample_file
            )
        )

        sample_results = (
            evaluate_all_algorithms(

                sample=sample,

                sample_id=sample_id,

                parameter_sets=(
                    parameter_sets
                ),
            )
        )

        all_results.extend(
            sample_results
        )

        print(
            f"[{sample_number:03d}/"
            f"{total_samples:03d}] "
            f"{sample_id}"
        )

    return all_results


# ============================================================
# CSV output
# ============================================================

def save_results_csv(
    results: List[GatingMetrics],
    output_file: Path,
) -> None:
    """
    Save per-sample evaluation results.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [

        asdict(
            result
        )

        for result
        in results
    ]

    if len(rows) == 0:
        return

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Dataset summary
# ============================================================

def summarise_results(
    results: List[GatingMetrics],
) -> Dict[str, Dict[str, float]]:
    """
    Calculate average metrics for each algorithm.
    """

    algorithms = sorted(
        set(
            result.algorithm
            for result
            in results
        )
    )

    summary = {}

    for algorithm in algorithms:

        algorithm_results = [

            result
            for result
            in results

            if result.algorithm
            == algorithm
        ]

        retention_values = np.asarray(
            [
                result
                .true_match_retention_rate

                for result
                in algorithm_results
            ],
            dtype=float,
        )

        reduction_values = np.asarray(
            [
                result
                .candidate_reduction_rate

                for result
                in algorithm_results
            ],
            dtype=float,
        )

        precision_values = np.asarray(
            [
                result.precision

                for result
                in algorithm_results
            ],
            dtype=float,
        )

        f1_values = np.asarray(
            [
                result.f1_score

                for result
                in algorithm_results
            ],
            dtype=float,
        )

        runtime_values = np.asarray(
            [
                result.runtime_ms

                for result
                in algorithm_results
            ],
            dtype=float,
        )

        summary[
            algorithm
        ] = {

            "mean_retention":
                float(
                    np.mean(
                        retention_values
                    )
                ),

            "std_retention":
                float(
                    np.std(
                        retention_values
                    )
                ),

            "mean_reduction":
                float(
                    np.mean(
                        reduction_values
                    )
                ),

            "std_reduction":
                float(
                    np.std(
                        reduction_values
                    )
                ),

            "mean_precision":
                float(
                    np.mean(
                        precision_values
                    )
                ),

            "mean_f1":
                float(
                    np.mean(
                        f1_values
                    )
                ),

            "mean_runtime_ms":
                float(
                    np.mean(
                        runtime_values
                    )
                ),
        }

    return summary


# ============================================================
# Pretty terminal output
# ============================================================

def print_summary(
    summary: Dict[
        str,
        Dict[str, float]
    ]
) -> None:
    """
    Print dataset-level summary.
    """

    print()
    print("=" * 100)

    print(
        "Candidate Gating DEV Baseline Evaluation"
    )

    print("=" * 100)

    header = (

        f"{'Algorithm':<25}"

        f"{'Retention':>12}"

        f"{'Reduction':>12}"

        f"{'Precision':>12}"

        f"{'F1':>10}"

        f"{'Runtime ms':>14}"
    )

    print(
        header
    )

    print(
        "-" * 100
    )

    for (
        algorithm,
        values,
    ) in summary.items():

        print(

            f"{algorithm:<25}"

            f"{values['mean_retention']:>12.3f}"

            f"{values['mean_reduction']:>12.3f}"

            f"{values['mean_precision']:>12.3f}"

            f"{values['mean_f1']:>10.3f}"

            f"{values['mean_runtime_ms']:>14.4f}"
        )

    print(
        "-" * 100
    )

    print()

    print(
        "Retention = proportion of GT matches surviving the gate."
    )

    print(
        "Reduction = proportion of all possible pairs removed."
    )

    print("=" * 100)
    print()


# ============================================================
# Direct execution
# ============================================================

def main() -> None:

    print()
    print("=" * 70)

    print(
        "Candidate Gating Evaluation"
    )

    print("=" * 70)

    print(
        f"Dataset split : {SPLIT.upper()}"
    )

    print(
        f"Dataset path  : {DATA_DIR}"
    )

    print()

    # --------------------------------------------------------
    # Evaluate complete DEV dataset using current defaults
    # --------------------------------------------------------

    results = evaluate_dataset(
        data_dir=DATA_DIR,
        parameter_sets=None,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = summarise_results(
        results
    )

    print_summary(
        summary
    )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    if SAVE_RESULTS:

        save_results_csv(
            results=results,
            output_file=RESULTS_FILE,
        )

        print(
            "Detailed results saved to:"
        )

        print(
            RESULTS_FILE
        )

        print()


if __name__ == "__main__":
    main()