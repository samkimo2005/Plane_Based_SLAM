"""
run_candidate_gating.py

Final benchmark runner for Candidate Gating.

This program:

    1. Loads the independent FINAL dataset.
    2. Loads the frozen parameters selected on DEV.
    3. Runs all four Candidate Gating algorithms.
    4. Evaluates them using the common evaluation procedure.
    5. Saves per-sample FINAL results.
    6. Saves an overall FINAL summary.

IMPORTANT
---------
No parameter tuning is performed here.

The FINAL dataset must not be used to modify algorithm
parameters.

Algorithms:

    [29] Motion-model-based distance gating
    [28] Contour-point gating
    [71] Normal-based statistical gating
    [72] PCPM-inspired gating
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from candidate_gating_common import (
    PROJECT_ROOT,
    load_candidate_sample,
)

from candidate_gating_evaluation import (
    GatingMetrics,
    evaluate_all_algorithms,
)


# ============================================================
# Paths
# ============================================================

FINAL_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "candidate_gating"
    / "final"
)

TUNING_DIR = (
    PROJECT_ROOT
    / "results"
    / "candidate_gating"
    / "tuning"
)

BEST_PARAMETERS_FILE = (
    TUNING_DIR
    / "best_candidate_gating_parameters.json"
)

FINAL_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "candidate_gating"
    / "final"
)

DETAILED_RESULTS_FILE = (
    FINAL_RESULTS_DIR
    / "candidate_gating_final_results.csv"
)

SUMMARY_FILE = (
    FINAL_RESULTS_DIR
    / "candidate_gating_final_summary.csv"
)

USED_PARAMETERS_FILE = (
    FINAL_RESULTS_DIR
    / "frozen_parameters_used.json"
)


# ============================================================
# Load frozen parameters
# ============================================================

def load_frozen_parameters() -> Dict[str, Dict]:
    """
    Load the parameter sets selected by tune_candidate_gating.py.

    Expected JSON structure:

        {
            "algorithms": {
                "motion_model": {
                    "parameters": {...}
                },
                ...
            }
        }
    """

    if not BEST_PARAMETERS_FILE.exists():

        raise FileNotFoundError(
            "\nFrozen parameter file was not found:\n"
            f"{BEST_PARAMETERS_FILE}\n\n"
            "Run tune_candidate_gating.py first."
        )

    with open(
        BEST_PARAMETERS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        tuning_data = json.load(
            file
        )

    if "algorithms" not in tuning_data:

        raise KeyError(
            "Frozen parameter file does not contain "
            "an 'algorithms' section."
        )

    algorithm_data = tuning_data[
        "algorithms"
    ]

    required_algorithms = [
        "motion_model",
        "contour_point",
        "normal_based",
        "pcpm",
    ]

    parameter_sets = {}

    for algorithm in required_algorithms:

        if algorithm not in algorithm_data:

            raise KeyError(
                f"Missing frozen parameters for: {algorithm}"
            )

        if "parameters" not in algorithm_data[
            algorithm
        ]:

            raise KeyError(
                f"Missing 'parameters' field for: {algorithm}"
            )

        parameter_sets[
            algorithm
        ] = algorithm_data[
            algorithm
        ][
            "parameters"
        ]

    return parameter_sets


# ============================================================
# Find FINAL samples
# ============================================================

def find_final_samples() -> List[Path]:
    """
    Find all FINAL .npz samples.
    """

    if not FINAL_DATA_DIR.exists():

        raise FileNotFoundError(
            "\nFINAL dataset folder does not exist:\n"
            f"{FINAL_DATA_DIR}\n\n"
            "Generate the FINAL dataset first."
        )

    sample_files = sorted(
        FINAL_DATA_DIR.glob(
            "sample_*.npz"
        )
    )

    if len(sample_files) == 0:

        raise FileNotFoundError(
            "\nNo FINAL samples were found in:\n"
            f"{FINAL_DATA_DIR}"
        )

    return sample_files


# ============================================================
# Run complete FINAL benchmark
# ============================================================

def run_final_benchmark(
    parameter_sets: Dict[str, Dict],
) -> List[GatingMetrics]:
    """
    Run all four algorithms over all FINAL samples.
    """

    sample_files = (
        find_final_samples()
    )

    total_samples = len(
        sample_files
    )

    all_results = []

    print()
    print(
        f"FINAL samples found : {total_samples}"
    )

    print()

    for (
        sample_number,
        sample_file,
    ) in enumerate(
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

                parameter_sets=parameter_sets,
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
# Save detailed results
# ============================================================

def save_detailed_results(
    results: List[GatingMetrics],
) -> None:
    """
    Save one row for every sample-algorithm combination.
    """

    FINAL_RESULTS_DIR.mkdir(
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
        DETAILED_RESULTS_FILE,
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
# Condition robustness
# ============================================================

def calculate_worst_condition_retention(
    algorithm_results: List[GatingMetrics],
) -> float:
    """
    Calculate the lowest mean retention across the
    synthetic experimental conditions.
    """

    conditions = sorted(
        set(
            result.condition
            for result
            in algorithm_results
        )
    )

    condition_means = []

    for condition in conditions:

        retention_values = [

            result.true_match_retention_rate

            for result
            in algorithm_results

            if result.condition
            == condition
        ]

        if len(
            retention_values
        ) > 0:

            condition_means.append(
                float(
                    np.mean(
                        retention_values
                    )
                )
            )

    if len(
        condition_means
    ) == 0:

        return 0.0

    return float(
        min(
            condition_means
        )
    )


# ============================================================
# Final summary
# ============================================================

def create_summary(
    results: List[GatingMetrics],
) -> List[Dict]:
    """
    Create overall FINAL statistics for each algorithm.
    """

    algorithms = sorted(
        set(
            result.algorithm
            for result
            in results
        )
    )

    summary_rows = []

    for algorithm in algorithms:

        algorithm_results = [

            result
            for result
            in results

            if result.algorithm
            == algorithm
        ]

        retention = np.asarray(
            [
                result.true_match_retention_rate

                for result
                in algorithm_results
            ],
            dtype=float,
        )

        reduction = np.asarray(
            [
                result.candidate_reduction_rate

                for result
                in algorithm_results
            ],
            dtype=float,
        )

        precision = np.asarray(
            [
                result.precision

                for result
                in algorithm_results
            ],
            dtype=float,
        )

        f1 = np.asarray(
            [
                result.f1_score

                for result
                in algorithm_results
            ],
            dtype=float,
        )

        runtime = np.asarray(
            [
                result.runtime_ms

                for result
                in algorithm_results
            ],
            dtype=float,
        )

        surviving_pairs = np.asarray(
            [
                result.surviving_pairs

                for result
                in algorithm_results
            ],
            dtype=float,
        )

        row = {

            "algorithm":
                algorithm,

            "samples":
                len(
                    algorithm_results
                ),

            "mean_retention":
                float(
                    np.mean(
                        retention
                    )
                ),

            "std_retention":
                float(
                    np.std(
                        retention
                    )
                ),

            "worst_condition_retention":
                calculate_worst_condition_retention(
                    algorithm_results
                ),

            "mean_reduction":
                float(
                    np.mean(
                        reduction
                    )
                ),

            "std_reduction":
                float(
                    np.std(
                        reduction
                    )
                ),

            "mean_precision":
                float(
                    np.mean(
                        precision
                    )
                ),

            "mean_f1":
                float(
                    np.mean(
                        f1
                    )
                ),

            "mean_surviving_pairs":
                float(
                    np.mean(
                        surviving_pairs
                    )
                ),

            "mean_runtime_ms":
                float(
                    np.mean(
                        runtime
                    )
                ),

            "std_runtime_ms":
                float(
                    np.std(
                        runtime
                    )
                ),
        }

        summary_rows.append(
            row
        )

    return summary_rows


# ============================================================
# Save summary
# ============================================================

def save_summary(
    summary_rows: List[Dict],
) -> None:
    """
    Save overall algorithm summary.
    """

    if len(
        summary_rows
    ) == 0:

        return

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )


# ============================================================
# Save exact frozen parameters used
# ============================================================

def save_used_parameters(
    parameter_sets: Dict[str, Dict],
) -> None:
    """
    Save a copy of the exact parameter configuration
    used for the FINAL benchmark.
    """

    with open(
        USED_PARAMETERS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "dataset": "FINAL",
                "parameter_status": "FROZEN",
                "parameters": parameter_sets,
            },
            file,
            indent=4,
        )


# ============================================================
# Terminal summary
# ============================================================

def print_summary(
    summary_rows: List[Dict],
) -> None:
    """
    Display overall FINAL benchmark results.
    """

    print()
    print("=" * 115)

    print(
        "Candidate Gating FINAL Benchmark"
    )

    print("=" * 115)

    print(

        f"{'Algorithm':<25}"

        f"{'Retention':>12}"

        f"{'Worst Ret.':>12}"

        f"{'Reduction':>12}"

        f"{'Precision':>12}"

        f"{'F1':>10}"

        f"{'Runtime ms':>14}"
    )

    print(
        "-" * 115
    )

    for row in summary_rows:

        print(

            f"{row['algorithm']:<25}"

            f"{row['mean_retention']:>12.3f}"

            f"{row['worst_condition_retention']:>12.3f}"

            f"{row['mean_reduction']:>12.3f}"

            f"{row['mean_precision']:>12.3f}"

            f"{row['mean_f1']:>10.3f}"

            f"{row['mean_runtime_ms']:>14.4f}"
        )

    print(
        "-" * 115
    )

    print()

    print(
        "Retention  = proportion of true plane matches preserved."
    )

    print(
        "Worst Ret. = lowest mean retention among test conditions."
    )

    print(
        "Reduction  = proportion of all possible candidate pairs removed."
    )

    print("=" * 115)
    print()


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print("=" * 80)

    print(
        "Candidate Gating FINAL Test"
    )

    print("=" * 80)

    print(
        f"FINAL dataset:"
    )

    print(
        FINAL_DATA_DIR
    )

    print()

    print(
        "Frozen parameter file:"
    )

    print(
        BEST_PARAMETERS_FILE
    )

    print()

    # --------------------------------------------------------
    # Load frozen DEV-selected parameters
    # --------------------------------------------------------

    parameter_sets = (
        load_frozen_parameters()
    )

    print(
        "Frozen parameters loaded:"
    )

    for (
        algorithm,
        parameters,
    ) in parameter_sets.items():

        print()

        print(
            f"  {algorithm}"
        )

        for (
            parameter_name,
            parameter_value,
        ) in parameters.items():

            print(
                f"    {parameter_name}: "
                f"{parameter_value}"
            )

    # --------------------------------------------------------
    # FINAL benchmark
    # --------------------------------------------------------

    results = (
        run_final_benchmark(
            parameter_sets
        )
    )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    FINAL_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_detailed_results(
        results
    )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    summary_rows = (
        create_summary(
            results
        )
    )

    save_summary(
        summary_rows
    )

    save_used_parameters(
        parameter_sets
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print_summary(
        summary_rows
    )

    print(
        "Detailed FINAL results:"
    )

    print(
        DETAILED_RESULTS_FILE
    )

    print()

    print(
        "FINAL summary:"
    )

    print(
        SUMMARY_FILE
    )

    print()

    print(
        "Frozen parameters used:"
    )

    print(
        USED_PARAMETERS_FILE
    )

    print()
    print("=" * 80)

    print(
        "FINAL Candidate Gating benchmark complete."
    )

    print("=" * 80)
    print()


if __name__ == "__main__":
    main()