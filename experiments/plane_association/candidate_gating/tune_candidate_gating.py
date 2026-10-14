"""
tune_candidate_gating.py

Parameter tuning for the Candidate Gating stage.

This program:

    1. Loads the DEV synthetic benchmark.
    2. Tests multiple parameter combinations for each algorithm.
    3. Evaluates every configuration using the common evaluation module.
    4. Prioritises True Match Retention over Candidate Reduction.
    5. Saves all tuning results.
    6. Saves the best parameter set for each algorithm.

IMPORTANT
---------
Only DEV data are used here.

FINAL data must NOT be used for tuning.

After this program has been completed, the selected parameters
will be frozen before the FINAL dataset is generated and tested.
"""

from __future__ import annotations

import csv
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from candidate_gating_common import (
    PROJECT_ROOT,
    CandidateSample,
    load_candidate_sample,
)

from candidate_gating_evaluation import (
    GatingMetrics,
    evaluate_algorithm,
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
# Paths
# ============================================================

DEV_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "candidate_gating"
    / "dev"
)

TUNING_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "candidate_gating"
    / "tuning"
)

ALL_RESULTS_FILE = (
    TUNING_RESULTS_DIR
    / "candidate_gating_tuning_results.csv"
)

BEST_PARAMETERS_FILE = (
    TUNING_RESULTS_DIR
    / "best_candidate_gating_parameters.json"
)


# ============================================================
# Tuning objective
# ============================================================

# Candidate Gating must strongly prioritise keeping real matches.
#
# A false candidate can still be removed later by:
#
#     Similarity
#     Assignment
#     Bad-Match Rejection
#
# but a true match rejected here is permanently lost.

WEIGHT_MEAN_RETENTION = 0.60
WEIGHT_WORST_CONDITION_RETENTION = 0.20
WEIGHT_REDUCTION = 0.15
WEIGHT_F1 = 0.05


# ============================================================
# Parameter search spaces
# ============================================================

MOTION_MODEL_PARAMETER_GRID = {

    "normal_dot_threshold": [
        0.90,
        0.925,
        0.95,
        0.965,
        0.98,
    ],

    "offset_threshold_m": [
        0.03,
        0.05,
        0.08,
        0.10,
        0.15,
    ],
}


CONTOUR_POINT_PARAMETER_GRID = {

    "contour_distance_threshold_m": [
        0.05,
        0.08,
        0.10,
        0.15,
        0.20,
    ],

    "normal_angle_threshold_deg": [
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
    ],
}


NORMAL_BASED_PARAMETER_GRID = {

    # Different chi-square compatibility levels.
    #
    # 4.605 : approximately 90%
    # 5.991 : approximately 95%
    # 7.378 : approximately 97.5%
    # 9.210 : approximately 99%
    #
    "chi_square_threshold": [
        4.605,
        5.991,
        7.378,
        9.210,
    ],
}


PCPM_PARAMETER_GRID = {

    "centroid_top_k": [
        1,
        2,
        3,
        4,
    ],

    "normal_top_k": [
        1,
        2,
        3,
        4,
    ],
}


# ============================================================
# Fixed normal-based settings
# ============================================================

# These are not being searched.
#
# They describe how the project-adapted [71] method obtains
# uncertainty information.

NORMAL_BASED_FIXED_PARAMETERS = {

    "default_normal_sigma_deg": 2.0,

    "default_pose_rotation_sigma_deg": 2.0,

    "minimum_sigma_deg": 0.10,

    "use_sample_uncertainty": True,
}


# ============================================================
# Parameter grid helper
# ============================================================

def generate_parameter_combinations(
    parameter_grid: Dict,
) -> List[Dict]:
    """
    Convert a parameter grid into every possible combination.

    Example:

        {
            "a": [1, 2],
            "b": [3, 4]
        }

    becomes:

        {"a": 1, "b": 3}
        {"a": 1, "b": 4}
        {"a": 2, "b": 3}
        {"a": 2, "b": 4}
    """

    parameter_names = list(
        parameter_grid.keys()
    )

    parameter_values = [

        parameter_grid[name]

        for name
        in parameter_names
    ]

    combinations = []

    for values in itertools.product(
        *parameter_values
    ):

        configuration = {

            name: value

            for name, value
            in zip(
                parameter_names,
                values,
            )
        }

        combinations.append(
            configuration
        )

    return combinations


# ============================================================
# Load DEV benchmark once
# ============================================================

def load_dev_dataset() -> List[
    Tuple[str, CandidateSample]
]:
    """
    Load all DEV samples once so tuning does not repeatedly
    read the same files from disk.
    """

    sample_files = sorted(
        DEV_DATA_DIR.glob(
            "sample_*.npz"
        )
    )

    if len(sample_files) == 0:

        raise FileNotFoundError(
            "\nNo DEV Candidate Gating dataset found:\n"
            f"{DEV_DATA_DIR}\n\n"
            "Run generate_candidate_gating.py first "
            "with SPLIT = 'dev'."
        )

    dataset = []

    print(
        f"Loading {len(sample_files)} DEV samples..."
    )

    for sample_file in sample_files:

        sample = load_candidate_sample(
            sample_file
        )

        dataset.append(
            (
                sample_file.stem,
                sample,
            )
        )

    print(
        "DEV dataset loaded successfully."
    )

    print()

    return dataset


# ============================================================
# Evaluate one parameter configuration
# ============================================================

def evaluate_configuration(
    dataset: List[
        Tuple[str, CandidateSample]
    ],
    algorithm_function: Callable,
    algorithm_name: str,
    parameters: Dict,
) -> List[GatingMetrics]:
    """
    Evaluate one parameter configuration across the
    complete DEV dataset.
    """

    results = []

    for (
        sample_id,
        sample,
    ) in dataset:

        metrics = evaluate_algorithm(

            sample=sample,

            algorithm_function=(
                algorithm_function
            ),

            algorithm_name=(
                algorithm_name
            ),

            parameters=parameters,

            sample_id=sample_id,
        )

        results.append(
            metrics
        )

    return results


# ============================================================
# Condition-level robustness
# ============================================================

def calculate_condition_retention(
    results: List[GatingMetrics],
) -> Dict[str, float]:
    """
    Calculate mean True Match Retention for every
    experimental condition.
    """

    condition_names = sorted(

        set(
            result.condition
            for result
            in results
        )
    )

    condition_retention = {}

    for condition in condition_names:

        values = [

            result
            .true_match_retention_rate

            for result
            in results

            if result.condition
            == condition
        ]

        condition_retention[
            condition
        ] = float(
            np.mean(
                values
            )
        )

    return condition_retention


# ============================================================
# Summarise one parameter configuration
# ============================================================

def summarise_configuration(
    results: List[GatingMetrics],
) -> Dict[str, float]:
    """
    Convert sample-level results into one set of
    tuning statistics.
    """

    retention = np.asarray(
        [
            result
            .true_match_retention_rate

            for result
            in results
        ],
        dtype=float,
    )

    reduction = np.asarray(
        [
            result
            .candidate_reduction_rate

            for result
            in results
        ],
        dtype=float,
    )

    precision = np.asarray(
        [
            result.precision

            for result
            in results
        ],
        dtype=float,
    )

    f1 = np.asarray(
        [
            result.f1_score

            for result
            in results
        ],
        dtype=float,
    )

    runtime = np.asarray(
        [
            result.runtime_ms

            for result
            in results
        ],
        dtype=float,
    )

    condition_retention = (
        calculate_condition_retention(
            results
        )
    )

    if len(condition_retention) > 0:

        worst_condition_retention = min(
            condition_retention.values()
        )

    else:

        worst_condition_retention = 0.0

    mean_retention = float(
        np.mean(
            retention
        )
    )

    mean_reduction = float(
        np.mean(
            reduction
        )
    )

    mean_precision = float(
        np.mean(
            precision
        )
    )

    mean_f1 = float(
        np.mean(
            f1
        )
    )

    mean_runtime_ms = float(
        np.mean(
            runtime
        )
    )

    # --------------------------------------------------------
    # Candidate Gating tuning score
    # --------------------------------------------------------

    score = (

        WEIGHT_MEAN_RETENTION
        * mean_retention

        +

        WEIGHT_WORST_CONDITION_RETENTION
        * worst_condition_retention

        +

        WEIGHT_REDUCTION
        * mean_reduction

        +

        WEIGHT_F1
        * mean_f1
    )

    return {

        "score":
            float(score),

        "mean_retention":
            mean_retention,

        "worst_condition_retention":
            float(
                worst_condition_retention
            ),

        "mean_reduction":
            mean_reduction,

        "mean_precision":
            mean_precision,

        "mean_f1":
            mean_f1,

        "mean_runtime_ms":
            mean_runtime_ms,
    }


# ============================================================
# Tune one algorithm
# ============================================================

def tune_algorithm(
    dataset: List[
        Tuple[str, CandidateSample]
    ],
    algorithm_name: str,
    algorithm_function: Callable,
    parameter_grid: Dict,
    fixed_parameters: Dict | None = None,
) -> Tuple[
    Dict,
    Dict,
    List[Dict],
]:
    """
    Grid-search one Candidate Gating algorithm.

    Returns
    -------
    best_parameters

    best_summary

    all_configuration_rows
    """

    parameter_combinations = (
        generate_parameter_combinations(
            parameter_grid
        )
    )

    number_configurations = len(
        parameter_combinations
    )

    print()
    print("=" * 90)

    print(
        f"Tuning: {algorithm_name}"
    )

    print(
        f"Parameter configurations: "
        f"{number_configurations}"
    )

    print("=" * 90)

    all_rows = []

    best_parameters = None
    best_summary = None

    for (
        configuration_number,
        searched_parameters,
    ) in enumerate(
        parameter_combinations,
        start=1,
    ):

        # ----------------------------------------------------
        # Merge searched + fixed parameters
        # ----------------------------------------------------

        parameters = {}

        if fixed_parameters is not None:

            parameters.update(
                fixed_parameters
            )

        parameters.update(
            searched_parameters
        )

        # ----------------------------------------------------
        # Evaluate complete DEV benchmark
        # ----------------------------------------------------

        sample_results = (
            evaluate_configuration(

                dataset=dataset,

                algorithm_function=(
                    algorithm_function
                ),

                algorithm_name=(
                    algorithm_name
                ),

                parameters=parameters,
            )
        )

        summary = (
            summarise_configuration(
                sample_results
            )
        )

        # ----------------------------------------------------
        # Store row
        # ----------------------------------------------------

        row = {

            "algorithm":
                algorithm_name,

            "configuration":
                configuration_number,

            "parameters":
                json.dumps(
                    parameters,
                    sort_keys=True,
                ),

            **summary,
        }

        all_rows.append(
            row
        )

        # ----------------------------------------------------
        # Display progress
        # ----------------------------------------------------

        print(

            f"[{configuration_number:02d}/"
            f"{number_configurations:02d}] "

            f"score="
            f"{summary['score']:.4f} | "

            f"ret="
            f"{summary['mean_retention']:.3f} | "

            f"worst="
            f"{summary['worst_condition_retention']:.3f} | "

            f"red="
            f"{summary['mean_reduction']:.3f} | "

            f"{searched_parameters}"
        )

        # ----------------------------------------------------
        # Select best configuration
        #
        # Primary:
        #     tuning score
        #
        # Tie-break:
        #     higher retention
        #     higher worst-condition retention
        #     higher reduction
        # ----------------------------------------------------

        if best_summary is None:

            best_parameters = (
                parameters.copy()
            )

            best_summary = (
                summary.copy()
            )

        else:

            current_key = (

                summary[
                    "score"
                ],

                summary[
                    "mean_retention"
                ],

                summary[
                    "worst_condition_retention"
                ],

                summary[
                    "mean_reduction"
                ],
            )

            best_key = (

                best_summary[
                    "score"
                ],

                best_summary[
                    "mean_retention"
                ],

                best_summary[
                    "worst_condition_retention"
                ],

                best_summary[
                    "mean_reduction"
                ],
            )

            if current_key > best_key:

                best_parameters = (
                    parameters.copy()
                )

                best_summary = (
                    summary.copy()
                )

    print()
    print(
        f"Best parameters for {algorithm_name}:"
    )

    print(
        json.dumps(
            best_parameters,
            indent=4,
        )
    )

    print()

    print(
        f"Best tuning score : "
        f"{best_summary['score']:.4f}"
    )

    print(
        f"Mean retention    : "
        f"{best_summary['mean_retention']:.4f}"
    )

    print(
        f"Worst condition   : "
        f"{best_summary['worst_condition_retention']:.4f}"
    )

    print(
        f"Mean reduction    : "
        f"{best_summary['mean_reduction']:.4f}"
    )

    return (
        best_parameters,
        best_summary,
        all_rows,
    )


# ============================================================
# Save tuning table
# ============================================================

def save_tuning_results(
    rows: List[Dict],
) -> None:
    """
    Save every tested parameter configuration.
    """

    TUNING_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if len(rows) == 0:
        return

    with open(
        ALL_RESULTS_FILE,
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
# Save selected parameters
# ============================================================

def save_best_parameters(
    best_results: Dict,
) -> None:
    """
    Save parameters that will later be frozen for FINAL testing.
    """

    TUNING_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {

        "dataset":
            "DEV",

        "selection_method": {

            "mean_retention_weight":
                WEIGHT_MEAN_RETENTION,

            "worst_condition_retention_weight":
                WEIGHT_WORST_CONDITION_RETENTION,

            "candidate_reduction_weight":
                WEIGHT_REDUCTION,

            "f1_weight":
                WEIGHT_F1,
        },

        "algorithms":
            best_results,
    }

    with open(
        BEST_PARAMETERS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print("=" * 90)

    print(
        "Candidate Gating Parameter Tuning"
    )

    print("=" * 90)

    print(
        "Dataset:"
    )

    print(
        DEV_DATA_DIR
    )

    print()

    print(
        "Tuning objective:"
    )

    print(
        f"  Mean retention             : "
        f"{WEIGHT_MEAN_RETENTION:.2f}"
    )

    print(
        f"  Worst-condition retention  : "
        f"{WEIGHT_WORST_CONDITION_RETENTION:.2f}"
    )

    print(
        f"  Candidate reduction        : "
        f"{WEIGHT_REDUCTION:.2f}"
    )

    print(
        f"  F1                         : "
        f"{WEIGHT_F1:.2f}"
    )

    print()

    # --------------------------------------------------------
    # Load DEV benchmark
    # --------------------------------------------------------

    dataset = load_dev_dataset()

    all_tuning_rows = []

    best_results = {}

    # ========================================================
    # [29] Motion-model-based distance gating
    # ========================================================

    (
        best_parameters,
        best_summary,
        rows,
    ) = tune_algorithm(

        dataset=dataset,

        algorithm_name=(
            "motion_model"
        ),

        algorithm_function=(
            run_motion_model
        ),

        parameter_grid=(
            MOTION_MODEL_PARAMETER_GRID
        ),
    )

    all_tuning_rows.extend(
        rows
    )

    best_results[
        "motion_model"
    ] = {

        "parameters":
            best_parameters,

        "dev_metrics":
            best_summary,
    }

    # ========================================================
    # [28] Contour-point gating
    # ========================================================

    (
        best_parameters,
        best_summary,
        rows,
    ) = tune_algorithm(

        dataset=dataset,

        algorithm_name=(
            "contour_point"
        ),

        algorithm_function=(
            run_contour_point
        ),

        parameter_grid=(
            CONTOUR_POINT_PARAMETER_GRID
        ),
    )

    all_tuning_rows.extend(
        rows
    )

    best_results[
        "contour_point"
    ] = {

        "parameters":
            best_parameters,

        "dev_metrics":
            best_summary,
    }

    # ========================================================
    # [71] Normal-based statistical gating
    # ========================================================

    (
        best_parameters,
        best_summary,
        rows,
    ) = tune_algorithm(

        dataset=dataset,

        algorithm_name=(
            "normal_based"
        ),

        algorithm_function=(
            run_normal_based
        ),

        parameter_grid=(
            NORMAL_BASED_PARAMETER_GRID
        ),

        fixed_parameters=(
            NORMAL_BASED_FIXED_PARAMETERS
        ),
    )

    all_tuning_rows.extend(
        rows
    )

    best_results[
        "normal_based"
    ] = {

        "parameters":
            best_parameters,

        "dev_metrics":
            best_summary,
    }

    # ========================================================
    # [72] PCPM-inspired gating
    # ========================================================

    (
        best_parameters,
        best_summary,
        rows,
    ) = tune_algorithm(

        dataset=dataset,

        algorithm_name=(
            "pcpm"
        ),

        algorithm_function=(
            run_pcpm
        ),

        parameter_grid=(
            PCPM_PARAMETER_GRID
        ),
    )

    all_tuning_rows.extend(
        rows
    )

    best_results[
        "pcpm"
    ] = {

        "parameters":
            best_parameters,

        "dev_metrics":
            best_summary,
    }

    # ========================================================
    # Save
    # ========================================================

    save_tuning_results(
        all_tuning_rows
    )

    save_best_parameters(
        best_results
    )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 90)

    print(
        "TUNING COMPLETE"
    )

    print("=" * 90)

    for (
        algorithm,
        result,
    ) in best_results.items():

        print()
        print(
            algorithm
        )

        print(
            "  Parameters:"
        )

        for (
            parameter_name,
            parameter_value,
        ) in (
            result[
                "parameters"
            ].items()
        ):

            print(
                f"    {parameter_name}: "
                f"{parameter_value}"
            )

        print(
            "  DEV score: "
            f"{result['dev_metrics']['score']:.4f}"
        )

        print(
            "  Retention: "
            f"{result['dev_metrics']['mean_retention']:.4f}"
        )

        print(
            "  Reduction: "
            f"{result['dev_metrics']['mean_reduction']:.4f}"
        )

    print()
    print(
        "All tuning results:"
    )

    print(
        ALL_RESULTS_FILE
    )

    print()

    print(
        "Selected parameters:"
    )

    print(
        BEST_PARAMETERS_FILE
    )

    print()
    print("=" * 90)

    print(
        "Parameters are now ready to be frozen "
        "before FINAL testing."
    )

    print("=" * 90)
    print()


if __name__ == "__main__":
    main()