from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np

import run_multiplane_plane_detection_benchmark as mp


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEV_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic_multiplane_benchmark"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "plane_detection"
    / "tuning"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "plane_detection_tuning_history.csv"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "best_detection_parameters.json"
)


# ============================================================
# 2. DEV subset
#
# Existing multi-plane benchmark:
#
#   3 plane counts
# × 2 noise levels
# × 2 outlier levels
# × 5 repeats
# = 60 samples
#
# For minimal-compute tuning, use repeats 1 and 2 only:
#
#   3 × 2 × 2 × 2 = 24 DEV cases
#
# The remaining old samples are still development/preliminary
# data and are NOT treated as untouched FINAL data.
# ============================================================

DEV_REPEAT_IDS = {
    1,
    2,
}

EXPECTED_DEV_CASES = 24

NUM_COORDINATE_PASSES = 2


# ============================================================
# 3. Tuning philosophy
#
# RANSAC and Hough already showed stable behaviour in the
# preliminary multi-plane benchmark, so their parameters are
# retained.
#
# Only the methods that showed a clear problem are calibrated:
#
#   Multi-Object RANSAC:
#       over-detection / false planes
#
#   Surface-Normal / Surfel:
#       some over-segmentation
#
#   Region Growing:
#       missed planes, especially harder tilted scenes
#
# Common evaluation rules are NEVER tuned here:
#
#   FINAL_DISTANCE_THRESHOLD
#   MATCH_NORMAL_THRESHOLD_DEG
#   MATCH_OFFSET_THRESHOLD_M
#   MIN_PLANE_SUPPORT
#   MAX_PLANES
#   K_NEIGHBORS
# ============================================================


TUNING_GROUPS = {
    "Multi-Object RANSAC": [
        (
            "MULTI_NORMAL_CLUSTER_DEG",
            [
                3.0,
                5.0,
                8.0,
                10.0,
            ],
        ),
        (
            "MULTI_OFFSET_CLUSTER_M",
            [
                0.03,
                0.05,
                0.07,
                0.10,
            ],
        ),
        (
            "MULTI_MIN_CLUSTER_VOTES",
            [
                3,
                4,
                5,
                6,
                8,
            ],
        ),
    ],

    "Surface-Normal / Surfel": [
        (
            "SURFEL_NORMAL_GROUP_DEG",
            [
                8.0,
                12.0,
                15.0,
                20.0,
            ],
        ),
        (
            "SURFEL_OFFSET_NMS_M",
            [
                0.05,
                0.08,
                0.10,
                0.12,
            ],
        ),
    ],

    "Region Growing": [
        (
            "RG_SMOOTHNESS_DEG",
            [
                10.0,
                15.0,
                20.0,
                25.0,
            ],
        ),
        (
            "RG_LOCAL_DISTANCE_M",
            [
                0.02,
                0.03,
                0.04,
                0.05,
            ],
        ),
    ],
}


# ============================================================
# 4. Parameters to record / freeze
# ============================================================

PARAMETER_NAMES = [
    # Common evaluation / structure
    "FINAL_DISTANCE_THRESHOLD",
    "MATCH_NORMAL_THRESHOLD_DEG",
    "MATCH_OFFSET_THRESHOLD_M",
    "MIN_PLANE_SUPPORT",
    "MAX_PLANES",
    "K_NEIGHBORS",

    # RANSAC
    "RANSAC_ITERATIONS",

    # Multi-Object RANSAC
    "MULTI_HYPOTHESES",
    "MULTI_MIN_HYPOTHESIS_SUPPORT",
    "MULTI_NORMAL_CLUSTER_DEG",
    "MULTI_OFFSET_CLUSTER_M",
    "MULTI_MIN_CLUSTER_VOTES",

    # Surfel
    "SURFEL_MAX_CURVATURE",
    "SURFEL_NORMAL_GROUP_DEG",
    "SURFEL_OFFSET_BIN_M",
    "SURFEL_OFFSET_GROUP_M",
    "SURFEL_OFFSET_NMS_M",

    # Region Growing
    "RG_SMOOTHNESS_DEG",
    "RG_LOCAL_DISTANCE_M",
    "RG_MIN_REGION_SIZE",

    # Hough
    "HOUGH_MAX_CURVATURE",
    "HOUGH_THETA_BIN_DEG",
    "HOUGH_PHI_BIN_DEG",
    "HOUGH_D_BIN_M",
    "HOUGH_MIN_VOTES",
]


# ============================================================
# 5. Method index
#
# Keep the same RNG convention as the original benchmark:
#
#   seed = metadata["seed"] + method_index * 100000
# ============================================================

METHOD_INDEX = {
    "Iterative RANSAC": 0,
    "Multi-Object RANSAC": 1,
    "Surface-Normal / Surfel": 2,
    "Region Growing": 3,
    "3D Hough": 4,
}


# ============================================================
# 6. Module-parameter helpers
# ============================================================

def capture_parameters():
    values = {}

    for name in PARAMETER_NAMES:
        if not hasattr(
            mp,
            name,
        ):
            raise AttributeError(
                f"run_multiplane_plane_detection_benchmark.py "
                f"does not contain '{name}'."
            )

        values[
            name
        ] = deepcopy(
            getattr(
                mp,
                name,
            )
        )

    return values


INITIAL_PARAMETERS = capture_parameters()


def apply_parameters(
    full_configuration,
):
    """
    Set every recorded parameter before each trial.

    This prevents one candidate configuration from leaking
    into the next trial.
    """

    for name in PARAMETER_NAMES:
        setattr(
            mp,
            name,
            deepcopy(
                full_configuration[
                    name
                ]
            ),
        )


def current_parameter_snapshot():
    return {
        name:
            deepcopy(
                getattr(
                    mp,
                    name,
                )
            )
        for name in PARAMETER_NAMES
    }


# ============================================================
# 7. Load DEV cases
# ============================================================

def load_dev_cases():
    sample_files = sorted(
        DEV_DIR.glob(
            "P*_N*_O*_R*.npz"
        )
    )

    if len(
        sample_files
    ) != 60:
        raise RuntimeError(
            "Expected the existing 60-case multi-plane "
            f"benchmark, but found {len(sample_files)} in:\n"
            f"{DEV_DIR}"
        )

    cases = []

    for npz_file in sample_files:
        (
            points,
            labels,
            metadata,
            gt_planes,
        ) = mp.load_sample(
            npz_file
        )

        repeat_id = int(
            metadata[
                "repeat_id"
            ]
        )

        if repeat_id not in DEV_REPEAT_IDS:
            continue

        cases.append(
            {
                "path":
                    npz_file,

                "points":
                    points,

                "labels":
                    labels,

                "metadata":
                    metadata,

                "gt_planes":
                    gt_planes,
            }
        )

    if len(
        cases
    ) != EXPECTED_DEV_CASES:
        raise RuntimeError(
            f"Expected {EXPECTED_DEV_CASES} DEV cases from "
            f"repeat IDs {sorted(DEV_REPEAT_IDS)}, "
            f"but found {len(cases)}."
        )

    return cases


# ============================================================
# 8. Run one method on one case
# ============================================================

def run_method_case(
    method_name,
    case,
):
    detector = mp.METHODS[
        method_name
    ]

    method_index = METHOD_INDEX[
        method_name
    ]

    seed = (
        int(
            case[
                "metadata"
            ][
                "seed"
            ]
        )
        + method_index
        * 100000
    )

    rng = np.random.default_rng(
        seed
    )

    start = perf_counter()

    models = detector(
        case[
            "points"
        ],
        rng,
    )

    end = perf_counter()

    runtime_ms = (
        end
        - start
    ) * 1000.0

    evaluation = mp.match_planes(
        models,
        case[
            "gt_planes"
        ],
        case[
            "labels"
        ],
    )

    num_gt = len(
        case[
            "gt_planes"
        ]
    )

    num_detected = len(
        models
    )

    matched_count = evaluation[
        "matched_count"
    ]

    return {
        "plane_precision":
            float(
                evaluation[
                    "plane_precision"
                ]
            ),

        "plane_recall":
            float(
                evaluation[
                    "plane_recall"
                ]
            ),

        "plane_f1":
            float(
                evaluation[
                    "plane_f1"
                ]
            ),

        "mean_iou":
            float(
                evaluation[
                    "mean_iou"
                ]
            ),

        "mean_normal_error_deg":
            (
                np.nan
                if np.isnan(
                    evaluation[
                        "mean_normal_error_deg"
                    ]
                )
                else float(
                    evaluation[
                        "mean_normal_error_deg"
                    ]
                )
            ),

        "mean_offset_error_m":
            (
                np.nan
                if np.isnan(
                    evaluation[
                        "mean_offset_error_m"
                    ]
                )
                else float(
                    evaluation[
                        "mean_offset_error_m"
                    ]
                )
            ),

        "detected_plane_count":
            int(
                num_detected
            ),

        "matched_plane_count":
            int(
                matched_count
            ),

        "false_plane_count":
            int(
                num_detected
                - matched_count
            ),

        "missed_plane_count":
            int(
                num_gt
                - matched_count
            ),

        "correct_plane_count":
            int(
                num_detected
                == num_gt
            ),

        "runtime_ms":
            float(
                runtime_ms
            ),
    }


# ============================================================
# 9. Evaluate one method configuration on all DEV cases
# ============================================================

def evaluate_method_configuration(
    method_name,
    cases,
    full_configuration,
):
    apply_parameters(
        full_configuration
    )

    rows = []

    for case in cases:
        result = run_method_case(
            method_name,
            case,
        )

        rows.append(
            result
        )

    def mean(
        key,
    ):
        values = np.asarray(
            [
                row[
                    key
                ]
                for row in rows
            ],
            dtype=float,
        )

        finite = np.isfinite(
            values
        )

        if not np.any(
            finite
        ):
            return float(
                "nan"
            )

        return float(
            np.mean(
                values[
                    finite
                ]
            )
        )

    return {
        "num_cases":
            int(
                len(
                    rows
                )
            ),

        "mean_plane_precision":
            mean(
                "plane_precision"
            ),

        "mean_plane_recall":
            mean(
                "plane_recall"
            ),

        "mean_plane_f1":
            mean(
                "plane_f1"
            ),

        "correct_plane_count_rate":
            mean(
                "correct_plane_count"
            ),

        "mean_iou":
            mean(
                "mean_iou"
            ),

        "mean_normal_error_deg":
            mean(
                "mean_normal_error_deg"
            ),

        "mean_offset_error_m":
            mean(
                "mean_offset_error_m"
            ),

        "mean_false_plane_count":
            mean(
                "false_plane_count"
            ),

        "mean_missed_plane_count":
            mean(
                "missed_plane_count"
            ),

        "mean_runtime_ms":
            mean(
                "runtime_ms"
            ),
    }


# ============================================================
# 10. Selection rule
#
# Runtime is NOT used to select parameters.
#
# Primary:
#   1. mean plane F1
#   2. correct plane-count rate
#   3. mean IoU
#
# Tie-break:
#   4. fewer false planes
#   5. fewer missed planes
#   6. lower normal error
#   7. lower offset error
# ============================================================

def safe_negative(
    value,
):
    if value is None:
        return -1e12

    if not np.isfinite(
        value
    ):
        return -1e12

    return -float(
        value
    )


def selection_key(
    evaluation,
):
    return (
        evaluation[
            "mean_plane_f1"
        ],

        evaluation[
            "correct_plane_count_rate"
        ],

        evaluation[
            "mean_iou"
        ],

        -evaluation[
            "mean_false_plane_count"
        ],

        -evaluation[
            "mean_missed_plane_count"
        ],

        safe_negative(
            evaluation[
                "mean_normal_error_deg"
            ]
        ),

        safe_negative(
            evaluation[
                "mean_offset_error_m"
            ]
        ),
    )


# ============================================================
# 11. Console helper
# ============================================================

def print_evaluation(
    label,
    evaluation,
):
    print(
        f"{label:<20} "
        f"F1={evaluation['mean_plane_f1']:.4f} "
        f"Count={evaluation['correct_plane_count_rate']:.4f} "
        f"IoU={evaluation['mean_iou']:.4f} "
        f"FP={evaluation['mean_false_plane_count']:.3f} "
        f"FN={evaluation['mean_missed_plane_count']:.3f}"
    )


# ============================================================
# 12. Baseline evaluation of all five methods
# ============================================================

def evaluate_all_initial_methods(
    cases,
):
    results = {}

    for method_name in mp.METHODS:
        evaluation = evaluate_method_configuration(
            method_name,
            cases,
            INITIAL_PARAMETERS,
        )

        results[
            method_name
        ] = evaluation

    return results


# ============================================================
# 13. Tune one method by coordinate sweep
# ============================================================

def tune_method(
    method_name,
    cases,
    starting_configuration,
):
    current_configuration = deepcopy(
        starting_configuration
    )

    history = []

    baseline = evaluate_method_configuration(
        method_name,
        cases,
        current_configuration,
    )

    print()
    print(
        f"{method_name} baseline"
    )

    print_evaluation(
        "initial",
        baseline,
    )

    history.append(
        {
            "method":
                method_name,

            "pass_number":
                0,

            "parameter":
                "baseline",

            "candidate":
                "",

            "selected":
                1,

            "evaluation":
                baseline,

            "configuration":
                deepcopy(
                    current_configuration
                ),
        }
    )

    for pass_number in range(
        1,
        NUM_COORDINATE_PASSES
        + 1,
    ):
        print()
        print(
            f"{method_name} "
            f"- coordinate pass {pass_number}"
        )

        changed_this_pass = False

        for (
            parameter_name,
            candidate_values,
        ) in TUNING_GROUPS[
            method_name
        ]:

            print()
            print(
                f"[{parameter_name}]"
            )

            candidate_rows = []

            old_value = current_configuration[
                parameter_name
            ]

            for candidate in candidate_values:

                trial_configuration = deepcopy(
                    current_configuration
                )

                trial_configuration[
                    parameter_name
                ] = candidate

                evaluation = (
                    evaluate_method_configuration(
                        method_name,
                        cases,
                        trial_configuration,
                    )
                )

                row = {
                    "method":
                        method_name,

                    "pass_number":
                        int(
                            pass_number
                        ),

                    "parameter":
                        parameter_name,

                    "candidate":
                        candidate,

                    "selected":
                        0,

                    "evaluation":
                        evaluation,

                    "configuration":
                        deepcopy(
                            trial_configuration
                        ),
                }

                candidate_rows.append(
                    row
                )

                print_evaluation(
                    str(
                        candidate
                    ),
                    evaluation,
                )

            best_row = max(
                candidate_rows,
                key=lambda row:
                    selection_key(
                        row[
                            "evaluation"
                        ]
                    ),
            )

            best_row[
                "selected"
            ] = 1

            best_value = best_row[
                "candidate"
            ]

            current_configuration = deepcopy(
                best_row[
                    "configuration"
                ]
            )

            history.extend(
                candidate_rows
            )

            print(
                "  -> selected:",
                best_value
            )

            if best_value != old_value:
                changed_this_pass = True

        if not changed_this_pass:
            print()
            print(
                "No parameter changed in this pass. "
                "Stopping early."
            )
            break

    final_evaluation = (
        evaluate_method_configuration(
            method_name,
            cases,
            current_configuration,
        )
    )

    return (
        current_configuration,
        final_evaluation,
        history,
    )


# ============================================================
# 14. Save CSV history
# ============================================================

def save_history_csv(
    history,
):
    fieldnames = [
        "method",
        "pass_number",
        "parameter",
        "candidate",
        "selected",

        "mean_plane_precision",
        "mean_plane_recall",
        "mean_plane_f1",

        "correct_plane_count_rate",
        "mean_iou",

        "mean_normal_error_deg",
        "mean_offset_error_m",

        "mean_false_plane_count",
        "mean_missed_plane_count",

        "mean_runtime_ms",

        "configuration_json",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in history:
            evaluation = row[
                "evaluation"
            ]

            writer.writerow(
                {
                    "method":
                        row[
                            "method"
                        ],

                    "pass_number":
                        row[
                            "pass_number"
                        ],

                    "parameter":
                        row[
                            "parameter"
                        ],

                    "candidate":
                        row[
                            "candidate"
                        ],

                    "selected":
                        row[
                            "selected"
                        ],

                    "mean_plane_precision":
                        evaluation[
                            "mean_plane_precision"
                        ],

                    "mean_plane_recall":
                        evaluation[
                            "mean_plane_recall"
                        ],

                    "mean_plane_f1":
                        evaluation[
                            "mean_plane_f1"
                        ],

                    "correct_plane_count_rate":
                        evaluation[
                            "correct_plane_count_rate"
                        ],

                    "mean_iou":
                        evaluation[
                            "mean_iou"
                        ],

                    "mean_normal_error_deg":
                        evaluation[
                            "mean_normal_error_deg"
                        ],

                    "mean_offset_error_m":
                        evaluation[
                            "mean_offset_error_m"
                        ],

                    "mean_false_plane_count":
                        evaluation[
                            "mean_false_plane_count"
                        ],

                    "mean_missed_plane_count":
                        evaluation[
                            "mean_missed_plane_count"
                        ],

                    "mean_runtime_ms":
                        evaluation[
                            "mean_runtime_ms"
                        ],

                    "configuration_json":
                        json.dumps(
                            row[
                                "configuration"
                            ],
                            ensure_ascii=False,
                        ),
                }
            )


# ============================================================
# 15. Main
# ============================================================

def main():
    print(
        "=" * 80
    )

    print(
        "Plane Detection DEV Parameter Tuning"
    )

    print(
        "=" * 80
    )

    print()
    print(
        "Source benchmark:"
    )

    print(
        DEV_DIR
    )

    print()
    print(
        "DEV repeats:",
        sorted(
            DEV_REPEAT_IDS
        )
    )

    print(
        "Expected DEV cases:",
        EXPECTED_DEV_CASES
    )

    print()
    print(
        "Common evaluation thresholds are fixed and "
        "are not tuned."
    )

    cases = load_dev_cases()

    # --------------------------------------------------------
    # Baseline check for all five methods
    # --------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "INITIAL DEV PERFORMANCE"
    )

    print(
        "=" * 80
    )

    initial_method_evaluations = (
        evaluate_all_initial_methods(
            cases
        )
    )

    for (
        method_name,
        evaluation
    ) in initial_method_evaluations.items():
        print_evaluation(
            method_name,
            evaluation,
        )

    # --------------------------------------------------------
    # RANSAC and Hough remain fixed.
    # --------------------------------------------------------

    frozen_configuration = deepcopy(
        INITIAL_PARAMETERS
    )

    tuning_history = []

    tuned_method_results = {}

    for method_name in [
        "Multi-Object RANSAC",
        "Surface-Normal / Surfel",
        "Region Growing",
    ]:

        (
            tuned_configuration,
            tuned_evaluation,
            method_history,
        ) = tune_method(
            method_name,
            cases,
            frozen_configuration,
        )

        # Only copy the parameters belonging to this method.
        for (
            parameter_name,
            _
        ) in TUNING_GROUPS[
            method_name
        ]:
            frozen_configuration[
                parameter_name
            ] = tuned_configuration[
                parameter_name
            ]

        tuned_method_results[
            method_name
        ] = tuned_evaluation

        tuning_history.extend(
            method_history
        )

    # --------------------------------------------------------
    # Evaluate all five methods using final frozen config.
    # --------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "FROZEN DEV CONFIGURATION"
    )

    print(
        "=" * 80
    )

    frozen_method_evaluations = {}

    for method_name in mp.METHODS:
        evaluation = (
            evaluate_method_configuration(
                method_name,
                cases,
                frozen_configuration,
            )
        )

        frozen_method_evaluations[
            method_name
        ] = evaluation

        print_evaluation(
            method_name,
            evaluation,
        )

    # --------------------------------------------------------
    # Save tuning evidence
    # --------------------------------------------------------

    save_history_csv(
        tuning_history
    )

    tuned_parameters = {}

    for (
        method_name,
        groups
    ) in TUNING_GROUPS.items():

        tuned_parameters[
            method_name
        ] = {
            parameter_name:
                frozen_configuration[
                    parameter_name
                ]
            for (
                parameter_name,
                _
            ) in groups
        }

    output_data = {
        "tuning_dataset":
            "synthetic_multiplane_benchmark",

        "dev_repeat_ids":
            sorted(
                DEV_REPEAT_IDS
            ),

        "num_dev_cases":
            EXPECTED_DEV_CASES,

        "strategy":
            (
                "Minimal controlled DEV calibration. "
                "Only methods/parameters that showed clear "
                "preliminary instability were tuned. "
                "RANSAC and Hough were retained because "
                "their preliminary multi-plane behaviour "
                "was already stable."
            ),

        "selection_priority":
            [
                "maximise mean plane F1",
                "maximise correct plane-count rate",
                "maximise mean matched-plane IoU",
                "minimise mean false-plane count",
                "minimise mean missed-plane count",
                "minimise mean normal error",
                "minimise mean offset error",
            ],

        "common_parameters_not_tuned":
            {
                name:
                    frozen_configuration[
                        name
                    ]
                for name in [
                    "FINAL_DISTANCE_THRESHOLD",
                    "MATCH_NORMAL_THRESHOLD_DEG",
                    "MATCH_OFFSET_THRESHOLD_M",
                    "MIN_PLANE_SUPPORT",
                    "MAX_PLANES",
                    "K_NEIGHBORS",
                ]
            },

        "methods_retained_without_tuning":
            {
                "Iterative RANSAC":
                    {
                        "RANSAC_ITERATIONS":
                            frozen_configuration[
                                "RANSAC_ITERATIONS"
                            ]
                    },

                "3D Hough":
                    {
                        "HOUGH_MAX_CURVATURE":
                            frozen_configuration[
                                "HOUGH_MAX_CURVATURE"
                            ],

                        "HOUGH_THETA_BIN_DEG":
                            frozen_configuration[
                                "HOUGH_THETA_BIN_DEG"
                            ],

                        "HOUGH_PHI_BIN_DEG":
                            frozen_configuration[
                                "HOUGH_PHI_BIN_DEG"
                            ],

                        "HOUGH_D_BIN_M":
                            frozen_configuration[
                                "HOUGH_D_BIN_M"
                            ],

                        "HOUGH_MIN_VOTES":
                            frozen_configuration[
                                "HOUGH_MIN_VOTES"
                            ],
                    },
            },

        "tuned_parameters":
            tuned_parameters,

        "full_frozen_multiplane_configuration":
            frozen_configuration,

        "initial_dev_evaluation":
            initial_method_evaluations,

        "frozen_dev_evaluation":
            frozen_method_evaluations,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output_data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Restore source-module values after tuning program exits.
    apply_parameters(
        INITIAL_PARAMETERS
    )

    print()
    print(
        "=" * 80
    )

    print(
        "TUNING COMPLETE"
    )

    print(
        "=" * 80
    )

    print()
    print(
        "Tuning history:"
    )

    print(
        OUTPUT_CSV
    )

    print()
    print(
        "Frozen detection configuration:"
    )

    print(
        OUTPUT_JSON
    )

    print()
    print(
        "The original detector source files were not edited."
    )

    print(
        "Next step: generate NEW-seed FINAL benchmark data, "
        "then run the five frozen methods once."
    )


if __name__ == "__main__":
    main()
