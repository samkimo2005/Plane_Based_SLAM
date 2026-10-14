from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

import plane_to_plane_distance_similarity as distance_module
import hard_threshold_similarity as hard_module
import descriptor_similarity as descriptor_module

from similarity_common import load_similarity_case
from similarity_evaluation import evaluate_similarity_candidates


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEV_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic_similarity_benchmark"
    / "dev"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "plane_similarity"
    / "tuning"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. General tuning settings
# ============================================================

NUM_SWEEP_PASSES = 2

EXPECTED_DEV_CASES = 27


# ============================================================
# 3. Resolve symbols from the three method files
#
# This deliberately accepts a few naming variants so that
# small naming differences in the previously written scripts
# do not force you to rewrite those scripts.
# ============================================================

def resolve_symbol(
    module,
    possible_names,
):
    for name in possible_names:
        if hasattr(
            module,
            name,
        ):
            return getattr(
                module,
                name,
            )

    available = sorted(
        name
        for name in dir(
            module
        )
        if not name.startswith(
            "_"
        )
    )

    raise AttributeError(
        "Could not find any of these symbols:\n"
        f"{possible_names}\n\n"
        f"Available public names in {module.__name__}:\n"
        f"{available}"
    )


PlaneDistanceParameterClass = resolve_symbol(
    distance_module,
    [
        "PlaneToPlaneDistanceParameters",
    ],
)

PlaneDistanceRunner = resolve_symbol(
    distance_module,
    [
        "run_plane_to_plane_distance_case",
    ],
)

HardParameterClass = resolve_symbol(
    hard_module,
    [
        "HardThresholdSimilarityParameters",
        "HardThresholdParameters",
        "GeometricThresholdParameters",
    ],
)

HardRunner = resolve_symbol(
    hard_module,
    [
        "run_hard_threshold_similarity_case",
        "run_hard_threshold_case",
        "run_geometric_threshold_similarity_case",
    ],
)

DescriptorParameterClass = resolve_symbol(
    descriptor_module,
    [
        "DescriptorSimilarityParameters",
        "DescriptorParameters",
    ],
)

DescriptorRunner = resolve_symbol(
    descriptor_module,
    [
        "run_descriptor_similarity_case",
        "run_descriptor_case",
    ],
)


# ============================================================
# 4. Load DEV cases only
# ============================================================

def load_dev_cases():
    json_files = sorted(
        DEV_DIR.glob(
            "*.json"
        )
    )

    if len(
        json_files
    ) != EXPECTED_DEV_CASES:

        raise RuntimeError(
            f"Expected {EXPECTED_DEV_CASES} DEV cases, "
            f"but found {len(json_files)} in:\n"
            f"{DEV_DIR}"
        )

    cases = []

    for json_file in json_files:
        case_data = load_similarity_case(
            json_file
        )

        cases.append(
            {
                "path": json_file,
                "data": case_data,
            }
        )

    return cases


# ============================================================
# 5. Parameter-object helpers
# ============================================================

def make_parameter_object(
    parameter_class,
    updates,
):
    """
    Start from the method's current default parameter object,
    then overwrite only the parameters being swept.
    """

    params = parameter_class()

    for name, value in updates.items():

        if not hasattr(
            params,
            name,
        ):
            raise AttributeError(
                f"{parameter_class.__name__} has no parameter "
                f"named '{name}'.\n"
                "Check that the method file matches the version "
                "used for this tuning script."
            )

        setattr(
            params,
            name,
            value,
        )

    return params


def parameter_object_to_dict(
    params,
):
    """
    Convert an ordinary dataclass-style parameter object
    to a plain dictionary.
    """

    if hasattr(
        params,
        "__dict__",
    ):
        return dict(
            vars(
                params
            )
        )

    raise TypeError(
        "Parameter object does not expose __dict__."
    )


# ============================================================
# 6. JSON-safe conversion
# ============================================================

def json_safe(
    value,
):
    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): json_safe(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            json_safe(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        np.ndarray,
    ):
        return json_safe(
            value.tolist()
        )

    if isinstance(
        value,
        (
            np.floating,
            float,
        ),
    ):
        value = float(
            value
        )

        if math.isinf(
            value
        ):
            return (
                "inf"
                if value > 0
                else "-inf"
            )

        if math.isnan(
            value
        ):
            return "nan"

        return value

    if isinstance(
        value,
        (
            np.integer,
            int,
        ),
    ):
        return int(
            value
        )

    if isinstance(
        value,
        (
            np.bool_,
            bool,
        ),
    ):
        return bool(
            value
        )

    return value


# ============================================================
# 7. Evaluate one complete parameter configuration
# ============================================================

def evaluate_configuration(
    cases,
    parameter_class,
    runner,
    parameter_updates,
):
    params = make_parameter_object(
        parameter_class,
        parameter_updates,
    )

    case_f1 = []
    case_precision = []
    case_recall = []

    case_mean_candidates = []
    case_ambiguity = []
    case_missed = []

    valid_unmatched_rejection = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_algorithm_runtime_ms = 0.0

    for case_item in cases:

        case_data = case_item[
            "data"
        ]

        # ----------------------------------------------------
        # Run similarity method only
        # ----------------------------------------------------

        start_time = perf_counter()

        result = runner(
            case_data,
            params,
        )

        end_time = perf_counter()

        total_algorithm_runtime_ms += (
            end_time
            - start_time
        ) * 1000.0

        # ----------------------------------------------------
        # Common candidate-set evaluator
        # ----------------------------------------------------

        evaluation = evaluate_similarity_candidates(
            case_data,
            result[
                "candidate_pairs"
            ],
        )

        case_f1.append(
            evaluation[
                "f1"
            ]
        )

        case_precision.append(
            evaluation[
                "precision"
            ]
        )

        case_recall.append(
            evaluation[
                "recall"
            ]
        )

        case_mean_candidates.append(
            evaluation[
                "mean_candidates_per_a"
            ]
        )

        case_ambiguity.append(
            evaluation[
                "ambiguous_matched_a_rate"
            ]
        )

        case_missed.append(
            evaluation[
                "missed_matched_a_rate"
            ]
        )

        if (
            evaluation[
                "num_gt_unmatched_a"
            ]
            > 0
        ):
            valid_unmatched_rejection.append(
                evaluation[
                    "unmatched_rejection_rate"
                ]
            )

        total_tp += int(
            evaluation[
                "tp"
            ]
        )

        total_fp += int(
            evaluation[
                "fp"
            ]
        )

        total_fn += int(
            evaluation[
                "fn"
            ]
        )

    # --------------------------------------------------------
    # Micro metrics
    # --------------------------------------------------------

    micro_precision_denominator = (
        total_tp
        + total_fp
    )

    if micro_precision_denominator > 0:
        micro_precision = (
            total_tp
            / micro_precision_denominator
        )
    else:
        micro_precision = 0.0

    micro_recall_denominator = (
        total_tp
        + total_fn
    )

    if micro_recall_denominator > 0:
        micro_recall = (
            total_tp
            / micro_recall_denominator
        )
    else:
        micro_recall = 0.0

    if (
        micro_precision
        + micro_recall
    ) > 0:
        micro_f1 = (
            2.0
            * micro_precision
            * micro_recall
            / (
                micro_precision
                + micro_recall
            )
        )
    else:
        micro_f1 = 0.0

    # --------------------------------------------------------
    # Macro / diagnostic metrics
    # --------------------------------------------------------

    mean_unmatched_rejection = None

    if valid_unmatched_rejection:
        mean_unmatched_rejection = float(
            np.mean(
                valid_unmatched_rejection
            )
        )

    return {
        "mean_f1": float(
            np.mean(
                case_f1
            )
        ),

        "mean_precision": float(
            np.mean(
                case_precision
            )
        ),

        "mean_recall": float(
            np.mean(
                case_recall
            )
        ),

        "micro_f1": float(
            micro_f1
        ),

        "micro_precision": float(
            micro_precision
        ),

        "micro_recall": float(
            micro_recall
        ),

        "mean_candidates_per_a": float(
            np.mean(
                case_mean_candidates
            )
        ),

        "mean_ambiguous_matched_a_rate": float(
            np.mean(
                case_ambiguity
            )
        ),

        "mean_missed_matched_a_rate": float(
            np.mean(
                case_missed
            )
        ),

        "mean_unmatched_rejection_rate":
            mean_unmatched_rejection,

        "total_tp": int(
            total_tp
        ),

        "total_fp": int(
            total_fp
        ),

        "total_fn": int(
            total_fn
        ),

        "mean_algorithm_runtime_ms": float(
            total_algorithm_runtime_ms
            / len(
                cases
            )
        ),
    }


# ============================================================
# 8. Selection rule
# ============================================================

def selection_key(
    evaluation,
):
    """
    Configuration-selection priority:

        1. highest mean F1 across the 27 DEV cases
        2. highest micro F1
        3. highest mean precision
        4. highest mean recall
        5. lowest ambiguity among matched A planes
        6. lowest mean number of candidates per A plane
        7. highest unmatched-plane rejection rate

    Runtime is deliberately NOT used to select tuning
    parameters. Speed remains a separate final comparison
    criterion between methods.
    """

    unmatched_rejection = evaluation[
        "mean_unmatched_rejection_rate"
    ]

    if unmatched_rejection is None:
        unmatched_rejection = -1.0

    return (
        evaluation[
            "mean_f1"
        ],

        evaluation[
            "micro_f1"
        ],

        evaluation[
            "mean_precision"
        ],

        evaluation[
            "mean_recall"
        ],

        -evaluation[
            "mean_ambiguous_matched_a_rate"
        ],

        -evaluation[
            "mean_candidates_per_a"
        ],

        unmatched_rejection,
    )


# ============================================================
# 9. Save one tuning history CSV
# ============================================================

def save_history_csv(
    method_name,
    history,
):
    output_file = (
        OUTPUT_DIR
        / (
            method_name
            .lower()
            .replace(
                " ",
                "_",
            )
            .replace(
                "-",
                "_",
            )
            + "_tuning.csv"
        )
    )

    fieldnames = [
        "method",
        "pass_number",
        "parameter_group",
        "candidate_label",
        "selected_for_group",
        "parameter_updates_json",
        "full_parameters_json",
        "mean_f1",
        "mean_precision",
        "mean_recall",
        "micro_f1",
        "micro_precision",
        "micro_recall",
        "mean_candidates_per_a",
        "mean_ambiguous_matched_a_rate",
        "mean_missed_matched_a_rate",
        "mean_unmatched_rejection_rate",
        "total_tp",
        "total_fp",
        "total_fn",
        "mean_algorithm_runtime_ms",
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in history:

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

                    "parameter_group":
                        row[
                            "parameter_group"
                        ],

                    "candidate_label":
                        row[
                            "candidate_label"
                        ],

                    "selected_for_group":
                        int(
                            row[
                                "selected_for_group"
                            ]
                        ),

                    "parameter_updates_json":
                        json.dumps(
                            json_safe(
                                row[
                                    "parameter_updates"
                                ]
                            ),
                            ensure_ascii=False,
                        ),

                    "full_parameters_json":
                        json.dumps(
                            json_safe(
                                row[
                                    "full_parameters"
                                ]
                            ),
                            ensure_ascii=False,
                        ),

                    "mean_f1":
                        row[
                            "evaluation"
                        ][
                            "mean_f1"
                        ],

                    "mean_precision":
                        row[
                            "evaluation"
                        ][
                            "mean_precision"
                        ],

                    "mean_recall":
                        row[
                            "evaluation"
                        ][
                            "mean_recall"
                        ],

                    "micro_f1":
                        row[
                            "evaluation"
                        ][
                            "micro_f1"
                        ],

                    "micro_precision":
                        row[
                            "evaluation"
                        ][
                            "micro_precision"
                        ],

                    "micro_recall":
                        row[
                            "evaluation"
                        ][
                            "micro_recall"
                        ],

                    "mean_candidates_per_a":
                        row[
                            "evaluation"
                        ][
                            "mean_candidates_per_a"
                        ],

                    "mean_ambiguous_matched_a_rate":
                        row[
                            "evaluation"
                        ][
                            "mean_ambiguous_matched_a_rate"
                        ],

                    "mean_missed_matched_a_rate":
                        row[
                            "evaluation"
                        ][
                            "mean_missed_matched_a_rate"
                        ],

                    "mean_unmatched_rejection_rate":
                        (
                            ""
                            if row[
                                "evaluation"
                            ][
                                "mean_unmatched_rejection_rate"
                            ] is None
                            else row[
                                "evaluation"
                            ][
                                "mean_unmatched_rejection_rate"
                            ]
                        ),

                    "total_tp":
                        row[
                            "evaluation"
                        ][
                            "total_tp"
                        ],

                    "total_fp":
                        row[
                            "evaluation"
                        ][
                            "total_fp"
                        ],

                    "total_fn":
                        row[
                            "evaluation"
                        ][
                            "total_fn"
                        ],

                    "mean_algorithm_runtime_ms":
                        row[
                            "evaluation"
                        ][
                            "mean_algorithm_runtime_ms"
                        ],
                }
            )

    return output_file


# ============================================================
# 10. Generic coordinate parameter sweep
# ============================================================

def coordinate_parameter_sweep(
    method_name,
    cases,
    parameter_class,
    runner,
    parameter_groups,
):
    print()
    print(
        "=" * 78
    )

    print(
        f"Tuning: {method_name}"
    )

    print(
        "=" * 78
    )

    current_updates = {}

    history = []

    # --------------------------------------------------------
    # Baseline = current default parameters
    # --------------------------------------------------------

    baseline_evaluation = evaluate_configuration(
        cases,
        parameter_class,
        runner,
        current_updates,
    )

    baseline_params = make_parameter_object(
        parameter_class,
        current_updates,
    )

    history.append(
        {
            "method":
                method_name,

            "pass_number":
                0,

            "parameter_group":
                "baseline",

            "candidate_label":
                "current defaults",

            "selected_for_group":
                True,

            "parameter_updates":
                dict(
                    current_updates
                ),

            "full_parameters":
                parameter_object_to_dict(
                    baseline_params
                ),

            "evaluation":
                baseline_evaluation,
        }
    )

    print(
        "Baseline DEV mean F1:",
        f"{baseline_evaluation['mean_f1']:.4f}",
    )

    print(
        "Baseline DEV precision / recall:",
        f"{baseline_evaluation['mean_precision']:.4f}",
        "/",
        f"{baseline_evaluation['mean_recall']:.4f}",
    )

    # --------------------------------------------------------
    # Coordinate sweeps
    # --------------------------------------------------------

    for pass_number in range(
        1,
        NUM_SWEEP_PASSES + 1,
    ):

        print()
        print(
            f"--- Sweep pass {pass_number}/{NUM_SWEEP_PASSES} ---"
        )

        for (
            group_name,
            candidates,
        ) in parameter_groups:

            print()
            print(
                f"[{group_name}]"
            )

            group_trials = []

            for (
                candidate_label,
                candidate_updates,
            ) in candidates:

                trial_updates = dict(
                    current_updates
                )

                trial_updates.update(
                    candidate_updates
                )

                evaluation = evaluate_configuration(
                    cases,
                    parameter_class,
                    runner,
                    trial_updates,
                )

                params = make_parameter_object(
                    parameter_class,
                    trial_updates,
                )

                trial = {
                    "method":
                        method_name,

                    "pass_number":
                        pass_number,

                    "parameter_group":
                        group_name,

                    "candidate_label":
                        candidate_label,

                    "selected_for_group":
                        False,

                    "parameter_updates":
                        trial_updates,

                    "full_parameters":
                        parameter_object_to_dict(
                            params
                        ),

                    "evaluation":
                        evaluation,
                }

                group_trials.append(
                    trial
                )

                print(
                    f"  {candidate_label:<28} "
                    f"F1={evaluation['mean_f1']:.4f} "
                    f"P={evaluation['mean_precision']:.4f} "
                    f"R={evaluation['mean_recall']:.4f} "
                    f"Amb={evaluation['mean_ambiguous_matched_a_rate']:.4f}"
                )

            best_trial = max(
                group_trials,
                key=lambda row: selection_key(
                    row[
                        "evaluation"
                    ]
                ),
            )

            best_trial[
                "selected_for_group"
            ] = True

            current_updates = dict(
                best_trial[
                    "parameter_updates"
                ]
            )

            history.extend(
                group_trials
            )

            print(
                "  -> selected:",
                best_trial[
                    "candidate_label"
                ],
                "| mean F1 =",
                f"{best_trial['evaluation']['mean_f1']:.4f}",
            )

    # --------------------------------------------------------
    # Final evaluation after all sweep passes
    # --------------------------------------------------------

    final_evaluation = evaluate_configuration(
        cases,
        parameter_class,
        runner,
        current_updates,
    )

    final_params = make_parameter_object(
        parameter_class,
        current_updates,
    )

    final_full_parameters = parameter_object_to_dict(
        final_params
    )

    print()
    print(
        f"Final {method_name} DEV configuration:"
    )

    for name, value in final_full_parameters.items():
        print(
            f"  {name}: {value}"
        )

    print()
    print(
        "Final DEV mean F1:",
        f"{final_evaluation['mean_f1']:.4f}",
    )

    print(
        "Final DEV mean precision:",
        f"{final_evaluation['mean_precision']:.4f}",
    )

    print(
        "Final DEV mean recall:",
        f"{final_evaluation['mean_recall']:.4f}",
    )

    print(
        "Final DEV ambiguity rate:",
        f"{final_evaluation['mean_ambiguous_matched_a_rate']:.4f}",
    )

    print(
        "Final DEV mean candidates/A:",
        f"{final_evaluation['mean_candidates_per_a']:.4f}",
    )

    history_file = save_history_csv(
        method_name,
        history,
    )

    print(
        "Tuning history saved to:",
        history_file,
    )

    return {
        "method":
            method_name,

        "parameter_updates":
            current_updates,

        "full_parameters":
            final_full_parameters,

        "evaluation":
            final_evaluation,

        "history_file":
            str(
                history_file
            ),
    }


# ============================================================
# 11. Plane-to-plane distance sweep
# ============================================================

PLANE_DISTANCE_GROUPS = [
    (
        "max_distance_m",
        [
            (
                f"{value:.2f} m",
                {
                    "max_distance_m":
                        value
                },
            )
            for value in [
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.40,
                0.50,
                0.75,
                1.00,
            ]
        ],
    ),
]


# ============================================================
# 12. Hard-threshold sweep
# ============================================================

HARD_GROUPS = [
    (
        "max_normal_angle_deg",
        [
            (
                f"{value:.0f} deg",
                {
                    "max_normal_angle_deg":
                        value
                },
            )
            for value in [
                3.0,
                5.0,
                10.0,
                15.0,
                20.0,
            ]
        ],
    ),
    (
        "max_offset_difference_m",
        [
            (
                f"{value:.2f} m",
                {
                    "max_offset_difference_m":
                        value
                },
            )
            for value in [
                0.03,
                0.05,
                0.10,
                0.15,
                0.20,
            ]
        ],
    ),
    (
        "max_centroid_distance_m",
        [
            (
                f"{value:.2f} m",
                {
                    "max_centroid_distance_m":
                        value
                },
            )
            for value in [
                0.25,
                0.50,
                1.00,
                1.50,
                2.00,
            ]
        ],
    ),
    (
        "max_area_difference_ratio",
        [
            (
                f"{value:.2f}",
                {
                    "max_area_difference_ratio":
                        value
                },
            )
            for value in [
                0.15,
                0.25,
                0.50,
                0.75,
                1.00,
            ]
        ],
    ),
]


# ============================================================
# 13. Descriptor sweep
# ============================================================

DESCRIPTOR_GROUPS = [
    (
        "descriptor_weights",
        [
            (
                "A.25 R.15 N.30 D.30",
                {
                    "area_weight": 0.25,
                    "aspect_weight": 0.15,
                    "normal_context_weight": 0.30,
                    "distance_context_weight": 0.30,
                },
            ),
            (
                "A.20 R.10 N.35 D.35",
                {
                    "area_weight": 0.20,
                    "aspect_weight": 0.10,
                    "normal_context_weight": 0.35,
                    "distance_context_weight": 0.35,
                },
            ),
            (
                "A.30 R.10 N.30 D.30",
                {
                    "area_weight": 0.30,
                    "aspect_weight": 0.10,
                    "normal_context_weight": 0.30,
                    "distance_context_weight": 0.30,
                },
            ),
            (
                "A.20 R.20 N.30 D.30",
                {
                    "area_weight": 0.20,
                    "aspect_weight": 0.20,
                    "normal_context_weight": 0.30,
                    "distance_context_weight": 0.30,
                },
            ),
            (
                "A.20 R.10 N.30 D.40",
                {
                    "area_weight": 0.20,
                    "aspect_weight": 0.10,
                    "normal_context_weight": 0.30,
                    "distance_context_weight": 0.40,
                },
            ),
        ],
    ),
    (
        "area_log_scale",
        [
            (
                f"{value:.2f}",
                {
                    "area_log_scale":
                        value
                },
            )
            for value in [
                0.25,
                0.50,
                0.75,
                1.00,
            ]
        ],
    ),
    (
        "aspect_log_scale",
        [
            (
                f"{value:.2f}",
                {
                    "aspect_log_scale":
                        value
                },
            )
            for value in [
                0.20,
                0.40,
                0.60,
                0.80,
            ]
        ],
    ),
    (
        "normal_angle_bins_deg",
        [
            (
                "fine-low-angle",
                {
                    "normal_angle_bins_deg":
                        (
                            0.0,
                            10.0,
                            30.0,
                            60.0,
                            90.000001,
                        )
                },
            ),
            (
                "baseline",
                {
                    "normal_angle_bins_deg":
                        (
                            0.0,
                            15.0,
                            45.0,
                            75.0,
                            90.000001,
                        )
                },
            ),
            (
                "balanced",
                {
                    "normal_angle_bins_deg":
                        (
                            0.0,
                            20.0,
                            40.0,
                            65.0,
                            90.000001,
                        )
                },
            ),
        ],
    ),
    (
        "centroid_distance_bins_m",
        [
            (
                "compact",
                {
                    "centroid_distance_bins_m":
                        (
                            0.0,
                            0.50,
                            1.50,
                            3.00,
                            float(
                                "inf"
                            ),
                        )
                },
            ),
            (
                "baseline",
                {
                    "centroid_distance_bins_m":
                        (
                            0.0,
                            1.00,
                            2.00,
                            4.00,
                            float(
                                "inf"
                            ),
                        )
                },
            ),
            (
                "medium",
                {
                    "centroid_distance_bins_m":
                        (
                            0.0,
                            1.00,
                            2.50,
                            4.50,
                            float(
                                "inf"
                            ),
                        )
                },
            ),
            (
                "wide",
                {
                    "centroid_distance_bins_m":
                        (
                            0.0,
                            1.50,
                            3.00,
                            5.00,
                            float(
                                "inf"
                            ),
                        )
                },
            ),
        ],
    ),
    (
        "max_descriptor_distance",
        [
            (
                f"{value:.2f}",
                {
                    "max_descriptor_distance":
                        value
                },
            )
            for value in [
                0.20,
                0.30,
                0.40,
                0.50,
                0.60,
                0.80,
                1.00,
            ]
        ],
    ),
]


# ============================================================
# 14. Save frozen parameter summary
# ============================================================

def save_best_parameters(
    results,
):
    output_json = (
        OUTPUT_DIR
        / "best_similarity_parameters.json"
    )

    data = {
        "tuning_dataset":
            "synthetic_similarity_benchmark/dev",

        "num_dev_cases":
            EXPECTED_DEV_CASES,

        "sweep_type":
            (
                "two-pass coordinate parameter sweep; "
                "final benchmark data not accessed"
            ),

        "selection_rule":
            [
                "maximise mean DEV F1",
                "then maximise micro F1",
                "then maximise mean precision",
                "then maximise mean recall",
                "then minimise matched-plane ambiguity",
                "then minimise mean candidates per A plane",
                "then maximise unmatched-plane rejection",
            ],

        "methods":
            {},
    }

    for result in results:

        data[
            "methods"
        ][
            result[
                "method"
            ]
        ] = {
            "parameters":
                json_safe(
                    result[
                        "full_parameters"
                    ]
                ),

            "dev_evaluation":
                json_safe(
                    result[
                        "evaluation"
                    ]
                ),

            "tuning_history_file":
                result[
                    "history_file"
                ],
        }

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return output_json


# ============================================================
# 15. Main
# ============================================================

def main():
    print(
        "=" * 78
    )

    print(
        "Plane Similarity DEV Parameter Tuning"
    )

    print(
        "=" * 78
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  - Only the DEV directory is used."
    )

    print(
        "  - The FINAL 90 cases are NOT accessed."
    )

    print(
        "  - Assignment is NOT performed here."
    )

    print(
        "  - All three methods use the same common candidate evaluator."
    )

    print(
        "  - Two coordinate-sweep passes are used to limit computation."
    )

    print(
        "  - Plane-to-Plane Distance tunes only max_distance_m."
    )

    print(
        "  - No one-to-one Assignment is performed during similarity tuning."
    )

    print()

    cases = load_dev_cases()

    print(
        "DEV cases loaded:",
        len(
            cases
        )
    )

    distance_result = coordinate_parameter_sweep(
        method_name="Plane-to-Plane Distance",
        cases=cases,
        parameter_class=PlaneDistanceParameterClass,
        runner=PlaneDistanceRunner,
        parameter_groups=PLANE_DISTANCE_GROUPS,
    )

    hard_result = coordinate_parameter_sweep(
        method_name="Hard Geometric Thresholds",
        cases=cases,
        parameter_class=HardParameterClass,
        runner=HardRunner,
        parameter_groups=HARD_GROUPS,
    )

    descriptor_result = coordinate_parameter_sweep(
        method_name="Descriptor Similarity",
        cases=cases,
        parameter_class=DescriptorParameterClass,
        runner=DescriptorRunner,
        parameter_groups=DESCRIPTOR_GROUPS,
    )

    results = [
        distance_result,
        hard_result,
        descriptor_result,
    ]

    summary_file = save_best_parameters(
        results
    )

    print()
    print(
        "=" * 78
    )

    print(
        "TUNING COMPLETE"
    )

    print(
        "=" * 78
    )

    for result in results:

        evaluation = result[
            "evaluation"
        ]

        print()
        print(
            result[
                "method"
            ]
        )

        print(
            "  Mean DEV F1:",
            f"{evaluation['mean_f1']:.4f}"
        )

        print(
            "  Mean precision:",
            f"{evaluation['mean_precision']:.4f}"
        )

        print(
            "  Mean recall:",
            f"{evaluation['mean_recall']:.4f}"
        )

        print(
            "  Ambiguity:",
            f"{evaluation['mean_ambiguous_matched_a_rate']:.4f}"
        )

        print(
            "  Candidates/A:",
            f"{evaluation['mean_candidates_per_a']:.4f}"
        )

    print()
    print(
        "Frozen DEV-selected parameters saved to:"
    )

    print(
        summary_file
    )

    print()
    print(
        "Do NOT modify these parameters after looking at FINAL results."
    )


if __name__ == "__main__":
    main()
