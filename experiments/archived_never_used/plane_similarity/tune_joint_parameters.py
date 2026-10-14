from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import joint_association_pose_similarity_v2 as joint_module

from similarity_common import load_similarity_case
from similarity_evaluation import evaluate_predicted_pairs


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

OUTPUT_CSV = (
    OUTPUT_DIR
    / "joint_v2_tuning.csv"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "best_joint_v2_parameters.json"
)

EXPECTED_DEV_CASES = 27

# Two coordinate passes are enough for this small calibration.
NUM_SWEEP_PASSES = 2


# ============================================================
# 2. Minimal tuning philosophy
#
# We do NOT tune every parameter.
#
# Parameters that already have stable, reasonable behaviour
# remain fixed.
#
# Only parameters that directly control:
#   - unmatched decisions,
#   - partial-visibility centroid influence,
#   - pose-prior regularisation
#
# are calibrated here.
# ============================================================

TUNING_GROUPS = [
    (
        "unmatched_cost",
        [
            (
                f"{value:.2f}",
                {
                    "unmatched_cost": value
                },
            )
            for value in [
                0.60,
                0.80,
                1.00,
                1.20,
            ]
        ],
    ),

    (
        "pose_centroid_weight",
        [
            (
                f"{value:.2f}",
                {
                    "pose_centroid_weight": value
                },
            )
            for value in [
                0.00,
                0.10,
                0.25,
                0.50,
            ]
        ],
    ),

    (
        "prior_rotation_scale_deg",
        [
            (
                f"{value:.1f} deg",
                {
                    "prior_rotation_scale_deg": value
                },
            )
            for value in [
                0.5,
                1.0,
                2.0,
                3.0,
                5.0,
            ]
        ],
    ),

    (
        "prior_translation_scale_m",
        [
            (
                f"{value:.3f} m",
                {
                    "prior_translation_scale_m": value
                },
            )
            for value in [
                0.010,
                0.020,
                0.030,
                0.050,
                0.080,
            ]
        ],
    ),
]


# ============================================================
# 3. Load DEV cases
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
# 4. Parameter helpers
# ============================================================

def make_params(
    updates,
):
    params = (
        joint_module
        .JointAssociationPoseParameters()
    )

    for name, value in updates.items():

        if not hasattr(
            params,
            name,
        ):
            raise AttributeError(
                "JointAssociationPoseParameters "
                f"does not contain '{name}'."
            )

        setattr(
            params,
            name,
            value,
        )

    return params


def params_to_dict(
    params,
):
    return dict(
        vars(
            params
        )
    )


# ============================================================
# 5. Micro metric helper
# ============================================================

def calculate_micro_metrics(
    total_tp,
    total_fp,
    total_fn,
):
    if (
        total_tp
        + total_fp
    ) > 0:

        precision = (
            total_tp
            / (
                total_tp
                + total_fp
            )
        )

    else:
        precision = 0.0

    if (
        total_tp
        + total_fn
    ) > 0:

        recall = (
            total_tp
            / (
                total_tp
                + total_fn
            )
        )

    else:
        recall = 0.0

    if (
        precision
        + recall
    ) > 0:

        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    else:
        f1 = 0.0

    return (
        float(
            precision
        ),
        float(
            recall
        ),
        float(
            f1
        ),
    )


# ============================================================
# 6. Evaluate one parameter configuration
# ============================================================

def evaluate_configuration(
    cases,
    parameter_updates,
):
    params = make_params(
        parameter_updates
    )

    case_f1 = []
    case_precision = []
    case_recall = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    runtimes_ms = []

    # Pose diagnostics
    prior_rotation_errors = []
    final_rotation_errors = []

    prior_translation_errors = []
    final_translation_errors = []

    rotation_improvements = []
    translation_improvements = []

    # Separate by prior level
    per_prior = {
        "exact": [],
        "mild": [],
        "moderate": [],
    }

    for case_item in cases:

        case_data = case_item[
            "data"
        ]

        # ----------------------------------------------------
        # Run Joint V2
        # ----------------------------------------------------

        t0 = perf_counter()

        result = (
            joint_module
            .run_joint_association_pose_case(
                case_data,
                params,
            )
        )

        t1 = perf_counter()

        runtimes_ms.append(
            (
                t1
                - t0
            )
            * 1000.0
        )

        # ----------------------------------------------------
        # Correspondence evaluation
        # ----------------------------------------------------

        evaluation = (
            evaluate_predicted_pairs(
                case_data,
                result[
                    "predicted_pairs"
                ],
            )
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

        # ----------------------------------------------------
        # Pose evaluation
        # ----------------------------------------------------

        (
            gt_rotation,
            gt_translation,
        ) = joint_module.parse_transform(
            case_data[
                "gt_transform_b_to_a"
            ]
        )

        (
            prior_rotation,
            prior_translation,
        ) = joint_module.parse_transform(
            case_data[
                "prior_transform_b_to_a"
            ]
        )

        prior_rotation_error = (
            joint_module.rotation_error_deg(
                prior_rotation,
                gt_rotation,
            )
        )

        final_rotation_error = (
            joint_module.rotation_error_deg(
                result[
                    "rotation_b_to_a"
                ],
                gt_rotation,
            )
        )

        prior_translation_error = (
            joint_module.translation_error_m(
                prior_translation,
                gt_translation,
            )
        )

        final_translation_error = (
            joint_module.translation_error_m(
                result[
                    "translation_b_to_a"
                ],
                gt_translation,
            )
        )

        rotation_improvement = (
            prior_rotation_error
            - final_rotation_error
        )

        translation_improvement = (
            prior_translation_error
            - final_translation_error
        )

        prior_rotation_errors.append(
            prior_rotation_error
        )

        final_rotation_errors.append(
            final_rotation_error
        )

        prior_translation_errors.append(
            prior_translation_error
        )

        final_translation_errors.append(
            final_translation_error
        )

        rotation_improvements.append(
            rotation_improvement
        )

        translation_improvements.append(
            translation_improvement
        )

        prior_level = str(
            case_data[
                "prior_level"
            ]
        ).lower()

        if prior_level not in per_prior:
            per_prior[
                prior_level
            ] = []

        per_prior[
            prior_level
        ].append(
            {
                "f1":
                    float(
                        evaluation[
                            "f1"
                        ]
                    ),

                "prior_rotation_error_deg":
                    float(
                        prior_rotation_error
                    ),

                "final_rotation_error_deg":
                    float(
                        final_rotation_error
                    ),

                "rotation_improvement_deg":
                    float(
                        rotation_improvement
                    ),

                "prior_translation_error_m":
                    float(
                        prior_translation_error
                    ),

                "final_translation_error_m":
                    float(
                        final_translation_error
                    ),

                "translation_improvement_m":
                    float(
                        translation_improvement
                    ),
            }
        )

    (
        micro_precision,
        micro_recall,
        micro_f1,
    ) = calculate_micro_metrics(
        total_tp,
        total_fp,
        total_fn,
    )

    # --------------------------------------------------------
    # Pose score
    #
    # 5 deg and 0.10 m are used only as fixed normalisation
    # scales so rotation and translation can be compared in
    # one diagnostic score.
    #
    # Lower = better.
    # --------------------------------------------------------

    pose_scores = (
        np.asarray(
            final_rotation_errors,
            dtype=float,
        )
        / 5.0

        +

        np.asarray(
            final_translation_errors,
            dtype=float,
        )
        / 0.10
    )

    # --------------------------------------------------------
    # Per-prior summaries
    # --------------------------------------------------------

    per_prior_summary = {}

    for prior_level, rows in per_prior.items():

        if not rows:
            continue

        per_prior_summary[
            prior_level
        ] = {
            "mean_f1":
                float(
                    np.mean(
                        [
                            row[
                                "f1"
                            ]
                            for row in rows
                        ]
                    )
                ),

            "mean_final_rotation_error_deg":
                float(
                    np.mean(
                        [
                            row[
                                "final_rotation_error_deg"
                            ]
                            for row in rows
                        ]
                    )
                ),

            "mean_rotation_improvement_deg":
                float(
                    np.mean(
                        [
                            row[
                                "rotation_improvement_deg"
                            ]
                            for row in rows
                        ]
                    )
                ),

            "mean_final_translation_error_m":
                float(
                    np.mean(
                        [
                            row[
                                "final_translation_error_m"
                            ]
                            for row in rows
                        ]
                    )
                ),

            "mean_translation_improvement_m":
                float(
                    np.mean(
                        [
                            row[
                                "translation_improvement_m"
                            ]
                            for row in rows
                        ]
                    )
                ),
        }

    return {
        "mean_f1":
            float(
                np.mean(
                    case_f1
                )
            ),

        "mean_precision":
            float(
                np.mean(
                    case_precision
                )
            ),

        "mean_recall":
            float(
                np.mean(
                    case_recall
                )
            ),

        "micro_precision":
            micro_precision,

        "micro_recall":
            micro_recall,

        "micro_f1":
            micro_f1,

        "total_tp":
            int(
                total_tp
            ),

        "total_fp":
            int(
                total_fp
            ),

        "total_fn":
            int(
                total_fn
            ),

        "mean_prior_rotation_error_deg":
            float(
                np.mean(
                    prior_rotation_errors
                )
            ),

        "mean_final_rotation_error_deg":
            float(
                np.mean(
                    final_rotation_errors
                )
            ),

        "mean_rotation_improvement_deg":
            float(
                np.mean(
                    rotation_improvements
                )
            ),

        "mean_prior_translation_error_m":
            float(
                np.mean(
                    prior_translation_errors
                )
            ),

        "mean_final_translation_error_m":
            float(
                np.mean(
                    final_translation_errors
                )
            ),

        "mean_translation_improvement_m":
            float(
                np.mean(
                    translation_improvements
                )
            ),

        "mean_normalised_pose_error":
            float(
                np.mean(
                    pose_scores
                )
            ),

        "mean_runtime_ms":
            float(
                np.mean(
                    runtimes_ms
                )
            ),

        "per_prior":
            per_prior_summary,
    }


# ============================================================
# 7. Selection rule
# ============================================================

def selection_key(
    evaluation,
):
    """
    Selection priority:

        1. maximise mean correspondence F1
        2. maximise micro F1
        3. minimise final normalised pose error
        4. maximise mean rotation improvement
        5. maximise mean translation improvement

    Runtime is NOT used for parameter selection.
    """

    return (
        evaluation[
            "mean_f1"
        ],

        evaluation[
            "micro_f1"
        ],

        -evaluation[
            "mean_normalised_pose_error"
        ],

        evaluation[
            "mean_rotation_improvement_deg"
        ],

        evaluation[
            "mean_translation_improvement_m"
        ],
    )


# ============================================================
# 8. Print one evaluation
# ============================================================

def print_evaluation(
    prefix,
    evaluation,
):
    print(
        f"{prefix:<26} "
        f"F1={evaluation['mean_f1']:.4f} "
        f"microF1={evaluation['micro_f1']:.4f} "
        f"pose={evaluation['mean_normalised_pose_error']:.4f} "
        f"dR={evaluation['mean_rotation_improvement_deg']:+.4f}deg "
        f"dT={evaluation['mean_translation_improvement_m']:+.5f}m"
    )


# ============================================================
# 9. Coordinate sweep
# ============================================================

def run_tuning(
    cases,
):
    current_updates = {}

    history = []

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline_evaluation = (
        evaluate_configuration(
            cases,
            current_updates,
        )
    )

    baseline_params = make_params(
        current_updates
    )

    history.append(
        {
            "pass_number":
                0,

            "parameter_group":
                "baseline",

            "candidate_label":
                "current V2 defaults",

            "selected":
                True,

            "updates":
                dict(
                    current_updates
                ),

            "full_parameters":
                params_to_dict(
                    baseline_params
                ),

            "evaluation":
                baseline_evaluation,
        }
    )

    print()
    print(
        "Baseline:"
    )

    print_evaluation(
        "current V2 defaults",
        baseline_evaluation,
    )

    # --------------------------------------------------------
    # Two coordinate passes
    # --------------------------------------------------------

    for pass_number in range(
        1,
        NUM_SWEEP_PASSES + 1,
    ):

        print()
        print(
            "=" * 78
        )

        print(
            f"Coordinate sweep pass "
            f"{pass_number}/{NUM_SWEEP_PASSES}"
        )

        print(
            "=" * 78
        )

        for (
            group_name,
            candidates,
        ) in TUNING_GROUPS:

            print()
            print(
                f"[{group_name}]"
            )

            group_rows = []

            for (
                candidate_label,
                candidate_update,
            ) in candidates:

                trial_updates = dict(
                    current_updates
                )

                trial_updates.update(
                    candidate_update
                )

                evaluation = (
                    evaluate_configuration(
                        cases,
                        trial_updates,
                    )
                )

                params = make_params(
                    trial_updates
                )

                row = {
                    "pass_number":
                        pass_number,

                    "parameter_group":
                        group_name,

                    "candidate_label":
                        candidate_label,

                    "selected":
                        False,

                    "updates":
                        dict(
                            trial_updates
                        ),

                    "full_parameters":
                        params_to_dict(
                            params
                        ),

                    "evaluation":
                        evaluation,
                }

                group_rows.append(
                    row
                )

                print_evaluation(
                    candidate_label,
                    evaluation,
                )

            best_row = max(
                group_rows,
                key=lambda row: selection_key(
                    row[
                        "evaluation"
                    ]
                ),
            )

            best_row[
                "selected"
            ] = True

            current_updates = dict(
                best_row[
                    "updates"
                ]
            )

            history.extend(
                group_rows
            )

            print(
                "  -> selected:",
                best_row[
                    "candidate_label"
                ]
            )

    # --------------------------------------------------------
    # Final selected configuration
    # --------------------------------------------------------

    final_evaluation = (
        evaluate_configuration(
            cases,
            current_updates,
        )
    )

    final_params = make_params(
        current_updates
    )

    return (
        current_updates,
        params_to_dict(
            final_params
        ),
        final_evaluation,
        history,
    )


# ============================================================
# 10. Save tuning history CSV
# ============================================================

def save_history_csv(
    history,
):
    fieldnames = [
        "pass_number",
        "parameter_group",
        "candidate_label",
        "selected",
        "updates_json",
        "full_parameters_json",

        "mean_f1",
        "mean_precision",
        "mean_recall",

        "micro_f1",

        "mean_final_rotation_error_deg",
        "mean_rotation_improvement_deg",

        "mean_final_translation_error_m",
        "mean_translation_improvement_m",

        "mean_normalised_pose_error",

        "exact_mean_f1",
        "exact_final_rotation_error_deg",
        "exact_final_translation_error_m",

        "mild_mean_f1",
        "mild_rotation_improvement_deg",
        "mild_translation_improvement_m",

        "moderate_mean_f1",
        "moderate_rotation_improvement_deg",
        "moderate_translation_improvement_m",

        "mean_runtime_ms",
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

            per_prior = evaluation[
                "per_prior"
            ]

            exact = per_prior.get(
                "exact",
                {},
            )

            mild = per_prior.get(
                "mild",
                {},
            )

            moderate = per_prior.get(
                "moderate",
                {},
            )

            writer.writerow(
                {
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

                    "selected":
                        int(
                            row[
                                "selected"
                            ]
                        ),

                    "updates_json":
                        json.dumps(
                            row[
                                "updates"
                            ],
                            ensure_ascii=False,
                        ),

                    "full_parameters_json":
                        json.dumps(
                            row[
                                "full_parameters"
                            ],
                            ensure_ascii=False,
                        ),

                    "mean_f1":
                        evaluation[
                            "mean_f1"
                        ],

                    "mean_precision":
                        evaluation[
                            "mean_precision"
                        ],

                    "mean_recall":
                        evaluation[
                            "mean_recall"
                        ],

                    "micro_f1":
                        evaluation[
                            "micro_f1"
                        ],

                    "mean_final_rotation_error_deg":
                        evaluation[
                            "mean_final_rotation_error_deg"
                        ],

                    "mean_rotation_improvement_deg":
                        evaluation[
                            "mean_rotation_improvement_deg"
                        ],

                    "mean_final_translation_error_m":
                        evaluation[
                            "mean_final_translation_error_m"
                        ],

                    "mean_translation_improvement_m":
                        evaluation[
                            "mean_translation_improvement_m"
                        ],

                    "mean_normalised_pose_error":
                        evaluation[
                            "mean_normalised_pose_error"
                        ],

                    "exact_mean_f1":
                        exact.get(
                            "mean_f1",
                            "",
                        ),

                    "exact_final_rotation_error_deg":
                        exact.get(
                            "mean_final_rotation_error_deg",
                            "",
                        ),

                    "exact_final_translation_error_m":
                        exact.get(
                            "mean_final_translation_error_m",
                            "",
                        ),

                    "mild_mean_f1":
                        mild.get(
                            "mean_f1",
                            "",
                        ),

                    "mild_rotation_improvement_deg":
                        mild.get(
                            "mean_rotation_improvement_deg",
                            "",
                        ),

                    "mild_translation_improvement_m":
                        mild.get(
                            "mean_translation_improvement_m",
                            "",
                        ),

                    "moderate_mean_f1":
                        moderate.get(
                            "mean_f1",
                            "",
                        ),

                    "moderate_rotation_improvement_deg":
                        moderate.get(
                            "mean_rotation_improvement_deg",
                            "",
                        ),

                    "moderate_translation_improvement_m":
                        moderate.get(
                            "mean_translation_improvement_m",
                            "",
                        ),

                    "mean_runtime_ms":
                        evaluation[
                            "mean_runtime_ms"
                        ],
                }
            )


# ============================================================
# 11. Save selected Joint V2 configuration
# ============================================================

def save_best_json(
    selected_updates,
    full_parameters,
    evaluation,
):
    data = {
        "method":
            "Joint Association + Pose Refinement V2",

        "tuning_dataset":
            "synthetic_similarity_benchmark/dev",

        "num_dev_cases":
            EXPECTED_DEV_CASES,

        "tuning_strategy":
            (
                "minimal two-pass coordinate sweep; "
                "only parameters requiring calibration "
                "were varied"
            ),

        "parameters_tuned":
            [
                group_name
                for (
                    group_name,
                    _
                ) in TUNING_GROUPS
            ],

        "parameters_left_fixed":
            [
                "normal_weight",
                "offset_weight",
                "centroid_weight",
                "area_weight",
                "normal_scale_deg",
                "offset_scale_m",
                "centroid_scale_m",
                "area_scale_ratio",
                "pose_normal_scale_deg",
                "pose_offset_scale_m",
                "pose_centroid_scale_m",
                "robust_loss",
                "robust_f_scale",
                "max_joint_iterations",
                "min_matches_for_pose",
                "rotation_convergence_deg",
                "translation_convergence_m",
                "max_rotation_update_deg",
                "max_translation_update_m",
            ],

        "selection_rule":
            [
                "maximise mean correspondence F1",
                "then maximise micro F1",
                "then minimise normalised final pose error",
                "then maximise rotation improvement",
                "then maximise translation improvement",
            ],

        "selected_parameter_updates":
            selected_updates,

        "full_frozen_parameters":
            full_parameters,

        "dev_evaluation":
            evaluation,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# 12. Main
# ============================================================

def main():

    print(
        "=" * 78
    )

    print(
        "Joint Association + Pose Refinement V2"
    )

    print(
        "DEV Parameter Tuning"
    )

    print(
        "=" * 78
    )

    print()
    print(
        "Only DEV cases are used."
    )

    print(
        "FINAL cases are not accessed."
    )

    print(
        "Only four high-impact parameter groups are tuned."
    )

    print(
        "Stable parameters remain fixed."
    )

    cases = load_dev_cases()

    print()
    print(
        "DEV cases loaded:",
        len(
            cases
        )
    )

    (
        selected_updates,
        full_parameters,
        final_evaluation,
        history,
    ) = run_tuning(
        cases
    )

    save_history_csv(
        history
    )

    save_best_json(
        selected_updates,
        full_parameters,
        final_evaluation,
    )

    print()
    print(
        "=" * 78
    )

    print(
        "JOINT V2 TUNING COMPLETE"
    )

    print(
        "=" * 78
    )

    print()
    print(
        "Selected full parameter set:"
    )

    for key, value in full_parameters.items():
        print(
            f"  {key}: {value}"
        )

    print()
    print(
        "Final DEV metrics:"
    )

    print(
        "  Mean F1:",
        f"{final_evaluation['mean_f1']:.4f}"
    )

    print(
        "  Micro F1:",
        f"{final_evaluation['micro_f1']:.4f}"
    )

    print(
        "  Mean final rotation error:",
        f"{final_evaluation['mean_final_rotation_error_deg']:.4f}",
        "deg"
    )

    print(
        "  Mean rotation improvement:",
        f"{final_evaluation['mean_rotation_improvement_deg']:+.4f}",
        "deg"
    )

    print(
        "  Mean final translation error:",
        f"{final_evaluation['mean_final_translation_error_m']:.5f}",
        "m"
    )

    print(
        "  Mean translation improvement:",
        f"{final_evaluation['mean_translation_improvement_m']:+.5f}",
        "m"
    )

    print()
    print(
        "Per-prior-level summary:"
    )

    for (
        prior_level,
        summary
    ) in final_evaluation[
        "per_prior"
    ].items():

        print()
        print(
            f"  [{prior_level}]"
        )

        print(
            "    Mean F1:",
            f"{summary['mean_f1']:.4f}"
        )

        print(
            "    Final rotation error:",
            f"{summary['mean_final_rotation_error_deg']:.4f}",
            "deg"
        )

        print(
            "    Rotation improvement:",
            f"{summary['mean_rotation_improvement_deg']:+.4f}",
            "deg"
        )

        print(
            "    Final translation error:",
            f"{summary['mean_final_translation_error_m']:.5f}",
            "m"
        )

        print(
            "    Translation improvement:",
            f"{summary['mean_translation_improvement_m']:+.5f}",
            "m"
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
        "Frozen Joint V2 parameters:"
    )

    print(
        OUTPUT_JSON
    )

    print()
    print(
        "Do not alter the selected configuration "
        "after inspecting FINAL results."
    )


if __name__ == "__main__":
    main()
