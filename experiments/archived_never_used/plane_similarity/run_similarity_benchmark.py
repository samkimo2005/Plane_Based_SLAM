from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

import weighted_geometric_similarity as weighted_module
import hard_threshold_similarity as hard_module
import descriptor_similarity as descriptor_module
import joint_association_pose_similarity_v2 as joint_module

from similarity_common import load_similarity_case
from similarity_evaluation import (
    evaluate_similarity_candidates,
    evaluate_predicted_pairs,
)


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic_similarity_benchmark"
    / "final"
)

TUNING_DIR = (
    PROJECT_ROOT
    / "results"
    / "plane_similarity"
    / "tuning"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "plane_similarity"
    / "final"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SIMILARITY_PARAMETER_FILE = (
    TUNING_DIR
    / "best_similarity_parameters.json"
)

JOINT_PARAMETER_FILE = (
    TUNING_DIR
    / "best_joint_v2_parameters.json"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "similarity_final_results.csv"
)

OUTPUT_PAIRS_JSONL = (
    OUTPUT_DIR
    / "similarity_final_pairs.jsonl"
)

OUTPUT_PARAMETER_SNAPSHOT = (
    OUTPUT_DIR
    / "frozen_similarity_parameters_used.json"
)

EXPECTED_FINAL_CASES = 90


# ============================================================
# 2. Method names
# ============================================================

WEIGHTED_NAME = "Weighted Geometric Cost"
HARD_NAME = "Hard Geometric Thresholds"
DESCRIPTOR_NAME = "Descriptor Similarity"
JOINT_NAME = "Joint Association + Pose Refinement V2"


# ============================================================
# 3. Resolve method symbols robustly
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
        f"Could not resolve symbol from {module.__name__}.\n"
        f"Tried: {possible_names}\n"
        f"Available public names:\n{available}"
    )


WeightedParameterClass = resolve_symbol(
    weighted_module,
    [
        "WeightedGeometricSimilarityParameters",
        "WeightedGeometricParameters",
        "WeightedSimilarityParameters",
    ],
)

WeightedRunner = resolve_symbol(
    weighted_module,
    [
        "run_weighted_geometric_similarity_case",
        "run_weighted_similarity_case",
        "run_weighted_geometric_case",
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
# 4. JSON helpers
# ============================================================

def load_json(
    path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(
            f
        )


def decode_special_json_values(
    value,
):
    """
    tune_similarity_parameters.py writes infinity as "inf".
    Convert special strings back to numerical values.
    """

    if isinstance(
        value,
        dict,
    ):
        return {
            key: decode_special_json_values(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            decode_special_json_values(
                item
            )
            for item in value
        ]

    if value == "inf":
        return float(
            "inf"
        )

    if value == "-inf":
        return float(
            "-inf"
        )

    if value == "nan":
        return float(
            "nan"
        )

    return value


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
        number = float(
            value
        )

        if math.isinf(
            number
        ):
            return (
                "inf"
                if number > 0
                else "-inf"
            )

        if math.isnan(
            number
        ):
            return "nan"

        return number

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
# 5. Parameter-object helpers
# ============================================================

def make_parameter_object(
    parameter_class,
    parameter_values,
):
    params = parameter_class()

    for name, value in parameter_values.items():

        if not hasattr(
            params,
            name,
        ):
            raise AttributeError(
                f"{parameter_class.__name__} has no parameter "
                f"named '{name}'."
            )

        # Histogram boundaries were originally tuples.
        if (
            isinstance(
                value,
                list,
            )
            and (
                "bins" in name
            )
        ):
            value = tuple(
                value
            )

        setattr(
            params,
            name,
            value,
        )

    return params


def load_frozen_parameter_objects():
    similarity_data = decode_special_json_values(
        load_json(
            SIMILARITY_PARAMETER_FILE
        )
    )

    joint_data = decode_special_json_values(
        load_json(
            JOINT_PARAMETER_FILE
        )
    )

    methods = similarity_data[
        "methods"
    ]

    weighted_parameters = methods[
        WEIGHTED_NAME
    ][
        "parameters"
    ]

    hard_parameters = methods[
        HARD_NAME
    ][
        "parameters"
    ]

    descriptor_parameters = methods[
        DESCRIPTOR_NAME
    ][
        "parameters"
    ]

    joint_parameters = joint_data[
        "full_frozen_parameters"
    ]

    parameter_objects = {
        WEIGHTED_NAME:
            make_parameter_object(
                WeightedParameterClass,
                weighted_parameters,
            ),

        HARD_NAME:
            make_parameter_object(
                HardParameterClass,
                hard_parameters,
            ),

        DESCRIPTOR_NAME:
            make_parameter_object(
                DescriptorParameterClass,
                descriptor_parameters,
            ),

        JOINT_NAME:
            make_parameter_object(
                joint_module
                .JointAssociationPoseParameters,
                joint_parameters,
            ),
    }

    snapshot = {
        "source_files":
            {
                "pairwise_similarity":
                    str(
                        SIMILARITY_PARAMETER_FILE
                    ),

                "joint_v2":
                    str(
                        JOINT_PARAMETER_FILE
                    ),
            },

        "parameters":
            {
                WEIGHTED_NAME:
                    weighted_parameters,

                HARD_NAME:
                    hard_parameters,

                DESCRIPTOR_NAME:
                    descriptor_parameters,

                JOINT_NAME:
                    joint_parameters,
            },
    }

    return (
        parameter_objects,
        snapshot,
    )


# ============================================================
# 6. Load FINAL cases only
# ============================================================

def load_final_cases():
    json_files = sorted(
        FINAL_DIR.glob(
            "*.json"
        )
    )

    if len(
        json_files
    ) != EXPECTED_FINAL_CASES:
        raise RuntimeError(
            f"Expected {EXPECTED_FINAL_CASES} FINAL cases, "
            f"but found {len(json_files)} in:\n"
            f"{FINAL_DIR}"
        )

    cases = []

    for path in json_files:
        cases.append(
            {
                "path":
                    path,

                "data":
                    load_similarity_case(
                        path
                    ),
            }
        )

    return cases


# ============================================================
# 7. Numeric helper
# ============================================================

def blank_if_none(
    value,
):
    if value is None:
        return ""

    return value


# ============================================================
# 8. Candidate-method evaluation row
# ============================================================

def run_candidate_method(
    method_name,
    runner,
    params,
    case_data,
):
    t0 = perf_counter()

    result = runner(
        case_data,
        params,
    )

    t1 = perf_counter()

    runtime_ms = (
        t1
        - t0
    ) * 1000.0

    evaluation = (
        evaluate_similarity_candidates(
            case_data,
            result[
                "candidate_pairs"
            ],
        )
    )

    row = {
        "method":
            method_name,

        "evaluation_scope":
            "candidate_generation",

        "sample_id":
            case_data[
                "sample_id"
            ],

        "scene_type":
            case_data[
                "scene_type"
            ],

        "prior_level":
            case_data[
                "prior_level"
            ],

        "tp":
            evaluation.get(
                "tp",
                0,
            ),

        "fp":
            evaluation.get(
                "fp",
                0,
            ),

        "fn":
            evaluation.get(
                "fn",
                0,
            ),

        "precision":
            evaluation.get(
                "precision",
                0.0,
            ),

        "recall":
            evaluation.get(
                "recall",
                0.0,
            ),

        "f1":
            evaluation.get(
                "f1",
                0.0,
            ),

        "num_gt_matches":
            evaluation.get(
                "num_gt_matches",
                "",
            ),

        "num_predicted_pairs":
            len(
                result[
                    "candidate_pairs"
                ]
            ),

        "num_gt_matched_a":
            evaluation.get(
                "num_gt_matched_a",
                "",
            ),

        "num_gt_unmatched_a":
            evaluation.get(
                "num_gt_unmatched_a",
                "",
            ),

        "mean_candidates_per_a":
            evaluation.get(
                "mean_candidates_per_a",
                "",
            ),

        "candidate_density":
            evaluation.get(
                "candidate_density",
                "",
            ),

        "ambiguous_matched_a_rate":
            evaluation.get(
                "ambiguous_matched_a_rate",
                "",
            ),

        "missed_matched_a_rate":
            evaluation.get(
                "missed_matched_a_rate",
                "",
            ),

        "unmatched_rejection_rate":
            blank_if_none(
                evaluation.get(
                    "unmatched_rejection_rate",
                    None,
                )
            ),

        "runtime_ms":
            float(
                runtime_ms
            ),

        # Joint-only columns remain blank.
        "num_joint_iterations":
            "",

        "prior_rotation_error_deg":
            "",

        "final_rotation_error_deg":
            "",

        "rotation_improvement_deg":
            "",

        "prior_translation_error_m":
            "",

        "final_translation_error_m":
            "",

        "translation_improvement_m":
            "",
    }

    audit = {
        "method":
            method_name,

        "evaluation_scope":
            "candidate_generation",

        "sample_id":
            case_data[
                "sample_id"
            ],

        "candidate_pairs":
            result[
                "candidate_pairs"
            ],
    }

    return (
        row,
        audit,
    )


# ============================================================
# 9. Joint-method evaluation row
# ============================================================

def run_joint_method(
    params,
    case_data,
):
    t0 = perf_counter()

    result = (
        joint_module
        .run_joint_association_pose_case(
            case_data,
            params,
        )
    )

    t1 = perf_counter()

    runtime_ms = (
        t1
        - t0
    ) * 1000.0

    evaluation = (
        evaluate_predicted_pairs(
            case_data,
            result[
                "predicted_pairs"
            ],
        )
    )

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

    row = {
        "method":
            JOINT_NAME,

        "evaluation_scope":
            "final_correspondence",

        "sample_id":
            case_data[
                "sample_id"
            ],

        "scene_type":
            case_data[
                "scene_type"
            ],

        "prior_level":
            case_data[
                "prior_level"
            ],

        "tp":
            evaluation.get(
                "tp",
                0,
            ),

        "fp":
            evaluation.get(
                "fp",
                0,
            ),

        "fn":
            evaluation.get(
                "fn",
                0,
            ),

        "precision":
            evaluation.get(
                "precision",
                0.0,
            ),

        "recall":
            evaluation.get(
                "recall",
                0.0,
            ),

        "f1":
            evaluation.get(
                "f1",
                0.0,
            ),

        "num_gt_matches":
            evaluation.get(
                "num_gt_matches",
                "",
            ),

        "num_predicted_pairs":
            evaluation.get(
                "num_predicted_pairs",
                len(
                    result[
                        "predicted_pairs"
                    ]
                ),
            ),

        # Candidate-structure columns are intentionally blank.
        "num_gt_matched_a":
            "",

        "num_gt_unmatched_a":
            "",

        "mean_candidates_per_a":
            "",

        "candidate_density":
            "",

        "ambiguous_matched_a_rate":
            "",

        "missed_matched_a_rate":
            "",

        "unmatched_rejection_rate":
            "",

        "runtime_ms":
            float(
                runtime_ms
            ),

        "num_joint_iterations":
            result[
                "num_iterations"
            ],

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

    audit = {
        "method":
            JOINT_NAME,

        "evaluation_scope":
            "final_correspondence",

        "sample_id":
            case_data[
                "sample_id"
            ],

        "predicted_pairs":
            result[
                "predicted_pairs"
            ],

        "iteration_history":
            result[
                "iteration_history"
            ],

        "final_transform_b_to_a":
            result[
                "transform_b_to_a"
            ],
    }

    return (
        row,
        audit,
    )


# ============================================================
# 10. Main
# ============================================================

def main():
    print(
        "=" * 78
    )

    print(
        "Plane Similarity FINAL Benchmark"
    )

    print(
        "=" * 78
    )

    print()
    print(
        "This program reads ONLY the frozen parameter files "
        "and the FINAL 90-case benchmark."
    )

    print(
        "Do not modify parameters after inspecting these results."
    )

    (
        params,
        parameter_snapshot,
    ) = load_frozen_parameter_objects()

    cases = load_final_cases()

    with open(
        OUTPUT_PARAMETER_SNAPSHOT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_safe(
                parameter_snapshot
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )

    fieldnames = [
        "method",
        "evaluation_scope",
        "sample_id",
        "scene_type",
        "prior_level",

        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",

        "num_gt_matches",
        "num_predicted_pairs",

        "num_gt_matched_a",
        "num_gt_unmatched_a",
        "mean_candidates_per_a",
        "candidate_density",
        "ambiguous_matched_a_rate",
        "missed_matched_a_rate",
        "unmatched_rejection_rate",

        "runtime_ms",

        "num_joint_iterations",
        "prior_rotation_error_deg",
        "final_rotation_error_deg",
        "rotation_improvement_deg",
        "prior_translation_error_m",
        "final_translation_error_m",
        "translation_improvement_m",
    ]

    rows = []

    with open(
        OUTPUT_PAIRS_JSONL,
        "w",
        encoding="utf-8",
    ) as audit_file:

        for case_index, case_item in enumerate(
            cases,
            start=1,
        ):
            case_data = case_item[
                "data"
            ]

            print(
                f"[{case_index:02d}/{EXPECTED_FINAL_CASES}] "
                f"{case_data['sample_id']}"
            )

            method_specs = [
                (
                    WEIGHTED_NAME,
                    WeightedRunner,
                ),
                (
                    HARD_NAME,
                    HardRunner,
                ),
                (
                    DESCRIPTOR_NAME,
                    DescriptorRunner,
                ),
            ]

            for (
                method_name,
                runner,
            ) in method_specs:

                row, audit = run_candidate_method(
                    method_name,
                    runner,
                    params[
                        method_name
                    ],
                    case_data,
                )

                rows.append(
                    row
                )

                audit_file.write(
                    json.dumps(
                        json_safe(
                            audit
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            row, audit = run_joint_method(
                params[
                    JOINT_NAME
                ],
                case_data,
            )

            rows.append(
                row
            )

            audit_file.write(
                json.dumps(
                    json_safe(
                        audit
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )

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

        writer.writerows(
            rows
        )

    print()
    print(
        "=" * 78
    )

    print(
        "FINAL BENCHMARK COMPLETE"
    )

    print(
        "=" * 78
    )

    print()
    print(
        "Per-case results:"
    )

    print(
        OUTPUT_CSV
    )

    print()
    print(
        "Pair/correspondence audit file:"
    )

    print(
        OUTPUT_PAIRS_JSONL
    )

    print()
    print(
        "Frozen parameter snapshot:"
    )

    print(
        OUTPUT_PARAMETER_SNAPSHOT
    )

    print()
    print(
        f"Expected rows: "
        f"{EXPECTED_FINAL_CASES * 4}"
    )

    print(
        f"Actual rows: "
        f"{len(rows)}"
    )


if __name__ == "__main__":
    main()
