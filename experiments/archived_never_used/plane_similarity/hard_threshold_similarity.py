from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from similarity_common import (
    calculate_pair_geometry,
    load_similarity_case,
    transform_frame_b_to_a,
)

from similarity_evaluation import (
    evaluate_similarity_candidates,
)


# ============================================================
# 1. Hard-threshold parameters
# ============================================================

@dataclass
class HardThresholdParameters:
    """
    Initial parameters for the hard geometric
    threshold similarity method.

    These are DEVELOPMENT starting values only.

    They will later be tuned using the DEV dataset
    and then frozen before the FINAL benchmark.
    """

    # Maximum allowed difference in plane orientation.
    max_normal_angle_deg: float = 10.0

    # Maximum allowed difference in plane offset.
    max_offset_difference_m: float = 0.15

    # Maximum allowed distance between plane centroids.
    max_centroid_distance_m: float = 1.00

    # Maximum relative visible-area difference.
    max_area_difference_ratio: float = 0.50


# ============================================================
# 2. Hard-threshold comparison for one plane pair
# ============================================================

def calculate_hard_threshold_similarity(
    plane_a: dict,
    plane_b_in_a: dict,
    params: HardThresholdParameters,
) -> dict:
    """
    Compare one plane from Frame A with one Frame-B plane
    already transformed into Frame A.

    A candidate is accepted only when ALL four
    geometric conditions pass.
    """

    # --------------------------------------------------------
    # Calculate the same common geometric quantities
    # used by the weighted method.
    # --------------------------------------------------------

    geometry = calculate_pair_geometry(
        plane_a,
        plane_b_in_a,
    )

    normal_angle_deg = geometry[
        "normal_angle_deg"
    ]

    offset_difference_m = geometry[
        "offset_difference_m"
    ]

    centroid_distance_m = geometry[
        "centroid_distance_m"
    ]

    area_difference_ratio = geometry[
        "area_difference_ratio"
    ]

    # --------------------------------------------------------
    # Independent hard geometric gates
    # --------------------------------------------------------

    pass_normal = (
        normal_angle_deg
        <= params.max_normal_angle_deg
    )

    pass_offset = (
        offset_difference_m
        <= params.max_offset_difference_m
    )

    pass_centroid = (
        centroid_distance_m
        <= params.max_centroid_distance_m
    )

    pass_area = (
        area_difference_ratio
        <= params.max_area_difference_ratio
    )

    # --------------------------------------------------------
    # Candidate decision
    # --------------------------------------------------------

    is_candidate = (
        pass_normal
        and pass_offset
        and pass_centroid
        and pass_area
    )

    # --------------------------------------------------------
    # Diagnostic ratios
    # --------------------------------------------------------
    #
    # These ratios do NOT decide the candidate using
    # weighted combination.
    #
    # They are only useful for inspection.
    #
    # Example:
    #
    # normal angle = 5 degrees
    # threshold    = 10 degrees
    #
    # ratio = 0.5
    #
    # A ratio > 1 means that particular threshold failed.
    # --------------------------------------------------------

    normal_ratio = (
        normal_angle_deg
        / params.max_normal_angle_deg
    )

    offset_ratio = (
        offset_difference_m
        / params.max_offset_difference_m
    )

    centroid_ratio = (
        centroid_distance_m
        / params.max_centroid_distance_m
    )

    area_ratio = (
        area_difference_ratio
        / params.max_area_difference_ratio
    )

    # Maximum ratio provides a simple diagnostic indication
    # of how close the pair is to violating the worst gate.
    diagnostic_score = max(
        normal_ratio,
        offset_ratio,
        centroid_ratio,
        area_ratio,
    )

    return {
        # Raw geometry
        "normal_angle_deg":
            float(normal_angle_deg),

        "offset_difference_m":
            float(offset_difference_m),

        "centroid_distance_m":
            float(centroid_distance_m),

        "area_difference_ratio":
            float(area_difference_ratio),

        # Individual gate results
        "pass_normal":
            bool(pass_normal),

        "pass_offset":
            bool(pass_offset),

        "pass_centroid":
            bool(pass_centroid),

        "pass_area":
            bool(pass_area),

        # Diagnostic values
        "normal_ratio":
            float(normal_ratio),

        "offset_ratio":
            float(offset_ratio),

        "centroid_ratio":
            float(centroid_ratio),

        "area_ratio":
            float(area_ratio),

        "diagnostic_score":
            float(diagnostic_score),

        # Final binary decision
        "is_candidate":
            bool(is_candidate),
    }


# ============================================================
# 3. Compare every possible A-B pair
# ============================================================

def compare_all_pairs(
    frame_a_planes: list,
    frame_b_planes: list,
    prior_transform: dict,
    params: HardThresholdParameters,
) -> list:
    """
    Compare every Frame-A plane against every Frame-B plane.

    This is still only the similarity stage.

    Therefore one A plane may have multiple candidate B planes.

    No Hungarian / MNN / Greedy assignment is performed.
    """

    # --------------------------------------------------------
    # Transform all B planes into Frame A
    # --------------------------------------------------------

    frame_b_in_a = transform_frame_b_to_a(
        frame_b_planes,
        prior_transform,
    )

    results = []

    # --------------------------------------------------------
    # Evaluate all possible plane pairs
    # --------------------------------------------------------

    for plane_a in frame_a_planes:

        for plane_b_original, plane_b_transformed in zip(
            frame_b_planes,
            frame_b_in_a,
        ):

            similarity = (
                calculate_hard_threshold_similarity(
                    plane_a,
                    plane_b_transformed,
                    params,
                )
            )

            row = {
                "a_obs_id":
                    plane_a["obs_id"],

                "b_obs_id":
                    plane_b_original["obs_id"],

                **similarity,
            }

            results.append(
                row
            )

    return results


# ============================================================
# 4. Extract accepted candidate pairs
# ============================================================

def get_candidate_pairs(
    pair_results: list,
) -> list:
    """
    Return only the A-B pairs that pass ALL
    hard geometric thresholds.
    """

    return [
        row
        for row in pair_results
        if row["is_candidate"]
    ]


# ============================================================
# 5. Build binary candidate matrix
# ============================================================

def build_candidate_matrix(
    frame_a_planes: list,
    frame_b_planes: list,
    pair_results: list,
):
    """
    Build a binary matrix:

                    B0   B1   B2
        A0           0    1    0
        A1           0    0    1
        A2           1    0    0

    1 = candidate
    0 = rejected

    This is NOT a one-to-one assignment matrix.
    """

    a_ids = [
        plane["obs_id"]
        for plane in frame_a_planes
    ]

    b_ids = [
        plane["obs_id"]
        for plane in frame_b_planes
    ]

    matrix = np.zeros(
        (
            len(a_ids),
            len(b_ids),
        ),
        dtype=int,
    )

    a_index = {
        obs_id: i
        for i, obs_id
        in enumerate(a_ids)
    }

    b_index = {
        obs_id: j
        for j, obs_id
        in enumerate(b_ids)
    }

    for row in pair_results:

        i = a_index[
            row["a_obs_id"]
        ]

        j = b_index[
            row["b_obs_id"]
        ]

        matrix[i, j] = int(
            row["is_candidate"]
        )

    return matrix, a_ids, b_ids


# ============================================================
# 6. Run one complete benchmark case
# ============================================================

def run_hard_threshold_case(
    case_data: dict,
    params: HardThresholdParameters,
) -> dict:
    """
    Run Hard Geometric Threshold similarity on one
    synthetic benchmark case.
    """

    frame_a_planes = (
        case_data[
            "frame_a"
        ][
            "planes"
        ]
    )

    frame_b_planes = (
        case_data[
            "frame_b"
        ][
            "planes"
        ]
    )

    prior_transform = (
        case_data[
            "prior_transform_b_to_a"
        ]
    )

    pair_results = compare_all_pairs(
        frame_a_planes,
        frame_b_planes,
        prior_transform,
        params,
    )

    candidate_pairs = (
        get_candidate_pairs(
            pair_results
        )
    )

    (
        candidate_matrix,
        a_ids,
        b_ids,
    ) = build_candidate_matrix(
        frame_a_planes,
        frame_b_planes,
        pair_results,
    )

    return {
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

        "pair_results":
            pair_results,

        "candidate_pairs":
            candidate_pairs,

        "candidate_matrix":
            candidate_matrix,

        "a_ids":
            a_ids,

        "b_ids":
            b_ids,
    }


# ============================================================
# 7. Standalone test
# ============================================================

def main():
    """
    Test Hard Geometric Threshold similarity using
    the first DEV benchmark case.

    The test:

        1. loads one DEV benchmark
        2. transforms Frame-B planes into Frame A
        3. applies four independent geometric thresholds
        4. generates candidate pairs
        5. evaluates candidates against ground truth
    """

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    dev_dir = (
        project_root
        / "data"
        / "synthetic_similarity_benchmark"
        / "dev"
    )

    json_files = sorted(
        dev_dir.glob(
            "*.json"
        )
    )

    if not json_files:

        raise FileNotFoundError(
            "No DEV benchmark JSON files were found.\n"
            "Run generate_similarity_benchmark.py first."
        )

    test_file = json_files[0]

    print("=" * 70)
    print("Hard Geometric Threshold Similarity Test")
    print("=" * 70)

    print(
        f"Input file: "
        f"{test_file.name}"
    )

    # --------------------------------------------------------
    # Load benchmark
    # --------------------------------------------------------

    case_data = (
        load_similarity_case(
            test_file
        )
    )

    # --------------------------------------------------------
    # Initial parameters
    # --------------------------------------------------------

    params = (
        HardThresholdParameters()
    )

    # --------------------------------------------------------
    # Run method
    # --------------------------------------------------------

    result = (
        run_hard_threshold_case(
            case_data,
            params,
        )
    )

    # --------------------------------------------------------
    # Common evaluator
    # --------------------------------------------------------

    evaluation = (
        evaluate_similarity_candidates(
            case_data,
            result[
                "candidate_pairs"
            ],
        )
    )

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print()

    print(
        f"Sample ID   : "
        f"{result['sample_id']}"
    )

    print(
        f"Scene       : "
        f"{result['scene_type']}"
    )

    print(
        f"Pose prior  : "
        f"{result['prior_level']}"
    )

    print(
        f"A planes    : "
        f"{len(result['a_ids'])}"
    )

    print(
        f"B planes    : "
        f"{len(result['b_ids'])}"
    )

    print(
        f"Total pairs : "
        f"{len(result['pair_results'])}"
    )

    print(
        f"Candidates  : "
        f"{len(result['candidate_pairs'])}"
    )

    # --------------------------------------------------------
    # Print thresholds
    # --------------------------------------------------------

    print()
    print("Thresholds:")
    print()

    print(
        f"Normal angle     <= "
        f"{params.max_normal_angle_deg:.2f} deg"
    )

    print(
        f"Offset difference <= "
        f"{params.max_offset_difference_m:.3f} m"
    )

    print(
        f"Centroid distance <= "
        f"{params.max_centroid_distance_m:.3f} m"
    )

    print(
        f"Area difference   <= "
        f"{params.max_area_difference_ratio:.3f}"
    )

    # --------------------------------------------------------
    # Print binary candidate matrix
    # --------------------------------------------------------

    print()
    print("Candidate matrix:")
    print()

    print(
        " " * 10,
        end="",
    )

    for b_id in result[
        "b_ids"
    ]:

        print(
            f"{b_id:>6}",
            end="",
        )

    print()

    for i, a_id in enumerate(
        result["a_ids"]
    ):

        print(
            f"{a_id:<10}",
            end="",
        )

        for j in range(
            len(
                result["b_ids"]
            )
        ):

            print(
                f"{result['candidate_matrix'][i, j]:6d}",
                end="",
            )

        print()

    # --------------------------------------------------------
    # Print accepted candidates
    # --------------------------------------------------------

    print()
    print("Candidate pairs:")
    print()

    if not result[
        "candidate_pairs"
    ]:

        print(
            "No candidate pairs passed all thresholds."
        )

    else:

        sorted_candidates = sorted(
            result[
                "candidate_pairs"
            ],
            key=lambda x: x[
                "diagnostic_score"
            ],
        )

        for candidate in sorted_candidates:

            print(
                f"{candidate['a_obs_id']} "
                f"<-> "
                f"{candidate['b_obs_id']} "
                f"| angle = "
                f"{candidate['normal_angle_deg']:.2f} deg "
                f"| offset = "
                f"{candidate['offset_difference_m']:.3f} m "
                f"| centroid = "
                f"{candidate['centroid_distance_m']:.3f} m "
                f"| area = "
                f"{candidate['area_difference_ratio']:.3f} "
                f"| diagnostic = "
                f"{candidate['diagnostic_score']:.3f}"
            )

    # --------------------------------------------------------
    # Common similarity evaluation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Similarity Evaluation")
    print("=" * 70)

    print(
        f"GT matches                  : "
        f"{evaluation['num_gt_matches']}"
    )

    print(
        f"Candidates                  : "
        f"{evaluation['num_candidates']}"
    )

    print(
        f"TP                          : "
        f"{evaluation['tp']}"
    )

    print(
        f"FP                          : "
        f"{evaluation['fp']}"
    )

    print(
        f"FN                          : "
        f"{evaluation['fn']}"
    )

    print(
        f"Precision                   : "
        f"{evaluation['precision']:.3f}"
    )

    print(
        f"Recall                      : "
        f"{evaluation['recall']:.3f}"
    )

    print(
        f"F1                          : "
        f"{evaluation['f1']:.3f}"
    )

    print(
        f"Mean candidates / A         : "
        f"{evaluation['mean_candidates_per_a']:.3f}"
    )

    print(
        f"Ambiguous matched A rate    : "
        f"{evaluation['ambiguous_matched_a_rate']:.3f}"
    )

    print(
        f"Missed matched A rate       : "
        f"{evaluation['missed_matched_a_rate']:.3f}"
    )

    # --------------------------------------------------------
    # Unmatched rejection
    # --------------------------------------------------------

    if (
        evaluation[
            "num_gt_unmatched_a"
        ]
        > 0
    ):

        print(
            f"Unmatched rejection rate    : "
            f"{evaluation['unmatched_rejection_rate']:.3f}"
        )

    else:

        print(
            "Unmatched rejection rate    : "
            "N/A"
        )

    # --------------------------------------------------------
    # Diagnostic errors
    # --------------------------------------------------------

    if evaluation[
        "false_positive_pairs"
    ]:

        print()
        print(
            "False-positive candidates:"
        )

        for pair in evaluation[
            "false_positive_pairs"
        ]:

            print(
                f"  {pair[0]} <-> {pair[1]}"
            )

    if evaluation[
        "false_negative_pairs"
    ]:

        print()
        print(
            "Missed GT correspondences:"
        )

        for pair in evaluation[
            "false_negative_pairs"
        ]:

            print(
                f"  {pair[0]} <-> {pair[1]}"
            )

    print()
    print("=" * 70)
    print(
        "Hard-threshold similarity test complete."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()