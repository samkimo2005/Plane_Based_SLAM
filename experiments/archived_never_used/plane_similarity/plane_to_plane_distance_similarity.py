from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from similarity_common import (
    align_plane_sign,
    calculate_pair_geometry,
    load_similarity_case,
    normalize_vector,
    transform_frame_b_to_a,
)

from similarity_evaluation import (
    evaluate_similarity_candidates,
)


# ============================================================
# 1. Plane-to-plane distance parameters
# ============================================================

@dataclass
class PlaneToPlaneDistanceParameters:
    """
    Parameters for the plane-to-plane distance similarity method.

    The pairwise distance follows the plane-to-plane distance
    formulation introduced by Proenca and Gao (2017):

        D_pi = || d_b * n_b - d_a * n_a ||

    for planes written as:

        n^T p + d = 0,  ||n|| = 1

    Lower distance means greater geometric similarity.

    max_distance_m is a project-specific candidate threshold.
    It must be calibrated using ONLY the DEV dataset and then
    frozen before the FINAL benchmark.
    """

    # Development starting value only.
    # Do not treat this as a literature-derived constant.
    max_distance_m: float = 0.30


# ============================================================
# 2. Plane-to-plane distance for one plane pair
# ============================================================

def calculate_plane_to_plane_similarity(
    plane_a: dict,
    plane_b_in_a: dict,
    params: PlaneToPlaneDistanceParameters,
) -> dict:
    """
    Calculate the plane-to-plane distance between one plane
    from Frame A and one Frame-B plane already transformed
    into Frame A.

    For the Hesse-form plane:

        n^T p + d = 0

    the closest point on the plane to the coordinate origin is:

        C = -d n

    Therefore:

        D_pi = ||C_a - C_b||
             = ||d_b n_b - d_a n_a||

    Lower D_pi means a more similar plane pair.
    """

    # --------------------------------------------------------
    # Common diagnostic geometry
    # --------------------------------------------------------

    geometry = calculate_pair_geometry(
        plane_a,
        plane_b_in_a,
    )

    # --------------------------------------------------------
    # Resolve plane-normal sign ambiguity
    # --------------------------------------------------------

    plane_b_aligned = align_plane_sign(
        plane_a,
        plane_b_in_a,
    )

    normal_a = normalize_vector(
        np.asarray(
            plane_a["normal"],
            dtype=float,
        )
    )

    normal_b = normalize_vector(
        np.asarray(
            plane_b_aligned["normal"],
            dtype=float,
        )
    )

    d_a = float(
        plane_a["offset"]
    )

    d_b = float(
        plane_b_aligned["offset"]
    )

    # --------------------------------------------------------
    # Closest points to the origin
    #
    # C = -d n
    # --------------------------------------------------------

    closest_point_a = (
        -d_a * normal_a
    )

    closest_point_b = (
        -d_b * normal_b
    )

    # --------------------------------------------------------
    # Plane-to-plane distance
    #
    # Equivalent to:
    #
    # || d_b n_b - d_a n_a ||
    # --------------------------------------------------------

    plane_distance_m = float(
        np.linalg.norm(
            closest_point_a
            - closest_point_b
        )
    )

    # --------------------------------------------------------
    # Candidate decision
    # --------------------------------------------------------

    is_candidate = (
        plane_distance_m
        <= params.max_distance_m
    )

    return {
        # Common diagnostics retained so the evaluation /
        # analysis format stays compatible with the other
        # similarity methods.
        "normal_angle_deg":
            float(
                geometry[
                    "normal_angle_deg"
                ]
            ),

        "offset_difference_m":
            float(
                geometry[
                    "offset_difference_m"
                ]
            ),

        "centroid_distance_m":
            float(
                geometry[
                    "centroid_distance_m"
                ]
            ),

        "area_difference_ratio":
            float(
                geometry[
                    "area_difference_ratio"
                ]
            ),

        # Main method output
        "plane_to_plane_distance_m":
            plane_distance_m,

        # Keep the generic key "cost" so that the later
        # Assignment experiment can consume the same matrix
        # interface as before.
        "cost":
            plane_distance_m,

        "is_candidate":
            bool(
                is_candidate
            ),
    }


# ============================================================
# 3. Compare every possible A-B plane pair
# ============================================================

def compare_all_pairs(
    frame_a_planes: list,
    frame_b_planes: list,
    prior_transform: dict,
    params: PlaneToPlaneDistanceParameters,
) -> list:
    """
    Compare every Frame-A plane against every Frame-B plane.

    IMPORTANT:

    This stage performs pairwise similarity comparison only.

    More than one Frame-B plane may remain a candidate for the
    same Frame-A plane. One-to-one selection belongs to the
    later Assignment of Matching experiment.
    """

    frame_b_in_a = transform_frame_b_to_a(
        frame_b_planes,
        prior_transform,
    )

    results = []

    for plane_a in frame_a_planes:

        for (
            plane_b_original,
            plane_b_transformed,
        ) in zip(
            frame_b_planes,
            frame_b_in_a,
        ):

            similarity = (
                calculate_plane_to_plane_similarity(
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
# 4. Convert results into a cost matrix
# ============================================================

def build_cost_matrix(
    frame_a_planes: list,
    frame_b_planes: list,
    pair_results: list,
):
    """
    Build a pairwise cost matrix.

    No assignment is performed here.
    """

    a_ids = [
        plane["obs_id"]
        for plane
        in frame_a_planes
    ]

    b_ids = [
        plane["obs_id"]
        for plane
        in frame_b_planes
    ]

    matrix = np.full(
        (
            len(a_ids),
            len(b_ids),
        ),
        np.inf,
        dtype=float,
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

        matrix[i, j] = float(
            row["cost"]
        )

    return (
        matrix,
        a_ids,
        b_ids,
    )


# ============================================================
# 5. Extract candidate pairs
# ============================================================

def get_candidate_pairs(
    pair_results: list,
) -> list:
    """
    Return only plane pairs whose plane-to-plane distance
    passes the DEV-calibrated candidate threshold.
    """

    return [
        row
        for row
        in pair_results
        if row[
            "is_candidate"
        ]
    ]


# ============================================================
# 6. Run the method on one benchmark case
# ============================================================

def run_plane_to_plane_distance_case(
    case_data: dict,
    params: PlaneToPlaneDistanceParameters,
) -> dict:
    """
    Run Plane-to-Plane Distance Similarity on one complete
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

    candidates = get_candidate_pairs(
        pair_results
    )

    (
        cost_matrix,
        a_ids,
        b_ids,
    ) = build_cost_matrix(
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
            candidates,

        "cost_matrix":
            cost_matrix,

        "a_ids":
            a_ids,

        "b_ids":
            b_ids,
    }


# ============================================================
# 7. Simple standalone DEV test
# ============================================================

def main():
    """
    Run the first DEV benchmark case as a basic verification.

    This verifies that:
        1. the benchmark JSON loads correctly
        2. Frame B is transformed into Frame A
        3. plane-to-plane distances are calculated
        4. candidate pairs are generated
        5. the existing similarity evaluator remains compatible
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

    test_file = (
        json_files[0]
    )

    print(
        "=" * 70
    )

    print(
        "Plane-to-Plane Distance Similarity Test"
    )

    print(
        "=" * 70
    )

    print(
        f"Input file: {test_file.name}"
    )

    case_data = load_similarity_case(
        test_file
    )

    params = (
        PlaneToPlaneDistanceParameters()
    )

    result = (
        run_plane_to_plane_distance_case(
            case_data,
            params,
        )
    )

    evaluation = (
        evaluate_similarity_candidates(
            case_data,
            result[
                "candidate_pairs"
            ],
        )
    )

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
    # Cost matrix
    # --------------------------------------------------------

    print()
    print(
        "Plane-to-plane distance matrix (m):"
    )

    print()

    print(
        " " * 10,
        end="",
    )

    for b_id in result[
        "b_ids"
    ]:

        print(
            f"{b_id:>10}",
            end="",
        )

    print()

    for i, a_id in enumerate(
        result[
            "a_ids"
        ]
    ):

        print(
            f"{a_id:<10}",
            end="",
        )

        for j in range(
            len(
                result[
                    "b_ids"
                ]
            )
        ):

            print(
                f"{result['cost_matrix'][i, j]:10.3f}",
                end="",
            )

        print()

    # --------------------------------------------------------
    # Candidate pairs
    # --------------------------------------------------------

    print()
    print(
        "Candidate pairs:"
    )

    print()

    if not result[
        "candidate_pairs"
    ]:

        print(
            "No candidate pairs passed the threshold."
        )

    else:

        sorted_candidates = sorted(
            result[
                "candidate_pairs"
            ],
            key=lambda x:
                x["cost"],
        )

        for candidate in sorted_candidates:

            print(
                f"{candidate['a_obs_id']} "
                f"<-> "
                f"{candidate['b_obs_id']} "
                f"| D_pi = "
                f"{candidate['plane_to_plane_distance_m']:.3f} m "
                f"| angle = "
                f"{candidate['normal_angle_deg']:.2f} deg "
                f"| offset = "
                f"{candidate['offset_difference_m']:.3f} m"
            )

    # --------------------------------------------------------
    # Existing similarity evaluation
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "Similarity Evaluation"
    )

    print(
        "=" * 70
    )

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

    unmatched_rejection_rate = evaluation.get(
        "unmatched_rejection_rate"
    )

    if (
        unmatched_rejection_rate
        is None
    ):
        print(
            "Unmatched rejection rate    : N/A"
        )

    else:
        print(
            f"Unmatched rejection rate    : "
            f"{unmatched_rejection_rate:.3f}"
        )

    print()
    print(
        "=" * 70
    )

    print(
        "Plane-to-plane distance test complete."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
