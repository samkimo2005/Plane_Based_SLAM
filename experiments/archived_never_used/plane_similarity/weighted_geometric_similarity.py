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
# 1. Weighted similarity parameters
# ============================================================

@dataclass
class WeightedSimilarityParameters:
    """
    Initial parameters for the normalised weighted
    geometric similarity method.

    These are development starting values.

    They will later be tuned using ONLY the DEV dataset
    and then frozen before the FINAL benchmark.
    """

    # --------------------------------------------------------
    # Main feature weights
    # --------------------------------------------------------

    # Plane normal is the most important term.
    normal_weight: float = 0.50

    # Plane offset is the second most important term.
    offset_weight: float = 0.40

    # Centroid + area form a weaker spatial term.
    spatial_weight: float = 0.10

    # --------------------------------------------------------
    # Internal spatial weights
    # --------------------------------------------------------

    centroid_weight: float = 0.50
    area_weight: float = 0.50

    # --------------------------------------------------------
    # Normalisation scales
    # --------------------------------------------------------
    #
    # Example:
    #
    #   normal difference = 5 degrees
    #   normal scale      = 10 degrees
    #
    # gives:
    #
    #   normalised normal difference = 0.5
    #
    # These values will later be tuned on DEV data.
    # --------------------------------------------------------

    normal_scale_deg: float = 10.0

    offset_scale_m: float = 0.15

    centroid_scale_m: float = 1.00

    area_scale_ratio: float = 0.50

    # --------------------------------------------------------
    # Candidate acceptance threshold
    # --------------------------------------------------------

    # Lower cost = more similar.
    #
    # A pair becomes a candidate if:
    #
    #     cost <= max_cost
    #
    max_cost: float = 1.00


# ============================================================
# 2. Normalisation
# ============================================================

def normalise_feature(
    value: float,
    scale: float,
) -> float:
    """
    Convert a physical difference into a dimensionless value.

    Values are NOT clipped to 1.

    Therefore a very poor match can produce a value > 1.
    """

    if scale <= 0.0:
        raise ValueError(
            "Normalisation scale must be greater than zero."
        )

    return float(
        value / scale
    )


# ============================================================
# 3. Weighted cost for one plane pair
# ============================================================

def calculate_weighted_similarity(
    plane_a: dict,
    plane_b_in_a: dict,
    params: WeightedSimilarityParameters,
) -> dict:
    """
    Calculate similarity between:

        one plane from Frame A

    and

        one Frame-B plane already transformed into Frame A.

    Lower cost means a better geometric match.
    """

    # --------------------------------------------------------
    # Raw geometric differences
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
    # Normalise features
    # --------------------------------------------------------

    normal_norm = normalise_feature(
        normal_angle_deg,
        params.normal_scale_deg,
    )

    offset_norm = normalise_feature(
        offset_difference_m,
        params.offset_scale_m,
    )

    centroid_norm = normalise_feature(
        centroid_distance_m,
        params.centroid_scale_m,
    )

    area_norm = normalise_feature(
        area_difference_ratio,
        params.area_scale_ratio,
    )

    # --------------------------------------------------------
    # Spatial term
    # --------------------------------------------------------

    spatial_term = (
        params.centroid_weight
        * centroid_norm
        +
        params.area_weight
        * area_norm
    )

    # --------------------------------------------------------
    # Final weighted cost
    # --------------------------------------------------------
    #
    # C =
    #     alpha * normal
    #   + beta  * offset
    #   + gamma * spatial
    #
    # Initial:
    #
    # alpha = 0.50
    # beta  = 0.40
    # gamma = 0.10
    # --------------------------------------------------------

    cost = (
        params.normal_weight
        * normal_norm
        +
        params.offset_weight
        * offset_norm
        +
        params.spatial_weight
        * spatial_term
    )

    # --------------------------------------------------------
    # Candidate decision
    # --------------------------------------------------------

    is_candidate = (
        cost <= params.max_cost
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

        # Normalised geometry
        "normal_normalised":
            float(normal_norm),

        "offset_normalised":
            float(offset_norm),

        "centroid_normalised":
            float(centroid_norm),

        "area_normalised":
            float(area_norm),

        # Combined quantities
        "spatial_term":
            float(spatial_term),

        "cost":
            float(cost),

        "is_candidate":
            bool(is_candidate),
    }


# ============================================================
# 4. Compare every possible A-B plane pair
# ============================================================

def compare_all_pairs(
    frame_a_planes: list,
    frame_b_planes: list,
    prior_transform: dict,
    params: WeightedSimilarityParameters,
) -> list:
    """
    Compare every Frame-A plane against every Frame-B plane.

    IMPORTANT:

    This is similarity comparison only.

    It is completely valid for:

        A0 -> B1

    and

        A0 -> B2

    to both be candidates.

    One-to-one matching belongs to the later
    Assignment experiment.
    """

    # --------------------------------------------------------
    # Transform B planes into A using the pose prior
    # --------------------------------------------------------

    frame_b_in_a = transform_frame_b_to_a(
        frame_b_planes,
        prior_transform,
    )

    results = []

    # --------------------------------------------------------
    # Test every possible A-B pair
    # --------------------------------------------------------

    for plane_a in frame_a_planes:

        for plane_b_original, plane_b_transformed in zip(
            frame_b_planes,
            frame_b_in_a,
        ):

            similarity = calculate_weighted_similarity(
                plane_a,
                plane_b_transformed,
                params,
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
# 5. Convert results into a cost matrix
# ============================================================

def build_cost_matrix(
    frame_a_planes: list,
    frame_b_planes: list,
    pair_results: list,
):
    """
    Build a matrix:

                    B0       B1       B2
        A0        cost     cost     cost
        A1        cost     cost     cost
        A2        cost     cost     cost

    This will be useful later for the Assignment experiment.

    No assignment is performed here.
    """

    a_ids = [
        plane["obs_id"]
        for plane in frame_a_planes
    ]

    b_ids = [
        plane["obs_id"]
        for plane in frame_b_planes
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

        matrix[i, j] = row[
            "cost"
        ]

    return matrix, a_ids, b_ids


# ============================================================
# 6. Extract candidate pairs
# ============================================================

def get_candidate_pairs(
    pair_results: list,
) -> list:
    """
    Return only plane pairs whose weighted cost
    passes the candidate threshold.
    """

    return [
        row
        for row in pair_results
        if row["is_candidate"]
    ]


# ============================================================
# 7. Run the method on one benchmark case
# ============================================================

def run_weighted_similarity_case(
    case_data: dict,
    params: WeightedSimilarityParameters,
) -> dict:
    """
    Run Weighted Geometric Similarity on one complete
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

    cost_matrix, a_ids, b_ids = (
        build_cost_matrix(
            frame_a_planes,
            frame_b_planes,
            pair_results,
        )
    )

    return {
        "sample_id":
            case_data["sample_id"],

        "scene_type":
            case_data["scene_type"],

        "prior_level":
            case_data["prior_level"],

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
# 8. Simple standalone test
# ============================================================

def main():
    """
    Test the method using the first DEV benchmark JSON.

    This does NOT evaluate against GT yet.

    It simply checks that:

        1. the JSON can be loaded
        2. Frame B can be transformed to Frame A
        3. all A-B pair costs can be calculated
        4. candidate pairs are produced
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
        dev_dir.glob("*.json")
    )

    if not json_files:
        raise FileNotFoundError(
            "No DEV benchmark JSON files were found.\n"
            "Run generate_similarity_benchmark.py first."
        )

    test_file = json_files[0]

    print("=" * 70)
    print("Weighted Geometric Similarity Test")
    print("=" * 70)

    print(
        f"Input file: {test_file.name}"
    )

    case_data = load_similarity_case(
        test_file
    )

    params = (
        WeightedSimilarityParameters()
    )

    result = (
        run_weighted_similarity_case(
            case_data,
            params,
        )
    )

    evaluation = evaluate_similarity_candidates(
        case_data,
        result["candidate_pairs"],
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
    # Print cost matrix
    # --------------------------------------------------------

    print()
    print("Cost matrix:")
    print()

    print(
        " " * 10,
        end="",
    )

    for b_id in result["b_ids"]:

        print(
            f"{b_id:>10}",
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
                f"{result['cost_matrix'][i, j]:10.3f}",
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
            "No candidate pairs passed the threshold."
        )

    else:

        sorted_candidates = sorted(
            result[
                "candidate_pairs"
            ],
            key=lambda x: x["cost"],
        )

        for candidate in sorted_candidates:

            print(
                f"{candidate['a_obs_id']} "
                f"<-> "
                f"{candidate['b_obs_id']} "
                f"| cost = "
                f"{candidate['cost']:.3f} "
                f"| angle = "
                f"{candidate['normal_angle_deg']:.2f} deg "
                f"| offset = "
                f"{candidate['offset_difference_m']:.3f} m "
                f"| centroid = "
                f"{candidate['centroid_distance_m']:.3f} m "
                f"| area = "
                f"{candidate['area_difference_ratio']:.3f}"
            )

    print()
    print("=" * 70)
    print(
        "Weighted similarity test complete."
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Similarity evaluation
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

    print(
        f"Unmatched rejection rate    : "
        f"{evaluation['unmatched_rejection_rate']:.3f}"
    )

    print()
    print("=" * 70)
    print(
        "Weighted similarity test complete."
    )
    print("=" * 70)

if __name__ == "__main__":
    main()
