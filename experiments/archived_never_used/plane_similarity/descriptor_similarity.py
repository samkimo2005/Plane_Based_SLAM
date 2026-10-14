from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from similarity_common import (
    load_similarity_case,
    normalize_vector,
    safe_acos_deg,
)

from similarity_evaluation import (
    evaluate_similarity_candidates,
)


# ============================================================
# 1. Descriptor parameters
# ============================================================

@dataclass
class DescriptorSimilarityParameters:
    """
    Initial parameters for the project-specific
    hand-crafted geometric context descriptor.

    These are DEVELOPMENT starting values only.

    They will later be tuned using ONLY the DEV dataset
    and then frozen before the FINAL benchmark.
    """

    # --------------------------------------------------------
    # Descriptor component weights
    # --------------------------------------------------------

    # Relative visible area
    area_weight: float = 0.25

    # Plane shape / aspect ratio
    aspect_weight: float = 0.15

    # Distribution of relative normal angles
    normal_context_weight: float = 0.30

    # Distribution of relative centroid distances
    distance_context_weight: float = 0.30

    # --------------------------------------------------------
    # Normalisation scales for intrinsic features
    # --------------------------------------------------------

    # Applied to:
    #
    # |log(A1 / A2)|
    #
    area_log_scale: float = 0.50

    # Applied to:
    #
    # |log(r1 / r2)|
    #
    # where r is aspect ratio.
    aspect_log_scale: float = 0.40

    # --------------------------------------------------------
    # Context histogram bins
    # --------------------------------------------------------

    # Plane normal angle is sign invariant and therefore
    # ranges from 0 to 90 degrees.
    normal_angle_bins_deg: tuple = (
        0.0,
        15.0,
        45.0,
        75.0,
        90.000001,
    )

    # Relative centroid-distance bins in metres.
    centroid_distance_bins_m: tuple = (
        0.0,
        1.0,
        2.0,
        4.0,
        float("inf"),
    )

    # --------------------------------------------------------
    # Candidate threshold
    # --------------------------------------------------------

    # Lower descriptor distance = more similar.
    #
    # This value is only an initial DEV setting.
    max_descriptor_distance: float = 0.80


# ============================================================
# 2. Plane intrinsic features
# ============================================================

def calculate_aspect_ratio(
    plane: dict,
) -> float:
    """
    Aspect ratio is defined as:

        max(width, height)
        ------------------
        min(width, height)

    This avoids dependence on which side is called
    width or height.
    """

    width = max(
        float(
            plane["width_m"]
        ),
        1e-12,
    )

    height = max(
        float(
            plane["height_m"]
        ),
        1e-12,
    )

    return max(
        width,
        height,
    ) / min(
        width,
        height,
    )


# ============================================================
# 3. Sign-invariant normal angle
# ============================================================

def plane_normal_angle_deg(
    normal_1,
    normal_2,
) -> float:
    """
    Calculate the angle between two plane normals.

    Because:

        n

    and:

        -n

    represent the same plane orientation, absolute dot
    product is used.

    The output therefore lies between 0 and 90 degrees.
    """

    n1 = normalize_vector(
        np.asarray(
            normal_1,
            dtype=float,
        )
    )

    n2 = normalize_vector(
        np.asarray(
            normal_2,
            dtype=float,
        )
    )

    dot = abs(
        float(
            np.dot(
                n1,
                n2,
            )
        )
    )

    dot = np.clip(
        dot,
        0.0,
        1.0,
    )

    return safe_acos_deg(
        dot
    )


# ============================================================
# 4. Normal-angle context histogram
# ============================================================

def build_normal_context_histogram(
    target_plane: dict,
    all_planes: list,
    params: DescriptorSimilarityParameters,
) -> np.ndarray:
    """
    Describe how the target plane is oriented relative
    to the OTHER planes in the same frame.

    Example:

        target wall
             |
             |  relative angles to surrounding planes
             v

        [0 deg, 90 deg, 90 deg, ...]

    The resulting histogram is normalised so that its
    entries sum to 1.
    """

    angles = []

    target_id = target_plane[
        "obs_id"
    ]

    for other_plane in all_planes:

        if (
            other_plane["obs_id"]
            == target_id
        ):
            continue

        angle = plane_normal_angle_deg(
            target_plane["normal"],
            other_plane["normal"],
        )

        angles.append(
            angle
        )

    num_bins = (
        len(
            params.normal_angle_bins_deg
        )
        - 1
    )

    if not angles:

        return np.zeros(
            num_bins,
            dtype=float,
        )

    histogram, _ = np.histogram(
        angles,
        bins=np.asarray(
            params.normal_angle_bins_deg,
            dtype=float,
        ),
    )

    histogram = histogram.astype(
        float
    )

    histogram /= histogram.sum()

    return histogram


# ============================================================
# 5. Centroid-distance context histogram
# ============================================================

def build_distance_context_histogram(
    target_plane: dict,
    all_planes: list,
    params: DescriptorSimilarityParameters,
) -> np.ndarray:
    """
    Describe the spatial relationship between one plane
    and the other planes in the SAME frame.

    Relative centroid distances are invariant to a global
    rigid camera transformation:

        ||c_i - c_j||

    remains unchanged after applying the same rigid
    transformation to both centroids.
    """

    target_centroid = np.asarray(
        target_plane["centroid"],
        dtype=float,
    )

    target_id = target_plane[
        "obs_id"
    ]

    distances = []

    for other_plane in all_planes:

        if (
            other_plane["obs_id"]
            == target_id
        ):
            continue

        other_centroid = np.asarray(
            other_plane["centroid"],
            dtype=float,
        )

        distance = float(
            np.linalg.norm(
                target_centroid
                - other_centroid
            )
        )

        distances.append(
            distance
        )

    num_bins = (
        len(
            params.centroid_distance_bins_m
        )
        - 1
    )

    if not distances:

        return np.zeros(
            num_bins,
            dtype=float,
        )

    histogram, _ = np.histogram(
        distances,
        bins=np.asarray(
            params.centroid_distance_bins_m,
            dtype=float,
        ),
    )

    histogram = histogram.astype(
        float
    )

    histogram /= histogram.sum()

    return histogram


# ============================================================
# 6. Build descriptor for one plane
# ============================================================

def build_plane_descriptor(
    target_plane: dict,
    all_planes: list,
    params: DescriptorSimilarityParameters,
) -> dict:
    """
    Build a project-specific plane descriptor.

    Descriptor information:

        1. visible plane area
        2. aspect ratio
        3. relative normal-angle context
        4. relative centroid-distance context

    No pose prior is used here.
    """

    area_m2 = max(
        float(
            target_plane["area_m2"]
        ),
        1e-12,
    )

    aspect_ratio = (
        calculate_aspect_ratio(
            target_plane
        )
    )

    normal_histogram = (
        build_normal_context_histogram(
            target_plane,
            all_planes,
            params,
        )
    )

    distance_histogram = (
        build_distance_context_histogram(
            target_plane,
            all_planes,
            params,
        )
    )

    return {
        "obs_id":
            target_plane["obs_id"],

        "area_m2":
            area_m2,

        "aspect_ratio":
            aspect_ratio,

        "normal_context":
            normal_histogram,

        "distance_context":
            distance_histogram,
    }


# ============================================================
# 7. Build descriptors for one complete frame
# ============================================================

def build_frame_descriptors(
    planes: list,
    params: DescriptorSimilarityParameters,
) -> list:
    """
    Build one descriptor for every plane in a frame.
    """

    descriptors = []

    for plane in planes:

        descriptor = (
            build_plane_descriptor(
                plane,
                planes,
                params,
            )
        )

        descriptors.append(
            descriptor
        )

    return descriptors


# ============================================================
# 8. Histogram difference
# ============================================================

def histogram_distance(
    histogram_a: np.ndarray,
    histogram_b: np.ndarray,
) -> float:
    """
    Use total-variation / half-L1 distance:

        0.5 * sum(|Ha - Hb|)

    For normalised histograms:

        0 = identical
        1 = completely different
    """

    histogram_a = np.asarray(
        histogram_a,
        dtype=float,
    )

    histogram_b = np.asarray(
        histogram_b,
        dtype=float,
    )

    return float(
        0.5
        * np.sum(
            np.abs(
                histogram_a
                - histogram_b
            )
        )
    )


# ============================================================
# 9. Compare two descriptors
# ============================================================

def calculate_descriptor_similarity(
    descriptor_a: dict,
    descriptor_b: dict,
    params: DescriptorSimilarityParameters,
) -> dict:
    """
    Compare two plane descriptors.

    Lower descriptor distance = more similar.
    """

    area_a = max(
        float(
            descriptor_a["area_m2"]
        ),
        1e-12,
    )

    area_b = max(
        float(
            descriptor_b["area_m2"]
        ),
        1e-12,
    )

    aspect_a = max(
        float(
            descriptor_a["aspect_ratio"]
        ),
        1e-12,
    )

    aspect_b = max(
        float(
            descriptor_b["aspect_ratio"]
        ),
        1e-12,
    )

    # --------------------------------------------------------
    # Area difference
    # --------------------------------------------------------
    #
    # Log ratio makes the comparison symmetric.
    #
    # Example:
    #
    # A1 = 2 * A2
    #
    # and
    #
    # A2 = 0.5 * A1
    #
    # produce the same magnitude.
    # --------------------------------------------------------

    area_log_difference = abs(
        np.log(
            area_a
            / area_b
        )
    )

    area_difference_normalised = (
        area_log_difference
        / params.area_log_scale
    )

    # --------------------------------------------------------
    # Aspect-ratio difference
    # --------------------------------------------------------

    aspect_log_difference = abs(
        np.log(
            aspect_a
            / aspect_b
        )
    )

    aspect_difference_normalised = (
        aspect_log_difference
        / params.aspect_log_scale
    )

    # --------------------------------------------------------
    # Structural context
    # --------------------------------------------------------

    normal_context_difference = (
        histogram_distance(
            descriptor_a[
                "normal_context"
            ],
            descriptor_b[
                "normal_context"
            ],
        )
    )

    distance_context_difference = (
        histogram_distance(
            descriptor_a[
                "distance_context"
            ],
            descriptor_b[
                "distance_context"
            ],
        )
    )

    # --------------------------------------------------------
    # Combined descriptor distance
    # --------------------------------------------------------

    descriptor_distance = (
        params.area_weight
        * area_difference_normalised

        + params.aspect_weight
        * aspect_difference_normalised

        + params.normal_context_weight
        * normal_context_difference

        + params.distance_context_weight
        * distance_context_difference
    )

    is_candidate = (
        descriptor_distance
        <= params.max_descriptor_distance
    )

    return {
        "area_log_difference":
            float(
                area_log_difference
            ),

        "area_difference_normalised":
            float(
                area_difference_normalised
            ),

        "aspect_log_difference":
            float(
                aspect_log_difference
            ),

        "aspect_difference_normalised":
            float(
                aspect_difference_normalised
            ),

        "normal_context_difference":
            float(
                normal_context_difference
            ),

        "distance_context_difference":
            float(
                distance_context_difference
            ),

        "descriptor_distance":
            float(
                descriptor_distance
            ),

        "is_candidate":
            bool(
                is_candidate
            ),
    }


# ============================================================
# 10. Compare every possible A-B descriptor pair
# ============================================================

def compare_all_pairs(
    frame_a_planes: list,
    frame_b_planes: list,
    params: DescriptorSimilarityParameters,
) -> list:
    """
    Build descriptors independently in Frame A and Frame B,
    then compare every possible descriptor pair.

    IMPORTANT:

    Unlike Weighted Geometric Cost and Hard Thresholds,
    this method does NOT use:

        prior_transform_b_to_a

    because the descriptor is based mainly on intrinsic and
    relative geometric information.
    """

    descriptors_a = (
        build_frame_descriptors(
            frame_a_planes,
            params,
        )
    )

    descriptors_b = (
        build_frame_descriptors(
            frame_b_planes,
            params,
        )
    )

    results = []

    for descriptor_a in descriptors_a:

        for descriptor_b in descriptors_b:

            similarity = (
                calculate_descriptor_similarity(
                    descriptor_a,
                    descriptor_b,
                    params,
                )
            )

            results.append(
                {
                    "a_obs_id":
                        descriptor_a[
                            "obs_id"
                        ],

                    "b_obs_id":
                        descriptor_b[
                            "obs_id"
                        ],

                    **similarity,
                }
            )

    return results


# ============================================================
# 11. Extract candidate pairs
# ============================================================

def get_candidate_pairs(
    pair_results: list,
) -> list:
    """
    Keep only pairs whose descriptor distance passes
    the candidate threshold.
    """

    return [
        row
        for row in pair_results
        if row[
            "is_candidate"
        ]
    ]


# ============================================================
# 12. Build descriptor-distance matrix
# ============================================================

def build_distance_matrix(
    frame_a_planes: list,
    frame_b_planes: list,
    pair_results: list,
):
    """
    Build:

                    B0       B1       B2
        A0        dist     dist     dist
        A1        dist     dist     dist
        A2        dist     dist     dist

    This is still NOT assignment.
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

        matrix[i, j] = (
            row[
                "descriptor_distance"
            ]
        )

    return matrix, a_ids, b_ids


# ============================================================
# 13. Run one benchmark case
# ============================================================

def run_descriptor_similarity_case(
    case_data: dict,
    params: DescriptorSimilarityParameters,
) -> dict:
    """
    Run descriptor-based similarity on one synthetic
    benchmark case.
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

    pair_results = compare_all_pairs(
        frame_a_planes,
        frame_b_planes,
        params,
    )

    candidate_pairs = (
        get_candidate_pairs(
            pair_results
        )
    )

    (
        distance_matrix,
        a_ids,
        b_ids,
    ) = build_distance_matrix(
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

        "distance_matrix":
            distance_matrix,

        "a_ids":
            a_ids,

        "b_ids":
            b_ids,
    }


# ============================================================
# 14. Standalone test
# ============================================================

def main():
    """
    Test descriptor-based similarity using the first
    DEV benchmark case.

    The test:

        1. loads one DEV benchmark
        2. builds plane descriptors in A and B independently
        3. compares every A-B descriptor pair
        4. extracts candidate pairs
        5. evaluates candidates against common GT
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
    print("Descriptor-Based Plane Similarity Test")
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
    # Initial descriptor parameters
    # --------------------------------------------------------

    params = (
        DescriptorSimilarityParameters()
    )

    # --------------------------------------------------------
    # Run method
    # --------------------------------------------------------

    result = (
        run_descriptor_similarity_case(
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

    print()

    print(
        "Pose prior used by descriptor method: NO"
    )

    # --------------------------------------------------------
    # Descriptor-distance matrix
    # --------------------------------------------------------

    print()
    print("Descriptor-distance matrix:")
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
                f"{result['distance_matrix'][i, j]:10.3f}",
                end="",
            )

        print()

    # --------------------------------------------------------
    # Candidate pairs
    # --------------------------------------------------------

    print()
    print("Candidate pairs:")
    print()

    if not result[
        "candidate_pairs"
    ]:

        print(
            "No candidate pairs passed the "
            "descriptor threshold."
        )

    else:

        sorted_candidates = sorted(
            result[
                "candidate_pairs"
            ],
            key=lambda x: x[
                "descriptor_distance"
            ],
        )

        for candidate in sorted_candidates:

            print(
                f"{candidate['a_obs_id']} "
                f"<-> "
                f"{candidate['b_obs_id']} "
                f"| distance = "
                f"{candidate['descriptor_distance']:.3f} "
                f"| area = "
                f"{candidate['area_difference_normalised']:.3f} "
                f"| aspect = "
                f"{candidate['aspect_difference_normalised']:.3f} "
                f"| normal context = "
                f"{candidate['normal_context_difference']:.3f} "
                f"| distance context = "
                f"{candidate['distance_context_difference']:.3f} "
                f"| candidate = "
                f"{candidate['is_candidate']}"
            )

    # --------------------------------------------------------
    # Common evaluation
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
        "Descriptor similarity test complete."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()