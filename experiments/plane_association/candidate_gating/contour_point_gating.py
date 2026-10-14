"""
contour_point_gating.py

Candidate Gating Method [28]

Based on:

X. Zhang, W. Wang, X. Qi, Z. Liao and R. Wei,
"Point-Plane SLAM Using Supposed Planes for Indoor Environments",
Sensors, 2019.

Candidate gating component:

For an observed plane contour C_m and a candidate map plane:

    s = min |n^T p_j + d|

        p_j in C_m

The paper uses:

    s < 0.1 m

to identify intersecting planes.

The angle between the two plane normals is then checked.
The paper reports an angle threshold of:

    30 degrees

for finding corresponding planes.

IMPORTANT
---------
The original paper subsequently selects the corresponding plane
with the smallest s.

That match-selection step is deliberately NOT implemented here,
because this project separates Plane Association into:

    Candidate Gating
        ->
    Pairwise Similarity / Cost
        ->
    Assignment

Therefore this file outputs every pair satisfying the contour
distance and angle gates.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from candidate_gating_common import (
    DEFAULT_SAMPLE,
    CandidateSample,
    PlaneSet,
    all_candidate_pairs,
    contour_to_plane_min_distance,
    load_candidate_sample,
    normal_angle_deg,
    transform_plane_set,
)


# ============================================================
# Default parameters from [28]
# ============================================================

DEFAULT_PARAMETERS = {

    # Equation (7) / experimental threshold in [28].
    "contour_distance_threshold_m": 0.10,

    # Correspondence angle threshold reported in [28].
    "normal_angle_threshold_deg": 30.0,
}


# ============================================================
# Candidate Gating algorithm
# ============================================================

def gate_candidates(
    source_planes: PlaneSet,
    target_planes: PlaneSet,
    prior_R: np.ndarray,
    prior_t: np.ndarray,
    parameters: Optional[Dict] = None,
) -> List[Dict]:
    """
    Apply contour-point Candidate Gating.

    Parameters
    ----------
    source_planes:
        Plane observations in the source frame.

    target_planes:
        Plane observations in the target frame.

    prior_R, prior_t:
        Estimated source-to-target transformation.

        x_target = prior_R @ x_source + prior_t

    parameters:
        Dictionary containing:

            contour_distance_threshold_m
            normal_angle_threshold_deg

    Returns
    -------
    candidate_pairs:
        All source-target plane pairs that pass:

            1. contour-to-plane distance gate
            2. plane-normal angle gate
    """

    if parameters is None:

        parameters = (
            DEFAULT_PARAMETERS.copy()
        )

    contour_threshold = float(
        parameters[
            "contour_distance_threshold_m"
        ]
    )

    angle_threshold = float(
        parameters[
            "normal_angle_threshold_deg"
        ]
    )

    # --------------------------------------------------------
    # Transform source observations into target frame
    #
    # The contour points and plane normals must be expressed
    # in the same coordinate frame as the target planes.
    # --------------------------------------------------------

    source_in_target = (
        transform_plane_set(
            planes=source_planes,
            R=prior_R,
            t=prior_t,
        )
    )

    candidate_pairs = []

    # --------------------------------------------------------
    # Test every possible source-target pair
    # --------------------------------------------------------

    for (
        source_idx,
        target_idx,
    ) in all_candidate_pairs(
        len(source_in_target),
        len(target_planes),
    ):

        source_normal = (
            source_in_target
            .normals[source_idx]
        )

        source_contour = (
            source_in_target
            .contours[source_idx]
        )

        target_normal = (
            target_planes
            .normals[target_idx]
        )

        target_offset = float(
            target_planes
            .offsets[target_idx]
        )

        # ----------------------------------------------------
        # STEP 1
        #
        # Contour-point distance test from [28]:
        #
        #     s = min |n_i^T p_j + d_i|
        #
        # where p_j belongs to the observed plane contour.
        #
        # In this implementation:
        #
        #     source contour = observed plane
        #     target plane   = candidate landmark
        # ----------------------------------------------------

        contour_distance = (
            contour_to_plane_min_distance(
                contour=source_contour,
                plane_normal=target_normal,
                plane_offset=target_offset,
            )
        )

        contour_pass = (
            contour_distance
            < contour_threshold
        )

        # ----------------------------------------------------
        # STEP 2
        #
        # Normal-angle compatibility.
        #
        # common.py handles plane-normal sign ambiguity.
        # ----------------------------------------------------

        angle_difference = (
            normal_angle_deg(
                source_normal,
                target_normal,
            )
        )

        angle_pass = (
            angle_difference
            < angle_threshold
        )

        # ----------------------------------------------------
        # Both gates must pass.
        # ----------------------------------------------------

        passed_gate = (
            contour_pass
            and angle_pass
        )

        if not passed_gate:
            continue

        # ----------------------------------------------------
        # Additional quantities are stored for later analysis.
        # They DO NOT affect the gate itself.
        # ----------------------------------------------------

        centroid_distance = float(
            np.linalg.norm(

                source_in_target
                .centroids[source_idx]

                -

                target_planes
                .centroids[target_idx]
            )
        )

        candidate_pairs.append(
            {

                "source_plane_id":
                    int(source_idx),

                "target_plane_id":
                    int(target_idx),

                "contour_min_distance_m":
                    float(
                        contour_distance
                    ),

                "normal_angle_deg":
                    float(
                        angle_difference
                    ),

                "centroid_distance_m":
                    centroid_distance,

                "passed_contour_gate":
                    bool(
                        contour_pass
                    ),

                "passed_angle_gate":
                    bool(
                        angle_pass
                    ),

                "passed_gate":
                    True,

                "algorithm":
                    "contour_point_gating",
            }
        )

    return candidate_pairs


# ============================================================
# Convenient wrapper
# ============================================================

def run_on_sample(
    sample: CandidateSample,
    parameters: Optional[Dict] = None,
) -> List[Dict]:
    """
    Run contour-point gating directly on one benchmark sample.
    """

    return gate_candidates(

        source_planes=sample.source,

        target_planes=sample.target,

        prior_R=sample.prior_R,

        prior_t=sample.prior_t,

        parameters=parameters,
    )


# ============================================================
# Temporary sanity evaluation
# ============================================================

def calculate_sanity_metrics(
    sample: CandidateSample,
    candidates: List[Dict],
) -> Dict:
    """
    Temporary development check.

    Formal evaluation will later be moved to:

        candidate_gating_evaluation.py
    """

    gt_pairs = {

        (
            int(source_idx),
            int(target_idx),
        )

        for (
            source_idx,
            target_idx
        ) in sample.gt_pairs
    }

    candidate_pairs = {

        (
            candidate[
                "source_plane_id"
            ],

            candidate[
                "target_plane_id"
            ],
        )

        for candidate
        in candidates
    }

    retained_gt = (
        gt_pairs
        & candidate_pairs
    )

    number_possible_pairs = (
        len(sample.source)
        * len(sample.target)
    )

    number_candidates = len(
        candidate_pairs
    )

    if len(gt_pairs) > 0:

        retention_rate = (
            len(retained_gt)
            / len(gt_pairs)
        )

    else:

        retention_rate = 0.0

    if number_possible_pairs > 0:

        reduction_rate = (
            1.0
            - (
                number_candidates
                / number_possible_pairs
            )
        )

    else:

        reduction_rate = 0.0

    return {

        "gt_pairs":
            len(gt_pairs),

        "gt_pairs_retained":
            len(retained_gt),

        "possible_pairs":
            number_possible_pairs,

        "candidate_pairs":
            number_candidates,

        "retention_rate":
            retention_rate,

        "reduction_rate":
            reduction_rate,
    }


# ============================================================
# Direct-run test
# ============================================================

def main() -> None:

    print()
    print("=" * 70)

    print(
        "Contour-Point Candidate Gating [28]"
    )

    print("=" * 70)

    print(
        f"Input sample : {DEFAULT_SAMPLE}"
    )

    print()

    # --------------------------------------------------------
    # Load default DEV sample
    # --------------------------------------------------------

    sample = (
        load_candidate_sample(
            DEFAULT_SAMPLE
        )
    )

    # --------------------------------------------------------
    # Run algorithm
    # --------------------------------------------------------

    candidates = run_on_sample(
        sample
    )

    number_before = (
        len(sample.source)
        * len(sample.target)
    )

    number_after = len(
        candidates
    )

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print(
        f"Source planes       : "
        f"{len(sample.source)}"
    )

    print(
        f"Target planes       : "
        f"{len(sample.target)}"
    )

    print(
        f"Possible pairs      : "
        f"{number_before}"
    )

    print(
        f"Surviving pairs     : "
        f"{number_after}"
    )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    print()

    print(
        "Parameters:"
    )

    print(
        "  Contour distance threshold : "
        f"{DEFAULT_PARAMETERS['contour_distance_threshold_m']:.3f} m"
    )

    print(
        "  Normal angle threshold     : "
        f"{DEFAULT_PARAMETERS['normal_angle_threshold_deg']:.1f} deg"
    )

    # --------------------------------------------------------
    # Print candidates
    # --------------------------------------------------------

    print()

    print(
        "Surviving candidate pairs:"
    )

    print(
        "-" * 70
    )

    if len(candidates) == 0:

        print(
            "No candidate pairs survived."
        )

    else:

        for candidate in candidates:

            print(

                f"S"
                f"{candidate['source_plane_id']}"

                f" -> "

                f"T"
                f"{candidate['target_plane_id']}"

                f" | contour="
                f"{candidate['contour_min_distance_m']:.4f} m"

                f" | angle="
                f"{candidate['normal_angle_deg']:.3f} deg"

                f" | centroid="
                f"{candidate['centroid_distance_m']:.4f} m"
            )

    print(
        "-" * 70
    )

    # --------------------------------------------------------
    # Temporary GT sanity check
    # --------------------------------------------------------

    metrics = (
        calculate_sanity_metrics(
            sample=sample,
            candidates=candidates,
        )
    )

    print()

    print(
        "Temporary GT sanity check:"
    )

    print(
        f"GT pairs             : "
        f"{metrics['gt_pairs']}"
    )

    print(
        f"GT pairs retained    : "
        f"{metrics['gt_pairs_retained']}"
    )

    print(
        f"GT retention         : "
        f"{metrics['retention_rate']:.3f}"
    )

    print(
        f"Candidate reduction  : "
        f"{metrics['reduction_rate']:.3f}"
    )

    print()

    print("=" * 70)

    print(
        "Algorithm test complete."
    )

    print("=" * 70)

    print()


if __name__ == "__main__":
    main()