"""
motion_model_distance_gating.py

Candidate Gating Method [29]

Motion-model-based distance constraint based on:

R. Guo et al.,
"RGB-D SLAM Using Point-Plane Constraints for Indoor Environments",
Sensors, 2019.

The method:

1. Uses the pose prior to transform source planes into
   the target coordinate frame.

2. Compares every source-target plane pair.

3. Keeps a pair if:

       |n_source^T n_target| > 0.95

   and:

       |d_source - d_target| < 0.05 m

These are the core plane-matching constraints used in [29].

No assignment or bad-match rejection is performed here.
This file implements Candidate Gating only.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from candidate_gating_common import (
    DEFAULT_SAMPLE,
    CandidateSample,
    PlaneSet,
    align_plane_sign,
    all_candidate_pairs,
    load_candidate_sample,
    transform_plane_set,
)


# ============================================================
# Default parameters
# ============================================================

# Literature baseline from [29].
DEFAULT_PARAMETERS = {
    "normal_dot_threshold": 0.95,
    "offset_threshold_m": 0.05,
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
    Apply the motion-model-based distance constraint.

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

            normal_dot_threshold
            offset_threshold_m

    Returns
    -------
    candidate_pairs:
        List of surviving source-target candidate pairs.
    """

    if parameters is None:
        parameters = DEFAULT_PARAMETERS.copy()

    normal_threshold = float(
        parameters["normal_dot_threshold"]
    )

    offset_threshold = float(
        parameters["offset_threshold_m"]
    )

    # --------------------------------------------------------
    # Transform source planes into predicted target frame
    # --------------------------------------------------------

    predicted_source = transform_plane_set(
        planes=source_planes,
        R=prior_R,
        t=prior_t,
    )

    candidate_pairs = []

    # --------------------------------------------------------
    # Test every possible source-target pair
    # --------------------------------------------------------

    for (
        source_idx,
        target_idx,
    ) in all_candidate_pairs(
        len(predicted_source),
        len(target_planes),
    ):

        source_normal = (
            predicted_source
            .normals[source_idx]
        )

        source_offset = float(
            predicted_source
            .offsets[source_idx]
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
        # Normal compatibility
        #
        # Plane normal sign is ambiguous:
        #
        #     (n, d)
        #     (-n, -d)
        #
        # therefore use absolute dot product.
        # ----------------------------------------------------

        normal_dot = abs(
            float(
                source_normal
                @ target_normal
            )
        )

        normal_pass = (
            normal_dot
            > normal_threshold
        )

        # ----------------------------------------------------
        # Resolve sign before comparing plane offsets
        # ----------------------------------------------------

        (
            aligned_target_normal,
            aligned_target_offset,
        ) = align_plane_sign(
            reference_normal=source_normal,
            normal=target_normal,
            offset=target_offset,
        )

        # Variable retained for clarity.
        _ = aligned_target_normal

        offset_difference = abs(
            source_offset
            - aligned_target_offset
        )

        offset_pass = (
            offset_difference
            < offset_threshold
        )

        # ----------------------------------------------------
        # Candidate survives only when BOTH conditions pass
        # ----------------------------------------------------

        passed_gate = (
            normal_pass
            and offset_pass
        )

        if not passed_gate:
            continue

        # ----------------------------------------------------
        # Additional quantities are recorded for later analysis,
        # but they DO NOT affect this algorithm's gate.
        # ----------------------------------------------------

        centroid_distance = float(
            np.linalg.norm(
                predicted_source
                .centroids[source_idx]

                - target_planes
                .centroids[target_idx]
            )
        )

        candidate_pairs.append(
            {
                "source_plane_id":
                    int(source_idx),

                "target_plane_id":
                    int(target_idx),

                "normal_dot":
                    float(normal_dot),

                "normal_angle_deg":
                    float(
                        np.degrees(
                            np.arccos(
                                np.clip(
                                    normal_dot,
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                    ),

                "offset_difference_m":
                    float(
                        offset_difference
                    ),

                "centroid_distance_m":
                    centroid_distance,

                "passed_normal_gate":
                    bool(normal_pass),

                "passed_offset_gate":
                    bool(offset_pass),

                "passed_gate":
                    True,

                "algorithm":
                    "motion_model_distance_gating",
            }
        )

    return candidate_pairs


# ============================================================
# Convenient wrapper for CandidateSample
# ============================================================

def run_on_sample(
    sample: CandidateSample,
    parameters: Optional[Dict] = None,
) -> List[Dict]:
    """
    Run the gating algorithm directly on a loaded sample.
    """

    return gate_candidates(
        source_planes=sample.source,
        target_planes=sample.target,
        prior_R=sample.prior_R,
        prior_t=sample.prior_t,
        parameters=parameters,
    )


# ============================================================
# Simple direct-run test
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print("Motion-Model-Based Distance Gating [29]")
    print("=" * 70)

    print(
        f"Input sample : {DEFAULT_SAMPLE}"
    )

    print()

    # --------------------------------------------------------
    # Load default DEV sample
    # --------------------------------------------------------

    sample = load_candidate_sample(
        DEFAULT_SAMPLE
    )

    # --------------------------------------------------------
    # Run candidate gating
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

    print(
        f"Source planes       : {len(sample.source)}"
    )

    print(
        f"Target planes       : {len(sample.target)}"
    )

    print(
        f"Possible pairs      : {number_before}"
    )

    print(
        f"Surviving pairs     : {number_after}"
    )

    print()

    print(
        "Parameters:"
    )

    print(
        "  Normal dot threshold : "
        f"{DEFAULT_PARAMETERS['normal_dot_threshold']}"
    )

    print(
        "  Offset threshold     : "
        f"{DEFAULT_PARAMETERS['offset_threshold_m']:.3f} m"
    )

    # --------------------------------------------------------
    # Display surviving candidates
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
                f"S{candidate['source_plane_id']} "
                f"-> "
                f"T{candidate['target_plane_id']} | "
                f"dot="
                f"{candidate['normal_dot']:.4f} | "
                f"angle="
                f"{candidate['normal_angle_deg']:.3f} deg | "
                f"offset="
                f"{candidate['offset_difference_m']:.4f} m"
            )

    print(
        "-" * 70
    )

    # --------------------------------------------------------
    # Temporary sanity check using GT
    #
    # This is NOT the final evaluation module.
    # It only helps us check whether the implementation works.
    # --------------------------------------------------------

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

    candidate_set = {
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
        & candidate_set
    )

    print()
    print(
        "Temporary GT sanity check:"
    )

    print(
        f"GT pairs             : {len(gt_pairs)}"
    )

    print(
        f"GT pairs retained    : {len(retained_gt)}"
    )

    if len(gt_pairs) > 0:

        retention = (
            len(retained_gt)
            / len(gt_pairs)
        )

    else:

        retention = 0.0

    if number_before > 0:

        reduction = (
            1.0
            - number_after
            / number_before
        )

    else:

        reduction = 0.0

    print(
        f"GT retention         : {retention:.3f}"
    )

    print(
        f"Candidate reduction  : {reduction:.3f}"
    )

    print()
    print("=" * 70)
    print("Algorithm test complete.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()