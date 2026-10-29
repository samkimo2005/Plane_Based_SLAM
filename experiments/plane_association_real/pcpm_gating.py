"""
pcpm_gating.py

Candidate Gating Method [72]

PCPM-inspired candidate gating based on:

J. Bao, X. Yuan, G. Huang and C.-T. Lam,
"Point Cloud Plane Segmentation-Based Robust Image Matching
for Camera Pose Estimation",
Remote Sensing, 2023.

Reference [72].

------------------------------------------------------------
Original PCPM
------------------------------------------------------------

The Point Cloud Plane Matching (PCPM) algorithm compares planes
using:

    1. centroid distance
    2. normal-vector angle

The original paper marks planes as matched when both quantities
are minimum.

------------------------------------------------------------
Project adaptation
------------------------------------------------------------

This project separates Plane Association into:

    Candidate Gating
        ->
    Pairwise Similarity / Cost
        ->
    Assignment

Therefore PCPM is adapted here as a candidate gate rather than
performing final plane matching.

For each source plane:

    - rank target planes by centroid distance
    - rank target planes by normal angle
    - keep pairs appearing in the top-K set of BOTH rankings

No final assignment is performed here.

Plane-normal sign ambiguity is handled using the common
sign-invariant normal-angle function.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from candidate_gating_common import (
    DEFAULT_SAMPLE,
    CandidateSample,
    PlaneSet,
    load_candidate_sample,
    normal_angle_deg,
    transform_plane_set,
)


# ============================================================
# Default parameters
# ============================================================

DEFAULT_PARAMETERS = {

    # Number of best centroid candidates retained per source.
    "centroid_top_k": 2,

    # Number of best normal-angle candidates retained per source.
    "normal_top_k": 2,
}


# ============================================================
# PCPM-inspired Candidate Gating
# ============================================================

def gate_candidates(
    source_planes: PlaneSet,
    target_planes: PlaneSet,
    prior_R: np.ndarray,
    prior_t: np.ndarray,
    parameters: Optional[Dict] = None,
) -> List[Dict]:
    """
    Apply PCPM-inspired centroid + normal candidate gating.

    Parameters
    ----------
    source_planes:
        Source-frame plane observations.

    target_planes:
        Target-frame plane observations.

    prior_R, prior_t:
        Estimated source-to-target pose.

        x_target = prior_R @ x_source + prior_t

    parameters:
        centroid_top_k
        normal_top_k

    Returns
    -------
    candidate_pairs:
        Candidate pairs surviving BOTH geometric rankings.
    """

    if parameters is None:
        parameters = DEFAULT_PARAMETERS.copy()

    centroid_top_k = int(
        parameters["centroid_top_k"]
    )

    normal_top_k = int(
        parameters["normal_top_k"]
    )

    if centroid_top_k < 1:
        raise ValueError(
            "centroid_top_k must be at least 1."
        )

    if normal_top_k < 1:
        raise ValueError(
            "normal_top_k must be at least 1."
        )

    # --------------------------------------------------------
    # Transform source planes into target frame
    # --------------------------------------------------------

    source_in_target = transform_plane_set(
        planes=source_planes,
        R=prior_R,
        t=prior_t,
    )

    n_source = len(
        source_in_target
    )

    n_target = len(
        target_planes
    )

    if n_target == 0:
        return []

    centroid_top_k = min(
        centroid_top_k,
        n_target,
    )

    normal_top_k = min(
        normal_top_k,
        n_target,
    )

    candidate_pairs = []

    # --------------------------------------------------------
    # Process each source plane independently
    # --------------------------------------------------------

    for source_idx in range(
        n_source
    ):

        source_centroid = (
            source_in_target
            .centroids[source_idx]
        )

        source_normal = (
            source_in_target
            .normals[source_idx]
        )

        centroid_distances = []
        normal_angles = []

        # ----------------------------------------------------
        # Compute PCPM geometric quantities for all targets
        # ----------------------------------------------------

        for target_idx in range(
            n_target
        ):

            target_centroid = (
                target_planes
                .centroids[target_idx]
            )

            target_normal = (
                target_planes
                .normals[target_idx]
            )

            # -----------------------------------------------
            # Equation corresponding to centroid distance
            # in PCPM.
            # -----------------------------------------------

            centroid_distance = float(
                np.linalg.norm(
                    source_centroid
                    - target_centroid
                )
            )

            # -----------------------------------------------
            # Normal-vector angular difference.
            #
            # common.py uses a sign-invariant angle because
            # n and -n represent the same geometric plane.
            # -----------------------------------------------

            angle_difference = (
                normal_angle_deg(
                    source_normal,
                    target_normal,
                )
            )

            centroid_distances.append(
                centroid_distance
            )

            normal_angles.append(
                angle_difference
            )

        centroid_distances = np.asarray(
            centroid_distances,
            dtype=float,
        )

        normal_angles = np.asarray(
            normal_angles,
            dtype=float,
        )

        # ----------------------------------------------------
        # Rank target planes by centroid distance
        # ----------------------------------------------------

        centroid_order = np.argsort(
            centroid_distances
        )

        centroid_candidates = set(
            int(index)
            for index
            in centroid_order[
                :centroid_top_k
            ]
        )

        # ----------------------------------------------------
        # Rank target planes by normal angle
        # ----------------------------------------------------

        normal_order = np.argsort(
            normal_angles
        )

        normal_candidates = set(
            int(index)
            for index
            in normal_order[
                :normal_top_k
            ]
        )

        # ----------------------------------------------------
        # PCPM-inspired joint compatibility:
        #
        # candidate must be good according to BOTH cues.
        # ----------------------------------------------------

        surviving_targets = (
            centroid_candidates
            & normal_candidates
        )

        # ----------------------------------------------------
        # Build candidate output
        # ----------------------------------------------------

        for target_idx in sorted(
            surviving_targets
        ):

            centroid_rank = int(
                np.where(
                    centroid_order
                    == target_idx
                )[0][0]
            ) + 1

            normal_rank = int(
                np.where(
                    normal_order
                    == target_idx
                )[0][0]
            ) + 1

            candidate_pairs.append(
                {

                    "source_plane_id":
                        int(source_idx),

                    "target_plane_id":
                        int(target_idx),

                    "centroid_distance_m":
                        float(
                            centroid_distances[
                                target_idx
                            ]
                        ),

                    "normal_angle_deg":
                        float(
                            normal_angles[
                                target_idx
                            ]
                        ),

                    "centroid_rank":
                        centroid_rank,

                    "normal_rank":
                        normal_rank,

                    "passed_centroid_gate":
                        True,

                    "passed_normal_gate":
                        True,

                    "passed_gate":
                        True,

                    "algorithm":
                        "pcpm_gating",
                }
            )

    return candidate_pairs


# ============================================================
# Convenient sample wrapper
# ============================================================

def run_on_sample(
    sample: CandidateSample,
    parameters: Optional[Dict] = None,
) -> List[Dict]:
    """
    Run PCPM-inspired gating on one generated sample.
    """

    return gate_candidates(
        source_planes=sample.source,
        target_planes=sample.target,

        prior_R=sample.prior_R,
        prior_t=sample.prior_t,

        parameters=parameters,
    )


# ============================================================
# Temporary sanity metrics
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

    possible_pairs = (
        len(sample.source)
        * len(sample.target)
    )

    surviving_pairs = len(
        candidate_pairs
    )

    if len(gt_pairs) > 0:

        retention = (
            len(retained_gt)
            / len(gt_pairs)
        )

    else:

        retention = 0.0

    if possible_pairs > 0:

        reduction = (
            1.0
            - surviving_pairs
            / possible_pairs
        )

    else:

        reduction = 0.0

    return {

        "gt_pairs":
            len(gt_pairs),

        "gt_pairs_retained":
            len(retained_gt),

        "possible_pairs":
            possible_pairs,

        "candidate_pairs":
            surviving_pairs,

        "retention_rate":
            retention,

        "reduction_rate":
            reduction,
    }


# ============================================================
# Direct-run test
# ============================================================

def main() -> None:

    print()
    print("=" * 70)

    print(
        "PCPM-Inspired Candidate Gating [72]"
    )

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
    # Run algorithm
    # --------------------------------------------------------

    candidates = run_on_sample(
        sample
    )

    possible_pairs = (
        len(sample.source)
        * len(sample.target)
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
        f"{possible_pairs}"
    )

    print(
        f"Surviving pairs     : "
        f"{len(candidates)}"
    )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    print()

    print(
        "Parameters:"
    )

    print(
        "  Centroid top-K : "
        f"{DEFAULT_PARAMETERS['centroid_top_k']}"
    )

    print(
        "  Normal top-K   : "
        f"{DEFAULT_PARAMETERS['normal_top_k']}"
    )

    # --------------------------------------------------------
    # Display surviving pairs
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

                f" | centroid="
                f"{candidate['centroid_distance_m']:.4f} m"

                f" | centroid rank="
                f"{candidate['centroid_rank']}"

                f" | angle="
                f"{candidate['normal_angle_deg']:.3f} deg"

                f" | angle rank="
                f"{candidate['normal_rank']}"
            )

    print(
        "-" * 70
    )

    # --------------------------------------------------------
    # Temporary GT sanity check
    # --------------------------------------------------------

    metrics = calculate_sanity_metrics(
        sample=sample,
        candidates=candidates,
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