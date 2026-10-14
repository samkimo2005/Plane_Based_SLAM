"""
normal_based_gating.py

Candidate Gating Method

Project-adapted implementation inspired by:

X. Zuo et al.,
"LIC-Fusion 2.0: LiDAR-Inertial-Camera Odometry with
Sliding-Window Plane-Feature Tracking",
IEEE/RSJ IROS, 2020.

------------------------------------------------------------
IMPORTANT
------------------------------------------------------------

LIC-Fusion 2.0 performs uncertainty-aware normal-based
plane data association using a Mahalanobis / chi-square test.

The original method propagates uncertainty from:

    - planar point measurements
    - relative orientation uncertainty
    - associated Jacobians

The present RGB-D synthetic benchmark does not contain
the original LiDAR point-level covariance or EKF covariance.

Therefore this implementation preserves the same principle
but uses an approximate covariance constructed from:

    - plane-normal measurement uncertainty
    - pose-prior rotational uncertainty

This is a project-adapted implementation, NOT an exact
reproduction of the complete LIC-Fusion 2.0 estimator.

Candidate Gating only:
    no similarity ranking
    no assignment
    no bad-match rejection
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from candidate_gating_common import (
    DEFAULT_SAMPLE,
    CandidateSample,
    PlaneSet,
    all_candidate_pairs,
    load_candidate_sample,
    normal_angle_deg,
    transform_plane_set,
)


# ============================================================
# Default parameters
# ============================================================

DEFAULT_PARAMETERS = {

    # 95% chi-square threshold for approximately
    # two rotational degrees of freedom.
    "chi_square_threshold": 5.991,

    # Used only if uncertainty cannot be obtained
    # from synthetic sample metadata.
    "default_normal_sigma_deg": 2.0,

    "default_pose_rotation_sigma_deg": 2.0,

    # Prevent numerically unrealistically tiny covariance.
    "minimum_sigma_deg": 0.10,

    # During our synthetic benchmark the configured
    # uncertainty levels are known.
    #
    # This plays the role that covariance estimates
    # would have in a real estimator.
    "use_sample_uncertainty": True,
}


# ============================================================
# Uncertainty utilities
# ============================================================

def get_sample_uncertainty(
    sample: CandidateSample,
    parameters: Dict,
) -> Tuple[float, float]:
    """
    Obtain uncertainty values used by the statistical gate.

    Returns
    -------
    normal_sigma_deg:
        Standard deviation associated with an individual
        plane-normal observation.

    pose_rotation_sigma_deg:
        Standard deviation associated with the relative
        rotation prior.

    Notes
    -----
    For the controlled synthetic benchmark, these values are
    stored in the condition metadata.

    In a real SLAM system they should instead come from the
    plane estimator and pose-estimator covariance.
    """

    normal_sigma_deg = float(
        parameters[
            "default_normal_sigma_deg"
        ]
    )

    pose_sigma_deg = float(
        parameters[
            "default_pose_rotation_sigma_deg"
        ]
    )

    use_metadata = bool(
        parameters[
            "use_sample_uncertainty"
        ]
    )

    if (
        use_metadata
        and sample.metadata is not None
    ):

        condition = sample.metadata.get(
            "condition",
            {},
        )

        if "normal_noise_deg" in condition:

            normal_sigma_deg = float(
                condition[
                    "normal_noise_deg"
                ]
            )

        if (
            "pose_rotation_error_deg"
            in condition
        ):

            pose_sigma_deg = float(
                condition[
                    "pose_rotation_error_deg"
                ]
            )

    minimum_sigma = float(
        parameters[
            "minimum_sigma_deg"
        ]
    )

    normal_sigma_deg = max(
        normal_sigma_deg,
        minimum_sigma,
    )

    pose_sigma_deg = max(
        pose_sigma_deg,
        minimum_sigma,
    )

    return (
        normal_sigma_deg,
        pose_sigma_deg,
    )


# ============================================================
# Normal covariance model
# ============================================================

def tangent_normal_covariance(
    normal: np.ndarray,
    sigma_rad: float,
) -> np.ndarray:
    """
    Approximate covariance of a unit plane normal.

    A unit normal has only two independent perturbation
    directions because motion along the normal itself does
    not change its orientation.

    The tangent-plane covariance is approximated by:

        P_n = sigma^2 (I - n n^T)
    """

    normal = np.asarray(
        normal,
        dtype=float,
    )

    normal = normal / np.linalg.norm(
        normal
    )

    I = np.eye(3)

    tangent_projector = (
        I
        - np.outer(
            normal,
            normal,
        )
    )

    covariance = (
        sigma_rad ** 2
        * tangent_projector
    )

    return covariance


def combined_normal_covariance(
    source_normal: np.ndarray,
    target_normal: np.ndarray,
    normal_sigma_deg: float,
    pose_rotation_sigma_deg: float,
) -> np.ndarray:
    """
    Construct an approximate covariance for the normal
    compatibility residual.

    Contributions:

        source normal measurement uncertainty
        target normal measurement uncertainty
        relative pose rotational uncertainty

    The source and target normal measurements are assumed
    independent.
    """

    normal_sigma_rad = np.deg2rad(
        normal_sigma_deg
    )

    pose_sigma_rad = np.deg2rad(
        pose_rotation_sigma_deg
    )

    source_measurement_cov = (
        tangent_normal_covariance(
            normal=source_normal,
            sigma_rad=normal_sigma_rad,
        )
    )

    target_measurement_cov = (
        tangent_normal_covariance(
            normal=target_normal,
            sigma_rad=normal_sigma_rad,
        )
    )

    pose_cov = (
        tangent_normal_covariance(
            normal=source_normal,
            sigma_rad=pose_sigma_rad,
        )
    )

    covariance = (
        source_measurement_cov
        + target_measurement_cov
        + pose_cov
    )

    # Very small numerical regularisation.
    covariance += (
        1e-12
        * np.eye(3)
    )

    return covariance


# ============================================================
# Normal compatibility
# ============================================================

def align_normal_direction(
    reference_normal: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    """
    Resolve normal sign ambiguity.

    n and -n represent the same geometric plane.
    """

    reference_normal = np.asarray(
        reference_normal,
        dtype=float,
    )

    normal = np.asarray(
        normal,
        dtype=float,
    )

    if float(
        reference_normal @ normal
    ) < 0.0:

        normal = -normal

    return normal


def normal_residual(
    source_normal: np.ndarray,
    target_normal: np.ndarray,
) -> np.ndarray:
    """
    Generate a residual expressing non-parallelity.

    Inspired by the normal "parallelity" measurement
    used in LIC-Fusion 2.0.

    Here:

        r = n_source x n_target

    For parallel planes:

        r -> 0
    """

    source_normal = np.asarray(
        source_normal,
        dtype=float,
    )

    target_normal = (
        align_normal_direction(
            reference_normal=source_normal,
            normal=target_normal,
        )
    )

    return np.cross(
        source_normal,
        target_normal,
    )


def mahalanobis_distance_squared(
    residual: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """
    Compute:

        D_M^2 = r^T P^-1 r

    Pseudoinverse is used because unit-normal covariance is
    naturally close to rank 2.
    """

    residual = np.asarray(
        residual,
        dtype=float,
    ).reshape(3)

    covariance = np.asarray(
        covariance,
        dtype=float,
    ).reshape(
        3,
        3,
    )

    covariance_inverse = (
        np.linalg.pinv(
            covariance
        )
    )

    distance_squared = float(

        residual.T
        @ covariance_inverse
        @ residual
    )

    return distance_squared


# ============================================================
# Candidate Gating algorithm
# ============================================================

def gate_candidates(
    source_planes: PlaneSet,
    target_planes: PlaneSet,
    prior_R: np.ndarray,
    prior_t: np.ndarray,
    normal_sigma_deg: float,
    pose_rotation_sigma_deg: float,
    parameters: Optional[Dict] = None,
) -> List[Dict]:
    """
    Apply uncertainty-aware normal-based Candidate Gating.

    Steps
    -----
    1. Transform source planes into target frame using
       the pose prior.

    2. For every source-target pair, calculate the
       normal-parallelity residual.

    3. Build approximate uncertainty covariance.

    4. Calculate squared Mahalanobis distance.

    5. Accept the pair if:

           D_M^2 < chi-square threshold
    """

    if parameters is None:

        parameters = (
            DEFAULT_PARAMETERS.copy()
        )

    chi_square_threshold = float(
        parameters[
            "chi_square_threshold"
        ]
    )

    # --------------------------------------------------------
    # Transform source planes to predicted target frame
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
    # Check every source-target pair
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

        target_normal = (
            target_planes
            .normals[target_idx]
        )

        # ----------------------------------------------------
        # Resolve normal sign ambiguity.
        # ----------------------------------------------------

        target_normal_aligned = (
            align_normal_direction(
                reference_normal=(
                    source_normal
                ),
                normal=(
                    target_normal
                ),
            )
        )

        # ----------------------------------------------------
        # Normal residual
        #
        # r = n_source x n_target
        # ----------------------------------------------------

        residual = (
            normal_residual(
                source_normal=(
                    source_normal
                ),
                target_normal=(
                    target_normal_aligned
                ),
            )
        )

        residual_norm = float(
            np.linalg.norm(
                residual
            )
        )

        # ----------------------------------------------------
        # Approximate residual covariance
        # ----------------------------------------------------

        covariance = (
            combined_normal_covariance(
                source_normal=(
                    source_normal
                ),

                target_normal=(
                    target_normal_aligned
                ),

                normal_sigma_deg=(
                    normal_sigma_deg
                ),

                pose_rotation_sigma_deg=(
                    pose_rotation_sigma_deg
                ),
            )
        )

        # ----------------------------------------------------
        # Mahalanobis compatibility
        # ----------------------------------------------------

        distance_squared = (
            mahalanobis_distance_squared(
                residual=residual,
                covariance=covariance,
            )
        )

        passed_gate = (
            distance_squared
            < chi_square_threshold
        )

        if not passed_gate:
            continue

        # ----------------------------------------------------
        # Extra quantities for analysis only
        # ----------------------------------------------------

        angle_difference = (
            normal_angle_deg(
                source_normal,
                target_normal_aligned,
            )
        )

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

                "normal_angle_deg":
                    float(
                        angle_difference
                    ),

                "normal_residual_norm":
                    float(
                        residual_norm
                    ),

                "mahalanobis_distance_squared":
                    float(
                        distance_squared
                    ),

                "chi_square_threshold":
                    float(
                        chi_square_threshold
                    ),

                "normal_sigma_deg":
                    float(
                        normal_sigma_deg
                    ),

                "pose_rotation_sigma_deg":
                    float(
                        pose_rotation_sigma_deg
                    ),

                "centroid_distance_m":
                    centroid_distance,

                "passed_gate":
                    True,

                "algorithm":
                    "normal_based_gating",
            }
        )

    return candidate_pairs


# ============================================================
# Wrapper for generated benchmark sample
# ============================================================

def run_on_sample(
    sample: CandidateSample,
    parameters: Optional[Dict] = None,
) -> List[Dict]:
    """
    Run [71]-inspired normal-based gating on one sample.
    """

    if parameters is None:

        parameters = (
            DEFAULT_PARAMETERS.copy()
        )

    (
        normal_sigma_deg,
        pose_rotation_sigma_deg,
    ) = get_sample_uncertainty(
        sample=sample,
        parameters=parameters,
    )

    return gate_candidates(

        source_planes=(
            sample.source
        ),

        target_planes=(
            sample.target
        ),

        prior_R=(
            sample.prior_R
        ),

        prior_t=(
            sample.prior_t
        ),

        normal_sigma_deg=(
            normal_sigma_deg
        ),

        pose_rotation_sigma_deg=(
            pose_rotation_sigma_deg
        ),

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
    Temporary test only.

    Formal evaluation will later be performed by:

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
        "Normal-Based Candidate Gating [71]"
    )

    print(
        "Uncertainty-Aware Mahalanobis Gate"
    )

    print("=" * 70)

    print(
        f"Input sample : {DEFAULT_SAMPLE}"
    )

    print()

    # --------------------------------------------------------
    # Load sample
    # --------------------------------------------------------

    sample = (
        load_candidate_sample(
            DEFAULT_SAMPLE
        )
    )

    # --------------------------------------------------------
    # Determine uncertainty
    # --------------------------------------------------------

    (
        normal_sigma_deg,
        pose_sigma_deg,
    ) = get_sample_uncertainty(
        sample=sample,
        parameters=DEFAULT_PARAMETERS,
    )

    # --------------------------------------------------------
    # Run gating
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

    print()

    print(
        "Statistical parameters:"
    )

    print(
        f"  Normal sigma      : "
        f"{normal_sigma_deg:.3f} deg"
    )

    print(
        f"  Pose rotation σ   : "
        f"{pose_sigma_deg:.3f} deg"
    )

    print(
        f"  Chi-square limit  : "
        f"{DEFAULT_PARAMETERS['chi_square_threshold']:.3f}"
    )

    # --------------------------------------------------------
    # Candidate output
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

                f" | angle="
                f"{candidate['normal_angle_deg']:.3f} deg"

                f" | residual="
                f"{candidate['normal_residual_norm']:.5f}"

                f" | Mahalanobis²="
                f"{candidate['mahalanobis_distance_squared']:.3f}"
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