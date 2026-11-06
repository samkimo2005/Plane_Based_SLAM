from __future__ import annotations

import numpy as np

import bad_match_rejection_common as common


# Frozen/tunable parameters.
ANGLE_THRESHOLD_DEG = 6.0
SEPARATION_THRESHOLD = 0.10
MIN_COMPATIBLE_FRACTION = 0.70

# Separation is only meaningful for approximately parallel planes.
# This is a structural guard rather than the main compatibility threshold.
PARALLEL_THRESHOLD_DEG = 10.0


def _aligned_plane_separation(
    plane_i: np.ndarray,
    plane_j: np.ndarray,
) -> float:
    """
    Distance between two approximately parallel planes after resolving
    the plane-sign ambiguity.

    Plane convention:
        n^T x + d = 0
    """

    n_i = np.asarray(
        plane_i[:3],
        dtype=float,
    )

    d_i = float(
        plane_i[3]
    )

    n_j = np.asarray(
        plane_j[:3],
        dtype=float,
    )

    d_j = float(
        plane_j[3]
    )

    if np.dot(
        n_i,
        n_j,
    ) < 0.0:

        n_j = -n_j
        d_j = -d_j

    return float(
        abs(
            d_i
            - d_j
        )
    )


def reject_bad_matches(
    planes_a,
    planes_b,
    assigned_pairs,
    angle_threshold_deg=ANGLE_THRESHOLD_DEG,
    separation_threshold=SEPARATION_THRESHOLD,
    min_compatible_fraction=MIN_COMPATIBLE_FRACTION,
    parallel_threshold_deg=PARALLEL_THRESHOLD_DEG,
):
    """
    SelMaP-inspired pairwise compatibility selection.

    For two assigned correspondences

        A_i <-> B_i
        A_j <-> B_j

    rigid motion should preserve the angle between the two source
    planes and the corresponding two target planes:

        |alpha_A - alpha_B| < tau_alpha.

    If both plane pairs are approximately parallel, their inter-plane
    separation should also be preserved:

        |s_A - s_B| < tau_s.

    A pairwise compatibility graph is constructed, then a mutually
    consistent subset is selected greedily from the highest-support
    correspondence.

    This is a project-specific SelMaP-inspired comparator; it is not
    claimed to reproduce the unpublished implementation details of
    SelMaP exactly.
    """

    pairs = np.asarray(
        assigned_pairs,
        dtype=int,
    ).reshape(-1, 2)

    n_pairs = len(
        pairs
    )

    if n_pairs == 0:

        return {
            "kept_pairs":
                pairs,

            "rejected_pairs":
                np.empty(
                    (0, 2),
                    dtype=int,
                ),

            "keep_mask":
                np.zeros(
                    0,
                    dtype=bool,
                ),

            "success":
                False,

            "diagnostics":
                {
                    "reason":
                        "no_pairs",
                },
        }

    if n_pairs < 2:

        return {
            "kept_pairs":
                pairs,

            "rejected_pairs":
                np.empty(
                    (0, 2),
                    dtype=int,
                ),

            "keep_mask":
                np.ones(
                    n_pairs,
                    dtype=bool,
                ),

            "success":
                False,

            "diagnostics":
                {
                    "reason":
                        "too_few_pairs_for_pairwise_test",
                },
        }

    C = np.eye(
        n_pairs,
        dtype=bool,
    )

    angle_difference = np.full(
        (n_pairs, n_pairs),
        np.nan,
        dtype=float,
    )

    separation_difference = np.full(
        (n_pairs, n_pairs),
        np.nan,
        dtype=float,
    )

    parallel_test_used = np.zeros(
        (n_pairs, n_pairs),
        dtype=bool,
    )

    for i in range(
        n_pairs
    ):

        ia, ib = pairs[i]

        for j in range(
            i + 1,
            n_pairs,
        ):

            ja, jb = pairs[j]

            angle_a = (
                common.normal_angle_deg(
                    planes_a[ia][:3],
                    planes_a[ja][:3],
                )
            )

            angle_b = (
                common.normal_angle_deg(
                    planes_b[ib][:3],
                    planes_b[jb][:3],
                )
            )

            delta_angle = float(
                abs(
                    angle_a
                    - angle_b
                )
            )

            angle_ok = (
                delta_angle
                < float(
                    angle_threshold_deg
                )
            )

            use_separation = (
                angle_a
                < float(
                    parallel_threshold_deg
                )
                and
                angle_b
                < float(
                    parallel_threshold_deg
                )
            )

            separation_ok = True
            delta_separation = np.nan

            if use_separation:

                separation_a = (
                    _aligned_plane_separation(
                        planes_a[ia],
                        planes_a[ja],
                    )
                )

                separation_b = (
                    _aligned_plane_separation(
                        planes_b[ib],
                        planes_b[jb],
                    )
                )

                delta_separation = float(
                    abs(
                        separation_a
                        - separation_b
                    )
                )

                separation_ok = (
                    delta_separation
                    < float(
                        separation_threshold
                    )
                )

            compatible = (
                angle_ok
                and separation_ok
            )

            C[i, j] = compatible
            C[j, i] = compatible

            angle_difference[
                i,
                j,
            ] = delta_angle

            angle_difference[
                j,
                i,
            ] = delta_angle

            separation_difference[
                i,
                j,
            ] = delta_separation

            separation_difference[
                j,
                i,
            ] = delta_separation

            parallel_test_used[
                i,
                j,
            ] = use_separation

            parallel_test_used[
                j,
                i,
            ] = use_separation

    support = (
        C.sum(
            axis=1
        )
        - 1
    )

    order = np.argsort(
        -support,
        kind="stable",
    )

    selected = [
        int(
            order[0]
        )
    ]

    for idx in order[1:]:

        compatibility_fraction = float(
            C[
                idx,
                selected,
            ].mean()
        )

        if (
            compatibility_fraction
            >= float(
                min_compatible_fraction
            )
        ):

            selected.append(
                int(idx)
            )

    keep_mask = np.zeros(
        n_pairs,
        dtype=bool,
    )

    keep_mask[
        selected
    ] = True

    return {
        "kept_pairs":
            pairs[
                keep_mask
            ],

        "rejected_pairs":
            pairs[
                ~keep_mask
            ],

        "keep_mask":
            keep_mask,

        "success":
            bool(
                keep_mask.sum()
                >= 3
            ),

        "diagnostics":
            {
                "compatibility_matrix":
                    C,

                "support":
                    support,

                "angle_difference_deg":
                    angle_difference,

                "separation_difference_m":
                    separation_difference,

                "parallel_test_used":
                    parallel_test_used,

                "selected_indices":
                    np.asarray(
                        selected,
                        dtype=int,
                    ),
            },
    }
