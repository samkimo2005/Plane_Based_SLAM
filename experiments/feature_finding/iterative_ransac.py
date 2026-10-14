from __future__ import annotations

import numpy as np

import plane_detection_common as common


# ==========================================================
# Iterative RANSAC parameters
# ==========================================================

RANSAC_ITERATIONS = 500


def detect_planes(points, rng):
    remaining_indices = np.arange(
        len(points)
    )

    models = []

    for _ in range(
        common.MAX_PLANES
    ):
        if (
            len(remaining_indices)
            < common.MIN_PLANE_SUPPORT
        ):
            break

        remaining_points = points[
            remaining_indices
        ]

        best_support = 0
        best_plane = None
        best_mask_local = None

        for _ in range(
            RANSAC_ITERATIONS
        ):
            sample_indices = rng.choice(
                len(remaining_points),
                size=3,
                replace=False,
            )

            fitted = common.fit_plane_three(
                remaining_points[
                    sample_indices[0]
                ],
                remaining_points[
                    sample_indices[1]
                ],
                remaining_points[
                    sample_indices[2]
                ],
            )

            if fitted is None:
                continue

            normal, d = fitted

            distances = np.abs(
                remaining_points @ normal + d
            )

            mask_local = (
                distances
                < common.FINAL_DISTANCE_THRESHOLD
            )

            support = int(
                np.sum(mask_local)
            )

            if support > best_support:
                best_support = support
                best_plane = (
                    normal,
                    d,
                )
                best_mask_local = mask_local

        if (
            best_plane is None
            or best_support
            < common.MIN_PLANE_SUPPORT
        ):
            break

        refined = common.fit_plane_svd(
            remaining_points[
                best_mask_local
            ]
        )

        if refined is None:
            break

        normal, d = refined

        model = common.build_model(
            points,
            normal,
            d,
            source_support=best_support,
        )

        if model is None:
            break

        models.append(model)

        remove_mask = np.abs(
            remaining_points @ normal + d
        ) < common.FINAL_DISTANCE_THRESHOLD

        remaining_indices = (
            remaining_indices[
                ~remove_mask
            ]
        )

    return common.merge_duplicate_models(
        models
    )


# Compatibility alias.
detect_iterative_ransac = detect_planes
