from __future__ import annotations

from collections import defaultdict

import numpy as np

import plane_detection_common_real as common


# ==========================================================
# 3D Hough parameters
# ==========================================================

HOUGH_MAX_CURVATURE = 0.08
HOUGH_THETA_BIN_DEG = 10.0
HOUGH_PHI_BIN_DEG = 10.0
HOUGH_D_BIN_M = 0.05
HOUGH_MIN_VOTES = 40


def detect_planes(points, rng=None):
    geometry = (
        common.estimate_local_geometry(
            points
        )
    )

    if geometry is None:
        return []

    normals, curvature, _ = geometry

    planar_indices = np.flatnonzero(
        curvature
        <= HOUGH_MAX_CURVATURE
    )

    if (
        len(planar_indices)
        < common.MIN_PLANE_SUPPORT
    ):
        return []

    accumulator = defaultdict(
        list
    )

    for point_index in planar_indices:
        normal = normals[
            point_index
        ]

        d = -float(
            np.dot(
                normal,
                points[
                    point_index
                ],
            )
        )

        normal, d = (
            common.canonicalise_plane(
                normal,
                d,
            )
        )

        theta = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        normal[2],
                        -1.0,
                        1.0,
                    )
                )
            )
        )

        phi = float(
            np.degrees(
                np.arctan2(
                    normal[1],
                    normal[0],
                )
            )
        )

        if phi < 0:
            phi += 360.0

        if (
            theta
            < HOUGH_THETA_BIN_DEG
        ):
            phi = 0.0

        theta_bin = int(
            np.floor(
                theta
                / HOUGH_THETA_BIN_DEG
            )
        )

        phi_bin = int(
            np.floor(
                phi
                / HOUGH_PHI_BIN_DEG
            )
        )

        d_bin = int(
            np.floor(
                d
                / HOUGH_D_BIN_M
            )
        )

        accumulator[
            (
                theta_bin,
                phi_bin,
                d_bin,
            )
        ].append(
            int(point_index)
        )

    sorted_bins = sorted(
        accumulator.items(),
        key=lambda item:
            len(item[1]),
        reverse=True,
    )

    models = []

    for _, indices in sorted_bins:
        if (
            len(indices)
            < HOUGH_MIN_VOTES
        ):
            break

        refined = common.fit_plane_svd(
            points[
                np.asarray(
                    indices,
                    dtype=np.int64,
                )
            ]
        )

        if refined is None:
            continue

        normal, d = refined

        model = common.build_model(
            points,
            normal,
            d,
            source_support=
                len(indices),
        )

        if model is None:
            continue

        models.append(model)

        models = (
            common.merge_duplicate_models(
                models
            )
        )

        if (
            len(models)
            >= common.MAX_PLANES
        ):
            break

    return common.merge_duplicate_models(
        models
    )


detect_hough = detect_planes
