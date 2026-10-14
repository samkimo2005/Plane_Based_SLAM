from __future__ import annotations

import numpy as np

import plane_detection_common as common


# ==========================================================
# Multi-Object RANSAC parameters
# ==========================================================

MULTI_HYPOTHESES = 1000
MULTI_MIN_HYPOTHESIS_SUPPORT = 120
MULTI_NORMAL_CLUSTER_DEG = 5.0
MULTI_OFFSET_CLUSTER_M = 0.05
MULTI_MIN_CLUSTER_VOTES = 4


def detect_planes(points, rng):
    hypothesis_normals = []
    hypothesis_ds = []
    hypothesis_supports = []

    for _ in range(
        MULTI_HYPOTHESES
    ):
        sample = rng.choice(
            len(points),
            size=3,
            replace=False,
        )

        fitted = common.fit_plane_three(
            points[sample[0]],
            points[sample[1]],
            points[sample[2]],
        )

        if fitted is None:
            continue

        normal, d = fitted

        support = int(
            np.sum(
                np.abs(
                    points @ normal + d
                )
                < common.FINAL_DISTANCE_THRESHOLD
            )
        )

        if (
            support
            >= MULTI_MIN_HYPOTHESIS_SUPPORT
        ):
            hypothesis_normals.append(
                normal
            )

            hypothesis_ds.append(
                d
            )

            hypothesis_supports.append(
                support
            )

    if len(hypothesis_normals) == 0:
        return []

    normals = np.asarray(
        hypothesis_normals
    )

    ds = np.asarray(
        hypothesis_ds
    )

    supports = np.asarray(
        hypothesis_supports
    )

    order = np.argsort(
        supports
    )[::-1]

    clusters = []

    for index in order:
        normal = normals[index]
        d = float(ds[index])
        support = int(
            supports[index]
        )

        assigned = False

        for cluster in clusters:
            n2, d2 = (
                common.align_plane_to_reference(
                    normal.copy(),
                    d,
                    cluster["normal"],
                )
            )

            if (
                common.plane_angle_deg(
                    cluster["normal"],
                    n2,
                )
                <= MULTI_NORMAL_CLUSTER_DEG
                and abs(
                    cluster["d"]
                    - d2
                )
                <= MULTI_OFFSET_CLUSTER_M
            ):
                cluster["votes"] += 1
                assigned = True
                break

        if not assigned:
            clusters.append(
                {
                    "normal":
                        normal.copy(),

                    "d":
                        d,

                    "representative_support":
                        support,

                    "votes":
                        1,
                }
            )

    models = []

    for cluster in clusters:
        if (
            cluster["votes"]
            < MULTI_MIN_CLUSTER_VOTES
        ):
            continue

        initial_mask = (
            np.abs(
                points
                @ cluster["normal"]
                + cluster["d"]
            )
            < common.FINAL_DISTANCE_THRESHOLD
        )

        if (
            np.sum(initial_mask)
            < common.MIN_PLANE_SUPPORT
        ):
            continue

        refined = common.fit_plane_svd(
            points[
                initial_mask
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
                cluster["votes"],
        )

        if model is not None:
            models.append(model)

    return common.merge_duplicate_models(
        models
    )


detect_multi_object_ransac = detect_planes
