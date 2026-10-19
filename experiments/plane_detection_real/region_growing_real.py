from __future__ import annotations

from collections import deque

import numpy as np

import plane_detection_common as common


# ==========================================================
# Region Growing parameters
# ==========================================================

RG_SMOOTHNESS_DEG = 15.0
RG_LOCAL_DISTANCE_M = 0.03
RG_MIN_REGION_SIZE = 100


def detect_planes(points, rng=None):
    geometry = (
        common.estimate_local_geometry(
            points
        )
    )

    if geometry is None:
        return []

    (
        normals,
        curvature,
        neighbour_indices,
    ) = geometry

    assigned = np.zeros(
        len(points),
        dtype=bool,
    )

    cos_threshold = np.cos(
        np.deg2rad(
            RG_SMOOTHNESS_DEG
        )
    )

    valid_regions = []

    for seed_index in np.argsort(
        curvature
    ):
        if assigned[
            seed_index
        ]:
            continue

        assigned[
            seed_index
        ] = True

        queue = deque(
            [int(seed_index)]
        )

        region = [
            int(seed_index)
        ]

        while queue:
            current = queue.popleft()

            current_normal = normals[
                current
            ]

            current_point = points[
                current
            ]

            for neighbour in (
                neighbour_indices[
                    current
                ]
            ):
                neighbour = int(
                    neighbour
                )

                if assigned[
                    neighbour
                ]:
                    continue

                normal_similarity = abs(
                    np.dot(
                        current_normal,
                        normals[
                            neighbour
                        ],
                    )
                )

                if (
                    normal_similarity
                    < cos_threshold
                ):
                    continue

                local_distance = abs(
                    np.dot(
                        current_normal,
                        points[
                            neighbour
                        ]
                        - current_point,
                    )
                )

                if (
                    local_distance
                    > RG_LOCAL_DISTANCE_M
                ):
                    continue

                assigned[
                    neighbour
                ] = True

                region.append(
                    neighbour
                )

                queue.append(
                    neighbour
                )

        if (
            len(region)
            >= RG_MIN_REGION_SIZE
        ):
            valid_regions.append(
                region
            )

    models = []

    for region in valid_regions:
        region_points = points[
            np.asarray(
                region,
                dtype=np.int64,
            )
        ]

        refined = common.fit_plane_svd(
            region_points
        )

        if refined is None:
            continue

        normal, d = refined

        model = common.build_model(
            points,
            normal,
            d,
            source_support=
                len(region),
        )

        if model is not None:
            models.append(model)

    return common.merge_duplicate_models(
        models
    )


detect_region_growing = detect_planes
