from __future__ import annotations

import numpy as np

import plane_detection_common as common


# ==========================================================
# Surface-normal / surfel parameters
# ==========================================================

SURFEL_MAX_CURVATURE = 0.08
SURFEL_NORMAL_GROUP_DEG = 12.0
SURFEL_OFFSET_BIN_M = 0.05
SURFEL_OFFSET_GROUP_M = 0.04
SURFEL_OFFSET_NMS_M = 0.08


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
        <= SURFEL_MAX_CURVATURE
    )

    if (
        len(planar_indices)
        < common.MIN_PLANE_SUPPORT
    ):
        return []

    groups = []

    for index in planar_indices[
        np.argsort(
            curvature[
                planar_indices
            ]
        )
    ]:
        normal = normals[index]

        assigned = False

        for group in groups:
            if (
                common.plane_angle_deg(
                    normal,
                    group["normal"],
                )
                <= SURFEL_NORMAL_GROUP_DEG
            ):
                group["indices"].append(
                    int(index)
                )

                assigned = True
                break

        if not assigned:
            groups.append(
                {
                    "normal":
                        normal.copy(),

                    "indices":
                        [int(index)],
                }
            )

    models = []

    for group in groups:
        indices = np.asarray(
            group["indices"],
            dtype=np.int64,
        )

        if (
            len(indices)
            < common.MIN_PLANE_SUPPORT
        ):
            continue

        representative_normal = (
            group["normal"]
        )

        offsets = -(
            points[indices]
            @ representative_normal
        )

        minimum = float(
            np.min(offsets)
        )

        maximum = float(
            np.max(offsets)
        )

        if (
            maximum
            - minimum
            < 1e-12
        ):
            candidate_centres = [
                float(
                    np.mean(offsets)
                )
            ]

        else:
            edges = np.arange(
                minimum,
                maximum
                + SURFEL_OFFSET_BIN_M,
                SURFEL_OFFSET_BIN_M,
            )

            if len(edges) < 2:
                edges = np.array(
                    [
                        minimum,
                        minimum
                        + SURFEL_OFFSET_BIN_M,
                    ]
                )

            histogram, edges = (
                np.histogram(
                    offsets,
                    bins=edges,
                )
            )

            bin_order = np.argsort(
                histogram
            )[::-1]

            candidate_centres = []

            for bin_index in bin_order:
                if (
                    histogram[bin_index]
                    < common.MIN_PLANE_SUPPORT
                ):
                    break

                centre = float(
                    0.5
                    * (
                        edges[bin_index]
                        + edges[
                            bin_index + 1
                        ]
                    )
                )

                if any(
                    abs(
                        centre
                        - old
                    )
                    < SURFEL_OFFSET_NMS_M
                    for old
                    in candidate_centres
                ):
                    continue

                candidate_centres.append(
                    centre
                )

                if (
                    len(candidate_centres)
                    >= common.MAX_PLANES
                ):
                    break

        for centre_d in candidate_centres:
            selected = indices[
                np.abs(
                    offsets
                    - centre_d
                )
                <= SURFEL_OFFSET_GROUP_M
            ]

            if (
                len(selected)
                < common.MIN_PLANE_SUPPORT
            ):
                continue

            refined = common.fit_plane_svd(
                points[
                    selected
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
                    len(selected),
            )

            if model is not None:
                models.append(model)

    return common.merge_duplicate_models(
        models
    )


detect_surface_normal = detect_planes
