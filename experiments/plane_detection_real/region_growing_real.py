"""Region-growing plane detector for real RGB-D point clouds.

Fixes a critical issue in the earlier implementation: neighbour-to-neighbour
smoothness alone allows region growth to drift across a wall corner. The old
version could join almost an entire non-planar cloud and then fit one SVD
plane through it, producing only a narrow stripe of geometric inliers.

This detector grows regions using both local continuity and *seed-anchored*
constraints. It implements the same callable API as the previous real-data
method, so evaluate_plane_detection_real.py requires no edits.
"""
from __future__ import annotations

from collections import deque
import numpy as np
import plane_detection_common_real as common

RG_SMOOTHNESS_DEG = 15.0       # Adjacent normal agreement
RG_SEED_NORMAL_DEG = 15.0     # Prevent cumulative normal drift
RG_LOCAL_DISTANCE_M = 0.03   # Local tangent-plane continuity
RG_SEED_DISTANCE_M = 0.03    # Prevent growth away from the seed plane
RG_MIN_REGION_SIZE = 100


def detect_planes(points: np.ndarray, rng=None) -> list[dict]:
    """Return candidate plane models in the format expected by the evaluator.

    The returned ``source_support`` records native connected-region size.
    The evaluator later computes its own disjoint *geometric* point labels;
    those labels are not the native spatial region memberships.
    """
    points = np.asarray(points, dtype=np.float64)
    geometry = common.estimate_local_geometry(points)
    if geometry is None:
        return []

    normals, curvature, neighbour_indices = geometry
    claimed = np.zeros(len(points), dtype=bool)
    cos_local = float(np.cos(np.deg2rad(RG_SMOOTHNESS_DEG)))
    cos_seed = float(np.cos(np.deg2rad(RG_SEED_NORMAL_DEG)))
    models: list[dict] = []

    for seed in np.argsort(curvature):
        seed = int(seed)
        if claimed[seed]:
            continue
        seed_normal = normals[seed]
        seed_point = points[seed]
        claimed[seed] = True
        queue = deque([seed])
        region = [seed]

        while queue:
            current = queue.popleft()
            normal = normals[current]
            point = points[current]
            for neighbour in neighbour_indices[current]:
                neighbour = int(neighbour)
                if claimed[neighbour]:
                    continue
                candidate_normal = normals[neighbour]
                candidate_point = points[neighbour]

                # Local continuity: adjacent smooth points only.
                if abs(float(normal @ candidate_normal)) < cos_local:
                    continue
                if abs(float(normal @ (candidate_point - point))) > RG_LOCAL_DISTANCE_M:
                    continue

                # Global consistency with the original seed: the missing
                # checks in the previous implementation.
                if abs(float(seed_normal @ candidate_normal)) < cos_seed:
                    continue
                if abs(float(seed_normal @ (candidate_point - seed_point))) > RG_SEED_DISTANCE_M:
                    continue

                claimed[neighbour] = True
                region.append(neighbour)
                queue.append(neighbour)

        if len(region) < RG_MIN_REGION_SIZE:
            continue

        fitted = common.fit_plane_svd(points[np.asarray(region, dtype=np.int64)])
        if fitted is None:
            continue
        normal, d = fitted
        # Preserve the existing cross-algorithm candidate-support rule.
        # build_model independently checks the 8 mm inlier threshold and the
        # minimum support, so a large but non-planar region cannot be accepted.
        model = common.build_model(points, normal, d, source_support=len(region))
        if model is not None:
            models.append(model)

    return common.merge_duplicate_models(models)


detect_region_growing = detect_planes
