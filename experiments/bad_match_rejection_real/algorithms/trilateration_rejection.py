from __future__ import annotations

from itertools import combinations
import numpy as np

import bad_match_rejection_common as common


CLUSTER_RADIUS = 0.15
MIN_SUPPORT = 0.50
MAX_SUBSETS = 200


def _estimate_pose(planes_a, planes_b, pairs):
    matched_a = planes_a[pairs[:, 0]]
    matched_b = planes_b[pairs[:, 1]]

    pose = common.estimate_pose_from_plane_matches(
        matched_a,
        matched_b,
    )

    if pose is None:
        return None

    if isinstance(pose, dict):
        return pose["R"], pose["t"]

    return pose[0], pose[1]


def reject_bad_matches(
    planes_a,
    planes_b,
    assigned_pairs,
):
    pairs = np.asarray(assigned_pairs, dtype=int)

    if len(pairs) < 3:
        return _failure(
            pairs,
            "too_few_pairs",
        )

    subsets = list(
        combinations(
            range(len(pairs)),
            3,
        )
    )

    # Keep runtime bounded.
    if len(subsets) > MAX_SUBSETS:
        rng = np.random.default_rng(20260927)

        ids = rng.choice(
            len(subsets),
            size=MAX_SUBSETS,
            replace=False,
        )

        subsets = [
            subsets[i]
            for i in ids
        ]

    translations = []
    valid_subsets = []

    for subset in subsets:

        subset_pairs = pairs[
            list(subset)
        ]

        try:
            pose = _estimate_pose(
                planes_a,
                planes_b,
                subset_pairs,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue

        if pose is None:
            continue

        _, t = pose

        translations.append(
            np.asarray(t, dtype=float)
        )

        valid_subsets.append(subset)

    if len(translations) == 0:
        return _failure(
            pairs,
            "no_valid_subset_solutions",
        )

    translations = np.asarray(
        translations
    )

    # Pairwise distance in translation-solution space.
    distances = np.linalg.norm(
        translations[:, None, :]
        - translations[None, :, :],
        axis=2,
    )

    neighbours = (
        distances < CLUSTER_RADIUS
    )

    cluster_size = neighbours.sum(
        axis=1
    )

    centre = int(
        np.argmax(cluster_size)
    )

    dominant_cluster = (
        neighbours[centre]
    )

    total_support = np.zeros(
        len(pairs)
    )

    cluster_support = np.zeros(
        len(pairs)
    )

    for k, subset in enumerate(
        valid_subsets
    ):
        for index in subset:

            total_support[index] += 1

            if dominant_cluster[k]:
                cluster_support[index] += 1

    support_ratio = np.divide(
        cluster_support,
        total_support,
        out=np.zeros_like(
            cluster_support
        ),
        where=total_support > 0,
    )

    keep_mask = (
        support_ratio >= MIN_SUPPORT
    )

    return {
        "kept_pairs": pairs[keep_mask],
        "rejected_pairs": pairs[~keep_mask],
        "keep_mask": keep_mask,
        "success": keep_mask.sum() >= 3,
        "diagnostics": {
            "valid_subsets":
                len(valid_subsets),
            "cluster_size":
                int(dominant_cluster.sum()),
            "support_ratio":
                support_ratio,
        },
    }


def _failure(pairs, reason):
    mask = np.zeros(len(pairs), dtype=bool)

    return {
        "kept_pairs": pairs[mask],
        "rejected_pairs": pairs,
        "keep_mask": mask,
        "success": False,
        "diagnostics": {
            "reason": reason,
        },
    }