from __future__ import annotations
import numpy as np

import bad_match_rejection_common as common


RANSAC_ITERATIONS = 200
NORMAL_THRESHOLD_DEG = 8.0
OFFSET_THRESHOLD = 0.08


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
    rng=None,
):
    pairs = np.asarray(assigned_pairs, dtype=int)

    if rng is None:
        rng = np.random.default_rng(20260927)

    if len(pairs) < 3:
        return _failure(pairs, "too_few_pairs")

    best_mask = np.zeros(
        len(pairs),
        dtype=bool,
    )

    best_error = np.inf

    for _ in range(RANSAC_ITERATIONS):

        ids = rng.choice(
            len(pairs),
            size=3,
            replace=False,
        )

        sample = pairs[ids]

        try:
            pose = _estimate_pose(
                planes_a,
                planes_b,
                sample,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue

        if pose is None:
            continue

        R, t = pose

        mask = np.zeros(
            len(pairs),
            dtype=bool,
        )

        total_error = 0.0

        for k, (ia, ib) in enumerate(pairs):
            n_a = planes_a[ia][:3]
            d_a = float(planes_a[ia][3])

            n_b = planes_b[ib][:3]
            d_b = float(planes_b[ib][3])

            predicted_n = R @ n_a
            predicted_d = (
                d_a - predicted_n @ t
            )

            if np.dot(predicted_n, n_b) < 0:
                predicted_n = -predicted_n
                predicted_d = -predicted_d

            angle = common.normal_angle_deg(
                predicted_n,
                n_b,
            )

            offset = abs(
                d_b - predicted_d
            )

            mask[k] = (
                angle < NORMAL_THRESHOLD_DEG
                and
                offset < OFFSET_THRESHOLD
            )

            if mask[k]:
                total_error += (
                    angle
                    + 10.0 * offset
                )

        better = (
            mask.sum() > best_mask.sum()
            or (
                mask.sum() == best_mask.sum()
                and total_error < best_error
            )
        )

        if better:
            best_mask = mask
            best_error = total_error

    if best_mask.sum() < 3:
        return _failure(
            pairs,
            "no_valid_consensus",
        )

    return {
        "kept_pairs": pairs[best_mask],
        "rejected_pairs": pairs[~best_mask],
        "keep_mask": best_mask,
        "success": True,
        "diagnostics": {
            "consensus_size":
                int(best_mask.sum()),
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