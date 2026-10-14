from __future__ import annotations
import numpy as np

import bad_match_rejection_common as common


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
    offset_threshold=OFFSET_THRESHOLD,
):
    pairs = np.asarray(assigned_pairs, dtype=int)

    if len(pairs) < 3:
        return _failure(pairs, "too_few_pairs")

    try:
        pose = _estimate_pose(
            planes_a,
            planes_b,
            pairs,
        )
    except (ValueError, np.linalg.LinAlgError):
        return _failure(
            pairs,
            "pose_estimation_failed",
        )

    if pose is None:
        return _failure(
            pairs,
            "pose_estimation_failed",
        )

    R, t = pose

    residuals = []

    for ia, ib in pairs:
        n_a = planes_a[ia][:3]
        d_a = float(planes_a[ia][3])

        n_b = planes_b[ib][:3]
        d_b = float(planes_b[ib][3])

        predicted_n = R @ n_a

        predicted_d = (
            d_a
            - predicted_n @ t
        )

        # Plane sign ambiguity.
        if np.dot(predicted_n, n_b) < 0:
            predicted_n = -predicted_n
            predicted_d = -predicted_d

        residuals.append(
            abs(d_b - predicted_d)
        )

    residuals = np.asarray(residuals)

    keep_mask = (
        residuals < offset_threshold
    )

    return {
        "kept_pairs": pairs[keep_mask],
        "rejected_pairs": pairs[~keep_mask],
        "keep_mask": keep_mask,
        "success": keep_mask.sum() >= 3,
        "diagnostics": {
            "offset_residual_m": residuals,
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
