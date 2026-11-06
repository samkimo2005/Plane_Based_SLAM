from __future__ import annotations
import numpy as np

import bad_match_rejection_common as common


NORMAL_THRESHOLD_DEG = 8.0


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
    normal_threshold_deg=NORMAL_THRESHOLD_DEG,
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

    R, _ = pose

    residuals = []

    for ia, ib in pairs:
        predicted_normal = (
            R @ planes_a[ia][:3]
        )

        angle = common.normal_angle_deg(
            predicted_normal,
            planes_b[ib][:3],
        )

        residuals.append(angle)

    residuals = np.asarray(residuals)

    keep_mask = (
        residuals < normal_threshold_deg
    )

    return {
        "kept_pairs": pairs[keep_mask],
        "rejected_pairs": pairs[~keep_mask],
        "keep_mask": keep_mask,
        "success": keep_mask.sum() >= 3,
        "diagnostics": {
            "normal_residual_deg": residuals,
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