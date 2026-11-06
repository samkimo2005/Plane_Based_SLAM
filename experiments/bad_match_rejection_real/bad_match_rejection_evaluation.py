from __future__ import annotations
import numpy as np

import bad_match_rejection_common as common


def evaluate_correspondences(
    keep_mask,
    gt_match_labels,
):
    """
    gt_match_labels:
        True  = correct correspondence
        False = bad correspondence
    """

    keep_mask = np.asarray(
        keep_mask,
        dtype=bool,
    )

    gt_correct = np.asarray(
        gt_match_labels,
        dtype=bool,
    )

    if len(keep_mask) != len(gt_correct):
        raise ValueError(
            "keep_mask and gt_match_labels "
            "must have the same length."
        )

    pred_bad = ~keep_mask
    gt_bad = ~gt_correct

    tp = int(np.sum(pred_bad & gt_bad))
    fp = int(np.sum(pred_bad & ~gt_bad))
    fn = int(np.sum(~pred_bad & gt_bad))
    tn = int(np.sum(~pred_bad & ~gt_bad))

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 1.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 1.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    correct_retention = (
        tn / (tn + fp)
        if tn + fp > 0
        else 1.0
    )

    accuracy = (
        (tp + tn) / len(gt_correct)
        if len(gt_correct) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "correct_retention": correct_retention,
    }


def evaluate_pose(
    planes_a,
    planes_b,
    assigned_pairs,
    keep_mask,
    R_gt,
    t_gt,
):
    pairs = np.asarray(
        assigned_pairs,
        dtype=int,
    )

    keep_mask = np.asarray(
        keep_mask,
        dtype=bool,
    )

    kept_pairs = pairs[keep_mask]

    if len(kept_pairs) < 3:
        return {
            "pose_success": False,
            "rotation_error_deg": np.nan,
            "translation_error_m": np.nan,
        }

    matched_a = planes_a[
        kept_pairs[:, 0]
    ]

    matched_b = planes_b[
        kept_pairs[:, 1]
    ]

    try:
        pose = common.estimate_pose_from_plane_matches(
            matched_a,
            matched_b,
        )
    except (ValueError, np.linalg.LinAlgError):
        pose = None

    if pose is None:
        return {
            "pose_success": False,
            "rotation_error_deg": np.nan,
            "translation_error_m": np.nan,
        }

    if isinstance(pose, dict):
        R_est = pose["R"]
        t_est = pose["t"]
    else:
        R_est = pose[0]
        t_est = pose[1]

    R_error = R_gt.T @ R_est

    cos_angle = (
        np.trace(R_error) - 1.0
    ) / 2.0

    cos_angle = np.clip(
        cos_angle,
        -1.0,
        1.0,
    )

    rotation_error_deg = np.degrees(
        np.arccos(cos_angle)
    )

    translation_error_m = np.linalg.norm(
        np.asarray(t_est)
        - np.asarray(t_gt)
    )

    return {
        "pose_success": True,
        "rotation_error_deg":
            float(rotation_error_deg),
        "translation_error_m":
            float(translation_error_m),
    }


def evaluate_result(
    planes_a,
    planes_b,
    assigned_pairs,
    keep_mask,
    gt_match_labels,
    R_gt,
    t_gt,
):
    correspondence = evaluate_correspondences(
        keep_mask,
        gt_match_labels,
    )

    pose = evaluate_pose(
        planes_a,
        planes_b,
        assigned_pairs,
        keep_mask,
        R_gt,
        t_gt,
    )

    return {
        **correspondence,
        **pose,
    }