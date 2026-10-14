from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import plane_detection_common as common


# ==========================================================
# Common matching rules
# ==========================================================

MATCH_NORMAL_THRESHOLD_DEG = 10.0
MATCH_OFFSET_THRESHOLD_M = 0.10


def load_sample(npz_file: Path | str):
    npz_file = Path(
        npz_file
    )

    data = np.load(
        npz_file
    )

    points = np.asarray(
        data["points"],
        dtype=np.float64,
    )

    labels = np.asarray(
        data["labels"],
        dtype=np.int32,
    )

    gt_file = npz_file.with_name(
        npz_file.stem
        + "_gt.json"
    )

    with open(
        gt_file,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    gt_planes = []

    for plane in metadata[
        "gt_planes"
    ]:
        normal = np.asarray(
            plane["normal"],
            dtype=np.float64,
        )

        d = float(
            plane["d"]
        )

        normal, d = (
            common.canonicalise_plane(
                normal,
                d,
            )
        )

        gt_planes.append(
            {
                "id":
                    int(
                        plane["id"]
                    ),

                "name":
                    plane["name"],

                "normal":
                    normal,

                "d":
                    d,
            }
        )

    return (
        points,
        labels,
        metadata,
        gt_planes,
    )


def match_planes(
    models,
    gt_planes,
    labels,
):
    num_detected = len(
        models
    )

    num_gt = len(
        gt_planes
    )

    if (
        num_detected == 0
        or num_gt == 0
    ):
        return {
            "matches": [],
            "matched_count": 0,
            "plane_precision": 0.0,
            "plane_recall": 0.0,
            "plane_f1": 0.0,
            "mean_normal_error_deg":
                np.nan,
            "mean_offset_error_m":
                np.nan,
            "mean_iou": 0.0,
        }

    cost = np.full(
        (
            num_detected,
            num_gt,
        ),
        1e6,
        dtype=np.float64,
    )

    pair_info = {}

    for i, model in enumerate(
        models
    ):
        for j, gt in enumerate(
            gt_planes
        ):
            n, d = (
                common.align_plane_to_reference(
                    model["normal"].copy(),
                    model["d"],
                    gt["normal"],
                )
            )

            angle = common.plane_angle_deg(
                n,
                gt["normal"],
            )

            offset_error = abs(
                d
                - gt["d"]
            )

            if (
                angle
                <= MATCH_NORMAL_THRESHOLD_DEG
                and offset_error
                <= MATCH_OFFSET_THRESHOLD_M
            ):
                pair_cost = (
                    angle
                    / MATCH_NORMAL_THRESHOLD_DEG
                    + offset_error
                    / MATCH_OFFSET_THRESHOLD_M
                )

                cost[
                    i,
                    j,
                ] = pair_cost

                gt_mask = (
                    labels
                    == gt["id"]
                )

                det_mask = model[
                    "mask"
                ]

                intersection = int(
                    np.sum(
                        gt_mask
                        & det_mask
                    )
                )

                union = int(
                    np.sum(
                        gt_mask
                        | det_mask
                    )
                )

                iou = (
                    intersection
                    / union
                    if union > 0
                    else 0.0
                )

                pair_info[
                    (
                        i,
                        j,
                    )
                ] = {
                    "normal_error_deg":
                        angle,

                    "offset_error_m":
                        offset_error,

                    "iou":
                        iou,
                }

    row_ind, col_ind = (
        linear_sum_assignment(
            cost
        )
    )

    matches = []

    for i, j in zip(
        row_ind,
        col_ind,
    ):
        if (
            cost[
                i,
                j,
            ]
            >= 1e5
        ):
            continue

        info = pair_info[
            (
                int(i),
                int(j),
            )
        ]

        matches.append(
            {
                "detected_index":
                    int(i),

                "gt_index":
                    int(j),

                **info,
            }
        )

    matched_count = len(
        matches
    )

    plane_precision = (
        matched_count
        / num_detected
        if num_detected > 0
        else 0.0
    )

    plane_recall = (
        matched_count
        / num_gt
        if num_gt > 0
        else 0.0
    )

    if (
        plane_precision
        + plane_recall
        > 0
    ):
        plane_f1 = (
            2.0
            * plane_precision
            * plane_recall
            / (
                plane_precision
                + plane_recall
            )
        )

    else:
        plane_f1 = 0.0

    if matched_count > 0:
        mean_normal_error = float(
            np.mean(
                [
                    match[
                        "normal_error_deg"
                    ]
                    for match
                    in matches
                ]
            )
        )

        mean_offset_error = float(
            np.mean(
                [
                    match[
                        "offset_error_m"
                    ]
                    for match
                    in matches
                ]
            )
        )

        mean_iou = float(
            np.mean(
                [
                    match["iou"]
                    for match
                    in matches
                ]
            )
        )

    else:
        mean_normal_error = np.nan
        mean_offset_error = np.nan
        mean_iou = 0.0

    return {
        "matches":
            matches,

        "matched_count":
            matched_count,

        "plane_precision":
            plane_precision,

        "plane_recall":
            plane_recall,

        "plane_f1":
            plane_f1,

        "mean_normal_error_deg":
            mean_normal_error,

        "mean_offset_error_m":
            mean_offset_error,

        "mean_iou":
            mean_iou,
    }
