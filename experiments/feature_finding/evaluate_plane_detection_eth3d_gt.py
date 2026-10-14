from __future__ import annotations

"""
evaluate_plane_detection_eth3d_gt.py

Evaluate the five Plane Detection methods on the TUM Freiburg1 XYZ frames
against the ETH3D-mesh-derived per-frame plane reference.

Place this file in:
    D:/PlaneBasedSLAM/experiments/feature_finding/

Expected detector modules in the same folder:
    iterative_ransac.py
    multi_object_ransac.py
    surface_normal_surfel.py
    region_growing.py
    hough_plane_detection.py
    plane_detection_common.py

Inputs:
    data/tum/plane_detection/rgbd_dataset_freiburg1_xyz/frames/frame_*.npz

    data/derived_gt/fr1_xyz/frame_plane_gt/frames/frame_*.npz

Outputs:
    results/plane_detection/tum_eth3d_gt/
        per_frame_results.csv
        matched_planes.csv
        overall_summary.csv
        evaluation_settings.json
        detector_parameter_snapshot.json

Important evaluation rule:
    The ETH3D-derived reference labels only part of the RGB-D point cloud.
    A predicted plane that lies mainly in the unlabelled/non-evaluable region
    is IGNORED rather than counted automatically as a false positive.

    Predictions with at least IGNORE_MIN_EVALUABLE_FRACTION of their support
    inside the evaluable GT region are scored.

Matching:
    - one-to-one Hungarian assignment
    - normal error <= MATCH_MAX_NORMAL_DEG
    - offset error <= MATCH_MAX_OFFSET_M
    - minimum supporting-point intersection
    - cost combines geometry + restricted support IoU

The reference is a derived evaluation reference, not official TUM plane GT.
"""

from pathlib import Path
from time import perf_counter
import csv
import json

import numpy as np
from scipy.optimize import linear_sum_assignment

import iterative_ransac
import multi_object_ransac
import surface_normal_surfel
import region_growing
import hough_plane_detection
import plane_detection_common as common


# ============================================================
# 1. PATHS
# ============================================================

SEQUENCE_NAME = "rgbd_dataset_freiburg1_xyz"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TUM_FRAMES_DIR = (
    PROJECT_ROOT
    / "data"
    / "tum"
    / "plane_detection"
    / SEQUENCE_NAME
    / "frames"
)

GT_FRAMES_DIR = (
    PROJECT_ROOT
    / "data"
    / "derived_gt"
    / "fr1_xyz"
    / "frame_plane_gt"
    / "frames"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "plane_detection"
    / "tum_eth3d_gt"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. EVALUATION SETTINGS
# ============================================================

# Prediction support must overlap the evaluable GT region by at least this
# fraction before it is allowed to become TP/FP.
#
# If below this value, the prediction is treated as "ignored" because the
# reference does not sufficiently cover the surface it detected.
IGNORE_MIN_EVALUABLE_FRACTION = 0.30

# Same geometric gates used by the original plane-detection evaluation logic.
MATCH_MAX_NORMAL_DEG = 10.0
MATCH_MAX_OFFSET_M = 0.10

# Prevent geometrically similar but spatially disjoint planes from matching.
MIN_MATCH_INTERSECTION_POINTS = 20

# Hungarian cost weights after candidate gating.
COST_WEIGHT_IOU = 0.50
COST_WEIGHT_NORMAL = 0.30
COST_WEIGHT_OFFSET = 0.20

# Large cost means "invalid candidate".
INVALID_COST = 1e6

# Reproducibility.
BASE_SEED = 20260919

# If desired, restrict evaluation to selected frame IDs.
# Example: FRAME_IDS = [3, 4, 5]
# None = use every frame for which both TUM and reference files exist.
FRAME_IDS = list(range(3, 79))


# ============================================================
# 3. DETECTOR DEFINITIONS
# ============================================================

DETECTORS = [
    (
        "Iterative RANSAC",
        iterative_ransac,
        iterative_ransac.detect_planes,
    ),
    (
        "Multi-Object RANSAC",
        multi_object_ransac,
        multi_object_ransac.detect_planes,
    ),
    (
        "Surface-Normal / Surfel",
        surface_normal_surfel,
        surface_normal_surfel.detect_planes,
    ),
    (
        "Region Growing",
        region_growing,
        region_growing.detect_planes,
    ),
    (
        "3D Hough",
        hough_plane_detection,
        hough_plane_detection.detect_planes,
    ),
]


# ============================================================
# 4. BASIC GEOMETRY
# ============================================================

def normal_error_deg(normal_a, normal_b):
    a = np.asarray(normal_a, dtype=np.float64)
    b = np.asarray(normal_b, dtype=np.float64)

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm < 1e-12 or b_norm < 1e-12:
        return float("inf")

    a = a / a_norm
    b = b / b_norm

    dot = np.clip(abs(float(np.dot(a, b))), 0.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def aligned_offset_error(det_normal, det_d, gt_normal, gt_d):
    """
    Plane sign ambiguity:
        (n, d) == (-n, -d)

    Align the detected plane sign to the GT normal before comparing d.
    """
    dn = np.asarray(det_normal, dtype=np.float64)
    gn = np.asarray(gt_normal, dtype=np.float64)

    if float(np.dot(dn, gn)) < 0.0:
        det_d = -float(det_d)

    return abs(float(det_d) - float(gt_d))


def mask_iou(mask_a, mask_b):
    intersection = int(np.sum(mask_a & mask_b))
    union = int(np.sum(mask_a | mask_b))

    if union == 0:
        return 0.0, intersection, union

    return float(intersection / union), intersection, union


def safe_precision(tp, fp):
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def safe_recall(tp, fn):
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def safe_f1(precision, recall):
    denom = precision + recall
    return (
        float(2.0 * precision * recall / denom)
        if denom > 0
        else 0.0
    )


# ============================================================
# 5. LOAD FRAME + GT
# ============================================================

def load_case(frame_path, gt_path):
    with np.load(frame_path, allow_pickle=False) as f:
        points = np.asarray(f["detector_points"], dtype=np.float64)

    with np.load(gt_path, allow_pickle=False) as g:
        gt_plane_ids = np.asarray(g["gt_plane_ids"], dtype=np.int32)
        gt_normals = np.asarray(g["gt_normals"], dtype=np.float64)
        gt_offsets = np.asarray(g["gt_offsets"], dtype=np.float64)
        gt_support_masks = np.asarray(g["gt_support_masks"], dtype=bool)

    if gt_support_masks.ndim != 2:
        raise ValueError(
            f"{gt_path.name}: gt_support_masks must have shape (G,N)."
        )

    if gt_support_masks.shape[1] != len(points):
        raise ValueError(
            f"{gt_path.name}: GT support length "
            f"{gt_support_masks.shape[1]} != detector point count {len(points)}."
        )

    if len(gt_plane_ids) != len(gt_support_masks):
        raise ValueError(
            f"{gt_path.name}: gt_plane_ids and gt_support_masks disagree."
        )

    if len(gt_normals) != len(gt_plane_ids):
        raise ValueError(
            f"{gt_path.name}: gt_normals and gt_plane_ids disagree."
        )

    if len(gt_offsets) != len(gt_plane_ids):
        raise ValueError(
            f"{gt_path.name}: gt_offsets and gt_plane_ids disagree."
        )

    return {
        "points": points,
        "gt_plane_ids": gt_plane_ids,
        "gt_normals": gt_normals,
        "gt_offsets": gt_offsets,
        "gt_support_masks": gt_support_masks,
    }


# ============================================================
# 6. PREDICTION CLASSIFICATION + MATCHING
# ============================================================

def prepare_predictions(models, evaluable_region):
    """
    Split predictions into:
      - scored predictions: enough support lies inside evaluable reference area
      - ignored predictions: mostly outside reference coverage
    """
    scored = []
    ignored = []

    for model_index, model in enumerate(models):
        mask = np.asarray(model["mask"], dtype=bool)

        support = int(np.sum(mask))

        if support <= 0:
            ignored.append(
                {
                    "model_index": model_index,
                    "reason": "zero_support",
                    "evaluable_fraction": 0.0,
                }
            )
            continue

        if len(mask) != len(evaluable_region):
            raise ValueError(
                "Detector model mask length does not match detector_points."
            )

        evaluable_support = int(np.sum(mask & evaluable_region))
        evaluable_fraction = float(evaluable_support / support)

        record = {
            "model_index": model_index,
            "model": model,
            "support": support,
            "evaluable_support": evaluable_support,
            "evaluable_fraction": evaluable_fraction,
        }

        if evaluable_fraction < IGNORE_MIN_EVALUABLE_FRACTION:
            record["reason"] = "mostly_unlabelled_region"
            ignored.append(record)
        else:
            scored.append(record)

    return scored, ignored


def match_predictions_to_gt(
    scored_predictions,
    gt_normals,
    gt_offsets,
    gt_masks,
    evaluable_region,
):
    pred_count = len(scored_predictions)
    gt_count = len(gt_masks)

    if pred_count == 0 or gt_count == 0:
        return [], set(), set()

    cost = np.full(
        (pred_count, gt_count),
        INVALID_COST,
        dtype=np.float64,
    )

    pair_data = {}

    for p_idx, pred_record in enumerate(scored_predictions):
        model = pred_record["model"]

        pred_mask_full = np.asarray(model["mask"], dtype=bool)

        # Restrict IoU scoring to the evaluable part of the reference.
        pred_mask_eval = pred_mask_full & evaluable_region

        for g_idx in range(gt_count):
            gt_mask = gt_masks[g_idx]

            n_err = normal_error_deg(
                model["normal"],
                gt_normals[g_idx],
            )

            d_err = aligned_offset_error(
                model["normal"],
                model["d"],
                gt_normals[g_idx],
                gt_offsets[g_idx],
            )

            iou, intersection, union = mask_iou(
                pred_mask_eval,
                gt_mask,
            )

            valid = (
                n_err <= MATCH_MAX_NORMAL_DEG
                and d_err <= MATCH_MAX_OFFSET_M
                and intersection >= MIN_MATCH_INTERSECTION_POINTS
            )

            pair_data[(p_idx, g_idx)] = {
                "normal_error_deg": n_err,
                "offset_error_m": d_err,
                "iou": iou,
                "intersection_points": intersection,
                "union_points": union,
                "valid": bool(valid),
            }

            if not valid:
                continue

            normalized_normal = n_err / MATCH_MAX_NORMAL_DEG
            normalized_offset = d_err / MATCH_MAX_OFFSET_M

            cost[p_idx, g_idx] = (
                COST_WEIGHT_IOU * (1.0 - iou)
                + COST_WEIGHT_NORMAL * normalized_normal
                + COST_WEIGHT_OFFSET * normalized_offset
            )

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    matched_pred = set()
    matched_gt = set()

    for p_idx, g_idx in zip(row_ind, col_ind):
        if cost[p_idx, g_idx] >= INVALID_COST:
            continue

        data = pair_data[(p_idx, g_idx)]

        matches.append(
            {
                "prediction_index": p_idx,
                "gt_index": g_idx,
                **data,
                "cost": float(cost[p_idx, g_idx]),
            }
        )

        matched_pred.add(int(p_idx))
        matched_gt.add(int(g_idx))

    return matches, matched_pred, matched_gt


# ============================================================
# 7. ONE DETECTOR / ONE FRAME
# ============================================================

def evaluate_one(
    method_name,
    detect_function,
    points,
    gt_plane_ids,
    gt_normals,
    gt_offsets,
    gt_masks,
    rng_seed,
):
    rng = np.random.default_rng(rng_seed)

    start = perf_counter()
    models = detect_function(points, rng)
    runtime_ms = (perf_counter() - start) * 1000.0

    if models is None:
        models = []

    # Union of only EVALUABLE GT plane supports.
    if len(gt_masks) > 0:
        evaluable_region = np.any(gt_masks, axis=0)
    else:
        evaluable_region = np.zeros(len(points), dtype=bool)

    scored_predictions, ignored_predictions = prepare_predictions(
        models,
        evaluable_region,
    )

    matches, matched_pred, matched_gt = match_predictions_to_gt(
        scored_predictions,
        gt_normals,
        gt_offsets,
        gt_masks,
        evaluable_region,
    )

    tp = len(matches)
    fp = len(scored_predictions) - tp
    fn = len(gt_masks) - tp

    precision = safe_precision(tp, fp)
    recall = safe_recall(tp, fn)
    f1 = safe_f1(precision, recall)

    matched_ious = [m["iou"] for m in matches]
    matched_normal_errors = [
        m["normal_error_deg"] for m in matches
    ]
    matched_offset_errors = [
        m["offset_error_m"] for m in matches
    ]

    scored_prediction_count = len(scored_predictions)
    gt_count = len(gt_masks)

    correct_plane_count = int(scored_prediction_count == gt_count)

    # Coverage diagnostics.
    evaluable_point_fraction = float(np.mean(evaluable_region))

    result = {
        "method": method_name,
        "gt_plane_count": int(gt_count),
        "raw_detected_plane_count": int(len(models)),
        "scored_prediction_count": int(scored_prediction_count),
        "ignored_prediction_count": int(len(ignored_predictions)),
        "matched_plane_count": int(tp),
        "false_plane_count": int(fp),
        "missed_plane_count": int(fn),
        "plane_precision": precision,
        "plane_recall": recall,
        "plane_f1": f1,
        "correct_plane_count": correct_plane_count,
        "mean_iou": (
            float(np.mean(matched_ious))
            if matched_ious
            else np.nan
        ),
        "mean_normal_error_deg": (
            float(np.mean(matched_normal_errors))
            if matched_normal_errors
            else np.nan
        ),
        "mean_offset_error_m": (
            float(np.mean(matched_offset_errors))
            if matched_offset_errors
            else np.nan
        ),
        "runtime_ms": float(runtime_ms),
        "evaluable_point_fraction": evaluable_point_fraction,
    }

    match_rows = []

    for m in matches:
        pred_record = scored_predictions[m["prediction_index"]]
        gt_index = m["gt_index"]

        match_rows.append(
            {
                "method": method_name,
                "prediction_index_within_scored":
                    int(m["prediction_index"]),
                "original_model_index":
                    int(pred_record["model_index"]),
                "gt_index":
                    int(gt_index),
                "gt_plane_id":
                    int(gt_plane_ids[gt_index]),
                "prediction_support":
                    int(pred_record["support"]),
                "prediction_evaluable_support":
                    int(pred_record["evaluable_support"]),
                "prediction_evaluable_fraction":
                    float(pred_record["evaluable_fraction"]),
                "iou":
                    float(m["iou"]),
                "normal_error_deg":
                    float(m["normal_error_deg"]),
                "offset_error_m":
                    float(m["offset_error_m"]),
                "intersection_points":
                    int(m["intersection_points"]),
                "union_points":
                    int(m["union_points"]),
                "assignment_cost":
                    float(m["cost"]),
            }
        )

    return result, match_rows


# ============================================================
# 8. FRAME DISCOVERY
# ============================================================

def frame_id_from_path(path):
    stem = path.stem  # frame_0000
    return int(stem.split("_")[-1])


def discover_cases():
    frame_files = {
        frame_id_from_path(p): p
        for p in TUM_FRAMES_DIR.glob("frame_*.npz")
    }

    gt_files = {
        frame_id_from_path(p): p
        for p in GT_FRAMES_DIR.glob("frame_*.npz")
    }

    common_ids = sorted(set(frame_files) & set(gt_files))

    if FRAME_IDS is not None:
        requested = set(int(x) for x in FRAME_IDS)
        common_ids = [x for x in common_ids if x in requested]

    if not common_ids:
        raise RuntimeError(
            "No matching TUM/GT frame pairs were found.\n"
            f"TUM: {TUM_FRAMES_DIR}\n"
            f"GT : {GT_FRAMES_DIR}"
        )

    return [
        (frame_id, frame_files[frame_id], gt_files[frame_id])
        for frame_id in common_ids
    ]


# ============================================================
# 9. PARAMETER SNAPSHOT
# ============================================================

def numeric_module_constants(module):
    result = {}

    for name in dir(module):
        if not name.isupper():
            continue

        value = getattr(module, name)

        if isinstance(value, (int, float, bool, str)):
            result[name] = value

    return result


def save_parameter_snapshot():
    snapshot = {
        "plane_detection_common": numeric_module_constants(common),
        "iterative_ransac": numeric_module_constants(iterative_ransac),
        "multi_object_ransac": numeric_module_constants(multi_object_ransac),
        "surface_normal_surfel": numeric_module_constants(surface_normal_surfel),
        "region_growing": numeric_module_constants(region_growing),
        "hough_plane_detection": numeric_module_constants(
            hough_plane_detection
        ),
    }

    path = OUTPUT_DIR / "detector_parameter_snapshot.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=4)

    return path


# ============================================================
# 10. CSV HELPERS
# ============================================================

def write_csv(path, rows):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def nanmean(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)

    if not np.any(finite):
        return np.nan

    return float(np.mean(arr[finite]))


# ============================================================
# 11. AGGREGATION
# ============================================================

def aggregate_method(rows):
    tp = int(sum(r["matched_plane_count"] for r in rows))
    fp = int(sum(r["false_plane_count"] for r in rows))
    fn = int(sum(r["missed_plane_count"] for r in rows))

    micro_precision = safe_precision(tp, fp)
    micro_recall = safe_recall(tp, fn)
    micro_f1 = safe_f1(micro_precision, micro_recall)

    return {
        "method": rows[0]["method"],
        "num_frames": len(rows),

        # Micro plane-level metrics.
        "total_tp": tp,
        "total_fp": fp,
        "total_fn": fn,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,

        # Mean frame-level metrics.
        "mean_frame_precision":
            float(np.mean([r["plane_precision"] for r in rows])),
        "mean_frame_recall":
            float(np.mean([r["plane_recall"] for r in rows])),
        "mean_frame_f1":
            float(np.mean([r["plane_f1"] for r in rows])),

        "correct_plane_count_rate":
            float(np.mean([r["correct_plane_count"] for r in rows])),

        "mean_iou":
            nanmean([r["mean_iou"] for r in rows]),

        "mean_normal_error_deg":
            nanmean([r["mean_normal_error_deg"] for r in rows]),

        "mean_offset_error_m":
            nanmean([r["mean_offset_error_m"] for r in rows]),

        "mean_runtime_ms":
            float(np.mean([r["runtime_ms"] for r in rows])),

        "median_runtime_ms":
            float(np.median([r["runtime_ms"] for r in rows])),

        "mean_raw_detected_planes":
            float(np.mean([r["raw_detected_plane_count"] for r in rows])),

        "mean_scored_predictions":
            float(np.mean([r["scored_prediction_count"] for r in rows])),

        "mean_ignored_predictions":
            float(np.mean([r["ignored_prediction_count"] for r in rows])),

        "mean_false_plane_count":
            float(np.mean([r["false_plane_count"] for r in rows])),

        "mean_missed_plane_count":
            float(np.mean([r["missed_plane_count"] for r in rows])),

        "mean_gt_plane_count":
            float(np.mean([r["gt_plane_count"] for r in rows])),

        "mean_evaluable_point_fraction":
            float(np.mean([r["evaluable_point_fraction"] for r in rows])),
    }


# ============================================================
# 12. MAIN
# ============================================================

def main():
    cases = discover_cases()

    parameter_snapshot_path = save_parameter_snapshot()

    settings = {
        "sequence": SEQUENCE_NAME,
        "reference":
            "ETH3D mesh-derived per-frame plane reference",
        "reference_is_not_official_tum_plane_gt": True,
        "ignore_min_evaluable_fraction":
            IGNORE_MIN_EVALUABLE_FRACTION,
        "match_max_normal_deg":
            MATCH_MAX_NORMAL_DEG,
        "match_max_offset_m":
            MATCH_MAX_OFFSET_M,
        "min_match_intersection_points":
            MIN_MATCH_INTERSECTION_POINTS,
        "cost_weights": {
            "iou": COST_WEIGHT_IOU,
            "normal": COST_WEIGHT_NORMAL,
            "offset": COST_WEIGHT_OFFSET,
        },
        "base_seed": BASE_SEED,
        "number_of_frame_pairs": len(cases),
        "common_final_distance_threshold_m":
            float(common.FINAL_DISTANCE_THRESHOLD),
        "common_min_plane_support":
            int(common.MIN_PLANE_SUPPORT),
        "common_max_planes":
            int(common.MAX_PLANES),
    }

    with open(
        OUTPUT_DIR / "evaluation_settings.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(settings, f, indent=4)

    print()
    print("=" * 96)
    print("TUM PLANE DETECTION EVALUATION — ETH3D-DERIVED GT")
    print("=" * 96)
    print(f"Frame pairs      : {len(cases)}")
    print(f"TUM frames       : {TUM_FRAMES_DIR}")
    print(f"GT frames        : {GT_FRAMES_DIR}")
    print(
        f"Ignore threshold : "
        f"{IGNORE_MIN_EVALUABLE_FRACTION:.2f} "
        f"prediction support in evaluable region"
    )
    print(
        f"Matching gates   : "
        f"{MATCH_MAX_NORMAL_DEG:.1f} deg, "
        f"{MATCH_MAX_OFFSET_M:.3f} m, "
        f"{MIN_MATCH_INTERSECTION_POINTS} shared points"
    )
    print()

    per_frame_rows = []
    all_match_rows = []

    for case_number, (frame_id, frame_path, gt_path) in enumerate(
        cases,
        start=1,
    ):
        case = load_case(frame_path, gt_path)

        points = case["points"]
        gt_plane_ids = case["gt_plane_ids"]
        gt_normals = case["gt_normals"]
        gt_offsets = case["gt_offsets"]
        gt_masks = case["gt_support_masks"]

        print(
            f"[{case_number:02d}/{len(cases):02d}] "
            f"{frame_path.name} "
            f"GT={len(gt_masks)}"
        )

        for method_index, (
            method_name,
            module,
            detect_function,
        ) in enumerate(DETECTORS):
            seed = (
                BASE_SEED
                + frame_id * 100
                + method_index
            )

            result, match_rows = evaluate_one(
                method_name,
                detect_function,
                points,
                gt_plane_ids,
                gt_normals,
                gt_offsets,
                gt_masks,
                seed,
            )

            result = {
                "frame_id": int(frame_id),
                "frame": frame_path.name,
                **result,
            }

            per_frame_rows.append(result)

            for row in match_rows:
                all_match_rows.append(
                    {
                        "frame_id": int(frame_id),
                        "frame": frame_path.name,
                        **row,
                    }
                )

            print(
                f"    {method_name:<24} "
                f"det={result['raw_detected_plane_count']:2d} "
                f"score={result['scored_prediction_count']:2d} "
                f"ign={result['ignored_prediction_count']:2d} "
                f"TP/FP/FN="
                f"{result['matched_plane_count']}/"
                f"{result['false_plane_count']}/"
                f"{result['missed_plane_count']} "
                f"F1={result['plane_f1']:.3f} "
                f"IoU="
                f"{result['mean_iou']:.3f} "
                if np.isfinite(result["mean_iou"])
                else
                f"    {method_name:<24} "
                f"det={result['raw_detected_plane_count']:2d} "
                f"score={result['scored_prediction_count']:2d} "
                f"ign={result['ignored_prediction_count']:2d} "
                f"TP/FP/FN="
                f"{result['matched_plane_count']}/"
                f"{result['false_plane_count']}/"
                f"{result['missed_plane_count']} "
                f"F1={result['plane_f1']:.3f} "
                f"IoU=nan "
            )

    # Save raw outputs.
    per_frame_csv = OUTPUT_DIR / "per_frame_results.csv"
    matched_csv = OUTPUT_DIR / "matched_planes.csv"

    write_csv(per_frame_csv, per_frame_rows)
    write_csv(matched_csv, all_match_rows)

    # Aggregate by method.
    overall_rows = []

    for method_name, _, _ in DETECTORS:
        method_rows = [
            r for r in per_frame_rows
            if r["method"] == method_name
        ]

        overall_rows.append(
            aggregate_method(method_rows)
        )

    # Sort for convenient reading by micro F1, but this is only an output
    # table ordering — it does not alter the evaluation.
    overall_rows.sort(
        key=lambda r: r["micro_f1"],
        reverse=True,
    )

    overall_csv = OUTPUT_DIR / "overall_summary.csv"
    write_csv(overall_csv, overall_rows)

    print()
    print("=" * 96)
    print("OVERALL SUMMARY")
    print("=" * 96)

    for row in overall_rows:
        print(
            f"{row['method']:<24} "
            f"P={row['micro_precision']:.3f} "
            f"R={row['micro_recall']:.3f} "
            f"F1={row['micro_f1']:.3f} "
            f"IoU={row['mean_iou']:.3f} "
            f"Nerr={row['mean_normal_error_deg']:.2f}deg "
            f"Derr={row['mean_offset_error_m']*1000:.1f}mm "
            f"Count={row['correct_plane_count_rate']*100:.1f}% "
            f"Runtime={row['mean_runtime_ms']:.1f}ms "
            f"Ignored={row['mean_ignored_predictions']:.2f}/frame"
        )

    gt_counts_by_frame = {}
    for r in per_frame_rows:
        gt_counts_by_frame[r["frame_id"]] = r["gt_plane_count"]

    frames_above_detector_cap = sum(
        count > common.MAX_PLANES
        for count in gt_counts_by_frame.values()
    )

    print()
    print(f"Saved per-frame results : {per_frame_csv}")
    print(f"Saved matched planes    : {matched_csv}")
    print(f"Saved overall summary   : {overall_csv}")
    print(f"Saved parameter snapshot: {parameter_snapshot_path}")
    print()
    print(
        f"Frames with GT plane count > detector MAX_PLANES "
        f"({common.MAX_PLANES}): {frames_above_detector_cap}"
    )
    print()
    print(
        "Evaluation complete. Predictions mainly outside the evaluable "
        "reference region were ignored rather than automatically counted "
        "as false positives."
    )


if __name__ == "__main__":
    main()
