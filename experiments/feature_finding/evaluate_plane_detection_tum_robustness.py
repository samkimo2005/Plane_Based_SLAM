# evaluate_plane_detection_tum_robustness_v2.py
#
# Controlled robustness evaluation for the five plane detectors.
#
# Key correction in v2:
#   IoU is computed only inside the evaluable reference region L:
#
#       pred_eval = pred_mask & evaluable_region
#
#       IoU = |pred_eval ∩ gt_mask| / |pred_eval ∪ gt_mask|
#
#   This makes the IoU calculation consistent with the partial-reference
#   ignore-region protocol. Prediction support outside L is not used to
#   penalise segmentation IoU.
#
# The perturbation is a controlled stress test on real TUM scene geometry.
# It is NOT claimed to reproduce the exact physical noise distribution of
# the RealSense D435i or the TUM sensor.

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from iterative_ransac import detect_planes as iterative_ransac
from multi_object_ransac import detect_planes as multi_object_ransac
from surface_normal_surfel import detect_planes as surface_normal_surfel
from region_growing import detect_planes as region_growing
from hough_plane_detection import detect_planes as hough_plane_detection


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TUM_FRAME_DIR = (
    PROJECT_ROOT
    / "data"
    / "tum"
    / "plane_detection"
    / "rgbd_dataset_freiburg1_xyz"
    / "frames"
)

GT_FRAME_DIR = (
    PROJECT_ROOT
    / "data"
    / "derived_gt"
    / "fr1_xyz"
    / "frame_plane_gt"
    / "frames"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "plane_detection"
    / "tum_robustness_v2"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Frame split
# ============================================================

# Use [0, 1, 2] only while choosing/fixing perturbation levels.
# FINAL held-out evaluation:
FRAME_IDS = list(range(3, 79))


# ============================================================
# 3. Controlled perturbation conditions
# ============================================================

CONDITIONS = {
    "clean": {
        "sigma_z_m": 0.000,
        "outlier_ratio": 0.00,
    },
    "moderate": {
        "sigma_z_m": 0.010,     # 10 mm
        "outlier_ratio": 0.10,  # 10%
    },
    "hard": {
        "sigma_z_m": 0.020,     # 20 mm
        "outlier_ratio": 0.20,  # 20%
    },
}

# Gross depth perturbation for selected outlier points.
# These are project-specific controlled stress-test values.
OUTLIER_MIN_M = 0.10
OUTLIER_MAX_M = 0.30

MIN_VALID_DEPTH_M = 0.10


# ============================================================
# 4. Evaluation settings
# ============================================================

# Prediction is ignored if less than 30% of its support lies in
# the evaluable planar-reference region.
IGNORE_COVERAGE_THRESHOLD = 0.30

# Same geometric eligibility gates as the main TUM evaluator.
MAX_NORMAL_ERROR_DEG = 10.0
MAX_OFFSET_ERROR_M = 0.10
MIN_INTERSECTION_POINTS = 20

# Evaluation matching cost.
W_IOU = 0.50
W_NORMAL = 0.30
W_OFFSET = 0.20

BIG_COST = 1.0e9


# ============================================================
# 5. Detectors
# ============================================================

DETECTORS = {
    "Iterative RANSAC": iterative_ransac,
    "Multi-Object RANSAC": multi_object_ransac,
    "Surface-Normal / Surfel": surface_normal_surfel,
    "Region Growing": region_growing,
    "3D Hough": hough_plane_detection,
}


# ============================================================
# 6. Utility functions
# ============================================================

def safe_divide(num, den):
    return float(num / den) if den != 0 else 0.0


def perturb_points(points, sigma_z_m, outlier_ratio, rng):
    """
    Controlled ray-preserving depth perturbation.

    For p=(x,y,z), change z to z' and preserve the original image ray:

        p' = (z'/z) p

    Advantages:
      - point count is unchanged
      - point ordering is unchanged
      - gt_detector_point_labels remain index-compatible

    This is a controlled stress test, not a physical sensor-noise model.
    """

    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got {points.shape}")

    perturbed = points.copy()

    z_original = points[:, 2].copy()

    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(z_original)
        & (z_original > MIN_VALID_DEPTH_M)
    )

    valid_indices = np.flatnonzero(valid)

    if len(valid_indices) == 0:
        return perturbed

    # A. Additive Gaussian depth noise
    if sigma_z_m > 0.0:
        dz = rng.normal(
            loc=0.0,
            scale=sigma_z_m,
            size=len(valid_indices),
        )

        z_new = z_original[valid_indices] + dz
        z_new = np.maximum(z_new, MIN_VALID_DEPTH_M)

        scale = z_new / z_original[valid_indices]
        perturbed[valid_indices] *= scale[:, None]

    # B. Gross depth outliers
    if outlier_ratio > 0.0:
        n_outliers = int(round(outlier_ratio * len(valid_indices)))

        if n_outliers > 0:
            outlier_indices = rng.choice(
                valid_indices,
                size=n_outliers,
                replace=False,
            )

            current_z = perturbed[outlier_indices, 2]

            magnitude = rng.uniform(
                OUTLIER_MIN_M,
                OUTLIER_MAX_M,
                size=n_outliers,
            )

            direction = rng.choice(
                [-1.0, 1.0],
                size=n_outliers,
            )

            z_outlier = current_z + direction * magnitude
            z_outlier = np.maximum(z_outlier, MIN_VALID_DEPTH_M)

            scale = z_outlier / current_z
            perturbed[outlier_indices] *= scale[:, None]

    return perturbed


def normalise_plane(normal, d):
    """
    Return plane parameters with ||n|| = 1.

    Important:
        If n is scaled, d must be scaled by the same factor.
    """

    normal = np.asarray(normal, dtype=np.float64).reshape(3)
    d = float(d)

    norm = np.linalg.norm(normal)

    if norm < 1e-12:
        raise ValueError("Zero-length plane normal.")

    return normal / norm, d / norm


def plane_errors(pred_normal, pred_d, gt_normal, gt_d):
    """
    Normal and offset errors, handling plane-sign ambiguity:

        (n, d) == (-n, -d)
    """

    pn, pd = normalise_plane(pred_normal, pred_d)
    gn, gd = normalise_plane(gt_normal, gt_d)

    if np.dot(pn, gn) < 0.0:
        pn = -pn
        pd = -pd

    dot = np.clip(np.dot(pn, gn), -1.0, 1.0)

    normal_error_deg = np.degrees(np.arccos(dot))
    offset_error_m = abs(pd - gd)

    return float(normal_error_deg), float(offset_error_m)


# ============================================================
# 7. Ground-truth masks
# ============================================================

def build_gt_masks(gt_data, n_points):
    """
    Build per-plane detector-point masks and the combined evaluable region L.
    """

    gt_plane_ids = np.asarray(
        gt_data["gt_plane_ids"]
    ).astype(int).reshape(-1)

    gt_normals = np.asarray(
        gt_data["gt_normals"],
        dtype=np.float64,
    )

    gt_offsets = np.asarray(
        gt_data["gt_offsets"],
        dtype=np.float64,
    ).reshape(-1)

    labels = np.asarray(
        gt_data["gt_detector_point_labels"]
    ).reshape(-1)

    if len(labels) != n_points:
        raise ValueError(
            "gt_detector_point_labels length does not match detector_points: "
            f"{len(labels)} vs {n_points}"
        )

    # Preferred interpretation:
    # labels store persistent global plane IDs.
    gt_masks = [
        np.asarray(labels == plane_id, dtype=bool)
        for plane_id in gt_plane_ids
    ]

    # Fallback only if the stored labels appear to use local 0..N-1 IDs.
    total_labelled = sum(np.count_nonzero(mask) for mask in gt_masks)

    if total_labelled == 0 and len(gt_plane_ids) > 0:
        gt_masks = [
            np.asarray(labels == local_id, dtype=bool)
            for local_id in range(len(gt_plane_ids))
        ]

    if len(gt_masks) > 0:
        evaluable_region = np.logical_or.reduce(gt_masks)
    else:
        evaluable_region = np.zeros(n_points, dtype=bool)

    return (
        gt_plane_ids,
        gt_normals,
        gt_offsets,
        gt_masks,
        evaluable_region,
    )


# ============================================================
# 8. Prediction/reference evaluation
# ============================================================

def evaluate_predictions(models, gt_data, n_points):
    """
    Evaluate one detector output for one frame.

    IMPORTANT v2 definition:
      1. Use coverage ratio to decide whether a prediction is evaluable.
      2. Once evaluable, restrict the prediction mask to L before IoU:

             pred_eval_mask = pred_mask & evaluable_region

         Therefore unlabelled support outside L is not added to the IoU union.
    """

    (
        gt_ids,
        gt_normals,
        gt_offsets,
        gt_masks,
        evaluable_region,
    ) = build_gt_masks(gt_data, n_points)

    n_gt = len(gt_ids)

    eligible_predictions = []
    ignored_count = 0

    # --------------------------------------------------------
    # 8A. Ignore-region filtering
    # --------------------------------------------------------

    for pred_index, model in enumerate(models):
        pred_mask = np.asarray(
            model["mask"],
            dtype=bool,
        ).reshape(-1)

        if len(pred_mask) != n_points:
            raise ValueError(
                f"Prediction mask length {len(pred_mask)} != {n_points}"
            )

        raw_support = int(np.count_nonzero(pred_mask))

        if raw_support == 0:
            ignored_count += 1
            continue

        evaluable_support = int(
            np.count_nonzero(pred_mask & evaluable_region)
        )

        coverage_ratio = evaluable_support / raw_support

        if coverage_ratio < IGNORE_COVERAGE_THRESHOLD:
            ignored_count += 1
            continue

        pred_eval_mask = pred_mask & evaluable_region

        eligible_predictions.append({
            "original_pred_index": pred_index,
            "model": model,
            "coverage_ratio": float(coverage_ratio),
            "raw_support": raw_support,
            "evaluable_support": evaluable_support,
            "pred_eval_mask": pred_eval_mask,
            "pred_raw_mask": pred_mask,
        })

    n_pred = len(eligible_predictions)

    # --------------------------------------------------------
    # 8B. Empty cases
    # --------------------------------------------------------

    if n_pred == 0:
        tp = 0
        fp = 0
        fn = n_gt

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "raw_predictions": len(models),
            "eligible_predictions": 0,
            "ignored_predictions": ignored_count,
        }, []

    if n_gt == 0:
        tp = 0
        fp = n_pred
        fn = 0

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": 0.0,
            "recall": np.nan,
            "f1": 0.0,
            "raw_predictions": len(models),
            "eligible_predictions": n_pred,
            "ignored_predictions": ignored_count,
        }, []

    # --------------------------------------------------------
    # 8C. Pairwise eligibility + matching cost
    # --------------------------------------------------------

    cost_matrix = np.full(
        (n_pred, n_gt),
        BIG_COST,
        dtype=np.float64,
    )

    pair_metrics = {}

    for pi, pred_info in enumerate(eligible_predictions):
        model = pred_info["model"]

        pred_eval_mask = pred_info["pred_eval_mask"]
        pred_raw_mask = pred_info["pred_raw_mask"]

        pred_normal = np.asarray(
            model["normal"],
            dtype=np.float64,
        )

        pred_d = float(model["d"])

        for gi in range(n_gt):
            gt_mask = gt_masks[gi]

            # Intersection is evaluated only inside L.
            intersection = int(
                np.count_nonzero(pred_eval_mask & gt_mask)
            )

            if intersection < MIN_INTERSECTION_POINTS:
                continue

            # CORRECTED IoU:
            # prediction support outside the evaluable region is excluded.
            eval_union = int(
                np.count_nonzero(pred_eval_mask | gt_mask)
            )

            if eval_union == 0:
                continue

            evaluable_iou = intersection / eval_union

            # Diagnostic only: old/raw definition.
            raw_union = int(
                np.count_nonzero(pred_raw_mask | gt_mask)
            )

            raw_iou = (
                intersection / raw_union
                if raw_union > 0
                else 0.0
            )

            normal_error_deg, offset_error_m = plane_errors(
                pred_normal,
                pred_d,
                gt_normals[gi],
                gt_offsets[gi],
            )

            if normal_error_deg > MAX_NORMAL_ERROR_DEG:
                continue

            if offset_error_m > MAX_OFFSET_ERROR_M:
                continue

            cost = (
                W_IOU * (1.0 - evaluable_iou)
                + W_NORMAL
                * (normal_error_deg / MAX_NORMAL_ERROR_DEG)
                + W_OFFSET
                * (offset_error_m / MAX_OFFSET_ERROR_M)
            )

            cost_matrix[pi, gi] = cost

            pair_metrics[(pi, gi)] = {
                "pred_index": int(pred_info["original_pred_index"]),
                "gt_plane_id": int(gt_ids[gi]),
                "coverage_ratio": float(pred_info["coverage_ratio"]),
                "raw_support": int(pred_info["raw_support"]),
                "evaluable_support": int(pred_info["evaluable_support"]),
                "intersection_points": int(intersection),
                "iou": float(evaluable_iou),
                "raw_iou_diagnostic": float(raw_iou),
                "normal_error_deg": float(normal_error_deg),
                "offset_error_m": float(offset_error_m),
                "cost": float(cost),
            }

    # --------------------------------------------------------
    # 8D. Hungarian one-to-one evaluation matching
    # --------------------------------------------------------

    rows, cols = linear_sum_assignment(cost_matrix)

    matches = []

    for row, col in zip(rows, cols):
        if cost_matrix[row, col] >= BIG_COST / 2.0:
            continue

        metric = pair_metrics.get((row, col))

        if metric is not None:
            matches.append(metric)

    tp = len(matches)
    fp = n_pred - tp
    fn = n_gt - tp

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "raw_predictions": len(models),
        "eligible_predictions": n_pred,
        "ignored_predictions": ignored_count,
    }, matches


# ============================================================
# 9. Load one frame
# ============================================================

def load_frame(frame_id):
    frame_name = f"frame_{frame_id:04d}.npz"

    frame_path = TUM_FRAME_DIR / frame_name
    gt_path = GT_FRAME_DIR / frame_name

    if not frame_path.exists():
        raise FileNotFoundError(frame_path)

    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    frame_data = np.load(frame_path, allow_pickle=False)
    gt_data = np.load(gt_path, allow_pickle=False)

    points = np.asarray(
        frame_data["detector_points"],
        dtype=np.float64,
    )

    return points, gt_data


# ============================================================
# 10. Save frozen settings
# ============================================================

def save_settings():
    settings = {
        "frame_ids": FRAME_IDS,
        "conditions": CONDITIONS,
        "outlier_min_m": OUTLIER_MIN_M,
        "outlier_max_m": OUTLIER_MAX_M,
        "min_valid_depth_m": MIN_VALID_DEPTH_M,
        "ignore_coverage_threshold": IGNORE_COVERAGE_THRESHOLD,
        "max_normal_error_deg": MAX_NORMAL_ERROR_DEG,
        "max_offset_error_m": MAX_OFFSET_ERROR_M,
        "min_intersection_points": MIN_INTERSECTION_POINTS,
        "matching_cost_weights": {
            "iou": W_IOU,
            "normal": W_NORMAL,
            "offset": W_OFFSET,
        },
        "iou_definition": (
            "IoU computed using pred_mask intersected with evaluable_region L "
            "before intersection/union with the GT plane mask."
        ),
        "perturbation_note": (
            "Controlled stress test only; not claimed to reproduce exact "
            "physical RGB-D sensor noise."
        ),
    }

    path = RESULT_DIR / "evaluation_settings.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

    return path


# ============================================================
# 11. Main experiment
# ============================================================

def main():
    print("Plane Detection TUM Robustness Evaluation v2")
    print(f"Frames: {FRAME_IDS[0]}..{FRAME_IDS[-1]} ({len(FRAME_IDS)} frames)")
    print(f"Results: {RESULT_DIR}")
    print()

    settings_path = save_settings()

    per_frame_rows = []
    match_rows = []

    condition_names = list(CONDITIONS.keys())

    for frame_id in FRAME_IDS:
        print(f"\nFrame {frame_id:04d}")

        clean_points, gt_data = load_frame(frame_id)

        for condition_index, condition_name in enumerate(condition_names):
            condition = CONDITIONS[condition_name]

            # Same corrupted cloud for all five detectors.
            perturb_rng = np.random.default_rng(
                100000
                + frame_id * 100
                + condition_index
            )

            test_points = perturb_points(
                clean_points,
                sigma_z_m=condition["sigma_z_m"],
                outlier_ratio=condition["outlier_ratio"],
                rng=perturb_rng,
            )

            print(
                f"  {condition_name:9s} "
                f"(sigma={condition['sigma_z_m']*1000:.0f} mm, "
                f"outliers={condition['outlier_ratio']*100:.0f}%)"
            )

            for method_index, (method_name, detector) in enumerate(
                DETECTORS.items()
            ):
                # Same detector RNG for the same frame/method across
                # clean/moderate/hard conditions.
                detector_rng = np.random.default_rng(
                    500000
                    + frame_id * 100
                    + method_index
                )

                start = time.perf_counter()

                models = detector(
                    test_points,
                    detector_rng,
                )

                runtime_ms = (
                    time.perf_counter() - start
                ) * 1000.0

                metrics, matches = evaluate_predictions(
                    models,
                    gt_data,
                    len(test_points),
                )

                per_frame_rows.append({
                    "frame_id": frame_id,
                    "condition": condition_name,
                    "sigma_z_m": condition["sigma_z_m"],
                    "outlier_ratio": condition["outlier_ratio"],
                    "method": method_name,
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "raw_predictions": metrics["raw_predictions"],
                    "eligible_predictions": metrics["eligible_predictions"],
                    "ignored_predictions": metrics["ignored_predictions"],
                    "runtime_ms": runtime_ms,
                })

                for match in matches:
                    match_rows.append({
                        "frame_id": frame_id,
                        "condition": condition_name,
                        "method": method_name,
                        **match,
                    })

                print(
                    f"    {method_name:25s} "
                    f"F1={metrics['f1']:.3f} "
                    f"TP={metrics['tp']:2d} "
                    f"FP={metrics['fp']:2d} "
                    f"FN={metrics['fn']:2d} "
                    f"{runtime_ms:8.1f} ms"
                )

    # --------------------------------------------------------
    # 11A. Save raw tables
    # --------------------------------------------------------

    per_frame_df = pd.DataFrame(per_frame_rows)
    matches_df = pd.DataFrame(match_rows)

    per_frame_path = RESULT_DIR / "per_frame_robustness.csv"
    matched_path = RESULT_DIR / "matched_planes_robustness.csv"

    per_frame_df.to_csv(per_frame_path, index=False)
    matches_df.to_csv(matched_path, index=False)

    # --------------------------------------------------------
    # 11B. Aggregate TP / FP / FN
    # --------------------------------------------------------

    totals = (
        per_frame_df
        .groupby(["condition", "method"], as_index=False)
        .agg(
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
            mean_runtime_ms=("runtime_ms", "mean"),
            mean_raw_predictions=("raw_predictions", "mean"),
            mean_eligible_predictions=("eligible_predictions", "mean"),
            mean_ignored_predictions=("ignored_predictions", "mean"),
        )
    )

    totals["precision"] = np.where(
        (totals["tp"] + totals["fp"]) > 0,
        totals["tp"] / (totals["tp"] + totals["fp"]),
        0.0,
    )

    totals["recall"] = np.where(
        (totals["tp"] + totals["fn"]) > 0,
        totals["tp"] / (totals["tp"] + totals["fn"]),
        0.0,
    )

    denom = totals["precision"] + totals["recall"]

    totals["f1"] = np.where(
        denom > 0,
        2.0 * totals["precision"] * totals["recall"] / denom,
        0.0,
    )

    # --------------------------------------------------------
    # 11C. Matched-plane geometry
    # --------------------------------------------------------

    if len(matches_df) > 0:
        matched_summary = (
            matches_df
            .groupby(["condition", "method"], as_index=False)
            .agg(
                matched_planes=("iou", "count"),
                mean_iou=("iou", "mean"),
                mean_raw_iou_diagnostic=("raw_iou_diagnostic", "mean"),
                mean_normal_error_deg=("normal_error_deg", "mean"),
                mean_offset_error_m=("offset_error_m", "mean"),
                mean_coverage_ratio=("coverage_ratio", "mean"),
            )
        )

        overall = totals.merge(
            matched_summary,
            on=["condition", "method"],
            how="left",
        )
    else:
        overall = totals.copy()

    overall_path = RESULT_DIR / "overall_robustness_summary.csv"
    overall.to_csv(overall_path, index=False)

    # --------------------------------------------------------
    # 11D. Clean -> hard degradation
    # --------------------------------------------------------

    degradation_rows = []

    for method_name in DETECTORS:
        sub = (
            overall[overall["method"] == method_name]
            .set_index("condition")
        )

        if "clean" not in sub.index or "hard" not in sub.index:
            continue

        clean = sub.loc["clean"]
        hard = sub.loc["hard"]

        moderate = (
            sub.loc["moderate"]
            if "moderate" in sub.index
            else None
        )

        clean_f1 = float(clean["f1"])
        hard_f1 = float(hard["f1"])

        row = {
            "method": method_name,
            "clean_f1": clean_f1,
            "moderate_f1": (
                float(moderate["f1"])
                if moderate is not None
                else np.nan
            ),
            "hard_f1": hard_f1,
            "hard_f1_drop": clean_f1 - hard_f1,
            "hard_f1_retention": (
                hard_f1 / clean_f1
                if clean_f1 > 0
                else np.nan
            ),
        }

        if "mean_iou" in overall.columns:
            row.update({
                "clean_iou": float(clean["mean_iou"]),
                "hard_iou": float(hard["mean_iou"]),
                "hard_iou_drop": (
                    float(clean["mean_iou"])
                    - float(hard["mean_iou"])
                ),
                "normal_error_increase_deg": (
                    float(hard["mean_normal_error_deg"])
                    - float(clean["mean_normal_error_deg"])
                ),
                "offset_error_increase_m": (
                    float(hard["mean_offset_error_m"])
                    - float(clean["mean_offset_error_m"])
                ),
            })

        degradation_rows.append(row)

    degradation_df = pd.DataFrame(degradation_rows)

    if len(degradation_df) > 0:
        degradation_df = degradation_df.sort_values(
            "hard_f1_drop",
            ascending=True,
        )

    degradation_path = RESULT_DIR / "robustness_degradation_summary.csv"
    degradation_df.to_csv(degradation_path, index=False)

    # --------------------------------------------------------
    # 11E. Console summary
    # --------------------------------------------------------

    print("\n\n========================================")
    print("OVERALL ROBUSTNESS RESULTS")
    print("========================================")

    print(
        overall[
            [
                "condition",
                "method",
                "precision",
                "recall",
                "f1",
                "mean_iou",
                "mean_normal_error_deg",
                "mean_offset_error_m",
                "mean_runtime_ms",
            ]
        ].to_string(index=False)
    )

    print("\n========================================")
    print("CLEAN -> HARD DEGRADATION")
    print("========================================")

    print(degradation_df.to_string(index=False))

    print("\nSaved:")
    print(per_frame_path)
    print(matched_path)
    print(overall_path)
    print(degradation_path)
    print(settings_path)


if __name__ == "__main__":
    main()
