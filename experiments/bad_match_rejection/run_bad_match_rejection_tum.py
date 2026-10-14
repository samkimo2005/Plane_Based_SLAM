from __future__ import annotations

import argparse
import inspect
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import bad_match_rejection_common as common
import normal_compatibility as normal
import distance_spatial_rejection as distance
import ransac_geometric_verification as ransac
import selmap_compatibility as selmap
import trilateration_rejection as trilateration


# ============================================================
# Paths
# ============================================================

HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent

DEFAULT_DATA_DIR = (
    EXPERIMENTS_DIR
    / "data"
    / "bad_match_rejection"
    / "tum"
)

DEFAULT_PARAMETER_FILE = (
    EXPERIMENTS_DIR
    / "results"
    / "bad_match_rejection"
    / "tuning_v2"
    / "best_parameters.json"
)

DEFAULT_RESULT_DIR = (
    EXPERIMENTS_DIR
    / "results"
    / "bad_match_rejection"
    / "tum_corrected"
)

RANSAC_BASE_SEED = 20261003


# ============================================================
# Basic metrics
# ============================================================

def safe_divide(a: float, b: float) -> float:
    return float(a / b) if b != 0 else np.nan


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator != 0 else np.nan


def correct_f1_from_counts(tn: int, fn: int, fp: int) -> float:
    denominator = 2 * tn + fn + fp
    return float(2 * tn / denominator) if denominator != 0 else np.nan


def percentile_or_nan(values, q: float) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.percentile(x, q))


def median_or_nan(values) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.median(x))


def mean_or_nan(values) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.mean(x))


# ============================================================
# Interface audit
# ============================================================

def audit_interfaces() -> None:
    """
    Stop immediately if the local algorithm files are not the final
    locked versions expected by this TUM runner.
    """

    # Distance must be the restored offset-only implementation.
    d_params = set(
        inspect.signature(
            distance.reject_bad_matches
        ).parameters
    )

    required_distance = {
        "planes_a",
        "planes_b",
        "assigned_pairs",
        "offset_threshold",
    }

    missing = required_distance - d_params

    if missing:
        raise RuntimeError(
            "Distance implementation does not match the final "
            "offset-only interface. Missing: "
            f"{sorted(missing)}"
        )

    forbidden_distance = {
        "centroids_a",
        "centroids_b",
        "spatial_threshold",
    }

    if d_params & forbidden_distance:
        raise RuntimeError(
            "The temporary centroid/spatial Distance implementation "
            "is still being imported. Replace it with the restored "
            "offset-only distance_spatial_rejection.py before running TUM."
        )

    # SelMaP must be the corrected version with separation consistency.
    s_params = set(
        inspect.signature(
            selmap.reject_bad_matches
        ).parameters
    )

    required_selmap = {
        "planes_a",
        "planes_b",
        "assigned_pairs",
        "angle_threshold_deg",
        "separation_threshold",
        "min_compatible_fraction",
    }

    missing = required_selmap - s_params

    if missing:
        raise RuntimeError(
            "SelMaP-inspired implementation is not the corrected "
            "angle + parallel-plane separation version. Missing: "
            f"{sorted(missing)}"
        )


# ============================================================
# Frozen parameters
# ============================================================

def load_frozen_parameters(parameter_file: Path) -> dict:
    if not parameter_file.exists():
        raise FileNotFoundError(
            f"Frozen parameter file not found:\n{parameter_file}"
        )

    with parameter_file.open(
        "r",
        encoding="utf-8",
    ) as f:
        best = json.load(f)

    # Set module globals for algorithms that use them internally.
    normal.NORMAL_THRESHOLD_DEG = float(
        best["Normal"]["normal_threshold_deg"]
    )

    distance.OFFSET_THRESHOLD = float(
        best["Distance"]["offset_threshold_m"]
    )

    ransac.NORMAL_THRESHOLD_DEG = float(
        best["RANSAC"]["normal_threshold_deg"]
    )

    ransac.OFFSET_THRESHOLD = float(
        best["RANSAC"]["offset_threshold_m"]
    )

    if "iterations" in best["RANSAC"]:
        ransac.RANSAC_ITERATIONS = int(
            best["RANSAC"]["iterations"]
        )

    selmap.ANGLE_THRESHOLD_DEG = float(
        best["SelMaP-inspired"]["angle_threshold_deg"]
    )

    selmap.SEPARATION_THRESHOLD = float(
        best["SelMaP-inspired"]["separation_threshold_m"]
    )

    selmap.MIN_COMPATIBLE_FRACTION = float(
        best["SelMaP-inspired"]["min_compatible_fraction"]
    )

    if "parallel_threshold_deg" in best["SelMaP-inspired"]:
        selmap.PARALLEL_THRESHOLD_DEG = float(
            best["SelMaP-inspired"]["parallel_threshold_deg"]
        )

    trilateration.CLUSTER_RADIUS = float(
        best["Trilateration-inspired"]["cluster_radius_m"]
    )

    trilateration.MIN_SUPPORT = float(
        best["Trilateration-inspired"]["min_support"]
    )

    if "max_subsets" in best["Trilateration-inspired"]:
        trilateration.MAX_SUBSETS = int(
            best["Trilateration-inspired"]["max_subsets"]
        )

    return best


# ============================================================
# TUM case loading
# ============================================================

def load_case(path: Path) -> dict:
    with np.load(
        path,
        allow_pickle=True,
    ) as data:
        required = {
            "planes_a",
            "planes_b",
            "assigned_pairs",
            "R_gt",
            "t_gt",
        }

        missing = required - set(data.files)

        if missing:
            raise KeyError(
                f"{path.name} missing required fields: {sorted(missing)}"
            )

        planes_a = np.asarray(
            data["planes_a"],
            dtype=float,
        )

        planes_b = np.asarray(
            data["planes_b"],
            dtype=float,
        )

        assigned_pairs = np.asarray(
            data["assigned_pairs"],
            dtype=int,
        ).reshape(-1, 2)

        R_gt = np.asarray(
            data["R_gt"],
            dtype=float,
        ).reshape(3, 3)

        t_gt = np.asarray(
            data["t_gt"],
            dtype=float,
        ).reshape(3)

        # Preferred representation from prepare_bad_match_rejection_tum.py:
        #   1 = known good
        #   0 = known bad
        #  -1 = ambiguous / ignore
        if "gt_match_labels_int8" in data.files:
            labels = np.asarray(
                data["gt_match_labels_int8"],
                dtype=np.int8,
            ).reshape(-1)

        elif (
            "gt_match_labels" in data.files
            and
            "gt_match_ambiguous" in data.files
        ):
            gt_good = np.asarray(
                data["gt_match_labels"],
                dtype=bool,
            ).reshape(-1)

            ambiguous = np.asarray(
                data["gt_match_ambiguous"],
                dtype=bool,
            ).reshape(-1)

            labels = np.where(
                ambiguous,
                -1,
                np.where(gt_good, 1, 0),
            ).astype(np.int8)

        elif "gt_match_labels" in data.files:
            # Only use this fallback if the file has no ambiguity field.
            labels = np.asarray(
                data["gt_match_labels"],
                dtype=np.int8,
            ).reshape(-1)

        else:
            raise KeyError(
                f"{path.name} contains no usable GT correspondence labels."
            )

        optional = {}

        for key in (
            "case_id",
            "frame_a",
            "frame_b",
            "timestamp_a",
            "timestamp_b",
            "assigned_similarity",
        ):
            if key in data.files:
                optional[key] = data[key]

    if len(labels) != len(assigned_pairs):
        raise ValueError(
            f"{path.name}: GT-label count does not match assigned_pairs."
        )

    if len(assigned_pairs) > 0:
        if assigned_pairs[:, 0].min() < 0 or assigned_pairs[:, 0].max() >= len(planes_a):
            raise IndexError(
                f"{path.name}: source-plane index out of range."
            )

        if assigned_pairs[:, 1].min() < 0 or assigned_pairs[:, 1].max() >= len(planes_b):
            raise IndexError(
                f"{path.name}: target-plane index out of range."
            )

    return {
        "path": path,
        "planes_a": planes_a,
        "planes_b": planes_b,
        "assigned_pairs": assigned_pairs,
        "labels": labels,
        "R_gt": R_gt,
        "t_gt": t_gt,
        **optional,
    }


# ============================================================
# Pose evaluation
# ============================================================

def estimate_pose(
    planes_a: np.ndarray,
    planes_b: np.ndarray,
    pairs: np.ndarray,
):
    pairs = np.asarray(
        pairs,
        dtype=int,
    ).reshape(-1, 2)

    if len(pairs) < 3:
        return None

    matched_a = planes_a[
        pairs[:, 0]
    ]

    matched_b = planes_b[
        pairs[:, 1]
    ]

    try:
        pose = common.estimate_pose_from_plane_matches(
            matched_a,
            matched_b,
        )
    except (
        ValueError,
        np.linalg.LinAlgError,
        FloatingPointError,
    ):
        return None

    if pose is None:
        return None

    if isinstance(pose, dict):
        R = pose.get("R")
        t = pose.get("t")
    else:
        try:
            R, t = pose[0], pose[1]
        except Exception:
            return None

    R = np.asarray(
        R,
        dtype=float,
    )

    t = np.asarray(
        t,
        dtype=float,
    ).reshape(-1)

    if R.shape != (3, 3) or t.shape != (3,):
        return None

    if not (
        np.all(np.isfinite(R))
        and
        np.all(np.isfinite(t))
    ):
        return None

    return R, t


def pose_error(
    pose,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
):
    if pose is None:
        return np.nan, np.nan

    R_est, t_est = pose

    R_err = (
        R_gt.T @ R_est
    )

    cos_angle = (
        np.trace(R_err) - 1.0
    ) / 2.0

    cos_angle = float(
        np.clip(
            cos_angle,
            -1.0,
            1.0,
        )
    )

    rotation_error_deg = math.degrees(
        math.acos(cos_angle)
    )

    translation_error_m = float(
        np.linalg.norm(
            t_est - t_gt
        )
    )

    return (
        rotation_error_deg,
        translation_error_m,
    )


# ============================================================
# Structural failure handling
# ============================================================

def extract_failure_reason(result: dict) -> str:
    diagnostics = result.get(
        "diagnostics",
        {},
    )

    if not isinstance(
        diagnostics,
        dict,
    ):
        return ""

    reason = diagnostics.get(
        "reason",
        "",
    )

    if reason is None:
        return ""

    return str(reason)


def run_method(
    method_name: str,
    case: dict,
    best: dict,
    case_index: int,
):
    planes_a = case["planes_a"]
    planes_b = case["planes_b"]
    pairs = case["assigned_pairs"]

    start = time.perf_counter()

    try:
        if method_name == "Normal":
            result = normal.reject_bad_matches(
                planes_a,
                planes_b,
                pairs,
                normal_threshold_deg=float(
                    best["Normal"]["normal_threshold_deg"]
                ),
            )

        elif method_name == "Distance":
            result = distance.reject_bad_matches(
                planes_a,
                planes_b,
                pairs,
                offset_threshold=float(
                    best["Distance"]["offset_threshold_m"]
                ),
            )

        elif method_name == "RANSAC":
            result = ransac.reject_bad_matches(
                planes_a,
                planes_b,
                pairs,
                rng=np.random.default_rng(
                    RANSAC_BASE_SEED + case_index
                ),
            )

        elif method_name == "SelMaP-inspired":
            result = selmap.reject_bad_matches(
                planes_a,
                planes_b,
                pairs,
                angle_threshold_deg=float(
                    best["SelMaP-inspired"]["angle_threshold_deg"]
                ),
                separation_threshold=float(
                    best["SelMaP-inspired"]["separation_threshold_m"]
                ),
                min_compatible_fraction=float(
                    best["SelMaP-inspired"]["min_compatible_fraction"]
                ),
            )

        elif method_name == "Trilateration-inspired":
            result = trilateration.reject_bad_matches(
                planes_a,
                planes_b,
                pairs,
            )

        else:
            raise KeyError(
                f"Unknown method: {method_name}"
            )

        runtime_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        keep_mask = np.asarray(
            result["keep_mask"],
            dtype=bool,
        ).reshape(-1)

        if len(keep_mask) != len(pairs):
            raise ValueError(
                "Algorithm returned keep_mask with incorrect length."
            )

        failure_reason = extract_failure_reason(
            result
        )

        structural_failure = (
            failure_reason != ""
        )

        # Important:
        # _failure() in several algorithm modules returns an all-False
        # sentinel. That must NOT be interpreted as a genuine decision
        # to reject every correspondence. For real-data evaluation,
        # preserve the upstream Assignment set and record the failure.
        if structural_failure:
            effective_keep_mask = np.ones(
                len(pairs),
                dtype=bool,
            )

            decision_available = False
            fallback_used = True

        else:
            effective_keep_mask = keep_mask
            decision_available = True
            fallback_used = False

        return {
            "raw_keep_mask": keep_mask,
            "effective_keep_mask": effective_keep_mask,
            "runtime_ms": runtime_ms,
            "decision_available": decision_available,
            "fallback_used": fallback_used,
            "failure_reason": failure_reason,
            "algorithm_success_flag": bool(
                result.get(
                    "success",
                    False,
                )
            ),
            "exception": "",
        }

    except Exception as exc:
        runtime_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        # Same safe real-data fallback for an execution failure.
        effective_keep_mask = np.ones(
            len(pairs),
            dtype=bool,
        )

        return {
            "raw_keep_mask": np.ones(
                len(pairs),
                dtype=bool,
            ),
            "effective_keep_mask": effective_keep_mask,
            "runtime_ms": runtime_ms,
            "decision_available": False,
            "fallback_used": True,
            "failure_reason": "exception",
            "algorithm_success_flag": False,
            "exception": repr(exc),
        }


# ============================================================
# Classification
# ============================================================

def classification_counts(
    effective_keep_mask: np.ndarray,
    labels: np.ndarray,
):
    """
    labels:
        1 = known good
        0 = known bad
       -1 = ambiguous / ignored

    Positive class:
        known bad correspondence.
    """

    keep = np.asarray(
        effective_keep_mask,
        dtype=bool,
    ).reshape(-1)

    labels = np.asarray(
        labels,
        dtype=np.int8,
    ).reshape(-1)

    evaluable = (
        labels >= 0
    )

    gt_good = (
        labels == 1
    )

    gt_bad = (
        labels == 0
    )

    pred_bad = (
        ~keep
    )

    tp = int(
        np.sum(
            evaluable
            & gt_bad
            & pred_bad
        )
    )

    fp = int(
        np.sum(
            evaluable
            & gt_good
            & pred_bad
        )
    )

    fn = int(
        np.sum(
            evaluable
            & gt_bad
            & keep
        )
    )

    tn = int(
        np.sum(
            evaluable
            & gt_good
            & keep
        )
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "evaluable_pairs": int(
            np.sum(evaluable)
        ),
        "known_good_pairs": int(
            np.sum(labels == 1)
        ),
        "known_bad_pairs": int(
            np.sum(labels == 0)
        ),
        "ambiguous_pairs": int(
            np.sum(labels < 0)
        ),
    }


# ============================================================
# Summaries
# ============================================================

def build_summary(
    df: pd.DataFrame,
    method_name: str,
    total_cases: int,
):
    x = df[
        df["method"] == method_name
    ].copy()

    tp = int(
        x["tp"].sum()
    )
    fp = int(
        x["fp"].sum()
    )
    fn = int(
        x["fn"].sum()
    )
    tn = int(
        x["tn"].sum()
    )

    bad_precision = safe_divide(
        tp,
        tp + fp,
    )

    bad_recall = safe_divide(
        tp,
        tp + fn,
    )

    bad_f1 = f1_from_counts(
        tp,
        fp,
        fn,
    )

    correct_retention = safe_divide(
        tn,
        tn + fp,
    )

    correct_f1 = correct_f1_from_counts(
        tn,
        fn,
        fp,
    )

    if (
        np.isfinite(bad_f1)
        and
        np.isfinite(correct_f1)
    ):
        macro_f1 = float(
            (bad_f1 + correct_f1) / 2.0
        )
    else:
        macro_f1 = np.nan

    pre_valid = x[
        "pre_pose_valid"
    ].astype(bool)

    post_valid = x[
        "post_pose_valid"
    ].astype(bool)

    pre_pose_availability = float(
        pre_valid.mean()
    )

    post_pose_availability_abs = float(
        post_valid.mean()
    )

    if int(pre_valid.sum()) > 0:
        post_pose_availability_conditional = float(
            (
                post_valid
                & pre_valid
            ).sum()
            /
            pre_valid.sum()
        )
    else:
        post_pose_availability_conditional = np.nan

    return {
        "method": method_name,
        "cases": total_cases,
        "assigned_pairs_total": int(
            x["assigned_pairs"].sum()
        ),
        "evaluable_pairs_total": int(
            x["evaluable_pairs"].sum()
        ),
        "known_good_total": int(
            x["known_good_pairs"].sum()
        ),
        "known_bad_total": int(
            x["known_bad_pairs"].sum()
        ),
        "ambiguous_total": int(
            x["ambiguous_pairs"].sum()
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "bad_precision": bad_precision,
        "bad_recall": bad_recall,
        "bad_f1": bad_f1,
        "correct_retention": correct_retention,
        "correct_f1": correct_f1,
        "macro_f1": macro_f1,
        "decision_available_rate": float(
            x["decision_available"].mean()
        ),
        "fallback_rate": float(
            x["fallback_used"].mean()
        ),
        "pre_pose_availability": pre_pose_availability,
        "post_pose_availability_abs": post_pose_availability_abs,
        "post_pose_availability_conditional": post_pose_availability_conditional,
        "median_rotation_error_deg": median_or_nan(
            x["post_rotation_error_deg"]
        ),
        "p95_rotation_error_deg": percentile_or_nan(
            x["post_rotation_error_deg"],
            95,
        ),
        "median_translation_error_m": median_or_nan(
            x["post_translation_error_m"]
        ),
        "p95_translation_error_m": percentile_or_nan(
            x["post_translation_error_m"],
            95,
        ),
        "median_runtime_ms": median_or_nan(
            x["runtime_ms"]
        ),
        "p95_runtime_ms": percentile_or_nan(
            x["runtime_ms"],
            95,
        ),
        "mean_runtime_ms": mean_or_nan(
            x["runtime_ms"]
        ),
        "mean_kept_pairs": mean_or_nan(
            x["kept_pairs"]
        ),
        "mean_rejected_pairs": mean_or_nan(
            x["rejected_pairs"]
        ),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Corrected real-data TUM evaluation for bad-match rejection."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )

    parser.add_argument(
        "--parameter-file",
        type=Path,
        default=DEFAULT_PARAMETER_FILE,
    )

    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
    )

    args = parser.parse_args()

    audit_interfaces()

    best = load_frozen_parameters(
        args.parameter_file
    )

    files = sorted(
        args.data_dir.glob(
            "tum_*.npz"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No TUM bad-match cases found in:\n{args.data_dir}"
        )

    args.result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    methods = [
        "Normal",
        "Distance",
        "RANSAC",
        "SelMaP-inspired",
        "Trilateration-inspired",
    ]

    print()
    print("=" * 78)
    print("BAD-MATCH REJECTION: TUM REAL-DATA EVALUATION")
    print("=" * 78)
    print(f"Cases: {len(files)}")
    print()
    print("Frozen parameters loaded from:")
    print(f"  {args.parameter_file}")
    print()
    print(
        "No TUM parameter tuning is performed. "
        "Ambiguous GT pairs are excluded from classification metrics."
    )
    print(
        "Structural algorithm failures use the upstream Assignment set "
        "unchanged and are recorded as fallback cases."
    )
    print()

    rows = []

    for case_index, path in enumerate(
        files
    ):
        case = load_case(
            path
        )

        pairs = case[
            "assigned_pairs"
        ]

        labels = case[
            "labels"
        ]

        # Baseline pose before bad-match rejection.
        pre_pose = estimate_pose(
            case["planes_a"],
            case["planes_b"],
            pairs,
        )

        pre_rot_err, pre_trans_err = pose_error(
            pre_pose,
            case["R_gt"],
            case["t_gt"],
        )

        for method_name in methods:
            run = run_method(
                method_name,
                case,
                best,
                case_index,
            )

            effective_keep_mask = run[
                "effective_keep_mask"
            ]

            kept_pairs = pairs[
                effective_keep_mask
            ]

            post_pose = estimate_pose(
                case["planes_a"],
                case["planes_b"],
                kept_pairs,
            )

            post_rot_err, post_trans_err = pose_error(
                post_pose,
                case["R_gt"],
                case["t_gt"],
            )

            counts = classification_counts(
                effective_keep_mask,
                labels,
            )

            row = {
                "case_file": path.name,
                "method": method_name,
                "assigned_pairs": int(
                    len(pairs)
                ),
                "kept_pairs": int(
                    effective_keep_mask.sum()
                ),
                "rejected_pairs": int(
                    (~effective_keep_mask).sum()
                ),
                **counts,
                "decision_available": bool(
                    run["decision_available"]
                ),
                "fallback_used": bool(
                    run["fallback_used"]
                ),
                "failure_reason": run[
                    "failure_reason"
                ],
                "algorithm_success_flag": bool(
                    run["algorithm_success_flag"]
                ),
                "exception": run[
                    "exception"
                ],
                "runtime_ms": float(
                    run["runtime_ms"]
                ),
                "pre_pose_valid": bool(
                    pre_pose is not None
                ),
                "post_pose_valid": bool(
                    post_pose is not None
                ),
                "pre_rotation_error_deg": pre_rot_err,
                "pre_translation_error_m": pre_trans_err,
                "post_rotation_error_deg": post_rot_err,
                "post_translation_error_m": post_trans_err,
            }

            # Optional audit metadata from prepared TUM cases.
            for key in (
                "case_id",
                "frame_a",
                "frame_b",
                "timestamp_a",
                "timestamp_b",
            ):
                if key in case:
                    value = case[key]

                    if isinstance(
                        value,
                        np.ndarray,
                    ) and value.ndim == 0:
                        value = value.item()

                    row[key] = value

            rows.append(
                row
            )

        if (
            (case_index + 1) % 10 == 0
            or
            case_index + 1 == len(files)
        ):
            print(
                f"Processed "
                f"{case_index + 1}/{len(files)}"
            )

    df = pd.DataFrame(
        rows
    )

    per_case_file = (
        args.result_dir
        / "per_case_results.csv"
    )

    df.to_csv(
        per_case_file,
        index=False,
    )

    summary_rows = [
        build_summary(
            df,
            method_name,
            len(files),
        )
        for method_name in methods
    ]

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_file = (
        args.result_dir
        / "summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    decision_columns = [
        "method",
        "macro_f1",
        "bad_precision",
        "bad_recall",
        "correct_retention",
        "post_pose_availability_abs",
        "post_pose_availability_conditional",
        "median_rotation_error_deg",
        "median_translation_error_m",
        "median_runtime_ms",
        "p95_runtime_ms",
        "decision_available_rate",
        "fallback_rate",
    ]

    decision_file = (
        args.result_dir
        / "decision_matrix_evidence.csv"
    )

    summary_df[
        decision_columns
    ].to_csv(
        decision_file,
        index=False,
    )

    parameters_used_file = (
        args.result_dir
        / "parameters_used.json"
    )

    with parameters_used_file.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "parameter_source": str(
                    args.parameter_file
                ),
                "dataset": str(
                    args.data_dir
                ),
                "ransac_base_seed": RANSAC_BASE_SEED,
                "frozen_parameters": best,
                "evaluation_policy": {
                    "positive_class": "known bad correspondence",
                    "ambiguous_label": -1,
                    "ambiguous_pairs": (
                        "excluded from TP/FP/FN/TN but retained "
                        "as algorithm input"
                    ),
                    "structural_failure_fallback": (
                        "preserve upstream Assignment correspondences"
                    ),
                    "classification_summary": "pooled over evaluable pairs",
                    "pose_summary": "median and P95",
                    "runtime_scope": "rejection algorithm only",
                },
            },
            f,
            indent=4,
        )

    print()
    print("=" * 78)
    print("TUM REAL-DATA EVALUATION COMPLETE")
    print("=" * 78)
    print()

    display_columns = [
        "method",
        "macro_f1",
        "bad_precision",
        "bad_recall",
        "correct_retention",
        "post_pose_availability_abs",
        "post_pose_availability_conditional",
        "median_rotation_error_deg",
        "median_translation_error_m",
        "median_runtime_ms",
        "decision_available_rate",
        "fallback_rate",
    ]

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        220,
        "display.precision",
        4,
    ):
        print(
            summary_df[
                display_columns
            ].to_string(
                index=False
            )
        )

    print()
    print("Saved:")
    print(f"  {per_case_file}")
    print(f"  {summary_file}")
    print(f"  {decision_file}")
    print(f"  {parameters_used_file}")


if __name__ == "__main__":
    main()
