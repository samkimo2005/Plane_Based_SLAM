from __future__ import annotations

"""
tune_plane_detection_tum.py

TUM-only parameter tuning for the five plane detectors in
experiments/feature_finding.

This version is written against the actual detector interfaces:
    iterative_ransac.detect_planes(points, rng)
    multi_object_ransac.detect_planes(points, rng)
    surface_normal_surfel.detect_planes(points, rng=None)
    region_growing.detect_planes(points, rng=None)
    hough_plane_detection.detect_planes(points, rng=None)

Important:
- The detector implementations store hyperparameters as module-level
  constants, so this tuner sets those constants directly.
- Synthetic and TUM tuning remain separate.
- TUM FINAL must never be used here.
"""

import csv
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


# ============================================================
# USER SETTINGS
# ============================================================

SEQUENCE_NAME = "rgbd_dataset_freiburg1_xyz"

# If dev_frame_ids.txt exists, only those frame IDs are used.
# Otherwise every frame that already has a reference file is used.
USE_REFERENCED_FRAMES_IF_NO_DEV_FILE = True

# Quick smoke test: set e.g. 1 or 3.
# Formal tuning: None.
MAX_DEV_FRAMES = None

# Reproducible RANSAC sampling.
BASE_RANDOM_SEED = 20260919

# Common evaluation support definition.
# IMPORTANT: this is an evaluation setting, not the tuned detector threshold.
EVAL_SUPPORT_THRESHOLD_M = 0.020

# Plane-to-reference matching criteria.
MATCH_ANGLE_THRESHOLD_DEG = 10.0
MATCH_OFFSET_THRESHOLD_M = 0.10

# Fixed detector-wide common constants for algorithms that do not
# explicitly tune them in their grid.
BASE_COMMON = {
    "FINAL_DISTANCE_THRESHOLD": 0.020,
    "MIN_PLANE_SUPPORT": 120,
    "MAX_PLANES": 6,
    "K_NEIGHBORS": 30,
}


# ============================================================
# PARAMETER GRIDS
# ============================================================

# The keys below are the ACTUAL module/global constant names used by
# the detector files.

ITERATIVE_GRID = {
    "common.FINAL_DISTANCE_THRESHOLD": [0.015, 0.020, 0.025, 0.030],
    "common.MIN_PLANE_SUPPORT": [80, 120, 160],
}
ITERATIVE_FIXED = {
    "iterative_ransac.RANSAC_ITERATIONS": 500,
}

MULTI_GRID = {
    "multi_object_ransac.MULTI_NORMAL_CLUSTER_DEG": [3.0, 5.0, 8.0],
    "multi_object_ransac.MULTI_OFFSET_CLUSTER_M": [0.030, 0.050, 0.080],
    "multi_object_ransac.MULTI_MIN_CLUSTER_VOTES": [3, 4, 6],
}
MULTI_FIXED = {
    "multi_object_ransac.MULTI_HYPOTHESES": 1000,
    "multi_object_ransac.MULTI_MIN_HYPOTHESIS_SUPPORT": 120,
}

SURFEL_GRID = {
    "surface_normal_surfel.SURFEL_MAX_CURVATURE": [0.04, 0.08, 0.12],
    "surface_normal_surfel.SURFEL_NORMAL_GROUP_DEG": [8.0, 12.0, 18.0],
    "surface_normal_surfel.SURFEL_OFFSET_NMS_M": [0.05, 0.08, 0.12],
}
SURFEL_FIXED = {
    "surface_normal_surfel.SURFEL_OFFSET_BIN_M": 0.05,
    "surface_normal_surfel.SURFEL_OFFSET_GROUP_M": 0.04,
}

REGION_GRID = {
    "region_growing.RG_SMOOTHNESS_DEG": [8.0, 15.0, 22.0],
    "region_growing.RG_LOCAL_DISTANCE_M": [0.015, 0.030, 0.050],
    "region_growing.RG_MIN_REGION_SIZE": [60, 100, 160],
}
REGION_FIXED = {}

HOUGH_GRID = {
    "hough_plane_detection.HOUGH_MAX_CURVATURE": [0.04, 0.08, 0.12],
    "hough_plane_detection.HOUGH_THETA_BIN_DEG": [5.0, 10.0, 15.0],
    "hough_plane_detection.HOUGH_D_BIN_M": [0.030, 0.050, 0.080],
}
HOUGH_FIXED = {
    # Keep theta/phi angular resolution matched during the sweep.
    "hough_plane_detection.HOUGH_MIN_VOTES": 40,
}


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_FINDING_DIR = PROJECT_ROOT / "experiments" / "feature_finding"

TUM_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "tum"
    / "plane_detection"
    / SEQUENCE_NAME
)
FRAMES_DIR = TUM_DATA_DIR / "frames"
REFERENCES_DIR = TUM_DATA_DIR / "references"
DEV_IDS_FILE = TUM_DATA_DIR / "dev_frame_ids.txt"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "plane_detection"
    / "tum"
    / "tuning"
)
RESULTS_CSV = OUTPUT_DIR / "tum_plane_detection_tuning_results.csv"
BEST_JSON = OUTPUT_DIR / "best_tum_detection_parameters.json"


# ============================================================
# IMPORT ACTUAL DETECTOR MODULES
# ============================================================

if str(FEATURE_FINDING_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_FINDING_DIR))

import plane_detection_common as common
import iterative_ransac
import multi_object_ransac
import surface_normal_surfel
import region_growing
import hough_plane_detection


DETECTORS = {
    "iterative_ransac": {
        "module": iterative_ransac,
        "grid": ITERATIVE_GRID,
        "fixed": ITERATIVE_FIXED,
    },
    "multi_object_ransac": {
        "module": multi_object_ransac,
        "grid": MULTI_GRID,
        "fixed": MULTI_FIXED,
    },
    "surface_normal_surfel": {
        "module": surface_normal_surfel,
        "grid": SURFEL_GRID,
        "fixed": SURFEL_FIXED,
    },
    "region_growing": {
        "module": region_growing,
        "grid": REGION_GRID,
        "fixed": REGION_FIXED,
    },
    "hough_plane_detection": {
        "module": hough_plane_detection,
        "grid": HOUGH_GRID,
        "fixed": HOUGH_FIXED,
    },
}

MODULES = {
    "common": common,
    "iterative_ransac": iterative_ransac,
    "multi_object_ransac": multi_object_ransac,
    "surface_normal_surfel": surface_normal_surfel,
    "region_growing": region_growing,
    "hough_plane_detection": hough_plane_detection,
}


# ============================================================
# VALIDATION
# ============================================================

def validate_detector_interfaces() -> None:
    required = {
        "iterative_ransac": ["RANSAC_ITERATIONS"],
        "multi_object_ransac": [
            "MULTI_HYPOTHESES",
            "MULTI_MIN_HYPOTHESIS_SUPPORT",
            "MULTI_NORMAL_CLUSTER_DEG",
            "MULTI_OFFSET_CLUSTER_M",
            "MULTI_MIN_CLUSTER_VOTES",
        ],
        "surface_normal_surfel": [
            "SURFEL_MAX_CURVATURE",
            "SURFEL_NORMAL_GROUP_DEG",
            "SURFEL_OFFSET_BIN_M",
            "SURFEL_OFFSET_GROUP_M",
            "SURFEL_OFFSET_NMS_M",
        ],
        "region_growing": [
            "RG_SMOOTHNESS_DEG",
            "RG_LOCAL_DISTANCE_M",
            "RG_MIN_REGION_SIZE",
        ],
        "hough_plane_detection": [
            "HOUGH_MAX_CURVATURE",
            "HOUGH_THETA_BIN_DEG",
            "HOUGH_PHI_BIN_DEG",
            "HOUGH_D_BIN_M",
            "HOUGH_MIN_VOTES",
        ],
    }

    common_required = [
        "FINAL_DISTANCE_THRESHOLD",
        "MIN_PLANE_SUPPORT",
        "MAX_PLANES",
        "K_NEIGHBORS",
    ]
    for name in common_required:
        if not hasattr(common, name):
            raise AttributeError(f"plane_detection_common.py is missing {name}")

    for detector_name, constants in required.items():
        module = MODULES[detector_name]
        if not hasattr(module, "detect_planes"):
            raise AttributeError(f"{detector_name}.py is missing detect_planes()")
        for name in constants:
            if not hasattr(module, name):
                raise AttributeError(f"{detector_name}.py is missing {name}")


# ============================================================
# PARAMETER PATCHING
# ============================================================

def reset_common_defaults() -> None:
    for name, value in BASE_COMMON.items():
        setattr(common, name, value)


def set_parameter(path: str, value) -> None:
    """
    path example:
        common.MIN_PLANE_SUPPORT
        region_growing.RG_MIN_REGION_SIZE
    """
    module_name, attribute = path.split(".", 1)
    module = MODULES[module_name]

    if not hasattr(module, attribute):
        raise AttributeError(
            f"Cannot tune {path}: {module_name} has no attribute {attribute}"
        )

    setattr(module, attribute, value)

    # Hough uses both theta and phi angular bins. Keep them matched when
    # the theta bin is varied.
    if path == "hough_plane_detection.HOUGH_THETA_BIN_DEG":
        hough_plane_detection.HOUGH_PHI_BIN_DEG = float(value)


def apply_configuration(parameters: dict) -> None:
    # Every configuration starts from the same common baseline so state
    # cannot leak from the previous detector/configuration.
    reset_common_defaults()

    for path, value in parameters.items():
        set_parameter(path, value)


def expand_grid(grid: dict, fixed: dict) -> list[dict]:
    keys = list(grid.keys())
    configs = []

    for values in itertools.product(*(grid[k] for k in keys)):
        parameters = dict(fixed)
        parameters.update(dict(zip(keys, values)))
        configs.append(parameters)

    return configs


# ============================================================
# DATA LOADING
# ============================================================

def frame_id(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def get_dev_frames() -> list[Path]:
    all_frames = sorted(FRAMES_DIR.glob("frame_*.npz"))

    if not all_frames:
        raise FileNotFoundError(
            f"No prepared TUM frames found in:\n{FRAMES_DIR}\n"
            "Run prepare_tum_plane_detection.py first."
        )

    if DEV_IDS_FILE.exists():
        ids = set()
        for line in DEV_IDS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ids.add(int(line))

        selected = [p for p in all_frames if frame_id(p) in ids]

        missing = sorted(
            i for i in ids
            if not (FRAMES_DIR / f"frame_{i:04d}.npz").exists()
        )
        if missing:
            raise FileNotFoundError(f"DEV frame IDs not found: {missing}")

    elif USE_REFERENCED_FRAMES_IF_NO_DEV_FILE:
        selected = [
            p for p in all_frames
            if (REFERENCES_DIR / p.name).exists()
        ]
        print("WARNING: dev_frame_ids.txt not found.")
        print("Using every currently referenced frame as TUM DEV.")
    else:
        raise FileNotFoundError(f"Missing DEV split file:\n{DEV_IDS_FILE}")

    if MAX_DEV_FRAMES is not None:
        selected = selected[:MAX_DEV_FRAMES]

    if not selected:
        raise RuntimeError("No TUM DEV frames selected.")

    for p in selected:
        ref = REFERENCES_DIR / p.name
        if not ref.exists():
            raise FileNotFoundError(f"Missing reference file:\n{ref}")

    return selected


def load_frame_and_reference(frame_file: Path) -> tuple[np.ndarray, dict]:
    with np.load(frame_file, allow_pickle=False) as data:
        if "detector_points" not in data:
            raise KeyError(f"{frame_file.name} has no detector_points")
        points = np.asarray(data["detector_points"], dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"{frame_file.name}: detector_points must have shape (N,3), "
            f"got {points.shape}"
        )
    if len(points) < 3:
        raise ValueError(f"{frame_file.name}: fewer than 3 detector points")

    ref_file = REFERENCES_DIR / frame_file.name
    with np.load(ref_file, allow_pickle=False) as ref:
        required = ["gt_normals", "gt_offsets"]
        for key in required:
            if key not in ref:
                raise KeyError(f"{ref_file.name} has no {key}")

        gt_normals = np.asarray(ref["gt_normals"], dtype=np.float64)
        gt_offsets = np.asarray(ref["gt_offsets"], dtype=np.float64)

        if "gt_support_masks" in ref:
            gt_masks = np.asarray(ref["gt_support_masks"], dtype=bool)
        elif "gt_labels" in ref:
            labels = np.asarray(ref["gt_labels"], dtype=np.int32)
            if len(labels) != len(points):
                raise ValueError(
                    f"{ref_file.name}: gt_labels length {len(labels)} "
                    f"!= detector point count {len(points)}"
                )
            gt_masks = np.stack(
                [labels == i for i in range(len(gt_normals))],
                axis=0,
            )
        else:
            raise KeyError(
                f"{ref_file.name} needs gt_support_masks or gt_labels"
            )

    if gt_normals.ndim != 2 or gt_normals.shape[1] != 3:
        raise ValueError(
            f"{ref_file.name}: gt_normals must have shape (K,3), "
            f"got {gt_normals.shape}"
        )
    if len(gt_offsets) != len(gt_normals):
        raise ValueError(f"{ref_file.name}: normals/offsets count mismatch")
    if gt_masks.shape != (len(gt_normals), len(points)):
        raise ValueError(
            f"{ref_file.name}: gt_support_masks shape {gt_masks.shape}; "
            f"expected {(len(gt_normals), len(points))}"
        )

    return points, {
        "normals": gt_normals,
        "offsets": gt_offsets,
        "masks": gt_masks,
    }


# ============================================================
# DETECTOR WRAPPERS — WRITTEN FOR THE ACTUAL FILES
# ============================================================

def run_detector(detector_name: str, points: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    module = DETECTORS[detector_name]["module"]

    # The uploaded implementations all expose detect_planes(points, rng)
    # (rng is optional in three methods but accepted).
    models = module.detect_planes(points, rng)

    if models is None:
        return []
    if not isinstance(models, list):
        raise TypeError(
            f"{detector_name}.detect_planes() returned {type(models).__name__}; "
            "expected list"
        )

    validated = []
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise TypeError(
                f"{detector_name} model {index} is not a dict"
            )
        if "normal" not in model or "d" not in model:
            raise KeyError(
                f"{detector_name} model {index} needs keys 'normal' and 'd'"
            )

        normal = np.asarray(model["normal"], dtype=np.float64)
        if normal.shape != (3,):
            raise ValueError(
                f"{detector_name} model {index}: normal shape {normal.shape}"
            )

        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm

        d = float(model["d"])

        # IMPORTANT:
        # Recompute support using a fixed evaluation threshold so a tuned
        # detection threshold cannot also change the evaluation definition.
        support_mask = np.abs(points @ normal + d) < EVAL_SUPPORT_THRESHOLD_M

        validated.append({
            "normal": normal,
            "d": d,
            "support_mask": support_mask,
        })

    return validated


# ============================================================
# EVALUATION
# ============================================================

def plane_error(det_n, det_d, gt_n, gt_d) -> tuple[float, float]:
    det_n = np.asarray(det_n, dtype=np.float64)
    gt_n = np.asarray(gt_n, dtype=np.float64)

    det_n = det_n / np.linalg.norm(det_n)
    gt_n = gt_n / np.linalg.norm(gt_n)

    dot = float(np.dot(det_n, gt_n))

    # (n,d) and (-n,-d) describe the same plane.
    if dot < 0.0:
        det_n = -det_n
        det_d = -det_d
        dot = -dot

    dot = float(np.clip(dot, 0.0, 1.0))
    angle = math.degrees(math.acos(dot))
    offset_error = abs(float(det_d) - float(gt_d))

    return angle, offset_error


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)
    return float(intersection / union) if union else 0.0


def evaluate_frame(models: list[dict], reference: dict) -> dict:
    n_det = len(models)
    n_gt = len(reference["normals"])

    if n_det == 0:
        return {
            "precision": 0.0,
            "recall": 0.0 if n_gt else 1.0,
            "f1": 0.0 if n_gt else 1.0,
            "correct_plane_count": float(n_gt == 0),
            "false_planes": 0.0,
            "missed_planes": float(n_gt),
            "mean_iou": 0.0,
            "normal_error_deg": np.nan,
            "offset_error_m": np.nan,
        }

    if n_gt == 0:
        return {
            "precision": 0.0,
            "recall": 1.0,
            "f1": 0.0,
            "correct_plane_count": 0.0,
            "false_planes": float(n_det),
            "missed_planes": 0.0,
            "mean_iou": 0.0,
            "normal_error_deg": np.nan,
            "offset_error_m": np.nan,
        }

    LARGE = 1e9
    cost = np.full((n_det, n_gt), LARGE, dtype=np.float64)
    angles = np.full((n_det, n_gt), np.nan, dtype=np.float64)
    offsets = np.full((n_det, n_gt), np.nan, dtype=np.float64)
    ious = np.zeros((n_det, n_gt), dtype=np.float64)

    for i, model in enumerate(models):
        for j in range(n_gt):
            angle, offset = plane_error(
                model["normal"],
                model["d"],
                reference["normals"][j],
                reference["offsets"][j],
            )
            iou = mask_iou(
                model["support_mask"],
                reference["masks"][j],
            )

            angles[i, j] = angle
            offsets[i, j] = offset
            ious[i, j] = iou

            if (
                angle <= MATCH_ANGLE_THRESHOLD_DEG
                and offset <= MATCH_OFFSET_THRESHOLD_M
            ):
                # Matching eligibility is geometric; IoU is then measured
                # independently as an evaluation metric.
                cost[i, j] = (
                    angle / MATCH_ANGLE_THRESHOLD_DEG
                    + offset / MATCH_OFFSET_THRESHOLD_M
                )

    rows, cols = linear_sum_assignment(cost)
    matches = [
        (i, j)
        for i, j in zip(rows, cols)
        if cost[i, j] < LARGE / 2
    ]

    tp = len(matches)
    fp = n_det - tp
    fn = n_gt - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    if matches:
        mean_iou = float(np.mean([ious[i, j] for i, j in matches]))
        mean_angle = float(np.mean([angles[i, j] for i, j in matches]))
        mean_offset = float(np.mean([offsets[i, j] for i, j in matches]))
    else:
        mean_iou = 0.0
        mean_angle = np.nan
        mean_offset = np.nan

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "correct_plane_count": float(n_det == n_gt),
        "false_planes": float(fp),
        "missed_planes": float(fn),
        "mean_iou": mean_iou,
        "normal_error_deg": mean_angle,
        "offset_error_m": mean_offset,
    }


def finite_mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else np.nan


def evaluate_configuration(
    detector_name: str,
    parameters: dict,
    dev_data: list[tuple[int, np.ndarray, dict]],
) -> dict:
    apply_configuration(parameters)

    frame_results = []

    for fid, points, reference in dev_data:
        seed = BASE_RANDOM_SEED + fid

        start = time.perf_counter()
        models = run_detector(detector_name, points, seed)
        runtime_ms = (time.perf_counter() - start) * 1000.0

        metrics = evaluate_frame(models, reference)
        metrics["runtime_ms"] = runtime_ms
        frame_results.append(metrics)

    return {
        "algorithm": detector_name,
        "parameters": dict(parameters),
        "number_frames": len(dev_data),
        "mean_precision": finite_mean(r["precision"] for r in frame_results),
        "mean_recall": finite_mean(r["recall"] for r in frame_results),
        "mean_f1": finite_mean(r["f1"] for r in frame_results),
        "correct_plane_count_rate": finite_mean(
            r["correct_plane_count"] for r in frame_results
        ),
        "mean_iou": finite_mean(r["mean_iou"] for r in frame_results),
        "mean_false_planes": finite_mean(
            r["false_planes"] for r in frame_results
        ),
        "mean_missed_planes": finite_mean(
            r["missed_planes"] for r in frame_results
        ),
        "mean_normal_error_deg": finite_mean(
            r["normal_error_deg"] for r in frame_results
        ),
        "mean_offset_error_m": finite_mean(
            r["offset_error_m"] for r in frame_results
        ),
        "mean_runtime_ms": finite_mean(
            r["runtime_ms"] for r in frame_results
        ),
    }


# ============================================================
# SELECTION
# ============================================================

def high(value: float) -> float:
    return float(value) if np.isfinite(value) else -1e12


def low(value: float) -> float:
    return float(value) if np.isfinite(value) else 1e12


def selection_key(result: dict):
    """
    Lexicographic DEV tuning priority:
      1 F1
      2 Correct plane-count rate
      3 IoU
      4 fewer false planes
      5 fewer missed planes
      6 lower normal error
      7 lower offset error

    Runtime is deliberately excluded from parameter selection.
    """
    return (
        high(result["mean_f1"]),
        high(result["correct_plane_count_rate"]),
        high(result["mean_iou"]),
        -low(result["mean_false_planes"]),
        -low(result["mean_missed_planes"]),
        -low(result["mean_normal_error_deg"]),
        -low(result["mean_offset_error_m"]),
    )


# ============================================================
# OUTPUT
# ============================================================

def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_results(all_results: list[dict], best_results: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for result in all_results:
        rows.append({
            "algorithm": result["algorithm"],
            "number_frames": result["number_frames"],
            "mean_precision": result["mean_precision"],
            "mean_recall": result["mean_recall"],
            "mean_f1": result["mean_f1"],
            "correct_plane_count_rate": result["correct_plane_count_rate"],
            "mean_iou": result["mean_iou"],
            "mean_false_planes": result["mean_false_planes"],
            "mean_missed_planes": result["mean_missed_planes"],
            "mean_normal_error_deg": result["mean_normal_error_deg"],
            "mean_offset_error_m": result["mean_offset_error_m"],
            "mean_runtime_ms": result["mean_runtime_ms"],
            "parameters_json": json.dumps(
                json_safe(result["parameters"]),
                sort_keys=True,
            ),
        })

    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    output = {
        "dataset": "TUM RGB-D DEV",
        "sequence": SEQUENCE_NAME,
        "selection_priority": [
            "mean_f1",
            "correct_plane_count_rate",
            "mean_iou",
            "minimise_mean_false_planes",
            "minimise_mean_missed_planes",
            "minimise_mean_normal_error_deg",
            "minimise_mean_offset_error_m",
        ],
        "runtime_used_for_parameter_selection": False,
        "evaluation_support_threshold_m": EVAL_SUPPORT_THRESHOLD_M,
        "match_angle_threshold_deg": MATCH_ANGLE_THRESHOLD_DEG,
        "match_offset_threshold_m": MATCH_OFFSET_THRESHOLD_M,
        "algorithms": {},
    }

    for name, result in best_results.items():
        output["algorithms"][name] = {
            "parameters": json_safe(result["parameters"]),
            "tum_dev_metrics": {
                key: json_safe(result[key])
                for key in [
                    "mean_precision",
                    "mean_recall",
                    "mean_f1",
                    "correct_plane_count_rate",
                    "mean_iou",
                    "mean_false_planes",
                    "mean_missed_planes",
                    "mean_normal_error_deg",
                    "mean_offset_error_m",
                    "mean_runtime_ms",
                ]
            },
        }

    BEST_JSON.write_text(
        json.dumps(output, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    validate_detector_interfaces()

    dev_frames = get_dev_frames()

    # Load once. This also validates every frame/reference before the
    # expensive grid search starts.
    dev_data = []
    for path in dev_frames:
        points, reference = load_frame_and_reference(path)
        dev_data.append((frame_id(path), points, reference))

    print()
    print("=" * 92)
    print("TUM PLANE DETECTION PARAMETER TUNING")
    print("=" * 92)
    print(f"Sequence        : {SEQUENCE_NAME}")
    print(f"DEV frames      : {len(dev_data)}")
    print(f"DEV frame IDs   : {[x[0] for x in dev_data]}")
    print(f"Evaluation gate : {EVAL_SUPPORT_THRESHOLD_M:.3f} m")
    print("=" * 92)

    all_results = []
    best_results = {}

    for detector_number, (name, spec) in enumerate(DETECTORS.items(), start=1):
        configurations = expand_grid(spec["grid"], spec["fixed"])

        print()
        print("=" * 92)
        print(
            f"[{detector_number}/{len(DETECTORS)}] {name} "
            f"— {len(configurations)} configurations"
        )
        print("=" * 92)

        detector_results = []

        for config_number, parameters in enumerate(configurations, start=1):
            print(
                f"[{config_number:02d}/{len(configurations):02d}] "
                f"{parameters}"
            )

            result = evaluate_configuration(
                name,
                parameters,
                dev_data,
            )

            detector_results.append(result)
            all_results.append(result)

            print(
                "    "
                f"F1={result['mean_f1']:.4f} | "
                f"Count={result['correct_plane_count_rate']:.4f} | "
                f"IoU={result['mean_iou']:.4f} | "
                f"FP={result['mean_false_planes']:.3f} | "
                f"FN={result['mean_missed_planes']:.3f} | "
                f"Runtime={result['mean_runtime_ms']:.2f} ms"
            )

        best = max(detector_results, key=selection_key)
        best_results[name] = best

        print()
        print(f"BEST — {name}")
        print(json.dumps(json_safe(best["parameters"]), indent=4))
        print(
            f"F1={best['mean_f1']:.4f}, "
            f"Count={best['correct_plane_count_rate']:.4f}, "
            f"IoU={best['mean_iou']:.4f}"
        )

    save_results(all_results, best_results)

    print()
    print("=" * 92)
    print("TUM TUNING COMPLETE")
    print("=" * 92)

    for name, result in best_results.items():
        print(
            f"{name:24s} "
            f"F1={result['mean_f1']:.4f} | "
            f"Count={result['correct_plane_count_rate']:.4f} | "
            f"IoU={result['mean_iou']:.4f} | "
            f"Runtime={result['mean_runtime_ms']:.2f} ms"
        )

    print()
    print(f"Full sweep : {RESULTS_CSV}")
    print(f"Frozen TUM : {BEST_JSON}")
    print()
    print("Do not retune these parameters using TUM FINAL.")
    print("=" * 92)


if __name__ == "__main__":
    main()
