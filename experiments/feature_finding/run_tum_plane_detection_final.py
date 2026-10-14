from __future__ import annotations

"""
run_tum_plane_detection_final.py

Run the five plane detectors on the frozen TUM FINAL split.

This script is written against the actual detector interfaces used in
experiments/feature_finding:

    iterative_ransac.detect_planes(points, rng)
    multi_object_ransac.detect_planes(points, rng)
    surface_normal_surfel.detect_planes(points, rng=None)
    region_growing.detect_planes(points, rng=None)
    hough_plane_detection.detect_planes(points, rng=None)

IMPORTANT
---------
1. This script NEVER tunes parameters.
2. It reads parameters only from:
       results/plane_detection/tum/tuning/best_tum_detection_parameters.json
3. FINAL frames must have independent reference files.
4. FINAL frame IDs must not overlap the DEV frames used for tuning.
5. In addition to evaluation CSV files, detector plane objects are saved so
   the selected plane detector can feed the next Candidate Gating stage.

Plane convention:
    n^T x + d = 0
"""

import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, QhullError


# ============================================================
# USER SETTINGS
# ============================================================

SEQUENCE_NAME = "rgbd_dataset_freiburg1_xyz"

# FINAL split handling is automatic by default.
# The runner uses every annotated reference frame that was NOT used for tuning.
# You do NOT need to create final_frame_ids.txt.
#
# Set an explicit list only if you want to restrict FINAL to specific frames.
FINAL_FRAME_IDS = None

# Your completed tuning run reported DEV frame IDs [0, 1, 2].
# This fallback is used only when neither dev_frame_ids.txt nor the frozen JSON
# records the DEV split.  Keep this equal to the frame IDs actually used for
# tuning.
DEV_FRAME_IDS_FALLBACK = [0, 1, 2]

# Automatically write discovered split files for reproducibility.
WRITE_DISCOVERED_SPLITS = True

# Deterministic RANSAC sampling. Keep identical to the TUM tuner.
BASE_RANDOM_SEED = 20260919

# Fixed number of contour samples saved for Candidate Gating.
CONTOUR_SAMPLES = 32

# Recreate final output folder on each run.
OVERWRITE_FINAL_OUTPUT = True


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
FINAL_IDS_FILE = TUM_DATA_DIR / "final_frame_ids.txt"

TUNING_DIR = (
    PROJECT_ROOT
    / "results"
    / "plane_detection"
    / "tum"
    / "tuning"
)

BEST_PARAMETERS_JSON = (
    TUNING_DIR
    / "best_tum_detection_parameters.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "plane_detection"
    / "tum"
    / "final"
)

DETECTIONS_DIR = OUTPUT_DIR / "detections"
FINAL_RESULTS_CSV = OUTPUT_DIR / "tum_plane_detection_final_results.csv"
FINAL_SUMMARY_CSV = OUTPUT_DIR / "tum_plane_detection_final_summary.csv"
FROZEN_PARAMETERS_USED_JSON = OUTPUT_DIR / "frozen_parameters_used.json"
RUN_METADATA_JSON = OUTPUT_DIR / "run_metadata.json"


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
    "iterative_ransac": iterative_ransac,
    "multi_object_ransac": multi_object_ransac,
    "surface_normal_surfel": surface_normal_surfel,
    "region_growing": region_growing,
    "hough_plane_detection": hough_plane_detection,
}

MODULES = {
    "common": common,
    "iterative_ransac": iterative_ransac,
    "multi_object_ransac": multi_object_ransac,
    "surface_normal_surfel": surface_normal_surfel,
    "region_growing": region_growing,
    "hough_plane_detection": hough_plane_detection,
}

# These are the same common baseline values used by the validated TUM tuner.
BASE_COMMON = {
    "FINAL_DISTANCE_THRESHOLD": 0.020,
    "MIN_PLANE_SUPPORT": 120,
    "MAX_PLANES": 6,
    "K_NEIGHBORS": 30,
}

EXPECTED_ALGORITHMS = list(DETECTORS.keys())


# ============================================================
# BASIC VALIDATION
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

    for name in BASE_COMMON:
        if not hasattr(common, name):
            raise AttributeError(
                f"plane_detection_common.py is missing {name}"
            )

    for detector_name, constants in required.items():
        module = DETECTORS[detector_name]

        if not hasattr(module, "detect_planes"):
            raise AttributeError(
                f"{detector_name}.py is missing detect_planes()"
            )

        for name in constants:
            if not hasattr(module, name):
                raise AttributeError(
                    f"{detector_name}.py is missing {name}"
                )


# ============================================================
# SPLIT HANDLING
# ============================================================

def read_id_file(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(path)

    ids = []
    seen = set()

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        frame_id = int(line)

        if frame_id not in seen:
            ids.append(frame_id)
            seen.add(frame_id)

    return ids


def reference_frame_ids() -> list[int]:
    ids = []
    for path in sorted(REFERENCES_DIR.glob("frame_*.npz")):
        try:
            ids.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return ids


def frozen_json_dev_ids() -> list[int]:
    if not BEST_PARAMETERS_JSON.exists():
        return []

    try:
        data = json.loads(BEST_PARAMETERS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []

    values = data.get("dev_frame_ids", [])
    if not isinstance(values, list):
        return []

    output = []
    for value in values:
        try:
            output.append(int(value))
        except (TypeError, ValueError):
            pass
    return output


def get_dev_ids() -> list[int]:
    if DEV_IDS_FILE.exists():
        ids = read_id_file(DEV_IDS_FILE)
        source = "dev_frame_ids.txt"
    else:
        ids = frozen_json_dev_ids()
        source = "frozen tuning JSON"

        if not ids:
            ids = [int(x) for x in DEV_FRAME_IDS_FALLBACK]
            source = "DEV_FRAME_IDS_FALLBACK"

    if not ids:
        raise SystemExit(
            "\nCannot determine which frames were used for TUM tuning.\n"
            "Set DEV_FRAME_IDS_FALLBACK near the top of this script.\n"
        )

    ids = list(dict.fromkeys(ids))

    if WRITE_DISCOVERED_SPLITS and not DEV_IDS_FILE.exists():
        DEV_IDS_FILE.write_text(
            "".join(f"{fid}\n" for fid in ids),
            encoding="utf-8",
        )

    print(f"DEV split source      : {source}")
    return ids


def get_final_ids(dev_ids: list[int]) -> list[int]:
    # 1. Explicit IDs in this script.
    if FINAL_FRAME_IDS is not None:
        ids = [int(value) for value in FINAL_FRAME_IDS]
        source = "FINAL_FRAME_IDS setting"

    # 2. Existing explicit file.
    elif FINAL_IDS_FILE.exists():
        ids = read_id_file(FINAL_IDS_FILE)
        source = "final_frame_ids.txt"

    # 3. Automatic discovery: every annotated reference not used in DEV.
    else:
        annotated = reference_frame_ids()
        dev_set = set(dev_ids)
        ids = [fid for fid in annotated if fid not in dev_set]
        source = "automatic: annotated references minus DEV"

        if WRITE_DISCOVERED_SPLITS and ids:
            FINAL_IDS_FILE.write_text(
                "".join(f"{fid}\n" for fid in ids),
                encoding="utf-8",
            )

    if len(set(ids)) != len(ids):
        raise SystemExit("TUM FINAL frame IDs contain duplicates.")

    overlap = sorted(set(dev_ids) & set(ids))
    if overlap:
        raise SystemExit(
            "\nDEV/FINAL DATA LEAKAGE DETECTED.\n"
            f"Overlapping frame IDs: {overlap}\n"
            "FINAL must use frames that were not used during tuning.\n"
        )

    if not ids:
        annotated = reference_frame_ids()
        print()
        print("=" * 88)
        print("NO HELD-OUT TUM FINAL REFERENCES YET")
        print("=" * 88)
        print(f"Annotated reference frames : {annotated}")
        print(f"Frames already used as DEV : {dev_ids}")
        print()
        print(
            "All currently annotated reference frames were already used for tuning, "
            "so there is no independent FINAL set to evaluate."
        )
        print()
        print(
            "Annotate additional TUM frames with build_tum_plane_reference.py "
            "(do not reuse the DEV frames)."
        )
        print(
            "After new reference files exist, simply run this script again; "
            "no final_frame_ids.txt is required."
        )
        print("=" * 88)
        raise SystemExit(0)

    print(f"FINAL split source    : {source}")
    return ids


# ============================================================
# FROZEN PARAMETER LOADING / PATCHING
# ============================================================

def load_frozen_configuration() -> tuple[dict, float, float, float]:
    if not BEST_PARAMETERS_JSON.exists():
        raise FileNotFoundError(
            "\nFrozen TUM parameters were not found:\n"
            f"  {BEST_PARAMETERS_JSON}\n\n"
            "Run tune_plane_detection_tum.py first."
        )

    data = json.loads(
        BEST_PARAMETERS_JSON.read_text(encoding="utf-8")
    )

    algorithms = data.get("algorithms")

    if not isinstance(algorithms, dict):
        raise KeyError(
            "best_tum_detection_parameters.json has no 'algorithms' object."
        )

    missing = [
        name for name in EXPECTED_ALGORITHMS
        if name not in algorithms
    ]

    if missing:
        raise KeyError(
            "Frozen parameter file is missing algorithms: "
            + ", ".join(missing)
        )

    frozen = {}

    for name in EXPECTED_ALGORITHMS:
        parameters = algorithms[name].get("parameters")

        if not isinstance(parameters, dict):
            raise TypeError(
                f"Frozen parameters for {name} are missing or invalid."
            )

        frozen[name] = parameters

    eval_support_threshold = float(
        data.get("evaluation_support_threshold_m", 0.020)
    )
    match_angle_threshold = float(
        data.get("match_angle_threshold_deg", 10.0)
    )
    match_offset_threshold = float(
        data.get("match_offset_threshold_m", 0.10)
    )

    return (
        frozen,
        eval_support_threshold,
        match_angle_threshold,
        match_offset_threshold,
    )


def reset_common_defaults() -> None:
    for name, value in BASE_COMMON.items():
        setattr(common, name, value)


def set_parameter(path: str, value) -> None:
    if "." not in path:
        raise ValueError(
            f"Frozen parameter key must be module.attribute, got: {path}"
        )

    module_name, attribute = path.split(".", 1)

    if module_name not in MODULES:
        raise KeyError(
            f"Unknown parameter module in frozen file: {module_name}"
        )

    module = MODULES[module_name]

    if not hasattr(module, attribute):
        raise AttributeError(
            f"Frozen parameter refers to missing attribute: {path}"
        )

    setattr(module, attribute, value)

    # The tuner deliberately keeps Hough theta and phi angular resolution
    # equal when HOUGH_THETA_BIN_DEG is tuned.
    if path == "hough_plane_detection.HOUGH_THETA_BIN_DEG":
        hough_plane_detection.HOUGH_PHI_BIN_DEG = float(value)


def apply_frozen_parameters(parameters: dict) -> None:
    # Prevent state leaking from whichever detector ran previously.
    reset_common_defaults()

    for path, value in parameters.items():
        set_parameter(path, value)


# ============================================================
# FRAME / REFERENCE LOADING
# ============================================================

def frame_file_for_id(frame_id: int) -> Path:
    return FRAMES_DIR / f"frame_{frame_id:04d}.npz"


def reference_file_for_id(frame_id: int) -> Path:
    return REFERENCES_DIR / f"frame_{frame_id:04d}.npz"


def load_frame_and_reference(
    frame_id: int,
) -> tuple[np.ndarray, dict, Path]:
    frame_file = frame_file_for_id(frame_id)
    reference_file = reference_file_for_id(frame_id)

    if not frame_file.exists():
        raise FileNotFoundError(
            f"FINAL frame does not exist:\n{frame_file}"
        )

    if not reference_file.exists():
        raise FileNotFoundError(
            "FINAL reference is missing:\n"
            f"{reference_file}\n\n"
            "Annotate this FINAL frame with build_tum_plane_reference.py "
            "before running FINAL evaluation."
        )

    with np.load(frame_file, allow_pickle=False) as data:
        if "detector_points" not in data:
            raise KeyError(f"{frame_file.name} has no detector_points")

        points = np.asarray(
            data["detector_points"],
            dtype=np.float64,
        )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"{frame_file.name}: detector_points must be (N,3), "
            f"got {points.shape}"
        )

    with np.load(reference_file, allow_pickle=False) as ref:
        for key in ["gt_normals", "gt_offsets"]:
            if key not in ref:
                raise KeyError(
                    f"{reference_file.name} has no {key}"
                )

        gt_normals = np.asarray(
            ref["gt_normals"],
            dtype=np.float64,
        )
        gt_offsets = np.asarray(
            ref["gt_offsets"],
            dtype=np.float64,
        )

        if "gt_support_masks" in ref:
            gt_masks = np.asarray(
                ref["gt_support_masks"],
                dtype=bool,
            )
        elif "gt_labels" in ref:
            labels = np.asarray(
                ref["gt_labels"],
                dtype=np.int32,
            )

            if len(labels) != len(points):
                raise ValueError(
                    f"{reference_file.name}: gt_labels length "
                    f"{len(labels)} != point count {len(points)}"
                )

            gt_masks = np.stack(
                [labels == i for i in range(len(gt_normals))],
                axis=0,
            )
        else:
            raise KeyError(
                f"{reference_file.name} needs gt_support_masks or gt_labels"
            )

    if gt_normals.ndim != 2 or gt_normals.shape[1] != 3:
        raise ValueError(
            f"{reference_file.name}: gt_normals must be (K,3)"
        )

    if len(gt_offsets) != len(gt_normals):
        raise ValueError(
            f"{reference_file.name}: GT normal/offset count mismatch"
        )

    expected_mask_shape = (len(gt_normals), len(points))

    if gt_masks.shape != expected_mask_shape:
        raise ValueError(
            f"{reference_file.name}: support masks are {gt_masks.shape}, "
            f"expected {expected_mask_shape}"
        )

    reference = {
        "normals": gt_normals,
        "offsets": gt_offsets,
        "masks": gt_masks,
    }

    return points, reference, frame_file


# ============================================================
# DETECTOR EXECUTION — ACTUAL INTERFACES
# ============================================================

def run_detector_raw(
    detector_name: str,
    points: np.ndarray,
    seed: int,
) -> list[dict]:
    module = DETECTORS[detector_name]
    rng = np.random.default_rng(seed)

    models = module.detect_planes(points, rng)

    if models is None:
        return []

    if not isinstance(models, list):
        raise TypeError(
            f"{detector_name}.detect_planes() returned "
            f"{type(models).__name__}; expected list"
        )

    output = []

    for model_index, model in enumerate(models):
        if not isinstance(model, dict):
            raise TypeError(
                f"{detector_name} model {model_index} is not a dict"
            )

        if "normal" not in model or "d" not in model:
            raise KeyError(
                f"{detector_name} model {model_index} needs normal and d"
            )

        normal = np.asarray(model["normal"], dtype=np.float64)

        if normal.shape != (3,):
            raise ValueError(
                f"{detector_name} model {model_index}: invalid normal shape "
                f"{normal.shape}"
            )

        norm = float(np.linalg.norm(normal))

        if norm < 1e-12:
            continue

        normal = normal / norm
        d = float(model["d"])

        if "mask" in model:
            detector_mask = np.asarray(model["mask"], dtype=bool)

            if detector_mask.shape != (len(points),):
                raise ValueError(
                    f"{detector_name} model {model_index}: mask shape "
                    f"{detector_mask.shape}, expected {(len(points),)}"
                )
        else:
            # All uploaded detectors currently use common.build_model(), so
            # this is only a defensive fallback.
            detector_mask = (
                np.abs(points @ normal + d)
                < float(common.FINAL_DISTANCE_THRESHOLD)
            )

        output.append(
            {
                "normal": normal,
                "d": d,
                "detector_mask": detector_mask,
                "source_support": model.get("source_support"),
            }
        )

    return output


def attach_evaluation_masks(
    models: list[dict],
    points: np.ndarray,
    threshold_m: float,
) -> list[dict]:
    for model in models:
        model["evaluation_mask"] = (
            np.abs(points @ model["normal"] + model["d"])
            < threshold_m
        )

    return models


# ============================================================
# PLANE GEOMETRY FOR NEXT PIPELINE STAGE
# ============================================================

def make_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)

    if abs(normal[2]) < 0.9:
        reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    u = np.cross(normal, reference)
    u_norm = np.linalg.norm(u)

    if u_norm < 1e-12:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = np.cross(normal, reference)
        u_norm = np.linalg.norm(u)

    u = u / u_norm
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    return u, v


def resample_closed_polygon(
    polygon_xyz: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    polygon_xyz = np.asarray(polygon_xyz, dtype=np.float64)

    if len(polygon_xyz) == 0:
        return np.zeros((sample_count, 3), dtype=np.float64)

    if len(polygon_xyz) == 1:
        return np.repeat(polygon_xyz, sample_count, axis=0)

    closed = np.vstack([polygon_xyz, polygon_xyz[0]])
    edge_vectors = np.diff(closed, axis=0)
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    perimeter = float(np.sum(edge_lengths))

    if perimeter < 1e-12:
        return np.repeat(polygon_xyz[:1], sample_count, axis=0)

    cumulative = np.concatenate(
        [[0.0], np.cumsum(edge_lengths)]
    )

    target_distances = np.linspace(
        0.0,
        perimeter,
        sample_count,
        endpoint=False,
    )

    samples = []

    for distance in target_distances:
        edge_index = int(
            np.searchsorted(cumulative, distance, side="right") - 1
        )
        edge_index = min(edge_index, len(edge_lengths) - 1)

        edge_start_distance = cumulative[edge_index]
        length = edge_lengths[edge_index]

        alpha = (
            (distance - edge_start_distance) / length
            if length > 1e-12
            else 0.0
        )

        point = (
            (1.0 - alpha) * closed[edge_index]
            + alpha * closed[edge_index + 1]
        )
        samples.append(point)

    return np.asarray(samples, dtype=np.float64)


def derive_bounded_plane_geometry(
    points: np.ndarray,
    normal: np.ndarray,
    support_mask: np.ndarray,
) -> dict:
    support_points = points[support_mask]

    if len(support_points) == 0:
        centroid = np.zeros(3, dtype=np.float64)
        contour = np.repeat(
            centroid[None, :],
            CONTOUR_SAMPLES,
            axis=0,
        )
        return {
            "centroid": centroid,
            "area": 0.0,
            "width": 0.0,
            "height": 0.0,
            "contour": contour,
            "support_count": 0,
        }

    centroid = np.mean(support_points, axis=0)
    centred = support_points - centroid

    # Build a stable in-plane basis. Prefer PCA's strongest in-plane
    # direction when possible so width/height are reproducible.
    u, v = make_plane_basis(normal)

    if len(support_points) >= 3:
        covariance = centred.T @ centred
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        principal = eigenvectors[:, int(np.argmax(eigenvalues))]
        principal = principal - normal * float(np.dot(principal, normal))
        principal_norm = np.linalg.norm(principal)

        if principal_norm > 1e-12:
            u = principal / principal_norm
            v = np.cross(normal, u)
            v = v / np.linalg.norm(v)

    coordinates = np.column_stack(
        [centred @ u, centred @ v]
    )

    width = float(np.ptp(coordinates[:, 0])) if len(coordinates) else 0.0
    height = float(np.ptp(coordinates[:, 1])) if len(coordinates) else 0.0

    polygon_xyz = None
    area = 0.0

    if len(coordinates) >= 3:
        try:
            hull = ConvexHull(coordinates)
            hull_coordinates = coordinates[hull.vertices]
            polygon_xyz = (
                centroid[None, :]
                + hull_coordinates[:, 0:1] * u[None, :]
                + hull_coordinates[:, 1:2] * v[None, :]
            )
            # For a 2-D scipy ConvexHull, .volume is polygon area.
            area = float(hull.volume)
        except QhullError:
            polygon_xyz = None

    if polygon_xyz is None or len(polygon_xyz) < 3:
        # Defensive fallback. Use an oriented bounding rectangle.
        min_u = float(np.min(coordinates[:, 0]))
        max_u = float(np.max(coordinates[:, 0]))
        min_v = float(np.min(coordinates[:, 1]))
        max_v = float(np.max(coordinates[:, 1]))

        rectangle = np.array(
            [
                [min_u, min_v],
                [max_u, min_v],
                [max_u, max_v],
                [min_u, max_v],
            ],
            dtype=np.float64,
        )

        polygon_xyz = (
            centroid[None, :]
            + rectangle[:, 0:1] * u[None, :]
            + rectangle[:, 1:2] * v[None, :]
        )
        area = max(0.0, width * height)

    contour = resample_closed_polygon(
        polygon_xyz,
        CONTOUR_SAMPLES,
    )

    return {
        "centroid": centroid,
        "area": area,
        "width": width,
        "height": height,
        "contour": contour,
        "support_count": int(len(support_points)),
    }


def save_detection_output(
    detector_name: str,
    frame_id: int,
    points: np.ndarray,
    models: list[dict],
) -> Path:
    algorithm_dir = DETECTIONS_DIR / detector_name
    algorithm_dir.mkdir(parents=True, exist_ok=True)

    output_file = algorithm_dir / f"frame_{frame_id:04d}.npz"

    if not models:
        np.savez_compressed(
            output_file,
            normals=np.empty((0, 3), dtype=np.float64),
            offsets=np.empty((0,), dtype=np.float64),
            centroids=np.empty((0, 3), dtype=np.float64),
            areas=np.empty((0,), dtype=np.float64),
            widths=np.empty((0,), dtype=np.float64),
            heights=np.empty((0,), dtype=np.float64),
            contours=np.empty((0, CONTOUR_SAMPLES, 3), dtype=np.float64),
            support_counts=np.empty((0,), dtype=np.int32),
            detector_support_masks=np.empty((0, len(points)), dtype=bool),
            evaluation_support_masks=np.empty((0, len(points)), dtype=bool),
        )
        return output_file

    bounded = [
        derive_bounded_plane_geometry(
            points,
            model["normal"],
            model["detector_mask"],
        )
        for model in models
    ]

    np.savez_compressed(
        output_file,
        normals=np.stack([m["normal"] for m in models], axis=0),
        offsets=np.asarray([m["d"] for m in models], dtype=np.float64),
        centroids=np.stack([g["centroid"] for g in bounded], axis=0),
        areas=np.asarray([g["area"] for g in bounded], dtype=np.float64),
        widths=np.asarray([g["width"] for g in bounded], dtype=np.float64),
        heights=np.asarray([g["height"] for g in bounded], dtype=np.float64),
        contours=np.stack([g["contour"] for g in bounded], axis=0),
        support_counts=np.asarray(
            [g["support_count"] for g in bounded],
            dtype=np.int32,
        ),
        detector_support_masks=np.stack(
            [m["detector_mask"] for m in models],
            axis=0,
        ),
        evaluation_support_masks=np.stack(
            [m["evaluation_mask"] for m in models],
            axis=0,
        ),
    )

    return output_file


# ============================================================
# EVALUATION
# ============================================================

def plane_error(
    det_n: np.ndarray,
    det_d: float,
    gt_n: np.ndarray,
    gt_d: float,
) -> tuple[float, float]:
    det_n = np.asarray(det_n, dtype=np.float64)
    gt_n = np.asarray(gt_n, dtype=np.float64)

    det_n = det_n / np.linalg.norm(det_n)
    gt_n = gt_n / np.linalg.norm(gt_n)

    dot = float(np.dot(det_n, gt_n))

    # (n,d) and (-n,-d) represent the same plane.
    if dot < 0.0:
        det_n = -det_n
        det_d = -det_d
        dot = -dot

    dot = float(np.clip(dot, 0.0, 1.0))

    return (
        math.degrees(math.acos(dot)),
        abs(float(det_d) - float(gt_d)),
    )


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)
    return float(intersection / union) if union else 0.0


def evaluate_frame(
    models: list[dict],
    reference: dict,
    match_angle_threshold_deg: float,
    match_offset_threshold_m: float,
) -> dict:
    n_det = len(models)
    n_gt = len(reference["normals"])

    if n_det == 0:
        return {
            "detected_count": 0,
            "gt_count": n_gt,
            "tp": 0,
            "fp": 0,
            "fn": n_gt,
            "precision": 0.0,
            "recall": 0.0 if n_gt else 1.0,
            "f1": 0.0 if n_gt else 1.0,
            "correct_plane_count": float(n_gt == 0),
            "mean_iou": 0.0,
            "mean_normal_error_deg": np.nan,
            "mean_offset_error_m": np.nan,
        }

    if n_gt == 0:
        return {
            "detected_count": n_det,
            "gt_count": 0,
            "tp": 0,
            "fp": n_det,
            "fn": 0,
            "precision": 0.0,
            "recall": 1.0,
            "f1": 0.0,
            "correct_plane_count": 0.0,
            "mean_iou": 0.0,
            "mean_normal_error_deg": np.nan,
            "mean_offset_error_m": np.nan,
        }

    large = 1e9
    cost = np.full((n_det, n_gt), large, dtype=np.float64)
    angles = np.full((n_det, n_gt), np.nan, dtype=np.float64)
    offsets = np.full((n_det, n_gt), np.nan, dtype=np.float64)
    ious = np.zeros((n_det, n_gt), dtype=np.float64)

    for i, model in enumerate(models):
        for j in range(n_gt):
            angle, offset_error = plane_error(
                model["normal"],
                model["d"],
                reference["normals"][j],
                reference["offsets"][j],
            )
            iou = mask_iou(
                model["evaluation_mask"],
                reference["masks"][j],
            )

            angles[i, j] = angle
            offsets[i, j] = offset_error
            ious[i, j] = iou

            if (
                angle <= match_angle_threshold_deg
                and offset_error <= match_offset_threshold_m
            ):
                cost[i, j] = (
                    angle / match_angle_threshold_deg
                    + offset_error / match_offset_threshold_m
                )

    rows, cols = linear_sum_assignment(cost)
    matches = [
        (i, j)
        for i, j in zip(rows, cols)
        if cost[i, j] < large / 2
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
        mean_normal_error = float(
            np.mean([angles[i, j] for i, j in matches])
        )
        mean_offset_error = float(
            np.mean([offsets[i, j] for i, j in matches])
        )
    else:
        mean_iou = 0.0
        mean_normal_error = np.nan
        mean_offset_error = np.nan

    return {
        "detected_count": n_det,
        "gt_count": n_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "correct_plane_count": float(n_det == n_gt),
        "mean_iou": mean_iou,
        "mean_normal_error_deg": mean_normal_error,
        "mean_offset_error_m": mean_offset_error,
    }


def finite_mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else np.nan


def finite_std(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.std(array, ddof=0)) if len(array) else np.nan


# ============================================================
# OUTPUT HELPERS
# ============================================================

def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def prepare_output_directory() -> None:
    if OUTPUT_DIR.exists():
        if OVERWRITE_FINAL_OUTPUT:
            shutil.rmtree(OUTPUT_DIR)
        else:
            raise FileExistsError(
                f"FINAL output folder already exists:\n{OUTPUT_DIR}"
            )

    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Cannot write empty CSV: {path}")

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN FINAL BENCHMARK
# ============================================================

def main() -> None:
    validate_detector_interfaces()

    dev_ids = get_dev_ids()
    final_ids = get_final_ids(dev_ids)

    (
        frozen_parameters,
        eval_support_threshold_m,
        match_angle_threshold_deg,
        match_offset_threshold_m,
    ) = load_frozen_configuration()

    # Validate every frame/reference before deleting or creating output.
    final_data = []

    for fid in final_ids:
        points, reference, frame_file = load_frame_and_reference(fid)
        final_data.append((fid, points, reference, frame_file))

    prepare_output_directory()

    print()
    print("=" * 96)
    print("TUM PLANE DETECTION — FINAL BENCHMARK")
    print("=" * 96)
    print(f"Sequence             : {SEQUENCE_NAME}")
    print(f"FINAL frames         : {len(final_data)}")
    print(f"FINAL frame IDs      : {final_ids}")
    print(f"DEV frame IDs        : {dev_ids if dev_ids else 'not checked'}")
    print(f"Eval support gate    : {eval_support_threshold_m:.3f} m")
    print(f"GT angle gate        : {match_angle_threshold_deg:.2f} deg")
    print(f"GT offset gate       : {match_offset_threshold_m:.3f} m")
    print("Parameters           : FROZEN — no tuning")
    print("=" * 96)

    result_rows = []
    summaries = []

    for detector_number, detector_name in enumerate(
        EXPECTED_ALGORITHMS,
        start=1,
    ):
        parameters = frozen_parameters[detector_name]
        apply_frozen_parameters(parameters)

        print()
        print("=" * 96)
        print(
            f"[{detector_number}/{len(EXPECTED_ALGORITHMS)}] "
            f"{detector_name}"
        )
        print("Frozen parameters:")
        print(json.dumps(json_safe(parameters), indent=4))
        print("=" * 96)

        detector_rows = []

        for frame_number, (fid, points, reference, _) in enumerate(
            final_data,
            start=1,
        ):
            # Apply again before every frame to guarantee no accidental state
            # mutation by a detector affects the next frame.
            apply_frozen_parameters(parameters)

            seed = BASE_RANDOM_SEED + fid

            start = time.perf_counter()
            models = run_detector_raw(detector_name, points, seed)
            runtime_ms = (time.perf_counter() - start) * 1000.0

            attach_evaluation_masks(
                models,
                points,
                eval_support_threshold_m,
            )

            metrics = evaluate_frame(
                models,
                reference,
                match_angle_threshold_deg,
                match_offset_threshold_m,
            )

            detection_file = save_detection_output(
                detector_name,
                fid,
                points,
                models,
            )

            row = {
                "algorithm": detector_name,
                "frame_id": fid,
                "detected_planes": metrics["detected_count"],
                "gt_planes": metrics["gt_count"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "correct_plane_count": metrics["correct_plane_count"],
                "mean_iou": metrics["mean_iou"],
                "mean_normal_error_deg": metrics[
                    "mean_normal_error_deg"
                ],
                "mean_offset_error_m": metrics[
                    "mean_offset_error_m"
                ],
                "runtime_ms": runtime_ms,
                "detection_file": str(
                    detection_file.relative_to(PROJECT_ROOT)
                ),
            }

            result_rows.append(row)
            detector_rows.append(row)

            print(
                f"[{frame_number:03d}/{len(final_data):03d}] "
                f"frame_{fid:04d} | "
                f"F1={metrics['f1']:.4f} | "
                f"IoU={metrics['mean_iou']:.4f} | "
                f"TP/FP/FN={metrics['tp']}/{metrics['fp']}/{metrics['fn']} | "
                f"{runtime_ms:.2f} ms"
            )

        total_tp = int(sum(row["tp"] for row in detector_rows))
        total_fp = int(sum(row["fp"] for row in detector_rows))
        total_fn = int(sum(row["fn"] for row in detector_rows))

        micro_precision = (
            total_tp / (total_tp + total_fp)
            if total_tp + total_fp
            else 0.0
        )
        micro_recall = (
            total_tp / (total_tp + total_fn)
            if total_tp + total_fn
            else 0.0
        )
        micro_f1 = (
            2.0 * micro_precision * micro_recall
            / (micro_precision + micro_recall)
            if micro_precision + micro_recall
            else 0.0
        )

        summary = {
            "algorithm": detector_name,
            "frames": len(detector_rows),
            "mean_precision": finite_mean(
                row["precision"] for row in detector_rows
            ),
            "mean_recall": finite_mean(
                row["recall"] for row in detector_rows
            ),
            "mean_f1": finite_mean(
                row["f1"] for row in detector_rows
            ),
            "std_f1": finite_std(
                row["f1"] for row in detector_rows
            ),
            "correct_plane_count_rate": finite_mean(
                row["correct_plane_count"] for row in detector_rows
            ),
            "mean_iou": finite_mean(
                row["mean_iou"] for row in detector_rows
            ),
            "std_iou": finite_std(
                row["mean_iou"] for row in detector_rows
            ),
            "mean_false_planes": finite_mean(
                row["fp"] for row in detector_rows
            ),
            "mean_missed_planes": finite_mean(
                row["fn"] for row in detector_rows
            ),
            "mean_normal_error_deg": finite_mean(
                row["mean_normal_error_deg"] for row in detector_rows
            ),
            "mean_offset_error_m": finite_mean(
                row["mean_offset_error_m"] for row in detector_rows
            ),
            "mean_runtime_ms": finite_mean(
                row["runtime_ms"] for row in detector_rows
            ),
            "std_runtime_ms": finite_std(
                row["runtime_ms"] for row in detector_rows
            ),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
        }

        summaries.append(summary)

        print()
        print(
            f"FINAL {detector_name}: "
            f"mean F1={summary['mean_f1']:.4f}, "
            f"mean IoU={summary['mean_iou']:.4f}, "
            f"micro F1={summary['micro_f1']:.4f}, "
            f"runtime={summary['mean_runtime_ms']:.2f} ms"
        )

    write_csv(FINAL_RESULTS_CSV, result_rows)
    write_csv(FINAL_SUMMARY_CSV, summaries)

    # Copy the exact parameter file used rather than reconstructing it.
    shutil.copy2(
        BEST_PARAMETERS_JSON,
        FROZEN_PARAMETERS_USED_JSON,
    )

    metadata = {
        "sequence": SEQUENCE_NAME,
        "final_frame_ids": final_ids,
        "dev_frame_ids_checked": dev_ids,
        "number_final_frames": len(final_ids),
        "evaluation_support_threshold_m": eval_support_threshold_m,
        "match_angle_threshold_deg": match_angle_threshold_deg,
        "match_offset_threshold_m": match_offset_threshold_m,
        "base_random_seed": BASE_RANDOM_SEED,
        "contour_samples": CONTOUR_SAMPLES,
        "parameter_source": str(
            BEST_PARAMETERS_JSON.relative_to(PROJECT_ROOT)
        ),
        "tuning_performed": False,
        "plane_output_convention": "n^T x + d = 0",
        "next_stage": "candidate_gating",
    }

    RUN_METADATA_JSON.write_text(
        json.dumps(json_safe(metadata), indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 96)
    print("TUM FINAL PLANE DETECTION COMPLETE")
    print("=" * 96)
    print(f"Per-frame results : {FINAL_RESULTS_CSV}")
    print(f"Summary           : {FINAL_SUMMARY_CSV}")
    print(f"Frozen params     : {FROZEN_PARAMETERS_USED_JSON}")
    print(f"Plane outputs     : {DETECTIONS_DIR}")
    print()
    print(
        "The saved plane outputs contain normals, offsets, centroids, "
        "areas, fixed-length contours and support masks, so the selected "
        "detector can feed Candidate Gating directly."
    )
    print("=" * 96)


if __name__ == "__main__":
    main()
