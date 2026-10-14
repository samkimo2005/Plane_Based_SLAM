"""
prepare_tum_plane_detection.py

Prepare TUM RGB-D data for the real-data Plane Detection benchmark.

Direct execution:
    Run this file directly from the IDE.

No command-line arguments are required.

Pipeline:

    TUM RGB + Depth + Ground Truth
                ↓
        Timestamp Association
                ↓
        Depth -> 3-D Point Cloud
                ↓
        Common Pre-processing
                ↓
        Saved Benchmark Frames

The generated point clouds are shared by ALL plane-detection
algorithms so that the detector comparison remains fair.

TUM ground-truth pose convention used here:

    T_world_cam

where a point in camera coordinates can be transformed into the
TUM world frame by:

    p_world = R_world_cam @ p_cam + t_world_cam
"""

from __future__ import annotations

import csv
import json
import math
import shutil

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError as exc:
    raise ImportError(
        "Pillow is required.\n"
        "Install it with:\n"
        "pip install pillow"
    ) from exc


# ============================================================
# USER SETTINGS
# ============================================================

# ------------------------------------------------------------
# TUM sequence
# ------------------------------------------------------------

SEQUENCE_NAME = "rgbd_dataset_freiburg1_xyz"

# Supported calibration families:
#
#   freiburg1
#   freiburg2
#   freiburg3
#
TUM_CAMERA = "freiburg1"


# ------------------------------------------------------------
# Frame sampling
# ------------------------------------------------------------

# Example:
#
#   FRAME_STRIDE = 10
#
# means keep one frame every 10 associated RGB-D frames.
#
# This is useful because TUM sequences may contain thousands
# of highly similar consecutive frames.
FRAME_STRIDE = 10

# None = no maximum.
#
# For an initial test you can temporarily use:
#
# MAX_FRAMES = 20
#
# After confirming the program works, set it back to None.
MAX_FRAMES = None


# ------------------------------------------------------------
# Timestamp association
# ------------------------------------------------------------

# Standard maximum timestamp difference used when associating
# TUM streams.
MAX_TIME_DIFFERENCE_SEC = 0.02


# ------------------------------------------------------------
# Depth filtering
# ------------------------------------------------------------

MIN_DEPTH_M = 0.30
MAX_DEPTH_M = 5.00


# ------------------------------------------------------------
# Detector cloud downsampling
# ------------------------------------------------------------

# Raw valid points are retained separately.
#
# detector_points are voxel-downsampled and are intended as the
# common input to all five plane detectors.
VOXEL_SIZE_M = 0.02


# ------------------------------------------------------------
# Output control
# ------------------------------------------------------------

OVERWRITE = True


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

RAW_SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "tum"
    / SEQUENCE_NAME
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "tum"
    / "plane_detection"
    / SEQUENCE_NAME
)

FRAMES_DIR = (
    OUTPUT_DIR
    / "frames"
)


# ============================================================
# TUM calibration
# ============================================================

TUM_CALIBRATION = {

    "freiburg1": {
        "fx": 517.3,
        "fy": 516.5,
        "cx": 318.6,
        "cy": 255.3,
        "depth_factor": 5000.0,
    },

    "freiburg2": {
        "fx": 520.9,
        "fy": 521.0,
        "cx": 325.1,
        "cy": 249.7,
        "depth_factor": 5000.0,
    },

    "freiburg3": {
        "fx": 535.4,
        "fy": 539.2,
        "cx": 320.1,
        "cy": 247.6,
        "depth_factor": 5000.0,
    },
}


# ============================================================
# Utility functions
# ============================================================

def quaternion_to_rotation_matrix(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> np.ndarray:
    """
    Convert quaternion (qx, qy, qz, qw) to 3x3 rotation matrix.
    """

    quaternion = np.array(
        [qx, qy, qz, qw],
        dtype=float,
    )

    norm = np.linalg.norm(
        quaternion
    )

    if norm < 1e-12:

        raise ValueError(
            "Invalid zero-length quaternion."
        )

    quaternion /= norm

    x, y, z, w = quaternion

    return np.array(
        [
            [
                1.0 - 2.0 * (y*y + z*z),
                2.0 * (x*y - z*w),
                2.0 * (x*z + y*w),
            ],

            [
                2.0 * (x*y + z*w),
                1.0 - 2.0 * (x*x + z*z),
                2.0 * (y*z - x*w),
            ],

            [
                2.0 * (x*z - y*w),
                2.0 * (y*z + x*w),
                1.0 - 2.0 * (x*x + y*y),
            ],
        ],
        dtype=float,
    )


def make_transform(
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:

    T = np.eye(
        4,
        dtype=float,
    )

    T[:3, :3] = R
    T[:3, 3] = t

    return T


# ============================================================
# TUM text-file readers
# ============================================================

def read_timestamp_file(
    file_path: Path,
) -> List[Tuple[float, str]]:
    """
    Read rgb.txt or depth.txt.

    Expected format:

        timestamp relative/path.png
    """

    records = []

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            timestamp = float(
                parts[0]
            )

            relative_path = (
                parts[1]
            )

            records.append(
                (
                    timestamp,
                    relative_path,
                )
            )

    return records


def read_groundtruth(
    file_path: Path,
) -> List[dict]:
    """
    Read TUM groundtruth.txt.

    Expected format:

    timestamp tx ty tz qx qy qz qw
    """

    records = []

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 8:
                continue

            values = [
                float(value)
                for value
                in parts[:8]
            ]

            (
                timestamp,
                tx,
                ty,
                tz,
                qx,
                qy,
                qz,
                qw,
            ) = values

            R = quaternion_to_rotation_matrix(
                qx,
                qy,
                qz,
                qw,
            )

            t = np.array(
                [
                    tx,
                    ty,
                    tz,
                ],
                dtype=float,
            )

            T_world_cam = make_transform(
                R,
                t,
            )

            records.append(
                {
                    "timestamp":
                        timestamp,

                    "translation":
                        t,

                    "quaternion":
                        np.array(
                            [
                                qx,
                                qy,
                                qz,
                                qw,
                            ],
                            dtype=float,
                        ),

                    "R_world_cam":
                        R,

                    "T_world_cam":
                        T_world_cam,
                }
            )

    return records


# ============================================================
# Timestamp association
# ============================================================

def associate_two_streams(
    first: List,
    second: List,
    first_timestamp_fn,
    second_timestamp_fn,
    max_difference: float,
) -> List[Tuple[int, int, float]]:
    """
    Greedy one-to-one nearest-timestamp association.
    """

    potential_matches = []

    for first_index, first_item in enumerate(
        first
    ):

        first_time = float(
            first_timestamp_fn(
                first_item
            )
        )

        for second_index, second_item in enumerate(
            second
        ):

            second_time = float(
                second_timestamp_fn(
                    second_item
                )
            )

            difference = abs(
                first_time
                - second_time
            )

            if difference <= max_difference:

                potential_matches.append(
                    (
                        difference,
                        first_index,
                        second_index,
                    )
                )

    potential_matches.sort(
        key=lambda item: item[0]
    )

    used_first = set()
    used_second = set()

    matches = []

    for (
        difference,
        first_index,
        second_index,
    ) in potential_matches:

        if first_index in used_first:
            continue

        if second_index in used_second:
            continue

        used_first.add(
            first_index
        )

        used_second.add(
            second_index
        )

        matches.append(
            (
                first_index,
                second_index,
                difference,
            )
        )

    matches.sort(
        key=lambda item: item[0]
    )

    return matches


def associate_tum_streams(
    rgb_records,
    depth_records,
    gt_records,
):
    """
    First associate RGB ↔ Depth, then associate the RGB timestamp
    of each RGB-D pair ↔ Ground Truth.
    """

    rgb_depth_matches = associate_two_streams(

        first=rgb_records,
        second=depth_records,

        first_timestamp_fn=lambda item: item[0],
        second_timestamp_fn=lambda item: item[0],

        max_difference=MAX_TIME_DIFFERENCE_SEC,
    )

    rgbd_records = []

    for (
        rgb_index,
        depth_index,
        difference,
    ) in rgb_depth_matches:

        rgb_timestamp, rgb_path = (
            rgb_records[
                rgb_index
            ]
        )

        depth_timestamp, depth_path = (
            depth_records[
                depth_index
            ]
        )

        rgbd_records.append(
            {
                "rgb_timestamp":
                    rgb_timestamp,

                "rgb_path":
                    rgb_path,

                "depth_timestamp":
                    depth_timestamp,

                "depth_path":
                    depth_path,

                "rgb_depth_dt":
                    difference,
            }
        )

    rgbd_gt_matches = associate_two_streams(

        first=rgbd_records,
        second=gt_records,

        first_timestamp_fn=(
            lambda item:
            item["rgb_timestamp"]
        ),

        second_timestamp_fn=(
            lambda item:
            item["timestamp"]
        ),

        max_difference=MAX_TIME_DIFFERENCE_SEC,
    )

    associated = []

    for (
        rgbd_index,
        gt_index,
        difference,
    ) in rgbd_gt_matches:

        record = dict(
            rgbd_records[
                rgbd_index
            ]
        )

        record[
            "groundtruth"
        ] = gt_records[
            gt_index
        ]

        record[
            "rgb_gt_dt"
        ] = difference

        associated.append(
            record
        )

    associated.sort(
        key=lambda item:
        item["rgb_timestamp"]
    )

    return associated


# ============================================================
# RGB-D loading
# ============================================================

def load_rgb_image(
    file_path: Path,
) -> np.ndarray:

    image = Image.open(
        file_path
    ).convert(
        "RGB"
    )

    return np.asarray(
        image,
        dtype=np.uint8,
    )


def load_depth_image(
    file_path: Path,
) -> np.ndarray:

    image = Image.open(
        file_path
    )

    depth = np.asarray(
        image
    )

    return depth


# ============================================================
# Depth -> point cloud
# ============================================================

def depth_to_point_cloud(
    depth_raw: np.ndarray,
    rgb: np.ndarray,
    calibration: dict,
):
    """
    Convert valid TUM depth pixels into 3-D camera-frame points.

    Camera convention:

        X right
        Y down
        Z forward
    """

    fx = float(
        calibration["fx"]
    )

    fy = float(
        calibration["fy"]
    )

    cx = float(
        calibration["cx"]
    )

    cy = float(
        calibration["cy"]
    )

    depth_factor = float(
        calibration[
            "depth_factor"
        ]
    )

    depth_m = (
        depth_raw.astype(
            np.float32
        )
        / depth_factor
    )

    valid = (
        np.isfinite(
            depth_m
        )
        & (depth_m >= MIN_DEPTH_M)
        & (depth_m <= MAX_DEPTH_M)
    )

    v, u = np.nonzero(
        valid
    )

    z = depth_m[
        v,
        u,
    ]

    x = (
        (u.astype(float) - cx)
        * z
        / fx
    )

    y = (
        (v.astype(float) - cy)
        * z
        / fy
    )

    points = np.column_stack(
        (
            x,
            y,
            z,
        )
    ).astype(
        np.float32
    )

    colours = (
        rgb[
            v,
            u,
        ]
        .astype(
            np.float32
        )
        / 255.0
    )

    pixels = np.column_stack(
        (
            u,
            v,
        )
    ).astype(
        np.int32
    )

    return (
        points,
        colours,
        pixels,
        valid,
        depth_m,
    )


# ============================================================
# Voxel downsampling
# ============================================================

def voxel_downsample(
    points: np.ndarray,
    colours: np.ndarray,
    voxel_size: float,
):
    """
    Deterministic NumPy voxel-grid downsampling.

    One centroid and mean colour are produced for each occupied
    voxel.
    """

    if voxel_size <= 0:

        return (
            points.copy(),
            colours.copy(),
        )

    if len(points) == 0:

        return (
            points.copy(),
            colours.copy(),
        )

    voxel_indices = np.floor(
        points
        / voxel_size
    ).astype(
        np.int64
    )

    unique_voxels, inverse = np.unique(
        voxel_indices,
        axis=0,
        return_inverse=True,
    )

    number_voxels = len(
        unique_voxels
    )

    downsampled_points = np.zeros(
        (
            number_voxels,
            3,
        ),
        dtype=np.float64,
    )

    downsampled_colours = np.zeros(
        (
            number_voxels,
            3,
        ),
        dtype=np.float64,
    )

    counts = np.bincount(
        inverse
    ).astype(
        np.float64
    )

    for dimension in range(
        3
    ):

        downsampled_points[
            :,
            dimension
        ] = np.bincount(
            inverse,
            weights=points[
                :,
                dimension
            ],
        )

        downsampled_colours[
            :,
            dimension
        ] = np.bincount(
            inverse,
            weights=colours[
                :,
                dimension
            ],
        )

    downsampled_points /= (
        counts[:, None]
    )

    downsampled_colours /= (
        counts[:, None]
    )

    return (
        downsampled_points.astype(
            np.float32
        ),

        downsampled_colours.astype(
            np.float32
        ),
    )


# ============================================================
# Output directory
# ============================================================

def prepare_output_directory():

    if OUTPUT_DIR.exists():

        if OVERWRITE:

            shutil.rmtree(
                OUTPUT_DIR
            )

        else:

            raise FileExistsError(
                "\nOutput directory already exists:\n"
                f"{OUTPUT_DIR}\n\n"
                "Set OVERWRITE = True if you want to regenerate it."
            )

    FRAMES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Input validation
# ============================================================

def validate_input_files():

    required_files = [

        RAW_SEQUENCE_DIR
        / "rgb.txt",

        RAW_SEQUENCE_DIR
        / "depth.txt",

        RAW_SEQUENCE_DIR
        / "groundtruth.txt",
    ]

    missing = [

        path
        for path
        in required_files
        if not path.exists()
    ]

    if missing:

        message = "\nMissing TUM files:\n"

        for path in missing:

            message += (
                f"  {path}\n"
            )

        message += (
            "\nCheck SEQUENCE_NAME and RAW_SEQUENCE_DIR."
        )

        raise FileNotFoundError(
            message
        )


# ============================================================
# Main preparation
# ============================================================

def prepare_tum_dataset():

    if TUM_CAMERA not in TUM_CALIBRATION:

        raise ValueError(
            f"Unknown TUM_CAMERA: {TUM_CAMERA}"
        )

    validate_input_files()

    calibration = (
        TUM_CALIBRATION[
            TUM_CAMERA
        ]
    )

    print()
    print("=" * 80)
    print(
        "TUM Plane Detection Dataset Preparation"
    )
    print("=" * 80)

    print(
        f"Sequence        : {SEQUENCE_NAME}"
    )

    print(
        f"Camera profile  : {TUM_CAMERA}"
    )

    print(
        f"Raw folder      : {RAW_SEQUENCE_DIR}"
    )

    print(
        f"Output folder   : {OUTPUT_DIR}"
    )

    print(
        f"Frame stride    : {FRAME_STRIDE}"
    )

    print(
        f"Depth range     : "
        f"{MIN_DEPTH_M:.2f} - "
        f"{MAX_DEPTH_M:.2f} m"
    )

    print(
        f"Voxel size      : "
        f"{VOXEL_SIZE_M:.3f} m"
    )

    print("=" * 80)
    print()

    # --------------------------------------------------------
    # Read TUM metadata
    # --------------------------------------------------------

    rgb_records = read_timestamp_file(
        RAW_SEQUENCE_DIR
        / "rgb.txt"
    )

    depth_records = read_timestamp_file(
        RAW_SEQUENCE_DIR
        / "depth.txt"
    )

    gt_records = read_groundtruth(
        RAW_SEQUENCE_DIR
        / "groundtruth.txt"
    )

    print(
        f"RGB entries         : {len(rgb_records)}"
    )

    print(
        f"Depth entries       : {len(depth_records)}"
    )

    print(
        f"Ground-truth poses  : {len(gt_records)}"
    )

    # --------------------------------------------------------
    # Associate streams
    # --------------------------------------------------------

    associated = associate_tum_streams(

        rgb_records,
        depth_records,
        gt_records,
    )

    print(
        f"Associated RGB-D-GT : {len(associated)}"
    )

    # --------------------------------------------------------
    # Temporal sampling
    # --------------------------------------------------------

    selected = associated[
        ::FRAME_STRIDE
    ]

    if MAX_FRAMES is not None:

        selected = selected[
            :MAX_FRAMES
        ]

    print(
        f"Selected frames     : {len(selected)}"
    )

    print()

    prepare_output_directory()

    manifest_rows = []

    # --------------------------------------------------------
    # Process each selected frame
    # --------------------------------------------------------

    for output_index, record in enumerate(
        selected
    ):

        rgb_path = (
            RAW_SEQUENCE_DIR
            / record[
                "rgb_path"
            ]
        )

        depth_path = (
            RAW_SEQUENCE_DIR
            / record[
                "depth_path"
            ]
        )

        if not rgb_path.exists():

            raise FileNotFoundError(
                rgb_path
            )

        if not depth_path.exists():

            raise FileNotFoundError(
                depth_path
            )

        rgb = load_rgb_image(
            rgb_path
        )

        depth_raw = load_depth_image(
            depth_path
        )

        (
            raw_points,
            raw_colours,
            raw_pixels,
            valid_depth_mask,
            depth_m,
        ) = depth_to_point_cloud(

            depth_raw,
            rgb,
            calibration,
        )

        (
            detector_points,
            detector_colours,
        ) = voxel_downsample(

            raw_points,
            raw_colours,
            VOXEL_SIZE_M,
        )

        gt = record[
            "groundtruth"
        ]

        frame_name = (
            f"frame_"
            f"{output_index:04d}"
        )

        output_file = (
            FRAMES_DIR
            / f"{frame_name}.npz"
        )

        # ----------------------------------------------------
        # Save everything needed by later benchmark stages
        # ----------------------------------------------------

        np.savez_compressed(

            output_file,

            # Common detector input.
            detector_points=(
                detector_points
            ),

            detector_colours=(
                detector_colours
            ),

            # Raw organised-depth-derived points.
            raw_points=(
                raw_points
            ),

            raw_colours=(
                raw_colours
            ),

            raw_pixels=(
                raw_pixels
            ),

            valid_depth_mask=(
                valid_depth_mask
            ),

            depth_m=(
                depth_m.astype(
                    np.float32
                )
            ),

            rgb=rgb,

            # Pose.
            R_world_cam=(
                gt[
                    "R_world_cam"
                ]
            ),

            T_world_cam=(
                gt[
                    "T_world_cam"
                ]
            ),

            t_world_cam=(
                gt[
                    "translation"
                ]
            ),

            quaternion_xyzw=(
                gt[
                    "quaternion"
                ]
            ),

            # Calibration.
            camera_intrinsics=np.array(
                [
                    [
                        calibration["fx"],
                        0.0,
                        calibration["cx"],
                    ],
                    [
                        0.0,
                        calibration["fy"],
                        calibration["cy"],
                    ],
                    [
                        0.0,
                        0.0,
                        1.0,
                    ],
                ],
                dtype=float,
            ),

            image_size=np.array(
                [
                    rgb.shape[1],
                    rgb.shape[0],
                ],
                dtype=np.int32,
            ),

            rgb_timestamp=np.array(
                record[
                    "rgb_timestamp"
                ],
                dtype=float,
            ),

            depth_timestamp=np.array(
                record[
                    "depth_timestamp"
                ],
                dtype=float,
            ),

            groundtruth_timestamp=np.array(
                gt[
                    "timestamp"
                ],
                dtype=float,
            ),
        )

        manifest_rows.append(
            {
                "frame_id":
                    output_index,

                "frame_file":
                    output_file.name,

                "rgb_timestamp":
                    record[
                        "rgb_timestamp"
                    ],

                "depth_timestamp":
                    record[
                        "depth_timestamp"
                    ],

                "gt_timestamp":
                    gt[
                        "timestamp"
                    ],

                "rgb_depth_dt":
                    record[
                        "rgb_depth_dt"
                    ],

                "rgb_gt_dt":
                    record[
                        "rgb_gt_dt"
                    ],

                "raw_valid_points":
                    len(
                        raw_points
                    ),

                "detector_points":
                    len(
                        detector_points
                    ),

                "rgb_path":
                    record[
                        "rgb_path"
                    ],

                "depth_path":
                    record[
                        "depth_path"
                    ],
            }
        )

        print(
            f"[{output_index + 1:4d}/"
            f"{len(selected):4d}] "
            f"{frame_name}: "
            f"{len(raw_points):7d} raw -> "
            f"{len(detector_points):6d} detector points"
        )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest_file = (
        OUTPUT_DIR
        / "manifest.csv"
    )

    with open(
        manifest_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "frame_id",
            "frame_file",
            "rgb_timestamp",
            "depth_timestamp",
            "gt_timestamp",
            "rgb_depth_dt",
            "rgb_gt_dt",
            "raw_valid_points",
            "detector_points",
            "rgb_path",
            "depth_path",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            manifest_rows
        )

    # --------------------------------------------------------
    # Dataset metadata
    # --------------------------------------------------------

    metadata = {

        "sequence_name":
            SEQUENCE_NAME,

        "camera_profile":
            TUM_CAMERA,

        "calibration":
            calibration,

        "frame_stride":
            FRAME_STRIDE,

        "maximum_frames":
            MAX_FRAMES,

        "maximum_timestamp_difference_sec":
            MAX_TIME_DIFFERENCE_SEC,

        "minimum_depth_m":
            MIN_DEPTH_M,

        "maximum_depth_m":
            MAX_DEPTH_M,

        "voxel_size_m":
            VOXEL_SIZE_M,

        "number_associated_frames":
            len(
                associated
            ),

        "number_selected_frames":
            len(
                selected
            ),

        "coordinate_system":
            {
                "camera_x":
                    "right",

                "camera_y":
                    "down",

                "camera_z":
                    "forward",
            },

        "pose_convention":
            "p_world = R_world_cam @ p_cam + t_world_cam",

        "purpose":
            (
                "Common TUM RGB-D input for all Plane Detection "
                "benchmark algorithms."
            ),
    }

    metadata_file = (
        OUTPUT_DIR
        / "dataset_metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print()
    print("=" * 80)

    print(
        "TUM PLANE DETECTION PREPARATION COMPLETE"
    )

    print("=" * 80)

    print(
        f"Frames prepared : {len(selected)}"
    )

    if manifest_rows:

        mean_raw = np.mean(
            [
                row[
                    "raw_valid_points"
                ]
                for row
                in manifest_rows
            ]
        )

        mean_detector = np.mean(
            [
                row[
                    "detector_points"
                ]
                for row
                in manifest_rows
            ]
        )

        print(
            f"Mean raw points : {mean_raw:.0f}"
        )

        print(
            f"Mean detector points : "
            f"{mean_detector:.0f}"
        )

    print()

    print(
        f"Frames:"
    )

    print(
        FRAMES_DIR
    )

    print()

    print(
        f"Manifest:"
    )

    print(
        manifest_file
    )

    print()

    print(
        f"Metadata:"
    )

    print(
        metadata_file
    )

    print("=" * 80)
    print()


# ============================================================
# Main
# ============================================================

def main():
    prepare_tum_dataset()


if __name__ == "__main__":
    main()