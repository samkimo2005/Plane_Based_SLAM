# run_downsampling_realdata.py

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

import downsampling_methods as dm


# ============================================================
# Project paths
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = THIS_DIR.parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

TUM_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "tum"
    / "rgbd_dataset_freiburg1_xyz"
)

OUTPUT_DIR = (
    EXPERIMENTS_DIR
    / "results"
    / "downsampling"
    / "real"
)

RESULT_FILE = (
    OUTPUT_DIR
    / "real_results.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "real_summary.csv"
)

CONFIG_FILE = (
    OUTPUT_DIR
    / "real_config.json"
)


# ============================================================
# Import shared TUM loader
# ============================================================

if not SRC_DIR.exists():
    raise FileNotFoundError(
        f"src directory not found:\n"
        f"{SRC_DIR}"
    )

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )

from tum_loader import TUMDataset


# ============================================================
# Experiment configuration
# ============================================================

METHODS = [
    "Voxel",
    "FPS",
    "Uniform",
    "Random",
]

RETENTION_LEVELS = [
    0.50,
    0.25,
    0.10,
]

# 79 real frames distributed across the sequence.
N_REAL_FRAMES = 79

# Raw TUM RGB-D frames can contain >200,000 valid points.
# FPS at 50% retention would otherwise become excessively slow.
#
# Therefore every selected frame is converted into one fixed
# benchmark cloud of at most 12,500 points.
#
# All four algorithms receive exactly the same benchmark cloud.
MAX_BENCHMARK_POINTS = 12_500

BENCHMARK_BASE_SEED = 20261005
RANDOM_BASE_SEED = 20261006

VOXEL_SEARCH_ITERATIONS = 20


# ============================================================
# TUM Freiburg 1 intrinsics
# ============================================================

FX = 517.3
FY = 516.5
CX = 318.6
CY = 255.3

DEPTH_SCALE = 5000.0

MIN_DEPTH_M = 0.30
MAX_DEPTH_M = 5.00


# ============================================================
# Validate fixed project paths
# ============================================================

def validate_paths() -> None:

    if not TUM_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"TUM dataset not found:\n"
            f"{TUM_DATASET_DIR}"
        )

    rgb_txt = (
        TUM_DATASET_DIR
        / "rgb.txt"
    )

    depth_txt = (
        TUM_DATASET_DIR
        / "depth.txt"
    )

    if not rgb_txt.exists():
        raise FileNotFoundError(
            f"rgb.txt not found:\n"
            f"{rgb_txt}"
        )

    if not depth_txt.exists():
        raise FileNotFoundError(
            f"depth.txt not found:\n"
            f"{depth_txt}"
        )

    tum_loader_file = (
        SRC_DIR
        / "tum_loader.py"
    )

    if not tum_loader_file.exists():
        raise FileNotFoundError(
            f"tum_loader.py not found:\n"
            f"{tum_loader_file}"
        )


# ============================================================
# Depth image -> 3D point cloud
# ============================================================

def depth_to_point_cloud(
    depth: np.ndarray,
) -> np.ndarray:

    depth = np.asarray(
        depth
    )

    if depth.ndim != 2:
        raise ValueError(
            "Depth image must be a 2-D array."
        )

    # TUM depth PNG:
    #
    # Z [m] = raw_depth / 5000
    #
    z = (
        depth.astype(
            np.float64
        )
        / DEPTH_SCALE
    )

    valid = (
        np.isfinite(z)
        & (z >= MIN_DEPTH_M)
        & (z <= MAX_DEPTH_M)
    )

    v, u = np.nonzero(
        valid
    )

    z_valid = z[
        v,
        u,
    ]

    x = (
        (
            u.astype(
                np.float64
            )
            - CX
        )
        * z_valid
        / FX
    )

    y = (
        (
            v.astype(
                np.float64
            )
            - CY
        )
        * z_valid
        / FY
    )

    points = np.column_stack(
        (
            x,
            y,
            z_valid,
        )
    )

    if len(points) == 0:
        raise RuntimeError(
            "No valid 3D points were generated "
            "from the depth image."
        )

    return points


# ============================================================
# Construct common benchmark cloud
# ============================================================

def create_benchmark_cloud(
    raw_points: np.ndarray,
    frame_index: int,
) -> np.ndarray:

    """
    Reduce a very large raw RGB-D cloud to a common,
    deterministic benchmark size before comparing the
    four downsampling methods.

    The same benchmark cloud is supplied to all methods.

    If the frame already contains <= MAX_BENCHMARK_POINTS,
    it is left unchanged.
    """

    raw_points = np.asarray(
        raw_points,
        dtype=np.float64,
    )

    if (
        len(raw_points)
        <= MAX_BENCHMARK_POINTS
    ):
        return raw_points.copy()

    rng = np.random.default_rng(
        BENCHMARK_BASE_SEED
        + int(frame_index)
    )

    indices = rng.choice(
        len(raw_points),
        size=MAX_BENCHMARK_POINTS,
        replace=False,
    )

    # Restore original RGB-D raster ordering.
    # This is important because Uniform is explicitly
    # an ordered every-k sampling method.
    indices.sort()

    return raw_points[
        indices
    ]


# ============================================================
# Open3D helper
# ============================================================

def make_cloud(
    points: np.ndarray,
) -> o3d.geometry.PointCloud:

    cloud = (
        o3d.geometry.PointCloud()
    )

    cloud.points = (
        o3d.utility.Vector3dVector(
            np.asarray(
                points,
                dtype=np.float64,
            )
        )
    )

    return cloud


# ============================================================
# Spatial-density regularity
# ============================================================

def nearest_neighbour_statistics(
    points: np.ndarray,
) -> tuple[
    float,
    float,
    float,
]:

    if len(points) < 2:

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    cloud = make_cloud(
        points
    )

    distances = np.asarray(
        cloud.compute_nearest_neighbor_distance(),
        dtype=np.float64,
    )

    if len(distances) == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    mean_distance = float(
        np.mean(
            distances
        )
    )

    std_distance = float(
        np.std(
            distances
        )
    )

    if mean_distance <= 1e-12:

        cv = np.nan

    else:

        cv = (
            std_distance
            / mean_distance
        )

    return (
        mean_distance,
        std_distance,
        float(cv),
    )


# ============================================================
# Geometric coverage preservation
# ============================================================

def coverage_error(
    original_points: np.ndarray,
    sampled_points: np.ndarray,
) -> tuple[
    float,
    float,
    float,
]:

    """
    Measure how well the downsampled point cloud covers
    the original benchmark cloud.

    For each original point p:

        e(p) = min ||p - q||

    where q belongs to the downsampled cloud.

    Returns:
        mean coverage error
        median coverage error
        P95 coverage error

    Lower is better.
    """

    if len(sampled_points) == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    original_cloud = make_cloud(
        original_points
    )

    sampled_cloud = make_cloud(
        sampled_points
    )

    distances = np.asarray(
        original_cloud.compute_point_cloud_distance(
            sampled_cloud
        ),
        dtype=np.float64,
    )

    distances = distances[
        np.isfinite(
            distances
        )
    ]

    if len(distances) == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    mean_error = float(
        np.mean(
            distances
        )
    )

    median_error = float(
        np.median(
            distances
        )
    )

    p95_error = float(
        np.percentile(
            distances,
            95,
        )
    )

    return (
        mean_error,
        median_error,
        p95_error,
    )


# ============================================================
# Fixed-parameter voxel operation
# ============================================================

def apply_voxel_downsample(
    points: np.ndarray,
    voxel_size: float,
) -> np.ndarray:

    cloud = make_cloud(
        points
    )

    sampled_cloud = (
        cloud.voxel_down_sample(
            voxel_size=float(
                voxel_size
            )
        )
    )

    return np.asarray(
        sampled_cloud.points,
        dtype=np.float64,
    )


# ============================================================
# Run one downsampling method
# ============================================================

def run_method(
    points: np.ndarray,
    method: str,
    retention_ratio: float,
    random_seed: int,
):

    calibration_ms = 0.0

    # --------------------------------------------------------
    # Voxel
    # --------------------------------------------------------

    if method == "Voxel":

        # Search for the voxel size necessary to achieve
        # approximately the requested retention ratio.
        #
        # This calibration/search cost is NOT included in
        # the operational runtime comparison.

        start = time.perf_counter()

        calibrated = (
            dm.voxel_downsample(
                points=points,
                retention_ratio=(
                    retention_ratio
                ),
                search_iterations=(
                    VOXEL_SEARCH_ITERATIONS
                ),
            )
        )

        calibration_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        voxel_size = float(
            calibrated.parameters[
                "voxel_size"
            ]
        )

        # Measure fixed-parameter voxel downsampling only.

        start = time.perf_counter()

        sampled = (
            apply_voxel_downsample(
                points,
                voxel_size,
            )
        )

        runtime_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        parameters = {
            "voxel_size":
                voxel_size,
        }

        return (
            sampled,
            runtime_ms,
            calibration_ms,
            parameters,
        )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    if method == "FPS":

        start = time.perf_counter()

        result = (
            dm.fps_downsample(
                points=points,
                retention_ratio=(
                    retention_ratio
                ),
            )
        )

        runtime_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        return (
            result.points,
            runtime_ms,
            calibration_ms,
            result.parameters,
        )

    # --------------------------------------------------------
    # Uniform
    # --------------------------------------------------------

    if method == "Uniform":

        start = time.perf_counter()

        result = (
            dm.uniform_downsample(
                points=points,
                retention_ratio=(
                    retention_ratio
                ),
            )
        )

        runtime_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        return (
            result.points,
            runtime_ms,
            calibration_ms,
            result.parameters,
        )

    # --------------------------------------------------------
    # Random
    # --------------------------------------------------------

    if method == "Random":

        rng = np.random.default_rng(
            random_seed
        )

        start = time.perf_counter()

        result = (
            dm.random_downsample(
                points=points,
                retention_ratio=(
                    retention_ratio
                ),
                rng=rng,
            )
        )

        runtime_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        return (
            result.points,
            runtime_ms,
            calibration_ms,
            result.parameters,
        )

    raise ValueError(
        f"Unknown method: {method}"
    )


# ============================================================
# Select 79 frames across complete sequence
# ============================================================

def select_frame_indices(
    dataset_length: int,
) -> np.ndarray:

    if dataset_length <= 0:

        raise ValueError(
            "TUM dataset contains no "
            "associated RGB-D frames."
        )

    n_frames = min(
        N_REAL_FRAMES,
        dataset_length,
    )

    indices = np.linspace(
        0,
        dataset_length - 1,
        num=n_frames,
        dtype=np.int64,
    )

    return np.unique(
        indices
    )


# ============================================================
# Safe finite values
# ============================================================

def finite_values(
    rows: list[dict],
    key: str,
) -> np.ndarray:

    values = np.asarray(
        [
            row[key]
            for row in rows
        ],
        dtype=np.float64,
    )

    return values[
        np.isfinite(
            values
        )
    ]


# ============================================================
# Overall method summary
# ============================================================

def build_summary(
    rows: list[dict],
) -> list[dict]:

    summary_rows = []

    for method in METHODS:

        method_rows = [
            row
            for row in rows
            if row[
                "method"
            ] == method
        ]

        retention = finite_values(
            method_rows,
            "actual_retention",
        )

        retention_error = finite_values(
            method_rows,
            "retention_error",
        )

        density = finite_values(
            method_rows,
            "density_cv",
        )

        coverage_mean = finite_values(
            method_rows,
            "coverage_mean_m",
        )

        coverage_p95 = finite_values(
            method_rows,
            "coverage_p95_m",
        )

        runtime = finite_values(
            method_rows,
            "runtime_ms",
        )

        calibration = finite_values(
            method_rows,
            "calibration_ms",
        )

        summary_rows.append(
            {
                "method":
                    method,

                "runs":
                    len(
                        method_rows
                    ),

                "mean_actual_retention":
                    float(
                        np.mean(
                            retention
                        )
                    ),

                "mean_retention_error":
                    float(
                        np.mean(
                            retention_error
                        )
                    ),

                "max_retention_error":
                    float(
                        np.max(
                            retention_error
                        )
                    ),

                "mean_density_cv":
                    float(
                        np.mean(
                            density
                        )
                    ),

                "median_density_cv":
                    float(
                        np.median(
                            density
                        )
                    ),

                "mean_coverage_error_mm":
                    float(
                        np.mean(
                            coverage_mean
                        )
                        * 1000.0
                    ),

                "median_coverage_error_mm":
                    float(
                        np.median(
                            coverage_mean
                        )
                        * 1000.0
                    ),

                "mean_p95_coverage_mm":
                    float(
                        np.mean(
                            coverage_p95
                        )
                        * 1000.0
                    ),

                "median_p95_coverage_mm":
                    float(
                        np.median(
                            coverage_p95
                        )
                        * 1000.0
                    ),

                "median_runtime_ms":
                    float(
                        np.median(
                            runtime
                        )
                    ),

                "p95_runtime_ms":
                    float(
                        np.percentile(
                            runtime,
                            95,
                        )
                    ),

                "mean_runtime_ms":
                    float(
                        np.mean(
                            runtime
                        )
                    ),

                "median_calibration_ms":
                    float(
                        np.median(
                            calibration
                        )
                    ),
            }
        )

    return summary_rows


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Validate fixed repository structure
    # --------------------------------------------------------

    validate_paths()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Camera intrinsics
    # --------------------------------------------------------

    intrinsics = {
        "fx": FX,
        "fy": FY,
        "cx": CX,
        "cy": CY,
    }

    # --------------------------------------------------------
    # Shared project TUM loader
    # --------------------------------------------------------

    dataset = TUMDataset(
        dataset_dir=TUM_DATASET_DIR,
        load_groundtruth=False,
        intrinsics=intrinsics,
    )

    frame_indices = (
        select_frame_indices(
            len(dataset)
        )
    )

    total_runs = (
        len(
            frame_indices
        )
        * len(
            METHODS
        )
        * len(
            RETENTION_LEVELS
        )
    )

    # --------------------------------------------------------
    # Experiment header
    # --------------------------------------------------------

    print()
    print(
        "=" * 84
    )

    print(
        "TUM REAL-DATA DOWNSAMPLING EVALUATION"
    )

    print(
        "=" * 84
    )

    print(
        f"Dataset               : "
        f"{TUM_DATASET_DIR}"
    )

    print(
        f"TUM loader            : "
        f"{SRC_DIR / 'tum_loader.py'}"
    )

    print(
        f"Available RGB-D pairs : "
        f"{len(dataset)}"
    )

    print(
        f"Selected frames       : "
        f"{len(frame_indices)}"
    )

    print(
        f"Benchmark max points  : "
        f"{MAX_BENCHMARK_POINTS}"
    )

    print(
        f"Methods               : "
        f"{len(METHODS)}"
    )

    print(
        f"Retention levels      : "
        f"{RETENTION_LEVELS}"
    )

    print(
        f"Total runs            : "
        f"{total_runs}"
    )

    print()

    rows = []

    run_index = 0

    # ========================================================
    # Run selected real frames
    # ========================================================

    for (
        selected_index,
        frame_index,
    ) in enumerate(
        frame_indices
    ):

        frame_index = int(
            frame_index
        )

        frame = (
            dataset.load_frame(
                frame_index
            )
        )

        # ----------------------------------------------------
        # Raw depth -> dense real point cloud
        # ----------------------------------------------------

        raw_points = (
            depth_to_point_cloud(
                frame[
                    "depth"
                ]
            )
        )

        raw_input_count = len(
            raw_points
        )

        # ----------------------------------------------------
        # Common benchmark cloud
        # ----------------------------------------------------

        points = (
            create_benchmark_cloud(
                raw_points=raw_points,
                frame_index=frame_index,
            )
        )

        input_count = len(
            points
        )

        if input_count < 100:

            raise RuntimeError(
                f"Frame {frame_index} "
                f"contains only "
                f"{input_count} usable points."
            )

        print(
            f"Frame {frame_index:04d}: "
            f"raw={raw_input_count}, "
            f"benchmark={input_count}"
        )

        # ----------------------------------------------------
        # 3 retention levels x 4 methods
        # ----------------------------------------------------

        for (
            retention_index,
            retention,
        ) in enumerate(
            RETENTION_LEVELS
        ):

            for (
                method_index,
                method,
            ) in enumerate(
                METHODS
            ):

                run_index += 1

                random_seed = (
                    RANDOM_BASE_SEED
                    + frame_index * 100
                    + retention_index * 10
                    + method_index
                )

                (
                    sampled_points,
                    runtime_ms,
                    calibration_ms,
                    parameters,
                ) = run_method(
                    points=points,
                    method=method,
                    retention_ratio=retention,
                    random_seed=random_seed,
                )

                output_count = len(
                    sampled_points
                )

                actual_retention = (
                    output_count
                    / input_count
                )

                # --------------------------------------------
                # Density regularity
                # --------------------------------------------

                (
                    nn_mean,
                    nn_std,
                    density_cv,
                ) = nearest_neighbour_statistics(
                    sampled_points
                )

                # --------------------------------------------
                # Geometry / coverage preservation
                # --------------------------------------------

                (
                    coverage_mean,
                    coverage_median,
                    coverage_p95,
                ) = coverage_error(
                    original_points=points,
                    sampled_points=sampled_points,
                )

                # --------------------------------------------
                # Store
                # --------------------------------------------

                row = {

                    "selected_index":
                        selected_index,

                    "frame_index":
                        frame_index,

                    "rgb_timestamp":
                        float(
                            frame[
                                "rgb_timestamp"
                            ]
                        ),

                    "depth_timestamp":
                        float(
                            frame[
                                "depth_timestamp"
                            ]
                        ),

                    "raw_input_count":
                        raw_input_count,

                    "benchmark_input_count":
                        input_count,

                    "method":
                        method,

                    "target_retention":
                        retention,

                    "output_count":
                        output_count,

                    "actual_retention":
                        actual_retention,

                    "retention_error":
                        abs(
                            actual_retention
                            - retention
                        ),

                    "density_nn_mean_m":
                        nn_mean,

                    "density_nn_std_m":
                        nn_std,

                    "density_cv":
                        density_cv,

                    "coverage_mean_m":
                        coverage_mean,

                    "coverage_median_m":
                        coverage_median,

                    "coverage_p95_m":
                        coverage_p95,

                    "runtime_ms":
                        runtime_ms,

                    "calibration_ms":
                        calibration_ms,

                    "parameters":
                        json.dumps(
                            parameters
                        ),
                }

                rows.append(
                    row
                )

                print(
                    f"  "
                    f"[{run_index:03d}/"
                    f"{total_runs:03d}] "
                    f"{method:7s} | "
                    f"r={retention:.2f} | "
                    f"N={input_count}"
                    f"->{output_count} | "
                    f"CV={density_cv:.4f} | "
                    f"coverage="
                    f"{coverage_mean * 1000.0:.3f} mm | "
                    f"time="
                    f"{runtime_ms:.3f} ms"
                )

    # ========================================================
    # Save detailed results
    # ========================================================

    result_fields = [
        "selected_index",
        "frame_index",
        "rgb_timestamp",
        "depth_timestamp",
        "raw_input_count",
        "benchmark_input_count",
        "method",
        "target_retention",
        "output_count",
        "actual_retention",
        "retention_error",
        "density_nn_mean_m",
        "density_nn_std_m",
        "density_cv",
        "coverage_mean_m",
        "coverage_median_m",
        "coverage_p95_m",
        "runtime_ms",
        "calibration_ms",
        "parameters",
    ]

    with RESULT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=result_fields,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # ========================================================
    # Build and save overall summary
    # ========================================================

    summary_rows = (
        build_summary(
            rows
        )
    )

    summary_fields = [
        "method",
        "runs",
        "mean_actual_retention",
        "mean_retention_error",
        "max_retention_error",
        "mean_density_cv",
        "median_density_cv",
        "mean_coverage_error_mm",
        "median_coverage_error_mm",
        "mean_p95_coverage_mm",
        "median_p95_coverage_mm",
        "median_runtime_ms",
        "p95_runtime_ms",
        "mean_runtime_ms",
        "median_calibration_ms",
    ]

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary_fields,
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )

    # ========================================================
    # Save configuration
    # ========================================================

    config = {

        "dataset":
            str(
                TUM_DATASET_DIR
            ),

        "sequence":
            "rgbd_dataset_freiburg1_xyz",

        "tum_loader":
            str(
                SRC_DIR
                / "tum_loader.py"
            ),

        "available_rgb_depth_pairs":
            len(
                dataset
            ),

        "selected_frame_count":
            len(
                frame_indices
            ),

        "selected_frame_indices":
            [
                int(index)
                for index
                in frame_indices
            ],

        "benchmark_max_points":
            MAX_BENCHMARK_POINTS,

        "benchmark_selection":
            (
                "A fixed-seed subset of valid RGB-D "
                "points is used when a raw frame "
                "contains more than 12,500 points. "
                "Selected indices are sorted to preserve "
                "the original raster ordering. All four "
                "downsampling methods receive exactly "
                "the same benchmark input cloud."
            ),

        "methods":
            METHODS,

        "retention_levels":
            RETENTION_LEVELS,

        "camera_intrinsics":
            intrinsics,

        "depth_scale":
            DEPTH_SCALE,

        "depth_range_m": [
            MIN_DEPTH_M,
            MAX_DEPTH_M,
        ],

        "benchmark_base_seed":
            BENCHMARK_BASE_SEED,

        "random_base_seed":
            RANDOM_BASE_SEED,

        "voxel_search_iterations":
            VOXEL_SEARCH_ITERATIONS,

        "runtime_policy":
            (
                "Voxel-size search is recorded in "
                "calibration_ms and excluded from "
                "runtime_ms. runtime_ms measures only "
                "the fixed-parameter downsampling "
                "operation."
            ),

        "coverage_metric":
            (
                "Nearest-neighbour distance from every "
                "benchmark input point to the "
                "downsampled point cloud."
            ),

        "density_metric":
            (
                "Coefficient of variation of "
                "nearest-neighbour distances in the "
                "downsampled point cloud."
            ),
    }

    with CONFIG_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print(
        "=" * 105
    )

    print(
        "REAL-DATA SUMMARY"
    )

    print(
        "=" * 105
    )

    print(
        f"{'Method':<9}"
        f"{'Density CV':>14}"
        f"{'Coverage(mm)':>16}"
        f"{'Median ms':>14}"
        f"{'P95 ms':>14}"
    )

    print(
        "-" * 105
    )

    for row in summary_rows:

        print(
            f"{row['method']:<9}"
            f"{row['mean_density_cv']:>14.4f}"
            f"{row['mean_coverage_error_mm']:>16.3f}"
            f"{row['median_runtime_ms']:>14.3f}"
            f"{row['p95_runtime_ms']:>14.3f}"
        )

    print()

    print(
        "=" * 84
    )

    print(
        "REAL-DATA EVALUATION COMPLETE"
    )

    print(
        "=" * 84
    )

    print(
        f"Detailed results : "
        f"{RESULT_FILE}"
    )

    print(
        f"Summary          : "
        f"{SUMMARY_FILE}"
    )

    print(
        f"Configuration    : "
        f"{CONFIG_FILE}"
    )


if __name__ == "__main__":
    main()