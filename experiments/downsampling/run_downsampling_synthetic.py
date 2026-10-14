# run_downsampling_synthetic.py

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d

import downsampling_methods as dm


# ============================================================
# Paths
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = THIS_DIR.parent

DATA_DIR = (
    EXPERIMENTS_DIR
    / "data"
    / "downsampling"
    / "synthetic"
)

SAMPLE_DIR = (
    DATA_DIR
    / "samples"
)

MANIFEST_FILE = (
    DATA_DIR
    / "manifest.csv"
)

OUTPUT_DIR = (
    EXPERIMENTS_DIR
    / "results"
    / "downsampling"
    / "synthetic"
)

RESULT_FILE = (
    OUTPUT_DIR
    / "results.csv"
)

PER_PLANE_FILE = (
    OUTPUT_DIR
    / "per_plane_results.csv"
)

CONFIG_FILE = (
    OUTPUT_DIR
    / "experiment_config.json"
)


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

RANDOM_BASE_SEED = 20260929

# Voxel size is calibrated only to obtain
# approximately the same number of retained
# points as the other methods.
VOXEL_SEARCH_ITERATIONS = 20


# ============================================================
# Plane geometry
# ============================================================

def normalise(
    vector: np.ndarray,
) -> np.ndarray:

    norm = float(
        np.linalg.norm(vector)
    )

    if norm <= 1e-12:
        raise ValueError(
            "Cannot normalise zero vector."
        )

    return (
        vector
        / norm
    )


def plane_basis(
    normal: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    """
    Reconstruct the same deterministic
    in-plane basis used by the generator.
    """

    normal = normalise(
        normal
    )

    axes = np.eye(
        3,
        dtype=np.float64,
    )

    helper = axes[
        np.argmin(
            np.abs(
                axes
                @ normal
            )
        )
    ]

    u = normalise(
        np.cross(
            normal,
            helper,
        )
    )

    v = normalise(
        np.cross(
            normal,
            u,
        )
    )

    return (
        u,
        v,
    )


def fit_plane_svd(
    points: np.ndarray,
):

    """
    Least-squares plane fit.

    Returns
        normal, offset, rmse

    for

        n^T p + d = 0.
    """

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    if len(points) < 3:
        return None

    centroid = np.mean(
        points,
        axis=0,
    )

    centred = (
        points
        - centroid
    )

    _, _, vh = np.linalg.svd(
        centred,
        full_matrices=False,
    )

    normal = normalise(
        vh[-1]
    )

    offset = -float(
        normal
        @ centroid
    )

    residuals = (
        points
        @ normal
        + offset
    )

    rmse = float(
        np.sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )

    return (
        normal,
        offset,
        rmse,
    )


def plane_errors(
    fitted_normal: np.ndarray,
    fitted_offset: float,
    gt_normal: np.ndarray,
    gt_offset: float,
) -> tuple[
    float,
    float,
]:

    fitted_normal = normalise(
        fitted_normal
    )

    gt_normal = normalise(
        gt_normal
    )

    # Align plane signs.
    if (
        fitted_normal
        @ gt_normal
    ) < 0.0:

        fitted_normal = (
            -fitted_normal
        )

        fitted_offset = (
            -fitted_offset
        )

    dot = float(
        np.clip(
            fitted_normal
            @ gt_normal,
            -1.0,
            1.0,
        )
    )

    normal_error_deg = float(
        np.degrees(
            np.arccos(
                dot
            )
        )
    )

    offset_error_m = abs(
        float(
            fitted_offset
            - gt_offset
        )
    )

    return (
        normal_error_deg,
        offset_error_m,
    )


# ============================================================
# Assign downsampled points to GT plane patches
# ============================================================

def assign_points_to_planes(
    points: np.ndarray,
    gt_normals: np.ndarray,
    plane_centres: np.ndarray,
    plane_widths: np.ndarray,
    plane_heights: np.ndarray,
) -> np.ndarray:

    """
    Assign each retained point to the nearest
    finite GT rectangular plane patch.

    This is needed particularly for voxel
    downsampling, because voxel centroids are
    not necessarily identical to any original
    point.
    """

    n_points = len(
        points
    )

    n_planes = len(
        gt_normals
    )

    distances = np.empty(
        (
            n_points,
            n_planes,
        ),
        dtype=np.float64,
    )

    for plane_id in range(
        n_planes
    ):

        normal = gt_normals[
            plane_id
        ]

        centre = plane_centres[
            plane_id
        ]

        width = float(
            plane_widths[
                plane_id
            ]
        )

        height = float(
            plane_heights[
                plane_id
            ]
        )

        u, v = plane_basis(
            normal
        )

        relative = (
            points
            - centre
        )

        coord_u = (
            relative
            @ u
        )

        coord_v = (
            relative
            @ v
        )

        coord_n = (
            relative
            @ normal
        )

        outside_u = np.maximum(
            np.abs(
                coord_u
            )
            - 0.5 * width,
            0.0,
        )

        outside_v = np.maximum(
            np.abs(
                coord_v
            )
            - 0.5 * height,
            0.0,
        )

        distances[
            :,
            plane_id
        ] = np.sqrt(
            outside_u ** 2
            + outside_v ** 2
            + coord_n ** 2
        )

    return np.argmin(
        distances,
        axis=1,
    ).astype(
        np.int32
    )


# ============================================================
# Spatial density regularity
# ============================================================

def nearest_neighbour_statistics(
    points: np.ndarray,
) -> tuple[
    float,
    float,
    float,
]:

    """
    Returns:
        mean NN distance,
        std NN distance,
        coefficient of variation.

    Lower CV means more regular spatial
    point spacing.
    """

    if len(points) < 2:

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    cloud = (
        o3d.geometry.PointCloud()
    )

    cloud.points = (
        o3d.utility.Vector3dVector(
            points
        )
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
# Full-cloud baseline
# ============================================================

def evaluate_original_planes(
    points: np.ndarray,
    plane_ids: np.ndarray,
    gt_normals: np.ndarray,
    gt_offsets: np.ndarray,
):

    results = {}

    for plane_id in range(
        len(
            gt_normals
        )
    ):

        plane_points = points[
            plane_ids
            == plane_id
        ]

        fitted = fit_plane_svd(
            plane_points
        )

        if fitted is None:

            raise RuntimeError(
                "Original synthetic plane "
                "contains fewer than 3 points."
            )

        normal, offset, rmse = (
            fitted
        )

        normal_error, offset_error = (
            plane_errors(
                normal,
                offset,
                gt_normals[
                    plane_id
                ],
                gt_offsets[
                    plane_id
                ],
            )
        )

        results[
            plane_id
        ] = {
            "normal_error_deg":
                normal_error,

            "offset_error_m":
                offset_error,

            "fit_rmse_m":
                rmse,
        }

    return results


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

        # First determine voxel size.
        start = time.perf_counter()

        calibrated = (
            dm.voxel_downsample(
                points=points,
                retention_ratio=retention_ratio,
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

        # Measure ONLY the actual fixed-parameter
        # voxel downsampling operation.
        start = time.perf_counter()

        sampled = (
            dm._voxel_downsample(
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

        rng = (
            np.random.default_rng(
                random_seed
            )
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
# Evaluate one downsampled cloud
# ============================================================

def evaluate_downsampled_cloud(
    sampled_points: np.ndarray,
    sampled_plane_ids: np.ndarray,
    gt_normals: np.ndarray,
    gt_offsets: np.ndarray,
    original_metrics: dict,
):

    plane_rows = []

    for plane_id in range(
        len(
            gt_normals
        )
    ):

        mask = (
            sampled_plane_ids
            == plane_id
        )

        plane_points = (
            sampled_points[
                mask
            ]
        )

        n_points = len(
            plane_points
        )

        fitted = fit_plane_svd(
            plane_points
        )

        if fitted is None:

            plane_rows.append(
                {
                    "plane_id":
                        plane_id,

                    "retained_points":
                        n_points,

                    "fit_success":
                        0,

                    "normal_error_deg":
                        np.nan,

                    "offset_error_m":
                        np.nan,

                    "fit_rmse_m":
                        np.nan,

                    "delta_normal_error_deg":
                        np.nan,

                    "delta_offset_error_m":
                        np.nan,

                    "delta_fit_rmse_m":
                        np.nan,

                    "nn_mean_m":
                        np.nan,

                    "nn_std_m":
                        np.nan,

                    "nn_cv":
                        np.nan,
                }
            )

            continue

        normal, offset, rmse = (
            fitted
        )

        normal_error, offset_error = (
            plane_errors(
                normal,
                offset,
                gt_normals[
                    plane_id
                ],
                gt_offsets[
                    plane_id
                ],
            )
        )

        (
            nn_mean,
            nn_std,
            nn_cv,
        ) = nearest_neighbour_statistics(
            plane_points
        )

        baseline = (
            original_metrics[
                plane_id
            ]
        )

        plane_rows.append(
            {
                "plane_id":
                    plane_id,

                "retained_points":
                    n_points,

                "fit_success":
                    1,

                "normal_error_deg":
                    normal_error,

                "offset_error_m":
                    offset_error,

                "fit_rmse_m":
                    rmse,

                "delta_normal_error_deg":
                    (
                        normal_error
                        - baseline[
                            "normal_error_deg"
                        ]
                    ),

                "delta_offset_error_m":
                    (
                        offset_error
                        - baseline[
                            "offset_error_m"
                        ]
                    ),

                "delta_fit_rmse_m":
                    (
                        rmse
                        - baseline[
                            "fit_rmse_m"
                        ]
                    ),

                "nn_mean_m":
                    nn_mean,

                "nn_std_m":
                    nn_std,

                "nn_cv":
                    nn_cv,
            }
        )

    return plane_rows


# ============================================================
# Safe mean
# ============================================================

def safe_mean(
    values,
) -> float:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    valid = np.isfinite(
        values
    )

    if not np.any(
        valid
    ):
        return np.nan

    return float(
        np.mean(
            values[
                valid
            ]
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    if not MANIFEST_FILE.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n"
            f"{MANIFEST_FILE}"
        )

    sample_files = sorted(
        SAMPLE_DIR.glob(
            "sample_*.npz"
        )
    )

    if not sample_files:

        raise FileNotFoundError(
            f"No synthetic samples found:\n"
            f"{SAMPLE_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_runs = (
        len(sample_files)
        * len(METHODS)
        * len(
            RETENTION_LEVELS
        )
    )

    print(
        "=" * 78
    )

    print(
        "DOWNSAMPLING SYNTHETIC EVALUATION"
    )

    print(
        "=" * 78
    )

    print(
        f"Samples           : "
        f"{len(sample_files)}"
    )

    print(
        f"Methods           : "
        f"{len(METHODS)}"
    )

    print(
        f"Retention levels  : "
        f"{RETENTION_LEVELS}"
    )

    print(
        f"Total runs        : "
        f"{total_runs}"
    )

    print()

    result_rows = []
    per_plane_rows = []

    run_index = 0

    for sample_index, path in enumerate(
        sample_files
    ):

        with np.load(
            path,
            allow_pickle=False,
        ) as data:

            points = np.asarray(
                data[
                    "points"
                ],
                dtype=np.float64,
            )

            plane_ids = np.asarray(
                data[
                    "plane_ids"
                ],
                dtype=np.int32,
            )

            gt_normals = np.asarray(
                data[
                    "gt_normals"
                ],
                dtype=np.float64,
            )

            gt_offsets = np.asarray(
                data[
                    "gt_offsets"
                ],
                dtype=np.float64,
            )

            plane_centres = np.asarray(
                data[
                    "plane_centres"
                ],
                dtype=np.float64,
            )

            plane_widths = np.asarray(
                data[
                    "plane_widths"
                ],
                dtype=np.float64,
            )

            plane_heights = np.asarray(
                data[
                    "plane_heights"
                ],
                dtype=np.float64,
            )

            n_planes = int(
                data[
                    "n_planes"
                ].item()
            )

            grid_size = int(
                data[
                    "grid_size"
                ].item()
            )

            noise_sigma_m = float(
                data[
                    "noise_sigma_m"
                ].item()
            )

            sample_seed = int(
                data[
                    "seed"
                ].item()
            )

        # ----------------------------------------------------
        # Baseline geometry before downsampling
        # ----------------------------------------------------

        original_metrics = (
            evaluate_original_planes(
                points=points,
                plane_ids=plane_ids,
                gt_normals=gt_normals,
                gt_offsets=gt_offsets,
            )
        )

        for retention_index, retention in enumerate(
            RETENTION_LEVELS
        ):

            for method_index, method in enumerate(
                METHODS
            ):

                run_index += 1

                random_seed = (
                    RANDOM_BASE_SEED
                    + sample_index * 100
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

                # ------------------------------------------------
                # Recover plane identity of retained points
                # ------------------------------------------------

                sampled_plane_ids = (
                    assign_points_to_planes(
                        points=sampled_points,
                        gt_normals=gt_normals,
                        plane_centres=(
                            plane_centres
                        ),
                        plane_widths=(
                            plane_widths
                        ),
                        plane_heights=(
                            plane_heights
                        ),
                    )
                )

                plane_results = (
                    evaluate_downsampled_cloud(
                        sampled_points=(
                            sampled_points
                        ),
                        sampled_plane_ids=(
                            sampled_plane_ids
                        ),
                        gt_normals=(
                            gt_normals
                        ),
                        gt_offsets=(
                            gt_offsets
                        ),
                        original_metrics=(
                            original_metrics
                        ),
                    )
                )

                # ------------------------------------------------
                # Aggregate plane-level metrics
                # ------------------------------------------------

                normal_error = safe_mean(
                    [
                        row[
                            "normal_error_deg"
                        ]
                        for row
                        in plane_results
                    ]
                )

                offset_error = safe_mean(
                    [
                        row[
                            "offset_error_m"
                        ]
                        for row
                        in plane_results
                    ]
                )

                fit_rmse = safe_mean(
                    [
                        row[
                            "fit_rmse_m"
                        ]
                        for row
                        in plane_results
                    ]
                )

                delta_normal = safe_mean(
                    [
                        row[
                            "delta_normal_error_deg"
                        ]
                        for row
                        in plane_results
                    ]
                )

                delta_offset = safe_mean(
                    [
                        row[
                            "delta_offset_error_m"
                        ]
                        for row
                        in plane_results
                    ]
                )

                delta_rmse = safe_mean(
                    [
                        row[
                            "delta_fit_rmse_m"
                        ]
                        for row
                        in plane_results
                    ]
                )

                density_cv = safe_mean(
                    [
                        row[
                            "nn_cv"
                        ]
                        for row
                        in plane_results
                    ]
                )

                fit_successes = sum(
                    row[
                        "fit_success"
                    ]
                    for row
                    in plane_results
                )

                valid_plane_fraction = (
                    fit_successes
                    / n_planes
                )

                input_count = len(
                    points
                )

                output_count = len(
                    sampled_points
                )

                actual_retention = (
                    output_count
                    / input_count
                )

                # ------------------------------------------------
                # Main result
                # ------------------------------------------------

                result_rows.append(
                    {
                        "sample":
                            path.name,

                        "sample_index":
                            sample_index,

                        "n_planes":
                            n_planes,

                        "grid_size":
                            grid_size,

                        "noise_sigma_m":
                            noise_sigma_m,

                        "sample_seed":
                            sample_seed,

                        "method":
                            method,

                        "target_retention":
                            retention,

                        "input_count":
                            input_count,

                        "output_count":
                            output_count,

                        "actual_retention":
                            actual_retention,

                        "retention_error":
                            abs(
                                actual_retention
                                - retention
                            ),

                        "normal_error_deg":
                            normal_error,

                        "offset_error_m":
                            offset_error,

                        "fit_rmse_m":
                            fit_rmse,

                        "delta_normal_error_deg":
                            delta_normal,

                        "delta_offset_error_m":
                            delta_offset,

                        "delta_fit_rmse_m":
                            delta_rmse,

                        "density_cv":
                            density_cv,

                        "valid_plane_fraction":
                            valid_plane_fraction,

                        "runtime_ms":
                            runtime_ms,

                        "calibration_ms":
                            calibration_ms,

                        "parameters":
                            json.dumps(
                                parameters
                            ),
                    }
                )

                # ------------------------------------------------
                # Detailed per-plane output
                # ------------------------------------------------

                for row in (
                    plane_results
                ):

                    per_plane_rows.append(
                        {
                            "sample":
                                path.name,

                            "sample_index":
                                sample_index,

                            "method":
                                method,

                            "target_retention":
                                retention,

                            "n_planes":
                                n_planes,

                            "grid_size":
                                grid_size,

                            "noise_sigma_m":
                                noise_sigma_m,

                            **row,
                        }
                    )

                print(
                    f"[{run_index:03d}/"
                    f"{total_runs:03d}] "
                    f"{path.name} | "
                    f"{method:7s} | "
                    f"r={retention:.2f} | "
                    f"N={input_count}"
                    f"->{output_count} | "
                    f"normal="
                    f"{normal_error:.4f} deg | "
                    f"offset="
                    f"{offset_error * 1000.0:.3f} mm | "
                    f"CV="
                    f"{density_cv:.4f} | "
                    f"time="
                    f"{runtime_ms:.3f} ms"
                )

    # ========================================================
    # Save main results
    # ========================================================

    result_fields = [
        "sample",
        "sample_index",
        "n_planes",
        "grid_size",
        "noise_sigma_m",
        "sample_seed",
        "method",
        "target_retention",
        "input_count",
        "output_count",
        "actual_retention",
        "retention_error",
        "normal_error_deg",
        "offset_error_m",
        "fit_rmse_m",
        "delta_normal_error_deg",
        "delta_offset_error_m",
        "delta_fit_rmse_m",
        "density_cv",
        "valid_plane_fraction",
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
            result_rows
        )

    # ========================================================
    # Save per-plane results
    # ========================================================

    per_plane_fields = [
        "sample",
        "sample_index",
        "method",
        "target_retention",
        "n_planes",
        "grid_size",
        "noise_sigma_m",
        "plane_id",
        "retained_points",
        "fit_success",
        "normal_error_deg",
        "offset_error_m",
        "fit_rmse_m",
        "delta_normal_error_deg",
        "delta_offset_error_m",
        "delta_fit_rmse_m",
        "nn_mean_m",
        "nn_std_m",
        "nn_cv",
    ]

    with PER_PLANE_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=per_plane_fields,
        )

        writer.writeheader()

        writer.writerows(
            per_plane_rows
        )

    # ========================================================
    # Save experiment configuration
    # ========================================================

    config = {
        "methods":
            METHODS,

        "retention_levels":
            RETENTION_LEVELS,

        "number_of_samples":
            len(
                sample_files
            ),

        "total_runs":
            total_runs,

        "random_base_seed":
            RANDOM_BASE_SEED,

        "voxel_search_iterations":
            VOXEL_SEARCH_ITERATIONS,

        "runtime_policy":
            (
                "Voxel calibration/search time is "
                "recorded separately and excluded "
                "from runtime_ms. runtime_ms measures "
                "the fixed-parameter downsampling "
                "operation."
            ),

        "geometry_policy":
            (
                "Downsampled planes are fitted by SVD "
                "and compared against analytical GT "
                "plane normals and offsets."
            ),

        "density_policy":
            (
                "Spatial regularity is measured using "
                "the coefficient of variation of "
                "nearest-neighbour distances within "
                "each GT plane, macro-averaged across "
                "planes."
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

    print()

    print(
        "=" * 78
    )

    print(
        "SYNTHETIC EVALUATION COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        f"Runs completed     : "
        f"{len(result_rows)}"
    )

    print(
        f"Main results       : "
        f"{RESULT_FILE}"
    )

    print(
        f"Per-plane results  : "
        f"{PER_PLANE_FILE}"
    )

    print(
        f"Experiment config  : "
        f"{CONFIG_FILE}"
    )


if __name__ == "__main__":
    main()