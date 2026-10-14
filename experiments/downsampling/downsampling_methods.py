# downsampling_methods.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


# ============================================================
# Result container
# ============================================================

@dataclass
class DownsamplingResult:
    points: np.ndarray

    method: str

    input_count: int
    output_count: int

    target_retention: float
    actual_retention: float

    parameters: dict


# ============================================================
# Common validation
# ============================================================

def _validate_points(
    points: np.ndarray,
) -> np.ndarray:

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    if points.ndim != 2:
        raise ValueError(
            "points must be a 2-D array."
        )

    if points.shape[1] != 3:
        raise ValueError(
            "points must have shape (N, 3)."
        )

    if len(points) == 0:
        raise ValueError(
            "points must not be empty."
        )

    if not np.all(
        np.isfinite(points)
    ):
        raise ValueError(
            "points contain non-finite values."
        )

    return points


def _validate_retention(
    retention_ratio: float,
) -> float:

    retention_ratio = float(
        retention_ratio
    )

    if not (
        0.0 < retention_ratio <= 1.0
    ):
        raise ValueError(
            "retention_ratio must satisfy "
            "0 < retention_ratio <= 1."
        )

    return retention_ratio


def _target_count(
    n_points: int,
    retention_ratio: float,
) -> int:

    target = int(
        round(
            n_points
            * retention_ratio
        )
    )

    return max(
        1,
        min(
            n_points,
            target,
        ),
    )


def _build_result(
    points_in: np.ndarray,
    points_out: np.ndarray,
    method: str,
    target_retention: float,
    parameters: dict,
) -> DownsamplingResult:

    n_in = len(
        points_in
    )

    n_out = len(
        points_out
    )

    return DownsamplingResult(
        points=np.asarray(
            points_out,
            dtype=np.float64,
        ),
        method=method,
        input_count=n_in,
        output_count=n_out,
        target_retention=float(
            target_retention
        ),
        actual_retention=(
            n_out / n_in
        ),
        parameters=parameters,
    )


# ============================================================
# 1. Random sampling
# ============================================================

def random_downsample(
    points: np.ndarray,
    retention_ratio: float,
    rng: np.random.Generator,
) -> DownsamplingResult:

    points = _validate_points(
        points
    )

    retention_ratio = (
        _validate_retention(
            retention_ratio
        )
    )

    n_target = _target_count(
        len(points),
        retention_ratio,
    )

    indices = rng.choice(
        len(points),
        size=n_target,
        replace=False,
    )

    sampled = points[
        indices
    ]

    return _build_result(
        points_in=points,
        points_out=sampled,
        method="Random",
        target_retention=(
            retention_ratio
        ),
        parameters={
            "target_count":
                n_target,
        },
    )


# ============================================================
# 2. Uniform every-k sampling
# ============================================================

def uniform_downsample(
    points: np.ndarray,
    retention_ratio: float,
) -> DownsamplingResult:

    points = _validate_points(
        points
    )

    retention_ratio = (
        _validate_retention(
            retention_ratio
        )
    )

    # For the planned experiment:
    #
    # r = 0.50 -> k = 2
    # r = 0.25 -> k = 4
    # r = 0.10 -> k = 10
    #
    # This preserves the intended
    # deterministic every-k definition.

    k = max(
        1,
        int(
            round(
                1.0
                / retention_ratio
            )
        ),
    )

    sampled = points[
        ::k
    ]

    return _build_result(
        points_in=points,
        points_out=sampled,
        method="Uniform",
        target_retention=(
            retention_ratio
        ),
        parameters={
            "every_k":
                k,
        },
    )


# ============================================================
# 3. Farthest Point Sampling
# ============================================================

def fps_downsample(
    points: np.ndarray,
    retention_ratio: float,
) -> DownsamplingResult:

    points = _validate_points(
        points
    )

    retention_ratio = (
        _validate_retention(
            retention_ratio
        )
    )

    n_target = _target_count(
        len(points),
        retention_ratio,
    )

    cloud = o3d.geometry.PointCloud()

    cloud.points = (
        o3d.utility.Vector3dVector(
            points
        )
    )

    sampled_cloud = (
        cloud.farthest_point_down_sample(
            n_target
        )
    )

    sampled = np.asarray(
        sampled_cloud.points
    ).copy()

    return _build_result(
        points_in=points,
        points_out=sampled,
        method="FPS",
        target_retention=(
            retention_ratio
        ),
        parameters={
            "target_count":
                n_target,
        },
    )


# ============================================================
# 4. Voxel Grid sampling
# ============================================================

def _voxel_downsample(
    points: np.ndarray,
    voxel_size: float,
) -> np.ndarray:

    cloud = o3d.geometry.PointCloud()

    cloud.points = (
        o3d.utility.Vector3dVector(
            points
        )
    )

    downsampled = (
        cloud.voxel_down_sample(
            voxel_size=float(
                voxel_size
            )
        )
    )

    return np.asarray(
        downsampled.points
    ).copy()


def voxel_downsample(
    points: np.ndarray,
    retention_ratio: float,
    search_iterations: int = 30,
) -> DownsamplingResult:

    points = _validate_points(
        points
    )

    retention_ratio = (
        _validate_retention(
            retention_ratio
        )
    )

    n_input = len(
        points
    )

    n_target = _target_count(
        n_input,
        retention_ratio,
    )

    # No downsampling required.
    if n_target >= n_input:

        return _build_result(
            points_in=points,
            points_out=points.copy(),
            method="Voxel",
            target_retention=(
                retention_ratio
            ),
            parameters={
                "voxel_size":
                    0.0,
                "target_count":
                    n_target,
            },
        )

    minimum = np.min(
        points,
        axis=0,
    )

    maximum = np.max(
        points,
        axis=0,
    )

    extent = (
        maximum
        - minimum
    )

    diagonal = float(
        np.linalg.norm(
            extent
        )
    )

    if diagonal <= 0.0:
        raise ValueError(
            "Point cloud has zero spatial extent."
        )

    # Lower bound:
    # almost no downsampling.
    low = (
        diagonal
        * 1e-6
    )

    # Upper bound:
    # large enough to strongly reduce
    # the number of occupied voxels.
    high = diagonal

    best_points = None
    best_voxel_size = None
    best_error = np.inf

    for _ in range(
        search_iterations
    ):

        voxel_size = (
            0.5
            * (
                low
                + high
            )
        )

        sampled = _voxel_downsample(
            points,
            voxel_size,
        )

        n_output = len(
            sampled
        )

        count_error = abs(
            n_output
            - n_target
        )

        if count_error < best_error:

            best_error = (
                count_error
            )

            best_points = (
                sampled
            )

            best_voxel_size = (
                voxel_size
            )

        # Larger voxel:
        # fewer output points.
        if n_output > n_target:

            low = voxel_size

        else:

            high = voxel_size

    if best_points is None:
        raise RuntimeError(
            "Voxel-size search failed."
        )

    return _build_result(
        points_in=points,
        points_out=best_points,
        method="Voxel",
        target_retention=(
            retention_ratio
        ),
        parameters={
            "voxel_size":
                float(
                    best_voxel_size
                ),
            "target_count":
                n_target,
            "count_error":
                int(
                    abs(
                        len(best_points)
                        - n_target
                    )
                ),
        },
    )


# ============================================================
# Unified interface
# ============================================================

def downsample_points(
    points: np.ndarray,
    method: str,
    retention_ratio: float,
    rng: np.random.Generator | None = None,
) -> DownsamplingResult:

    method_key = (
        method
        .strip()
        .lower()
    )

    if method_key == "voxel":

        return voxel_downsample(
            points,
            retention_ratio,
        )

    if method_key == "fps":

        return fps_downsample(
            points,
            retention_ratio,
        )

    if method_key == "uniform":

        return uniform_downsample(
            points,
            retention_ratio,
        )

    if method_key == "random":

        if rng is None:

            raise ValueError(
                "Random downsampling requires "
                "an explicit NumPy RNG."
            )

        return random_downsample(
            points,
            retention_ratio,
            rng,
        )

    raise ValueError(
        f"Unknown downsampling method: "
        f"{method}"
    )


# ============================================================
# Quick self-test
# ============================================================

if __name__ == "__main__":

    rng = np.random.default_rng(
        20260929
    )

    # Simple synthetic 3-D cloud.
    points = rng.uniform(
        low=-1.0,
        high=1.0,
        size=(
            10000,
            3,
        ),
    )

    methods = [
        "Voxel",
        "FPS",
        "Uniform",
        "Random",
    ]

    retention_levels = [
        0.50,
        0.25,
        0.10,
    ]

    print(
        "=" * 72
    )

    print(
        "DOWNSAMPLING METHOD SELF-TEST"
    )

    print(
        "=" * 72
    )

    for retention in (
        retention_levels
    ):

        print(
            f"\nTarget retention: "
            f"{retention:.2f}"
        )

        for method in methods:

            result = (
                downsample_points(
                    points=points,
                    method=method,
                    retention_ratio=retention,
                    rng=rng,
                )
            )

            print(
                f"{method:8s} | "
                f"{result.input_count:6d}"
                f" -> "
                f"{result.output_count:6d}"
                f" | "
                f"actual="
                f"{result.actual_retention:.4f}"
                f" | "
                f"{result.parameters}"
            )