import json
from pathlib import Path

import numpy as np


# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic_multiplane_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Benchmark design
#
# 3 plane counts x 2 noise levels x 2 outlier ratios
# x 5 repetitions = 60 point clouds.
# ==========================================================

PLANE_COUNTS = [2, 3, 4]
NOISE_LEVELS_M = [0.0, 0.02]
OUTLIER_RATIOS = [0.0, 0.20]
REPEATS = 5

TOTAL_INLIER_POINTS = 6000

BASE_SEED = 20260918


# ==========================================================
# Geometry helpers
# ==========================================================

def normalise(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def plane_basis(normal):
    """
    Return two orthonormal vectors spanning a plane.
    """
    n = normalise(normal)

    if abs(n[2]) < 0.9:
        helper = np.array([0.0, 0.0, 1.0])
    else:
        helper = np.array([1.0, 0.0, 0.0])

    u = np.cross(n, helper)
    u = normalise(u)

    v = np.cross(n, u)
    v = normalise(v)

    return u, v


def sample_plane(normal, d, num_points, extent_u, extent_v, noise_sigma, rng):
    """
    Sample a finite rectangular patch from:

        n^T p + d = 0

    Noise is added along the plane normal so that the
    benchmark does not privilege horizontal planes.
    """
    n = normalise(normal)

    # Closest point on plane to origin.
    centre = -d * n

    u, v = plane_basis(n)

    a = rng.uniform(-extent_u, extent_u, num_points)
    b = rng.uniform(-extent_v, extent_v, num_points)

    points = (
        centre[None, :]
        + a[:, None] * u[None, :]
        + b[:, None] * v[None, :]
    )

    if noise_sigma > 0:
        noise = rng.normal(0.0, noise_sigma, num_points)
        points = points + noise[:, None] * n[None, :]

    return points


def get_plane_library():
    """
    Fixed plane family. The first K planes are used.

    Planes are deliberately given different orientations.
    """
    tilted_n = normalise([0.40, 0.20, 0.8944271909999159])
    tilted_point = np.array([0.0, 0.0, 2.8])
    tilted_d = -float(np.dot(tilted_n, tilted_point))

    return [
        {
            "name": "floor_like",
            "normal": np.array([0.0, 0.0, 1.0]),
            "d": -2.0,
            "extent_u": 2.0,
            "extent_v": 2.0,
        },
        {
            "name": "wall_x",
            "normal": np.array([1.0, 0.0, 0.0]),
            "d": -1.5,
            "extent_u": 2.0,
            "extent_v": 2.0,
        },
        {
            "name": "wall_y",
            "normal": np.array([0.0, 1.0, 0.0]),
            "d": 1.5,   # y = -1.5
            "extent_u": 2.0,
            "extent_v": 2.0,
        },
        {
            "name": "tilted_plane",
            "normal": tilted_n,
            "d": tilted_d,
            "extent_u": 1.6,
            "extent_v": 1.6,
        },
    ]


# ==========================================================
# Main generation
# ==========================================================

def main():
    plane_library = get_plane_library()

    sample_index = 0

    for plane_count in PLANE_COUNTS:
        for noise_sigma in NOISE_LEVELS_M:
            for outlier_ratio in OUTLIER_RATIOS:
                for repeat_id in range(1, REPEATS + 1):
                    sample_index += 1
                    seed = BASE_SEED + sample_index
                    rng = np.random.default_rng(seed)

                    num_per_plane = TOTAL_INLIER_POINTS // plane_count

                    point_blocks = []
                    label_blocks = []
                    gt_planes = []

                    for plane_id in range(plane_count):
                        plane = plane_library[plane_id]

                        points = sample_plane(
                            plane["normal"],
                            plane["d"],
                            num_per_plane,
                            plane["extent_u"],
                            plane["extent_v"],
                            noise_sigma,
                            rng,
                        )

                        point_blocks.append(points)

                        # 0 is reserved for outliers.
                        label_blocks.append(
                            np.full(
                                len(points),
                                plane_id + 1,
                                dtype=np.int32,
                            )
                        )

                        gt_planes.append(
                            {
                                "id": plane_id + 1,
                                "name": plane["name"],
                                "normal": plane["normal"].tolist(),
                                "d": float(plane["d"]),
                            }
                        )

                    inlier_points = np.vstack(point_blocks)
                    inlier_labels = np.concatenate(label_blocks)

                    if outlier_ratio > 0:
                        num_outliers = int(
                            round(
                                len(inlier_points)
                                * outlier_ratio
                                / (1.0 - outlier_ratio)
                            )
                        )

                        # Structured clutter box. Unlike the old single-plane
                        # benchmark, these points are not forced to stay far
                        # away from every plane.
                        outliers = np.empty((num_outliers, 3), dtype=np.float64)
                        outliers[:, 0] = rng.uniform(-2.5, 2.5, num_outliers)
                        outliers[:, 1] = rng.uniform(-2.5, 2.5, num_outliers)
                        outliers[:, 2] = rng.uniform(0.2, 4.2, num_outliers)

                        outlier_labels = np.zeros(num_outliers, dtype=np.int32)

                        points = np.vstack([inlier_points, outliers])
                        labels = np.concatenate([inlier_labels, outlier_labels])
                    else:
                        points = inlier_points
                        labels = inlier_labels

                    permutation = rng.permutation(len(points))
                    points = points[permutation]
                    labels = labels[permutation]

                    noise_mm = int(round(noise_sigma * 1000))
                    outlier_pct = int(round(outlier_ratio * 100))

                    sample_id = (
                        f"P{plane_count}_"
                        f"N{noise_mm:02d}_"
                        f"O{outlier_pct:02d}_"
                        f"R{repeat_id:02d}"
                    )

                    npz_path = OUTPUT_DIR / f"{sample_id}.npz"
                    gt_path = OUTPUT_DIR / f"{sample_id}_gt.json"

                    np.savez_compressed(
                        npz_path,
                        points=points.astype(np.float64),
                        labels=labels.astype(np.int32),
                    )

                    metadata = {
                        "sample_id": sample_id,
                        "plane_count": plane_count,
                        "noise_sigma_m": float(noise_sigma),
                        "outlier_ratio": float(outlier_ratio),
                        "repeat_id": repeat_id,
                        "seed": seed,
                        "num_points": int(len(points)),
                        "num_inlier_points": int(np.sum(labels > 0)),
                        "num_outlier_points": int(np.sum(labels == 0)),
                        "gt_planes": gt_planes,
                    }

                    with open(gt_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2)

                    print(
                        f"[{sample_index:02d}/60] {sample_id} "
                        f"points={len(points)}"
                    )

    print("\nGeneration complete.")
    print("Output directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
