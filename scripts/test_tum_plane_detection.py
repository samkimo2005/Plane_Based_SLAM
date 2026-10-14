from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d


# ============================================================
# Project paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_DIR / "src"

PLANE_DETECTION_DIR = (
    PROJECT_DIR
    / "experiments"
    / "feature_finding"
)

sys.path.insert(
    0,
    str(SRC_DIR),
)

sys.path.insert(
    0,
    str(PLANE_DETECTION_DIR),
)


# ============================================================
# Existing modules
# ============================================================

from tum_loader import TUMDataset

from tum_pointcloud import (
    depth_to_point_cloud,
)

from iterative_ransac import (
    detect_iterative_ransac,
)


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "tum"
    / "rgbd_dataset_freiburg1_xyz"
)

FRAME_INDEX = 0

INTRINSICS = {
    "fx": 525.0,
    "fy": 525.0,
    "cx": 319.5,
    "cy": 239.5,
}

# TUM point-cloud preprocessing
VOXEL_SIZE_M = 0.02

# Fixed seed so that the RANSAC result is reproducible.
RNG_SEED = 20260919


# ============================================================
# Point-cloud preprocessing
# ============================================================

def voxel_downsample(
    points,
    voxel_size,
):
    """
    Downsample an N x 3 NumPy point cloud using Open3D.
    """

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    if (
        points.ndim != 2
        or points.shape[1] != 3
    ):
        raise ValueError(
            "points must have shape (N, 3)"
        )

    pcd = o3d.geometry.PointCloud()

    pcd.points = (
        o3d.utility.Vector3dVector(
            points
        )
    )

    downsampled = (
        pcd.voxel_down_sample(
            voxel_size=voxel_size
        )
    )

    return np.asarray(
        downsampled.points,
        dtype=np.float64,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("TUM + ITERATIVE RANSAC PLANE-DETECTION TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load TUM dataset
    # --------------------------------------------------------

    dataset = TUMDataset(
        dataset_dir=DATASET_DIR,
        load_groundtruth=True,
        intrinsics=INTRINSICS,
    )

    print(
        f"Associated frames   : {len(dataset)}"
    )

    # --------------------------------------------------------
    # 2. Load one RGB-D frame
    # --------------------------------------------------------

    frame = dataset.load_frame(
        FRAME_INDEX
    )

    print(
        f"Frame index         : {FRAME_INDEX}"
    )

    print(
        f"RGB timestamp       : "
        f"{frame['rgb_timestamp']:.6f}"
    )

    print(
        f"Depth timestamp     : "
        f"{frame['depth_timestamp']:.6f}"
    )

    print(
        f"RGB-depth dt        : "
        f"{frame['time_difference']:.6f} s"
    )

    # --------------------------------------------------------
    # 3. Depth -> point cloud
    # --------------------------------------------------------

    raw_points, pixels = (
        depth_to_point_cloud(
            depth=frame["depth"],
            intrinsics=frame[
                "intrinsics"
            ],
        )
    )

    print(
        f"Raw point count     : "
        f"{len(raw_points)}"
    )

    # --------------------------------------------------------
    # 4. Voxel downsampling
    # --------------------------------------------------------

    points = voxel_downsample(
        raw_points,
        VOXEL_SIZE_M,
    )

    print(
        f"Downsampled points  : "
        f"{len(points)}"
    )

    print(
        f"Voxel size          : "
        f"{VOXEL_SIZE_M:.3f} m"
    )

    if len(points) < 3:
        raise RuntimeError(
            "Too few points after preprocessing."
        )

    # --------------------------------------------------------
    # 5. Create deterministic RANSAC RNG
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RNG_SEED
    )

    # --------------------------------------------------------
    # 6. Run the EXISTING Iterative RANSAC
    # --------------------------------------------------------

    start_time = time.perf_counter()

    models = detect_iterative_ransac(
        points,
        rng,
    )

    end_time = time.perf_counter()

    runtime_ms = (
        end_time
        - start_time
    ) * 1000.0

    # --------------------------------------------------------
    # 7. Basic result summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ITERATIVE RANSAC RESULT")
    print("=" * 70)

    print(
        f"Detected planes     : "
        f"{len(models)}"
    )

    print(
        f"Runtime             : "
        f"{runtime_ms:.2f} ms"
    )

    # --------------------------------------------------------
    # 8. Print each detected plane
    # --------------------------------------------------------

    for index, model in enumerate(
        models
    ):

        normal = model["normal"]
        d = model["d"]
        support = model["support"]

        source_support = model.get(
            "source_support",
            None,
        )

        support_ratio = (
            support
            / len(points)
        )

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"Plane {index}"
        )

        print(
            f"  normal         : "
            f"["
            f"{normal[0]: .6f}, "
            f"{normal[1]: .6f}, "
            f"{normal[2]: .6f}"
            f"]"
        )

        print(
            f"  d              : "
            f"{d:.6f} m"
        )

        print(
            f"  support        : "
            f"{support}"
        )

        print(
            f"  support ratio  : "
            f"{support_ratio:.3f}"
        )

        print(
            f"  source support : "
            f"{source_support}"
        )

    # --------------------------------------------------------
    # 9. Sanity checks
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    if len(models) == 0:

        print(
            "WARNING: program ran successfully, "
            "but no planes were detected."
        )

    else:

        valid = True

        for model in models:

            normal = np.asarray(
                model["normal"],
                dtype=float,
            )

            if not np.isclose(
                np.linalg.norm(normal),
                1.0,
                atol=1e-6,
            ):
                valid = False

            if model["support"] <= 0:
                valid = False

        if valid:

            print(
                "PASS: Iterative RANSAC successfully "
                "detected planes from a real TUM frame."
            )

        else:

            print(
                "FAIL: detector returned invalid "
                "plane parameters."
            )

    print("=" * 70)


if __name__ == "__main__":
    main()