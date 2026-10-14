from pathlib import Path
import sys

import numpy as np


# ============================================================
# Project imports
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"

sys.path.insert(
    0,
    str(SRC_DIR),
)

from tum_loader import TUMDataset
from tum_pointcloud import depth_to_point_cloud


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


# TUM officially recommends the ROS-default parameters for
# the pre-registered RGB-D images.
INTRINSICS = {
    "fx": 525.0,
    "fy": 525.0,
    "cx": 319.5,
    "cy": 239.5,
}


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 68)
    print("TUM POINT-CLOUD SMOKE TEST")
    print("=" * 68)

    dataset = TUMDataset(
        dataset_dir=DATASET_DIR,
        load_groundtruth=True,
        intrinsics=INTRINSICS,
    )

    print(
        f"Associated frames : {len(dataset)}"
    )

    frame = dataset.load_frame(
        FRAME_INDEX
    )

    points, pixels = depth_to_point_cloud(
        depth=frame["depth"],
        intrinsics=frame["intrinsics"],
    )

    print(
        f"Frame index       : {frame['index']}"
    )

    print(
        f"RGB timestamp     : "
        f"{frame['rgb_timestamp']:.6f}"
    )

    print(
        f"Depth timestamp   : "
        f"{frame['depth_timestamp']:.6f}"
    )

    print(
        f"RGB-depth dt      : "
        f"{frame['time_difference']:.6f} s"
    )

    print(
        f"Point-cloud shape : {points.shape}"
    )

    print(
        f"Pixel-map shape   : {pixels.shape}"
    )

    print(
        f"Minimum XYZ       : "
        f"{np.min(points, axis=0)}"
    )

    print(
        f"Maximum XYZ       : "
        f"{np.max(points, axis=0)}"
    )

    if frame["groundtruth"] is not None:

        print(
            f"GT time delta     : "
            f"{frame['groundtruth']['time_difference']:.6f} s"
        )

        print(
            f"GT translation    : "
            f"{frame['groundtruth']['translation']}"
        )

        print(
            f"GT quaternion     : "
            f"{frame['groundtruth']['quaternion']}"
        )

    print("\n" + "-" * 68)

    if (
        points.ndim == 2
        and points.shape[1] == 3
        and len(points) > 0
        and np.all(np.isfinite(points))
    ):

        print(
            "PASS: TUM depth successfully converted "
            "to an N x 3 point cloud."
        )

    else:

        print(
            "FAIL: invalid point-cloud output."
        )

    print("-" * 68)


if __name__ == "__main__":
    main()