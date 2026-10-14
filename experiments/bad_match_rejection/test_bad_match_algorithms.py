from pathlib import Path
import numpy as np

from normal_compatibility import (
    reject_bad_matches as run_normal,
)
from distance_spatial_rejection import (
    reject_bad_matches as run_distance,
)
from ransac_geometric_verification import (
    reject_bad_matches as run_ransac,
)
from selmap_compatibility import (
    reject_bad_matches as run_selmap,
)
from trilateration_rejection import (
    reject_bad_matches as run_trilateration,
)


# ============================================================
# Paths
# ============================================================

HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent

DATA_DIR = (
    EXPERIMENTS_DIR
    / "data"
    / "bad_match_rejection"
    / "dev"
)


# ============================================================
# Algorithms
# ============================================================

METHODS = {
    "Normal": run_normal,
    "Distance": run_distance,
    "RANSAC": run_ransac,
    "SelMaP-inspired": run_selmap,
    "Trilateration-inspired": run_trilateration,
}


def main():

    files = sorted(DATA_DIR.glob("*.npz"))

    if not files:
        raise FileNotFoundError(
            f"No NPZ samples found in:\n{DATA_DIR}"
        )

    print(f"DEV samples found: {len(files)}")
    print(f"Data directory: {DATA_DIR}")
    print()

    # One sample only for smoke testing
    file = files[0]

    data = np.load(file)

    planes_a = data["planes_a"]
    planes_b = data["planes_b"]
    assigned_pairs = data["assigned_pairs"]

    print(f"Testing sample: {file.name}")
    print(f"Assigned pairs: {len(assigned_pairs)}")
    print("-" * 70)

    for name, method in METHODS.items():

        try:
            result = method(
                planes_a,
                planes_b,
                assigned_pairs,
            )

            keep_mask = np.asarray(
                result["keep_mask"],
                dtype=bool,
            )

            assert len(keep_mask) == len(assigned_pairs)

            print(
                f"{name:25s}"
                f"KEEP={keep_mask.sum():3d}   "
                f"REJECT={(~keep_mask).sum():3d}   "
                f"SUCCESS={result['success']}"
            )

        except Exception as error:

            print(
                f"{name:25s}"
                f"FAILED: {error}"
            )


if __name__ == "__main__":
    main()