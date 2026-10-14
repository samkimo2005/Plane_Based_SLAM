from pathlib import Path
import time

import numpy as np
import pandas as pd

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

from bad_match_rejection_evaluation import (
    evaluate_result,
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

RESULT_DIR = (
    EXPERIMENTS_DIR
    / "results"
    / "bad_match_rejection"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    RESULT_DIR
    / "dev_results.csv"
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


# ============================================================
# Main
# ============================================================

def main():

    files = sorted(
        DATA_DIR.glob("*.npz")
    )

    if not files:
        raise FileNotFoundError(
            f"No NPZ files found in {DATA_DIR}"
        )

    print(
        f"DEV samples found: {len(files)}"
    )

    rows = []

    for sample_index, file in enumerate(
        files,
        start=1,
    ):
        data = np.load(file)

        planes_a = data["planes_a"]
        planes_b = data["planes_b"]
        assigned_pairs = data["assigned_pairs"]

        gt_match_labels = data[
            "gt_match_labels"
        ]

        R_gt = data["R_gt"]
        t_gt = data["t_gt"]

        for method_name, method in METHODS.items():

            start = time.perf_counter()

            try:
                result = method(
                    planes_a,
                    planes_b,
                    assigned_pairs,
                )

                runtime_ms = (
                    time.perf_counter()
                    - start
                ) * 1000.0

                metrics = evaluate_result(
                    planes_a=planes_a,
                    planes_b=planes_b,
                    assigned_pairs=assigned_pairs,
                    keep_mask=result["keep_mask"],
                    gt_match_labels=gt_match_labels,
                    R_gt=R_gt,
                    t_gt=t_gt,
                )

                rows.append({
                    "sample": file.name,
                    "method": method_name,
                    "algorithm_success":
                        result["success"],
                    "runtime_ms":
                        runtime_ms,
                    "kept":
                        int(
                            np.sum(
                                result["keep_mask"]
                            )
                        ),
                    "rejected":
                        int(
                            np.sum(
                                ~result["keep_mask"]
                            )
                        ),
                    **metrics,
                })

            except Exception as error:

                runtime_ms = (
                    time.perf_counter()
                    - start
                ) * 1000.0

                rows.append({
                    "sample": file.name,
                    "method": method_name,
                    "algorithm_success": False,
                    "runtime_ms": runtime_ms,
                    "error": str(error),
                })

        if (
            sample_index % 25 == 0
            or sample_index == len(files)
        ):
            print(
                f"Processed "
                f"{sample_index}/{len(files)}"
            )

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()
    print("=" * 60)
    print("DEV evaluation complete")
    print(f"Rows: {len(df)}")
    print(f"Saved to: {OUTPUT_CSV}")
    print("=" * 60)

    print()
    print(
        df.groupby("method")[
            [
                "precision",
                "recall",
                "f1",
                "correct_retention",
                "rotation_error_deg",
                "translation_error_m",
                "runtime_ms",
            ]
        ].mean(numeric_only=True)
    )


if __name__ == "__main__":
    main()