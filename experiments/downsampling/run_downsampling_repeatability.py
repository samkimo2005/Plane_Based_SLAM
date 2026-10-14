# run_downsampling_repeatability.py

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

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
    / "samples"
)

OUTPUT_DIR = (
    EXPERIMENTS_DIR
    / "results"
    / "downsampling"
    / "repeatability"
)

RESULT_FILE = (
    OUTPUT_DIR
    / "repeatability_results.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "repeatability_summary.csv"
)


# ============================================================
# Configuration
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

N_REPEATS = 5

RANDOM_BASE_SEED = 20260930

ROUND_DECIMALS = 10


# ============================================================
# Point-set comparison
# ============================================================

def point_set(
    points: np.ndarray,
) -> set[tuple]:

    rounded = np.round(
        np.asarray(
            points,
            dtype=np.float64,
        ),
        decimals=ROUND_DECIMALS,
    )

    return {
        tuple(row)
        for row in rounded
    }


def overlap_ratio(
    reference: np.ndarray,
    comparison: np.ndarray,
) -> float:

    set_a = point_set(
        reference
    )

    set_b = point_set(
        comparison
    )

    denominator = max(
        len(set_a),
        len(set_b),
    )

    if denominator == 0:
        return np.nan

    intersection = len(
        set_a.intersection(
            set_b
        )
    )

    return (
        intersection
        / denominator
    )


# ============================================================
# Run one method
# ============================================================

def run_method(
    points: np.ndarray,
    method: str,
    retention: float,
    repeat_index: int,
):

    # --------------------------------------------------------
    # Voxel
    # --------------------------------------------------------

    if method == "Voxel":

        result = (
            dm.voxel_downsample(
                points=points,
                retention_ratio=retention,
            )
        )

        return result.points

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    if method == "FPS":

        result = (
            dm.fps_downsample(
                points=points,
                retention_ratio=retention,
            )
        )

        return result.points

    # --------------------------------------------------------
    # Uniform
    # --------------------------------------------------------

    if method == "Uniform":

        result = (
            dm.uniform_downsample(
                points=points,
                retention_ratio=retention,
            )
        )

        return result.points

    # --------------------------------------------------------
    # Random
    # --------------------------------------------------------

    if method == "Random":

        rng = np.random.default_rng(
            RANDOM_BASE_SEED
            + repeat_index
        )

        result = (
            dm.random_downsample(
                points=points,
                retention_ratio=retention,
                rng=rng,
            )
        )

        return result.points

    raise ValueError(
        f"Unknown method: {method}"
    )


# ============================================================
# Main
# ============================================================

def main():

    sample_files = sorted(
        DATA_DIR.glob(
            "sample_*.npz"
        )
    )

    if not sample_files:

        raise FileNotFoundError(
            f"No synthetic samples found:\n"
            f"{DATA_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_conditions = (
        len(sample_files)
        * len(METHODS)
        * len(RETENTION_LEVELS)
    )

    print(
        "=" * 76
    )

    print(
        "DOWNSAMPLING REPEATABILITY TEST"
    )

    print(
        "=" * 76
    )

    print(
        f"Samples           : "
        f"{len(sample_files)}"
    )

    print(
        f"Repeats/condition : "
        f"{N_REPEATS}"
    )

    print(
        f"Conditions        : "
        f"{total_conditions}"
    )

    print()

    rows = []

    condition_index = 0

    for sample_index, path in enumerate(
        sample_files
    ):

        with np.load(
            path,
            allow_pickle=False,
        ) as data:

            points = np.asarray(
                data["points"],
                dtype=np.float64,
            )

        for retention in (
            RETENTION_LEVELS
        ):

            for method in METHODS:

                condition_index += 1

                outputs = []

                for repeat_index in range(
                    N_REPEATS
                ):

                    output = run_method(
                        points=points,
                        method=method,
                        retention=retention,
                        repeat_index=(
                            repeat_index
                            + sample_index * N_REPEATS
                        ),
                    )

                    outputs.append(
                        output
                    )

                reference = outputs[0]

                overlap_values = [
                    overlap_ratio(
                        reference,
                        outputs[i],
                    )
                    for i in range(
                        1,
                        N_REPEATS
                    )
                ]

                mean_overlap = float(
                    np.mean(
                        overlap_values
                    )
                )

                min_overlap = float(
                    np.min(
                        overlap_values
                    )
                )

                output_counts = [
                    len(output)
                    for output in outputs
                ]

                count_cv = float(
                    np.std(
                        output_counts
                    )
                    / np.mean(
                        output_counts
                    )
                )

                rows.append(
                    {
                        "sample":
                            path.name,

                        "sample_index":
                            sample_index,

                        "method":
                            method,

                        "target_retention":
                            retention,

                        "repeats":
                            N_REPEATS,

                        "mean_overlap":
                            mean_overlap,

                        "min_overlap":
                            min_overlap,

                        "output_count_mean":
                            float(
                                np.mean(
                                    output_counts
                                )
                            ),

                        "output_count_cv":
                            count_cv,
                    }
                )

                print(
                    f"[{condition_index:03d}/"
                    f"{total_conditions:03d}] "
                    f"{path.name} | "
                    f"{method:7s} | "
                    f"r={retention:.2f} | "
                    f"mean overlap="
                    f"{mean_overlap:.4f}"
                )

    # ========================================================
    # Save detailed results
    # ========================================================

    fieldnames = [
        "sample",
        "sample_index",
        "method",
        "target_retention",
        "repeats",
        "mean_overlap",
        "min_overlap",
        "output_count_mean",
        "output_count_cv",
    ]

    with RESULT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    # ========================================================
    # Method summary
    # ========================================================

    summary_rows = []

    for method in METHODS:

        method_rows = [
            row
            for row in rows
            if row["method"] == method
        ]

        mean_overlap = float(
            np.mean(
                [
                    row["mean_overlap"]
                    for row in method_rows
                ]
            )
        )

        worst_overlap = float(
            np.min(
                [
                    row["min_overlap"]
                    for row in method_rows
                ]
            )
        )

        mean_count_cv = float(
            np.mean(
                [
                    row["output_count_cv"]
                    for row in method_rows
                ]
            )
        )

        summary_rows.append(
            {
                "method":
                    method,

                "mean_overlap":
                    mean_overlap,

                "worst_overlap":
                    worst_overlap,

                "mean_output_count_cv":
                    mean_count_cv,
            }
        )

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "mean_overlap",
                "worst_overlap",
                "mean_output_count_cv",
            ],
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )

    # ========================================================
    # Console summary
    # ========================================================

    print()

    print(
        "=" * 76
    )

    print(
        "REPEATABILITY SUMMARY"
    )

    print(
        "=" * 76
    )

    for row in summary_rows:

        print(
            f"{row['method']:7s} | "
            f"mean overlap="
            f"{row['mean_overlap']:.4f} | "
            f"worst="
            f"{row['worst_overlap']:.4f} | "
            f"count CV="
            f"{row['mean_output_count_cv']:.6f}"
        )

    print()

    print(
        f"Detailed results : "
        f"{RESULT_FILE}"
    )

    print(
        f"Summary          : "
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()