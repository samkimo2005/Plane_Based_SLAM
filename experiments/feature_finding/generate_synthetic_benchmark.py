import json
from pathlib import Path

import numpy as np


# --------------------------------------------------
# Project paths
# --------------------------------------------------

project_root = Path(__file__).resolve().parents[2]

output_dir = (
    project_root
    / "data"
    / "synthetic_plane_benchmark"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Benchmark configuration
# --------------------------------------------------

TOTAL_POINTS = 10000

PLANE_Z = 2.0

XY_MIN = -2.0
XY_MAX = 2.0


GT_NORMAL = np.array(
    [0.0, 0.0, 1.0]
)

GT_D = -2.0


NOISE_LEVELS = [
    0.000,
    0.005,
    0.010,
    0.020
]


OUTLIER_RATIOS = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4
]


REPEATS_PER_CONDITION = 10


# --------------------------------------------------
# Generate one benchmark sample
# --------------------------------------------------

def generate_sample(
    noise_sigma,
    outlier_ratio,
    seed
):
    rng = np.random.default_rng(
        seed
    )

    num_outliers = int(
        TOTAL_POINTS
        * outlier_ratio
    )

    num_inliers = (
        TOTAL_POINTS
        - num_outliers
    )


    # --------------------------------------------------
    # Generate true plane points
    # --------------------------------------------------

    x = rng.uniform(
        XY_MIN,
        XY_MAX,
        num_inliers
    )

    y = rng.uniform(
        XY_MIN,
        XY_MAX,
        num_inliers
    )

    z = (
        PLANE_Z
        + rng.normal(
            loc=0.0,
            scale=noise_sigma,
            size=num_inliers
        )
    )


    plane_points = np.column_stack(
        (
            x,
            y,
            z
        )
    )


    # --------------------------------------------------
    # Generate outliers
    # --------------------------------------------------

    if num_outliers > 0:

        outlier_x = rng.uniform(
            XY_MIN,
            XY_MAX,
            num_outliers
        )

        outlier_y = rng.uniform(
            XY_MIN,
            XY_MAX,
            num_outliers
        )

        lower_or_upper = rng.random(
            num_outliers
        )

        outlier_z = np.zeros(
            num_outliers,
            dtype=float
        )

        lower_mask = (
            lower_or_upper < 0.5
        )

        upper_mask = ~lower_mask


        num_lower = np.count_nonzero(
            lower_mask
        )

        if num_lower > 0:

            outlier_z[
                lower_mask
            ] = rng.uniform(
                0.0,
                1.5,
                num_lower
            )


        num_upper = np.count_nonzero(
            upper_mask
        )

        if num_upper > 0:

            outlier_z[
                upper_mask
            ] = rng.uniform(
                2.5,
                4.0,
                num_upper
            )


        outlier_points = np.column_stack(
            (
                outlier_x,
                outlier_y,
                outlier_z
            )
        )

    else:

        outlier_points = np.empty(
            (0, 3),
            dtype=float
        )


    # --------------------------------------------------
    # Combine points
    # --------------------------------------------------

    points = np.vstack(
        (
            plane_points,
            outlier_points
        )
    )


    # --------------------------------------------------
    # Ground-truth labels
    # --------------------------------------------------

    labels = np.concatenate(
        (
            np.ones(
                num_inliers,
                dtype=np.uint8
            ),
            np.zeros(
                num_outliers,
                dtype=np.uint8
            )
        )
    )


    # --------------------------------------------------
    # Shuffle
    # --------------------------------------------------

    permutation = rng.permutation(
        TOTAL_POINTS
    )

    points = points[
        permutation
    ]

    labels = labels[
        permutation
    ]


    # --------------------------------------------------
    # Safety checks
    # --------------------------------------------------

    if points.shape != (
        TOTAL_POINTS,
        3
    ):
        raise ValueError(
            "Generated point cloud has incorrect shape."
        )


    if labels.shape != (
        TOTAL_POINTS,
    ):
        raise ValueError(
            "Generated labels have incorrect shape."
        )


    if not np.all(
        np.isfinite(points)
    ):
        raise ValueError(
            "Generated point cloud contains NaN or infinity."
        )


    return (
        points,
        labels,
        num_inliers,
        num_outliers
    )


# --------------------------------------------------
# Helper for sample names
# --------------------------------------------------

def noise_tag(
    noise_sigma
):
    return int(
        round(
            noise_sigma
            * 1000
        )
    )


def outlier_tag(
    outlier_ratio
):
    return int(
        round(
            outlier_ratio
            * 100
        )
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print(
        "=== Synthetic Plane Benchmark Generation ==="
    )


    global_sample_id = 1


    for noise_sigma in NOISE_LEVELS:

        for outlier_ratio in OUTLIER_RATIOS:

            for repeat_id in range(
                1,
                REPEATS_PER_CONDITION + 1
            ):

                seed = (
                    1000
                    + global_sample_id
                )


                (
                    points,
                    labels,
                    num_inliers,
                    num_outliers
                ) = generate_sample(
                    noise_sigma,
                    outlier_ratio,
                    seed
                )


                sample_name = (
                    f"N{noise_tag(noise_sigma):03d}"
                    f"_O{outlier_tag(outlier_ratio):02d}"
                    f"_R{repeat_id:02d}"
                )


                # --------------------------------------------------
                # Save points and labels
                # --------------------------------------------------

                np.savez_compressed(
                    output_dir
                    / f"{sample_name}.npz",

                    points=points,
                    labels=labels
                )


                # --------------------------------------------------
                # Save metadata
                # --------------------------------------------------

                metadata = {
                    "sample_id": sample_name,

                    "noise_sigma_m": float(
                        noise_sigma
                    ),

                    "outlier_ratio": float(
                        outlier_ratio
                    ),

                    "repeat_id": int(
                        repeat_id
                    ),

                    "random_seed": int(
                        seed
                    ),

                    "num_points": int(
                        TOTAL_POINTS
                    ),

                    "num_true_inliers": int(
                        num_inliers
                    ),

                    "num_true_outliers": int(
                        num_outliers
                    ),

                    "ground_truth_plane": {
                        "normal": [
                            0.0,
                            0.0,
                            1.0
                        ],
                        "d": float(
                            GT_D
                        )
                    }
                }


                with open(
                    output_dir
                    / f"{sample_name}_gt.json",
                    "w",
                    encoding="utf-8"
                ) as f:

                    json.dump(
                        metadata,
                        f,
                        indent=4
                    )


                print(
                    sample_name,
                    "| noise =",
                    noise_sigma,
                    "| outlier ratio =",
                    outlier_ratio,
                    "| repeat =",
                    repeat_id,
                    "| inliers =",
                    num_inliers,
                    "| outliers =",
                    num_outliers
                )


                global_sample_id += 1


    total_samples = (
        len(NOISE_LEVELS)
        * len(OUTLIER_RATIOS)
        * REPEATS_PER_CONDITION
    )


    print(
        "\nSynthetic benchmark generation complete."
    )

    print(
        "Total samples:",
        total_samples
    )

    print(
        "Output directory:",
        output_dir
    )