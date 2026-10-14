import json
from pathlib import Path

import numpy as np


# --------------------------------------------------
# Project paths
# --------------------------------------------------

project_root = Path(__file__).resolve().parents[2]

data_dir = (
    project_root
    / "data"
    / "synthetic_plane_benchmark"
)


EXPECTED_SAMPLES = 200


# --------------------------------------------------
# Find all benchmark samples
# --------------------------------------------------

files = sorted(
    data_dir.glob(
        "N*_O*_R*.npz"
    )
)


print(
    "=== Synthetic Benchmark Verification ==="
)

print(
    "\nNumber of benchmark samples:",
    len(files)
)


assert len(files) == EXPECTED_SAMPLES, (
    f"Expected {EXPECTED_SAMPLES} samples, "
    f"but found {len(files)}."
)


# Track how many samples exist
# for every condition.
condition_counts = {}


for file in files:

    sample_name = file.stem


    data = np.load(
        file
    )

    points = data[
        "points"
    ]

    labels = data[
        "labels"
    ]


    # --------------------------------------------------
    # Basic shape checks
    # --------------------------------------------------

    assert points.shape == (
        10000,
        3
    ), (
        f"{sample_name}: "
        f"unexpected point shape {points.shape}"
    )


    assert labels.shape == (
        10000,
    ), (
        f"{sample_name}: "
        f"unexpected label shape {labels.shape}"
    )


    # --------------------------------------------------
    # Finite-value check
    # --------------------------------------------------

    assert np.all(
        np.isfinite(points)
    ), (
        f"{sample_name}: "
        "NaN or infinity found."
    )


    # --------------------------------------------------
    # Label check
    # --------------------------------------------------

    assert np.all(
        np.isin(
            labels,
            [0, 1]
        )
    ), (
        f"{sample_name}: "
        "invalid labels."
    )


    num_true_inliers = np.count_nonzero(
        labels == 1
    )

    num_true_outliers = np.count_nonzero(
        labels == 0
    )


    # --------------------------------------------------
    # Load metadata
    # --------------------------------------------------

    json_file = (
        data_dir
        / f"{sample_name}_gt.json"
    )


    assert json_file.exists(), (
        f"{sample_name}: "
        "metadata JSON missing."
    )


    with open(
        json_file,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(
            f
        )


    # --------------------------------------------------
    # Metadata checks
    # --------------------------------------------------

    assert metadata[
        "sample_id"
    ] == sample_name


    assert metadata[
        "num_points"
    ] == 10000


    assert metadata[
        "num_true_inliers"
    ] == num_true_inliers


    assert metadata[
        "num_true_outliers"
    ] == num_true_outliers


    assert 1 <= metadata[
        "repeat_id"
    ] <= 10


    # --------------------------------------------------
    # GT plane check
    # --------------------------------------------------

    gt_plane = metadata[
        "ground_truth_plane"
    ]

    gt_normal = np.array(
        gt_plane[
            "normal"
        ],
        dtype=float
    )

    gt_d = float(
        gt_plane[
            "d"
        ]
    )


    assert np.allclose(
        gt_normal,
        np.array(
            [0.0, 0.0, 1.0]
        )
    )


    assert np.isclose(
        gt_d,
        -2.0
    )


    # --------------------------------------------------
    # Outlier-count check
    # --------------------------------------------------

    expected_outliers = int(
        10000
        * metadata[
            "outlier_ratio"
        ]
    )


    assert num_true_outliers == (
        expected_outliers
    )


    # --------------------------------------------------
    # Count repetitions per condition
    # --------------------------------------------------

    condition = (
        metadata[
            "noise_sigma_m"
        ],
        metadata[
            "outlier_ratio"
        ]
    )


    condition_counts[
        condition
    ] = (
        condition_counts.get(
            condition,
            0
        )
        + 1
    )


print(
    "\nChecking repetitions per condition..."
)


assert len(
    condition_counts
) == 20, (
    "Expected 20 unique noise/outlier conditions."
)


for condition, count in sorted(
    condition_counts.items()
):

    print(
        "Noise:",
        condition[0],
        "| Outlier ratio:",
        condition[1],
        "| Samples:",
        count
    )

    assert count == 10, (
        f"Condition {condition} "
        f"has {count} samples instead of 10."
    )


print(
    "\n========================================"
)

print(
    "=== Synthetic Benchmark Verification PASSED ==="
)

print(
    "Total samples:",
    len(files)
)

print(
    "Unique conditions:",
    len(condition_counts)
)

print(
    "Samples per condition: 10"
)

print(
    "========================================"
)