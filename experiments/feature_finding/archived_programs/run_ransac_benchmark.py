import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np


# ==================================================
# Project paths
# ==================================================

project_root = Path(__file__).resolve().parents[2]

data_dir = (
    project_root
    / "data"
    / "synthetic_plane_benchmark"
)

results_dir = (
    project_root
    / "results"
    / "feature_finding"
)

results_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_csv = (
    results_dir
    / "ransac_results.csv"
)


# ==================================================
# RANSAC configuration
# ==================================================

DISTANCE_THRESHOLD = 0.02
NUM_ITERATIONS = 1000
RANSAC_SEED = 42


# ==================================================
# Fit a plane from three points
# ==================================================

def fit_plane_from_3_points(
    p1,
    p2,
    p3
):
    """
    Fit a plane from three 3D points.

    Plane equation:

        n_x x + n_y y + n_z z + d = 0

    Returns:
        normal, d

    or:
        None

    if the three points are approximately collinear.
    """

    v1 = p2 - p1
    v2 = p3 - p1

    normal = np.cross(
        v1,
        v2
    )

    norm = np.linalg.norm(
        normal
    )

    if norm < 1e-12:
        return None

    normal = (
        normal
        / norm
    )

    d = -np.dot(
        normal,
        p1
    )

    return (
        normal,
        d
    )


# ==================================================
# Point-to-plane distance
# ==================================================

def point_plane_distances(
    points,
    normal,
    d
):
    """
    Calculate perpendicular distances from
    all points to a plane.

    normal must already be normalised.
    """

    distances = np.abs(
        points @ normal
        + d
    )

    return distances


# ==================================================
# Refine plane using all detected inliers
# ==================================================

def refine_plane(
    points
):
    """
    Refine a plane model using all detected
    inlier points.

    SVD is used to find the direction with the
    smallest variance, which is the plane normal.
    """

    centroid = np.mean(
        points,
        axis=0
    )

    centered = (
        points
        - centroid
    )

    _, _, vh = np.linalg.svd(
        centered,
        full_matrices=False
    )

    normal = vh[
        -1
    ]

    normal_norm = np.linalg.norm(
        normal
    )

    if normal_norm < 1e-12:
        return None

    normal = (
        normal
        / normal_norm
    )

    d = -np.dot(
        normal,
        centroid
    )

    return (
        normal,
        d
    )


# ==================================================
# RANSAC plane detection
# ==================================================

def ransac_plane(
    points,
    distance_threshold,
    num_iterations,
    seed
):
    """
    Detect the dominant plane in a point cloud.

    Procedure:
        1. Randomly select three points.
        2. Fit a candidate plane.
        3. Count inliers.
        4. Keep the model with the largest
           consensus set.
        5. Refine the best model using all inliers.

    Returns:
        dictionary containing:
            normal
            d
            inlier_mask

        or None if fitting fails.
    """

    rng = np.random.default_rng(
        seed
    )

    num_points = len(
        points
    )

    if num_points < 3:
        return None

    best_inlier_mask = None
    best_normal = None
    best_d = None
    best_inlier_count = -1


    # --------------------------------------------------
    # RANSAC iterations
    # --------------------------------------------------

    for _ in range(
        num_iterations
    ):

        indices = rng.choice(
            num_points,
            size=3,
            replace=False
        )

        p1 = points[
            indices[0]
        ]

        p2 = points[
            indices[1]
        ]

        p3 = points[
            indices[2]
        ]


        result = fit_plane_from_3_points(
            p1,
            p2,
            p3
        )

        if result is None:
            continue


        normal, d = result


        distances = point_plane_distances(
            points,
            normal,
            d
        )


        inlier_mask = (
            distances
            <= distance_threshold
        )


        inlier_count = np.count_nonzero(
            inlier_mask
        )


        if inlier_count > best_inlier_count:

            best_inlier_count = (
                inlier_count
            )

            best_inlier_mask = (
                inlier_mask.copy()
            )

            best_normal = (
                normal.copy()
            )

            best_d = float(
                d
            )


    # --------------------------------------------------
    # Check whether RANSAC found any valid plane
    # --------------------------------------------------

    if best_inlier_mask is None:
        return None


    if np.count_nonzero(
        best_inlier_mask
    ) < 3:
        return None


    # --------------------------------------------------
    # Refine using all inliers
    # --------------------------------------------------

    inlier_points = points[
        best_inlier_mask
    ]


    refined_result = refine_plane(
        inlier_points
    )

    if refined_result is None:

        refined_normal = (
            best_normal
        )

        refined_d = (
            best_d
        )

    else:

        (
            refined_normal,
            refined_d
        ) = refined_result


    # --------------------------------------------------
    # Recalculate inlier mask using refined model
    # --------------------------------------------------

    refined_distances = point_plane_distances(
        points,
        refined_normal,
        refined_d
    )

    refined_inlier_mask = (
        refined_distances
        <= distance_threshold
    )


    return {
        "normal": refined_normal,
        "d": refined_d,
        "inlier_mask": refined_inlier_mask
    }


# ==================================================
# Plane orientation alignment
# ==================================================

def align_plane_direction(
    normal,
    d,
    gt_normal
):
    """
    A plane represented by:

        (n, d)

    is identical to:

        (-n, -d)

    Therefore align the estimated normal with the
    ground-truth normal before calculating errors.
    """

    if np.dot(
        normal,
        gt_normal
    ) < 0:

        normal = -normal
        d = -d

    return (
        normal,
        d
    )


# ==================================================
# Normal-vector angular error
# ==================================================

def calculate_normal_error(
    normal,
    gt_normal
):
    """
    Calculate angular error between estimated
    and ground-truth plane normals in degrees.
    """

    normal = (
        normal
        / np.linalg.norm(
            normal
        )
    )

    gt_normal = (
        gt_normal
        / np.linalg.norm(
            gt_normal
        )
    )


    cosine = np.clip(
        np.dot(
            normal,
            gt_normal
        ),
        -1.0,
        1.0
    )


    angle_rad = np.arccos(
        cosine
    )


    angle_deg = np.degrees(
        angle_rad
    )


    return float(
        angle_deg
    )


# ==================================================
# Classification metrics
# ==================================================

def calculate_classification_metrics(
    predicted_inliers,
    gt_labels
):
    """
    Compare detected RANSAC inliers with
    synthetic ground-truth point labels.

    Ground truth:
        1 = true plane point
        0 = outlier

    Prediction:
        True  = detected plane inlier
        False = rejected point
    """

    gt_inliers = (
        gt_labels == 1
    )

    gt_outliers = (
        gt_labels == 0
    )


    true_positive = np.count_nonzero(
        predicted_inliers
        & gt_inliers
    )


    false_positive = np.count_nonzero(
        predicted_inliers
        & gt_outliers
    )


    false_negative = np.count_nonzero(
        (~predicted_inliers)
        & gt_inliers
    )


    true_negative = np.count_nonzero(
        (~predicted_inliers)
        & gt_outliers
    )


    precision_denominator = (
        true_positive
        + false_positive
    )


    recall_denominator = (
        true_positive
        + false_negative
    )


    if precision_denominator > 0:

        precision = (
            true_positive
            / precision_denominator
        )

    else:

        precision = 0.0


    if recall_denominator > 0:

        recall = (
            true_positive
            / recall_denominator
        )

    else:

        recall = 0.0


    if (
        precision
        + recall
        > 0
    ):

        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    else:

        f1 = 0.0


    return {
        "tp": int(
            true_positive
        ),

        "fp": int(
            false_positive
        ),

        "fn": int(
            false_negative
        ),

        "tn": int(
            true_negative
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        )
    }


# ==================================================
# Main benchmark
# ==================================================

if __name__ == "__main__":

    print(
        "=== RANSAC Plane Detection Benchmark ==="
    )


    # --------------------------------------------------
    # Find all benchmark samples
    # --------------------------------------------------

    sample_files = sorted(
        data_dir.glob(
            "N*_O*_R*.npz"
        )
    )


    print(
        "\nNumber of samples:",
        len(sample_files)
    )


    assert len(
        sample_files
    ) == 200, (
        f"Expected 200 benchmark samples, "
        f"but found {len(sample_files)}."
    )


    # --------------------------------------------------
    # Define CSV fields
    # --------------------------------------------------

    fieldnames = [
        "method",
        "sample_id",

        "noise_sigma_m",
        "outlier_ratio",
        "repeat_id",
        "random_seed",

        "num_points",
        "num_true_inliers",
        "num_true_outliers",

        "distance_threshold_m",
        "num_iterations",
        "ransac_seed",

        "estimated_nx",
        "estimated_ny",
        "estimated_nz",
        "estimated_d",

        "detected_inlier_count",

        "normal_error_deg",
        "offset_error_m",

        "tp",
        "fp",
        "fn",
        "tn",

        "precision",
        "recall",
        "f1",

        "runtime_ms",

        "success"
    ]


    # --------------------------------------------------
    # Open output CSV
    # --------------------------------------------------

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()


        # --------------------------------------------------
        # Process all 200 synthetic samples
        # --------------------------------------------------

        for index, sample_file in enumerate(
            sample_files,
            start=1
        ):

            sample_id = (
                sample_file.stem
            )


            # ----------------------------------------------
            # Load synthetic points and labels
            # ----------------------------------------------

            data = np.load(
                sample_file
            )

            points = data[
                "points"
            ]

            labels = data[
                "labels"
            ]


            # ----------------------------------------------
            # Load metadata / ground truth
            # ----------------------------------------------

            gt_file = (
                data_dir
                / f"{sample_id}_gt.json"
            )


            with open(
                gt_file,
                "r",
                encoding="utf-8"
            ) as f:

                metadata = json.load(
                    f
                )


            gt_normal = np.array(
                metadata[
                    "ground_truth_plane"
                ][
                    "normal"
                ],
                dtype=float
            )


            gt_d = float(
                metadata[
                    "ground_truth_plane"
                ][
                    "d"
                ]
            )


            # ----------------------------------------------
            # Run and time RANSAC
            # ----------------------------------------------

            start_time = perf_counter()


            result = ransac_plane(
                points,
                distance_threshold=(
                    DISTANCE_THRESHOLD
                ),
                num_iterations=(
                    NUM_ITERATIONS
                ),
                seed=(
                    RANSAC_SEED
                )
            )


            end_time = perf_counter()


            runtime_ms = (
                end_time
                - start_time
            ) * 1000.0


            # ----------------------------------------------
            # Failure case
            # ----------------------------------------------

            if result is None:

                writer.writerow(
                    {
                        "method": "RANSAC",

                        "sample_id": sample_id,

                        "noise_sigma_m": metadata[
                            "noise_sigma_m"
                        ],

                        "outlier_ratio": metadata[
                            "outlier_ratio"
                        ],

                        "repeat_id": metadata[
                            "repeat_id"
                        ],

                        "random_seed": metadata[
                            "random_seed"
                        ],

                        "num_points": metadata[
                            "num_points"
                        ],

                        "num_true_inliers": metadata[
                            "num_true_inliers"
                        ],

                        "num_true_outliers": metadata[
                            "num_true_outliers"
                        ],

                        "distance_threshold_m": (
                            DISTANCE_THRESHOLD
                        ),

                        "num_iterations": (
                            NUM_ITERATIONS
                        ),

                        "ransac_seed": (
                            RANSAC_SEED
                        ),

                        "runtime_ms": (
                            runtime_ms
                        ),

                        "success": 0
                    }
                )


                print(
                    f"[{index:03d}/200]",
                    sample_id,
                    "FAILED"
                )


                continue


            # ----------------------------------------------
            # Extract estimated model
            # ----------------------------------------------

            normal = result[
                "normal"
            ]

            d = float(
                result[
                    "d"
                ]
            )

            predicted_inliers = result[
                "inlier_mask"
            ]


            # ----------------------------------------------
            # Align plane normal direction
            # ----------------------------------------------

            (
                normal,
                d
            ) = align_plane_direction(
                normal,
                d,
                gt_normal
            )


            # ----------------------------------------------
            # Calculate plane-model errors
            # ----------------------------------------------

            normal_error_deg = (
                calculate_normal_error(
                    normal,
                    gt_normal
                )
            )


            offset_error_m = abs(
                d
                - gt_d
            )


            # ----------------------------------------------
            # Classification metrics
            # ----------------------------------------------

            metrics = (
                calculate_classification_metrics(
                    predicted_inliers,
                    labels
                )
            )


            detected_inlier_count = (
                np.count_nonzero(
                    predicted_inliers
                )
            )


            # ----------------------------------------------
            # Save result
            # ----------------------------------------------

            writer.writerow(
                {
                    "method": "RANSAC",

                    "sample_id": sample_id,

                    "noise_sigma_m": metadata[
                        "noise_sigma_m"
                    ],

                    "outlier_ratio": metadata[
                        "outlier_ratio"
                    ],

                    "repeat_id": metadata[
                        "repeat_id"
                    ],

                    "random_seed": metadata[
                        "random_seed"
                    ],

                    "num_points": metadata[
                        "num_points"
                    ],

                    "num_true_inliers": metadata[
                        "num_true_inliers"
                    ],

                    "num_true_outliers": metadata[
                        "num_true_outliers"
                    ],

                    "distance_threshold_m": (
                        DISTANCE_THRESHOLD
                    ),

                    "num_iterations": (
                        NUM_ITERATIONS
                    ),

                    "ransac_seed": (
                        RANSAC_SEED
                    ),

                    "estimated_nx": float(
                        normal[0]
                    ),

                    "estimated_ny": float(
                        normal[1]
                    ),

                    "estimated_nz": float(
                        normal[2]
                    ),

                    "estimated_d": float(
                        d
                    ),

                    "detected_inlier_count": int(
                        detected_inlier_count
                    ),

                    "normal_error_deg": float(
                        normal_error_deg
                    ),

                    "offset_error_m": float(
                        offset_error_m
                    ),

                    "tp": metrics[
                        "tp"
                    ],

                    "fp": metrics[
                        "fp"
                    ],

                    "fn": metrics[
                        "fn"
                    ],

                    "tn": metrics[
                        "tn"
                    ],

                    "precision": metrics[
                        "precision"
                    ],

                    "recall": metrics[
                        "recall"
                    ],

                    "f1": metrics[
                        "f1"
                    ],

                    "runtime_ms": float(
                        runtime_ms
                    ),

                    "success": 1
                }
            )


            # ----------------------------------------------
            # Console output
            # ----------------------------------------------

            print(
                f"[{index:03d}/200]",
                sample_id,

                "| normal error =",
                f"{normal_error_deg:.4f}",
                "deg",

                "| offset error =",
                f"{offset_error_m:.6f}",
                "m",

                "| precision =",
                f"{metrics['precision']:.4f}",

                "| recall =",
                f"{metrics['recall']:.4f}",

                "| F1 =",
                f"{metrics['f1']:.4f}",

                "| runtime =",
                f"{runtime_ms:.2f}",
                "ms"
            )


    # --------------------------------------------------
    # Final message
    # --------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "=== RANSAC Benchmark Complete ==="
    )

    print(
        "========================================"
    )

    print(
        "\nResults saved to:"
    )

    print(
        output_csv
    )