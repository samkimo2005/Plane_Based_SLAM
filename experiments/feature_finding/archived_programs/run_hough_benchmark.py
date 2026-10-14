import csv
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree


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
    / "hough_results.csv"
)


# ==================================================
# Hough parameters
# ==================================================

K_NEIGHBORS = 30

THETA_BIN_DEG = 5.0
PHI_BIN_DEG = 5.0
D_BIN_M = 0.02

DISTANCE_THRESHOLD = 0.02

MIN_VOTES = 20


# ==================================================
# Estimate local normals
# ==================================================

def estimate_normals(
    points,
    k_neighbors
):
    """
    Estimate a local surface normal for every point
    using PCA on neighbouring points.
    """

    tree = cKDTree(
        points
    )

    distances, indices = tree.query(
        points,
        k=k_neighbors + 1
    )

    # First neighbour is the point itself
    neighbour_indices = indices[
        :,
        1:
    ]

    neighbours = points[
        neighbour_indices
    ]

    centroids = np.mean(
        neighbours,
        axis=1
    )

    centered = (
        neighbours
        - centroids[:, None, :]
    )

    covariance = np.einsum(
        "nki,nkj->nij",
        centered,
        centered
    )

    covariance = (
        covariance
        / k_neighbors
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    # Smallest eigenvalue corresponds
    # to the surface normal
    normals = eigenvectors[
        :,
        :,
        0
    ]

    normal_lengths = np.linalg.norm(
        normals,
        axis=1,
        keepdims=True
    )

    normals = (
        normals
        / normal_lengths
    )

    return normals


# ==================================================
# Make normal direction consistent
# ==================================================

def normalise_direction(
    normal
):
    """
    n and -n describe the same plane.

    This function forces a consistent direction
    before Hough voting.
    """

    normal = normal.copy()

    # For this benchmark the true plane normal
    # points mainly in the z direction.
    if normal[2] < 0:
        normal = -normal

    return normal


# ==================================================
# Convert normal to Hough angles
# ==================================================

def normal_to_angles(
    normal
):
    """
    Convert a unit normal vector to spherical angles.

    theta:
        angle from positive z axis

    phi:
        angle in x-y plane
    """

    nx = normal[0]
    ny = normal[1]
    nz = normal[2]

    nz = np.clip(
        nz,
        -1.0,
        1.0
    )

    theta = np.degrees(
        np.arccos(
            nz
        )
    )

    phi = np.degrees(
        np.arctan2(
            ny,
            nx
        )
    )

    return (
        theta,
        phi
    )


# ==================================================
# Fit plane using SVD
# ==================================================

def fit_plane_svd(
    points
):
    """
    Fit a plane to a group of points using SVD.
    """

    if len(points) < 3:
        return None

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

    normal_length = np.linalg.norm(
        normal
    )

    if normal_length < 1e-12:
        return None

    normal = (
        normal
        / normal_length
    )

    normal = normalise_direction(
        normal
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
# Point-to-plane distance
# ==================================================

def point_plane_distances(
    points,
    normal,
    d
):
    return np.abs(
        points @ normal
        + d
    )


# ==================================================
# Hough-style plane detection
# ==================================================

def hough_plane(
    points,
    k_neighbors,
    theta_bin_deg,
    phi_bin_deg,
    d_bin_m,
    distance_threshold,
    min_votes
):
    """
    Detect the dominant plane using a simple
    Hough-style voting method.

    Each point estimates a local normal.

    The local plane is represented using:
        theta
        phi
        d

    The parameters are quantised into bins.
    The bin receiving the largest number of votes
    is treated as the dominant plane candidate.
    """

    normals = estimate_normals(
        points,
        k_neighbors
    )

    votes = defaultdict(
        list
    )


    # --------------------------------------------------
    # Hough voting
    # --------------------------------------------------

    for index in range(
        len(points)
    ):

        normal = normalise_direction(
            normals[index]
        )

        theta, phi = normal_to_angles(
            normal
        )

        d = -np.dot(
            normal,
            points[index]
        )


        theta_bin = int(
            round(
                theta
                / theta_bin_deg
            )
        )

        phi_bin = int(
            round(
                phi
                / phi_bin_deg
            )
        )

        d_bin = int(
            round(
                d
                / d_bin_m
            )
        )


        key = (
            theta_bin,
            phi_bin,
            d_bin
        )


        votes[
            key
        ].append(
            index
        )


    # --------------------------------------------------
    # No votes
    # --------------------------------------------------

    if len(votes) == 0:
        return None


    # --------------------------------------------------
    # Find highest-vote Hough bin
    # --------------------------------------------------

    best_key = max(
        votes,
        key=lambda key: len(
            votes[key]
        )
    )


    voter_indices = votes[
        best_key
    ]


    num_votes = len(
        voter_indices
    )


    if num_votes < min_votes:
        return None


    voter_points = points[
        voter_indices
    ]


    # --------------------------------------------------
    # Initial plane fitting
    # --------------------------------------------------

    plane_result = fit_plane_svd(
        voter_points
    )

    if plane_result is None:
        return None


    normal, d = plane_result


    # --------------------------------------------------
    # Find all points close to the estimated plane
    # --------------------------------------------------

    distances = point_plane_distances(
        points,
        normal,
        d
    )

    inlier_mask = (
        distances
        <= distance_threshold
    )


    if np.count_nonzero(
        inlier_mask
    ) < 3:
        return None


    # --------------------------------------------------
    # Refine using all detected inliers
    # --------------------------------------------------

    refined_result = fit_plane_svd(
        points[
            inlier_mask
        ]
    )


    if refined_result is not None:

        normal, d = refined_result


        distances = point_plane_distances(
            points,
            normal,
            d
        )


        inlier_mask = (
            distances
            <= distance_threshold
        )


    return {
        "normal": normal,
        "d": d,
        "inlier_mask": inlier_mask,
        "num_votes": num_votes,
        "num_hough_bins": len(votes)
    }


# ==================================================
# Align plane direction with GT
# ==================================================

def align_plane_direction(
    normal,
    d,
    gt_normal
):
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
# Calculate normal error
# ==================================================

def calculate_normal_error(
    normal,
    gt_normal
):
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

    angle = np.degrees(
        np.arccos(
            cosine
        )
    )

    return float(
        angle
    )


# ==================================================
# Classification metrics
# ==================================================

def calculate_classification_metrics(
    predicted_inliers,
    gt_labels
):
    gt_inliers = (
        gt_labels == 1
    )

    gt_outliers = (
        gt_labels == 0
    )


    tp = np.count_nonzero(
        predicted_inliers
        & gt_inliers
    )

    fp = np.count_nonzero(
        predicted_inliers
        & gt_outliers
    )

    fn = np.count_nonzero(
        (~predicted_inliers)
        & gt_inliers
    )

    tn = np.count_nonzero(
        (~predicted_inliers)
        & gt_outliers
    )


    if (
        tp + fp
    ) > 0:

        precision = (
            tp
            / (
                tp + fp
            )
        )

    else:

        precision = 0.0


    if (
        tp + fn
    ) > 0:

        recall = (
            tp
            / (
                tp + fn
            )
        )

    else:

        recall = 0.0


    if (
        precision
        + recall
    ) > 0:

        f1 = (
            2
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
            tp
        ),

        "fp": int(
            fp
        ),

        "fn": int(
            fn
        ),

        "tn": int(
            tn
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
        "=== Hough Plane Detection Benchmark ==="
    )


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
        f"Expected 200 samples, "
        f"but found {len(sample_files)}"
    )


    # --------------------------------------------------
    # CSV columns
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

        "k_neighbors",
        "theta_bin_deg",
        "phi_bin_deg",
        "d_bin_m",
        "distance_threshold_m",
        "min_votes",

        "estimated_nx",
        "estimated_ny",
        "estimated_nz",
        "estimated_d",

        "detected_inlier_count",

        "num_votes",
        "num_hough_bins",

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
    # Open results file
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


        for index, sample_file in enumerate(
            sample_files,
            start=1
        ):

            sample_id = (
                sample_file.stem
            )


            # ----------------------------------------------
            # Load data
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
            # Load ground truth
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
            # Run Hough method
            # ----------------------------------------------

            start_time = perf_counter()


            result = hough_plane(
                points,
                k_neighbors=(
                    K_NEIGHBORS
                ),
                theta_bin_deg=(
                    THETA_BIN_DEG
                ),
                phi_bin_deg=(
                    PHI_BIN_DEG
                ),
                d_bin_m=(
                    D_BIN_M
                ),
                distance_threshold=(
                    DISTANCE_THRESHOLD
                ),
                min_votes=(
                    MIN_VOTES
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
                        "method": (
                            "Hough"
                        ),

                        "sample_id": (
                            sample_id
                        ),

                        "noise_sigma_m": (
                            metadata[
                                "noise_sigma_m"
                            ]
                        ),

                        "outlier_ratio": (
                            metadata[
                                "outlier_ratio"
                            ]
                        ),

                        "repeat_id": (
                            metadata[
                                "repeat_id"
                            ]
                        ),

                        "random_seed": (
                            metadata[
                                "random_seed"
                            ]
                        ),

                        "num_points": (
                            metadata[
                                "num_points"
                            ]
                        ),

                        "num_true_inliers": (
                            metadata[
                                "num_true_inliers"
                            ]
                        ),

                        "num_true_outliers": (
                            metadata[
                                "num_true_outliers"
                            ]
                        ),

                        "k_neighbors": (
                            K_NEIGHBORS
                        ),

                        "theta_bin_deg": (
                            THETA_BIN_DEG
                        ),

                        "phi_bin_deg": (
                            PHI_BIN_DEG
                        ),

                        "d_bin_m": (
                            D_BIN_M
                        ),

                        "distance_threshold_m": (
                            DISTANCE_THRESHOLD
                        ),

                        "min_votes": (
                            MIN_VOTES
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
            # Extract result
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


            (
                normal,
                d
            ) = align_plane_direction(
                normal,
                d,
                gt_normal
            )


            # ----------------------------------------------
            # Accuracy
            # ----------------------------------------------

            normal_error = (
                calculate_normal_error(
                    normal,
                    gt_normal
                )
            )


            offset_error = abs(
                d
                - gt_d
            )


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
            # Save results
            # ----------------------------------------------

            writer.writerow(
                {
                    "method": (
                        "Hough"
                    ),

                    "sample_id": (
                        sample_id
                    ),

                    "noise_sigma_m": (
                        metadata[
                            "noise_sigma_m"
                        ]
                    ),

                    "outlier_ratio": (
                        metadata[
                            "outlier_ratio"
                        ]
                    ),

                    "repeat_id": (
                        metadata[
                            "repeat_id"
                        ]
                    ),

                    "random_seed": (
                        metadata[
                            "random_seed"
                        ]
                    ),

                    "num_points": (
                        metadata[
                            "num_points"
                        ]
                    ),

                    "num_true_inliers": (
                        metadata[
                            "num_true_inliers"
                        ]
                    ),

                    "num_true_outliers": (
                        metadata[
                            "num_true_outliers"
                        ]
                    ),

                    "k_neighbors": (
                        K_NEIGHBORS
                    ),

                    "theta_bin_deg": (
                        THETA_BIN_DEG
                    ),

                    "phi_bin_deg": (
                        PHI_BIN_DEG
                    ),

                    "d_bin_m": (
                        D_BIN_M
                    ),

                    "distance_threshold_m": (
                        DISTANCE_THRESHOLD
                    ),

                    "min_votes": (
                        MIN_VOTES
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

                    "num_votes": int(
                        result[
                            "num_votes"
                        ]
                    ),

                    "num_hough_bins": int(
                        result[
                            "num_hough_bins"
                        ]
                    ),

                    "normal_error_deg": float(
                        normal_error
                    ),

                    "offset_error_m": float(
                        offset_error
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
            # Short console output
            # ----------------------------------------------

            print(
                f"[{index:03d}/200]",
                sample_id,

                "| normal error =",
                f"{normal_error:.4f}",
                "deg",

                "| precision =",
                f"{metrics['precision']:.4f}",

                "| recall =",
                f"{metrics['recall']:.4f}",

                "| F1 =",
                f"{metrics['f1']:.4f}",

                "| votes =",
                result[
                    "num_votes"
                ],

                "| runtime =",
                f"{runtime_ms:.2f}",
                "ms"
            )


    print(
        "\n========================================"
    )

    print(
        "=== Hough Benchmark Complete ==="
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