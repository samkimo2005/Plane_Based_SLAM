import csv
import json
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
    / "direct_plane_model_results.csv"
)


# ==================================================
# Direct plane model parameters
# ==================================================

K_NEIGHBORS = 30

NORMAL_THRESHOLD_DEG = 15.0

DISTANCE_THRESHOLD = 0.02

MIN_LOCAL_SUPPORT = 50


NORMAL_THRESHOLD_COS = np.cos(
    np.radians(
        NORMAL_THRESHOLD_DEG
    )
)


# ==================================================
# Estimate local planes
# ==================================================

def estimate_local_planes(
    points,
    k_neighbors
):
    """
    Estimate a local plane around every point.

    Local PCA is applied to the k nearest
    neighbours of each point.

    Returns:
        normals
        curvature
    """

    tree = cKDTree(
        points
    )


    distances, indices = tree.query(
        points,
        k=k_neighbors + 1
    )


    # Remove the point itself
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


    # Eigenvector belonging to smallest
    # eigenvalue is the local normal
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


    # Curvature estimate
    eigenvalue_sum = np.sum(
        eigenvalues,
        axis=1
    )


    curvature = np.zeros(
        len(points)
    )


    valid = (
        eigenvalue_sum > 1e-12
    )


    curvature[
        valid
    ] = (
        eigenvalues[
            valid,
            0
        ]
        / eigenvalue_sum[
            valid
        ]
    )


    return (
        normals,
        curvature
    )


# ==================================================
# Estimate dominant plane normal
# ==================================================

def estimate_dominant_normal(
    normals,
    curvature
):
    """
    Estimate one dominant plane direction.

    The sign of a plane normal is ambiguous:
        n and -n represent the same plane.

    Therefore, instead of directly averaging normals,
    outer products n*n^T are accumulated.

    Flatter local regions receive a slightly larger
    weight than highly curved regions.
    """

    # Lower curvature = more planar
    weights = (
        1.0
        / (
            curvature
            + 1e-6
        )
    )


    # Prevent a few extremely flat points from
    # receiving excessive weight
    weight_limit = np.percentile(
        weights,
        95
    )


    weights = np.minimum(
        weights,
        weight_limit
    )


    normal_matrix = np.zeros(
        (
            3,
            3
        )
    )


    for index in range(
        len(normals)
    ):

        normal = normals[
            index
        ]


        normal_matrix += (
            weights[index]
            * np.outer(
                normal,
                normal
            )
        )


    eigenvalues, eigenvectors = np.linalg.eigh(
        normal_matrix
    )


    # Largest eigenvalue gives the dominant
    # normal direction
    dominant_normal = eigenvectors[
        :,
        -1
    ]


    dominant_normal = (
        dominant_normal
        / np.linalg.norm(
            dominant_normal
        )
    )


    # Use a consistent sign
    largest_component = np.argmax(
        np.abs(
            dominant_normal
        )
    )


    if dominant_normal[
        largest_component
    ] < 0:

        dominant_normal = (
            -dominant_normal
        )


    return dominant_normal


# ==================================================
# Fit plane using SVD
# ==================================================

def fit_plane_svd(
    points
):
    """
    Fit one plane to points using SVD.
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


    d = -np.dot(
        normal,
        centroid
    )


    return (
        normal,
        d
    )


# ==================================================
# Point-to-plane distances
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
# Direct plane model detection
# ==================================================

def direct_plane_model(
    points,
    k_neighbors,
    normal_threshold_cos,
    distance_threshold,
    min_local_support
):
    """
    CPA-style direct plane-model baseline.

    The method:

    1. estimates local plane observations;
    2. finds a dominant normal direction;
    3. selects locally consistent plane observations;
    4. estimates plane offset robustly using a median;
    5. finds point-to-plane inliers;
    6. refines the final plane using SVD.
    """

    # --------------------------------------------------
    # Step 1: Local plane observations
    # --------------------------------------------------

    (
        local_normals,
        curvature
    ) = estimate_local_planes(
        points,
        k_neighbors
    )


    # --------------------------------------------------
    # Step 2: Dominant plane normal
    # --------------------------------------------------

    dominant_normal = estimate_dominant_normal(
        local_normals,
        curvature
    )


    # --------------------------------------------------
    # Step 3: Align local normal directions
    # --------------------------------------------------

    aligned_normals = (
        local_normals.copy()
    )


    for index in range(
        len(aligned_normals)
    ):

        if np.dot(
            aligned_normals[index],
            dominant_normal
        ) < 0:

            aligned_normals[index] = (
                -aligned_normals[index]
            )


    # --------------------------------------------------
    # Step 4: Select local planes whose normals
    # agree with dominant normal
    # --------------------------------------------------

    normal_similarity = (
        aligned_normals
        @ dominant_normal
    )


    local_support_mask = (
        normal_similarity
        >= normal_threshold_cos
    )


    local_support_count = np.count_nonzero(
        local_support_mask
    )


    if local_support_count < min_local_support:

        return {
            "success": False,
            "failure_reason": (
                "INSUFFICIENT_LOCAL_SUPPORT"
            ),
            "local_support_count": int(
                local_support_count
            )
        }


    # --------------------------------------------------
    # Step 5: Estimate plane offsets
    # --------------------------------------------------

    support_points = points[
        local_support_mask
    ]


    support_normals = aligned_normals[
        local_support_mask
    ]


    local_offsets = -np.sum(
        support_normals
        * support_points,
        axis=1
    )


    # Median is used instead of mean because it
    # is more resistant to outlier local planes.
    median_d = np.median(
        local_offsets
    )


    # Use the dominant normal and robust offset as
    # initial direct plane model
    initial_normal = (
        dominant_normal
    )

    initial_d = float(
        median_d
    )


    # --------------------------------------------------
    # Step 6: Initial inlier classification
    # --------------------------------------------------

    distances = point_plane_distances(
        points,
        initial_normal,
        initial_d
    )


    inlier_mask = (
        distances
        <= distance_threshold
    )


    initial_inlier_count = np.count_nonzero(
        inlier_mask
    )


    if initial_inlier_count < 3:

        return {
            "success": False,
            "failure_reason": (
                "TOO_FEW_INITIAL_INLIERS"
            ),
            "local_support_count": int(
                local_support_count
            )
        }


    # --------------------------------------------------
    # Step 7: Refine plane using detected inliers
    # --------------------------------------------------

    refined_result = fit_plane_svd(
        points[
            inlier_mask
        ]
    )


    if refined_result is None:

        return {
            "success": False,
            "failure_reason": (
                "PLANE_REFINEMENT_FAILED"
            ),
            "local_support_count": int(
                local_support_count
            )
        }


    normal, d = refined_result


    # Make refined normal agree with
    # the initial dominant direction
    if np.dot(
        normal,
        initial_normal
    ) < 0:

        normal = -normal
        d = -d


    # --------------------------------------------------
    # Step 8: Final inliers
    # --------------------------------------------------

    final_distances = point_plane_distances(
        points,
        normal,
        d
    )


    final_inlier_mask = (
        final_distances
        <= distance_threshold
    )


    return {
        "success": True,

        "failure_reason": "",

        "normal": normal,

        "d": float(
            d
        ),

        "inlier_mask": (
            final_inlier_mask
        ),

        "local_support_count": int(
            local_support_count
        ),

        "initial_inlier_count": int(
            initial_inlier_count
        )
    }


# ==================================================
# Align plane with ground truth
# ==================================================

def align_plane_direction(
    normal,
    d,
    gt_normal
):
    """
    n,d and -n,-d describe the same plane.
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
# Normal error
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
        precision + recall
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
        "=== CPA-style Direct Plane Model Benchmark ==="
    )


    print(
        "\nParameters:"
    )

    print(
        "K neighbours:",
        K_NEIGHBORS
    )

    print(
        "Normal threshold:",
        NORMAL_THRESHOLD_DEG,
        "degrees"
    )

    print(
        "Distance threshold:",
        DISTANCE_THRESHOLD,
        "m"
    )

    print(
        "Minimum local support:",
        MIN_LOCAL_SUPPORT
    )


    # --------------------------------------------------
    # Find samples
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
        "normal_threshold_deg",
        "distance_threshold_m",
        "min_local_support",

        "estimated_nx",
        "estimated_ny",
        "estimated_nz",
        "estimated_d",

        "local_support_count",
        "initial_inlier_count",
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

        "failure_reason",

        "success"
    ]


    # --------------------------------------------------
    # Open result CSV
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
        # Run all 200 samples
        # --------------------------------------------------

        for index, sample_file in enumerate(
            sample_files,
            start=1
        ):

            sample_id = (
                sample_file.stem
            )


            # ----------------------------------------------
            # Load points and labels
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
            # Load metadata / GT
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
            # Run direct plane model
            # ----------------------------------------------

            start_time = perf_counter()


            result = direct_plane_model(
                points,

                k_neighbors=(
                    K_NEIGHBORS
                ),

                normal_threshold_cos=(
                    NORMAL_THRESHOLD_COS
                ),

                distance_threshold=(
                    DISTANCE_THRESHOLD
                ),

                min_local_support=(
                    MIN_LOCAL_SUPPORT
                )
            )


            end_time = perf_counter()


            runtime_ms = (
                end_time
                - start_time
            ) * 1000.0


            # ----------------------------------------------
            # Failure
            # ----------------------------------------------

            if not result[
                "success"
            ]:

                writer.writerow(
                    {
                        "method": (
                            "CPA-style Direct Plane Model"
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

                        "normal_threshold_deg": (
                            NORMAL_THRESHOLD_DEG
                        ),

                        "distance_threshold_m": (
                            DISTANCE_THRESHOLD
                        ),

                        "min_local_support": (
                            MIN_LOCAL_SUPPORT
                        ),

                        "local_support_count": (
                            result.get(
                                "local_support_count",
                                0
                            )
                        ),

                        "runtime_ms": (
                            runtime_ms
                        ),

                        "failure_reason": (
                            result[
                                "failure_reason"
                            ]
                        ),

                        "success": 0
                    }
                )


                print(
                    f"[{index:03d}/200]",
                    sample_id,
                    "FAILED",

                    "| reason =",
                    result[
                        "failure_reason"
                    ],

                    "| runtime =",
                    f"{runtime_ms:.2f}",
                    "ms"
                )


                continue


            # ----------------------------------------------
            # Successful result
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


            normal, d = align_plane_direction(
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
            # Save result
            # ----------------------------------------------

            writer.writerow(
                {
                    "method": (
                        "CPA-style Direct Plane Model"
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

                    "normal_threshold_deg": (
                        NORMAL_THRESHOLD_DEG
                    ),

                    "distance_threshold_m": (
                        DISTANCE_THRESHOLD
                    ),

                    "min_local_support": (
                        MIN_LOCAL_SUPPORT
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

                    "local_support_count": int(
                        result[
                            "local_support_count"
                        ]
                    ),

                    "initial_inlier_count": int(
                        result[
                            "initial_inlier_count"
                        ]
                    ),

                    "detected_inlier_count": int(
                        detected_inlier_count
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

                    "failure_reason": "",

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
                f"{normal_error:.4f}",
                "deg",

                "| precision =",
                f"{metrics['precision']:.4f}",

                "| recall =",
                f"{metrics['recall']:.4f}",

                "| F1 =",
                f"{metrics['f1']:.4f}",

                "| support =",
                result[
                    "local_support_count"
                ],

                "| runtime =",
                f"{runtime_ms:.2f}",
                "ms"
            )


    # --------------------------------------------------
    # Finished
    # --------------------------------------------------

    print(
        "\n============================================"
    )

    print(
        "=== Direct Plane Model Benchmark Complete ==="
    )

    print(
        "============================================"
    )


    print(
        "\nResults saved to:"
    )

    print(
        output_csv
    )