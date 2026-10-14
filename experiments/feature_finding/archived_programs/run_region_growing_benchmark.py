import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


# ==========================================================
# Project paths
# ==========================================================

project_root = Path(
    __file__
).resolve().parents[2]


benchmark_dir = (
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


output_file = (
    results_dir
    / "region_growing_results.csv"
)


# ==========================================================
# Fixed Region Growing parameters
# ==========================================================

K_NEIGHBORS = 30

SMOOTHNESS_THRESHOLD_DEG = 15.0

REGION_DISTANCE_THRESHOLD = 0.02

MIN_REGION_SIZE = 100


# ==========================================================
# Common benchmark final threshold
#
# IMPORTANT:
# This is the mask used for Precision / Recall / F1.
# ==========================================================

FINAL_DISTANCE_THRESHOLD = 0.02


# ==========================================================
# Plane fitting
# ==========================================================

def fit_plane_svd(
    points
):
    """
    Fit:

        n^T p + d = 0

    using SVD.
    """

    if len(points) < 3:
        return None


    centroid = np.mean(
        points,
        axis=0
    )


    centred = (
        points
        - centroid
    )


    _, _, vh = np.linalg.svd(
        centred,
        full_matrices=False
    )


    normal = vh[-1]


    norm = np.linalg.norm(
        normal
    )


    if norm < 1e-10:
        return None


    normal = (
        normal
        / norm
    )


    d = -np.dot(
        normal,
        centroid
    )


    return (
        normal,
        d
    )


# ==========================================================
# Resolve plane sign
# ==========================================================

def canonicalise_plane(
    normal,
    d
):
    """
    Resolve sign ambiguity without using GT.
    """

    largest_component = int(
        np.argmax(
            np.abs(
                normal
            )
        )
    )


    if normal[
        largest_component
    ] < 0:

        normal = -normal
        d = -d


    return (
        normal,
        d
    )


# ==========================================================
# Plane-normal angle
# ==========================================================

def normal_angle_deg(
    normal_a,
    normal_b
):
    dot_value = np.dot(
        normal_a,
        normal_b
    )


    dot_value = np.clip(
        abs(
            dot_value
        ),
        0.0,
        1.0
    )


    return np.degrees(
        np.arccos(
            dot_value
        )
    )


# ==========================================================
# Estimate local normals and curvature
# ==========================================================

def estimate_normals_and_curvature(
    points
):
    """
    Estimate one local normal and curvature value
    for every point using PCA of its kNN neighbourhood.
    """

    num_points = len(
        points
    )


    if num_points <= K_NEIGHBORS:

        return None


    tree = cKDTree(
        points
    )


    _, neighbour_indices = tree.query(
        points,
        k=(
            K_NEIGHBORS
            + 1
        )
    )


    # Remove the point itself.

    neighbour_indices = (
        neighbour_indices[
            :,
            1:
        ]
    )


    neighbours = points[
        neighbour_indices
    ]


    centroids = np.mean(
        neighbours,
        axis=1
    )


    centred = (
        neighbours
        - centroids[
            :,
            None,
            :
        ]
    )


    covariance = np.einsum(
        "nki,nkj->nij",
        centred,
        centred
    )


    covariance /= float(
        K_NEIGHBORS
    )


    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )


    normals = (
        eigenvectors[
            :,
            :,
            0
        ]
    )


    eigenvalue_sum = np.sum(
        eigenvalues,
        axis=1
    )


    curvature = np.ones(
        num_points,
        dtype=np.float64
    )


    valid = (
        eigenvalue_sum
        > 1e-12
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
        curvature,
        neighbour_indices
    )


# ==========================================================
# Region Growing
# ==========================================================

def grow_regions(
    points,
    normals,
    curvature,
    neighbour_indices
):
    """
    Grow smooth locally coplanar regions.

    Curvature is used to choose flatter points first
    as seeds.

    A neighbouring point is accepted if:

      1. its local normal is sufficiently similar;
      2. it lies sufficiently close to the current
         local tangent plane.
    """

    num_points = len(
        points
    )


    assigned = np.zeros(
        num_points,
        dtype=bool
    )


    seed_order = np.argsort(
        curvature
    )


    cos_threshold = np.cos(
        np.deg2rad(
            SMOOTHNESS_THRESHOLD_DEG
        )
    )


    valid_regions = []


    for seed_index in seed_order:

        if assigned[
            seed_index
        ]:

            continue


        assigned[
            seed_index
        ] = True


        queue = deque(
            [
                int(
                    seed_index
                )
            ]
        )


        region_indices = [
            int(
                seed_index
            )
        ]


        while queue:

            current_index = (
                queue.popleft()
            )


            current_normal = (
                normals[
                    current_index
                ]
            )


            current_point = (
                points[
                    current_index
                ]
            )


            neighbours = (
                neighbour_indices[
                    current_index
                ]
            )


            for neighbour_index in neighbours:

                neighbour_index = int(
                    neighbour_index
                )


                if assigned[
                    neighbour_index
                ]:

                    continue


                neighbour_normal = (
                    normals[
                        neighbour_index
                    ]
                )


                # ------------------------------------------
                # Smoothness criterion
                # ------------------------------------------

                normal_similarity = abs(
                    np.dot(
                        current_normal,
                        neighbour_normal
                    )
                )


                if (
                    normal_similarity
                    < cos_threshold
                ):

                    continue


                # ------------------------------------------
                # Local point-to-plane criterion
                # ------------------------------------------

                displacement = (
                    points[
                        neighbour_index
                    ]
                    - current_point
                )


                local_distance = abs(
                    np.dot(
                        current_normal,
                        displacement
                    )
                )


                if (
                    local_distance
                    > REGION_DISTANCE_THRESHOLD
                ):

                    continue


                # ------------------------------------------
                # Accept neighbour
                # ------------------------------------------

                assigned[
                    neighbour_index
                ] = True


                region_indices.append(
                    neighbour_index
                )


                queue.append(
                    neighbour_index
                )


        if (
            len(
                region_indices
            )
            >= MIN_REGION_SIZE
        ):

            valid_regions.append(
                region_indices
            )


    return valid_regions


# ==========================================================
# Full Region Growing detector
# ==========================================================

def detect_plane_region_growing(
    points
):
    """
    Region Growing V2:

       kNN
        ↓
       local normals
        ↓
       smooth connected regions
        ↓
       choose largest region
        ↓
       SVD plane fit
        ↓
       COMMON 0.02 m full-cloud final mask
    """

    normal_result = (
        estimate_normals_and_curvature(
            points
        )
    )


    if normal_result is None:

        return None, (
            "normal_estimation_failed"
        )


    (
        normals,
        curvature,
        neighbour_indices
    ) = normal_result


    regions = grow_regions(
        points,
        normals,
        curvature,
        neighbour_indices
    )


    if len(
        regions
    ) == 0:

        return None, (
            "no_valid_region"
        )


    # ------------------------------------------------------
    # Largest detected region
    # ------------------------------------------------------

    largest_region = max(
        regions,
        key=len
    )


    native_region_mask = np.zeros(
        len(
            points
        ),
        dtype=bool
    )


    native_region_mask[
        largest_region
    ] = True


    region_points = points[
        native_region_mask
    ]


    # ------------------------------------------------------
    # SVD plane refinement
    # ------------------------------------------------------

    fitted_plane = fit_plane_svd(
        region_points
    )


    if fitted_plane is None:

        return None, (
            "svd_refinement_failed"
        )


    normal, d = fitted_plane


    normal, d = canonicalise_plane(
        normal,
        d
    )


    # ------------------------------------------------------
    # FAIR COMMON EVALUATION ADAPTER
    #
    # This is the major V2 change.
    #
    # Region membership is NOT directly used for common F1.
    #
    # Every point is re-evaluated using the same 0.02 m
    # point-to-plane criterion used by the other methods.
    # ------------------------------------------------------

    final_distances = np.abs(
        points
        @ normal
        + d
    )


    final_mask = (
        final_distances
        < FINAL_DISTANCE_THRESHOLD
    )


    final_support = int(
        np.sum(
            final_mask
        )
    )


    if final_support < 3:

        return None, (
            "insufficient_final_support"
        )


    return {
        "normal": normal,

        "d": d,

        "native_region_mask": (
            native_region_mask
        ),

        "final_mask": (
            final_mask
        ),

        "native_region_size": int(
            len(
                largest_region
            )
        ),

        "num_detected_regions": int(
            len(
                regions
            )
        ),

        "final_support": (
            final_support
        )
    }, ""


# ==========================================================
# Confusion matrix
# ==========================================================

def calculate_confusion_matrix(
    true_labels,
    predicted_mask
):
    true_plane = (
        true_labels
        == 1
    )


    tp = int(
        np.sum(
            true_plane
            & predicted_mask
        )
    )


    fp = int(
        np.sum(
            (~true_plane)
            & predicted_mask
        )
    )


    fn = int(
        np.sum(
            true_plane
            & (~predicted_mask)
        )
    )


    tn = int(
        np.sum(
            (~true_plane)
            & (~predicted_mask)
        )
    )


    return (
        tp,
        fp,
        fn,
        tn
    )


# ==========================================================
# Precision / Recall / F1
# ==========================================================

def calculate_precision_recall_f1(
    tp,
    fp,
    fn
):
    if (
        tp + fp
    ) > 0:

        precision = (
            tp
            / (
                tp
                + fp
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
                tp
                + fn
            )
        )

    else:

        recall = 0.0


    if (
        precision + recall
    ) > 0:

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


    return (
        precision,
        recall,
        f1
    )


# ==========================================================
# Load benchmark sample
# ==========================================================

def load_sample(
    npz_file
):
    data = np.load(
        npz_file
    )


    if "points" in data:

        points = data[
            "points"
        ]

    elif "point_cloud" in data:

        points = data[
            "point_cloud"
        ]

    else:

        raise KeyError(
            f"{npz_file.name}: "
            "cannot find points array"
        )


    if "labels" in data:

        labels = data[
            "labels"
        ]

    elif "gt_labels" in data:

        labels = data[
            "gt_labels"
        ]

    else:

        raise KeyError(
            f"{npz_file.name}: "
            "cannot find labels array"
        )


    return (
        np.asarray(
            points,
            dtype=np.float64
        ),

        np.asarray(
            labels,
            dtype=np.int32
        )
    )


# ==========================================================
# Load metadata
# ==========================================================

def load_metadata(
    npz_file
):
    metadata_file = (
        npz_file.with_name(
            npz_file.stem
            + "_gt.json"
        )
    )


    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(
            f
        )


def metadata_value(
    metadata,
    possible_keys
):
    for key in possible_keys:

        if key in metadata:

            return metadata[
                key
            ]


    raise KeyError(
        f"Cannot find metadata key: "
        f"{possible_keys}"
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print(
        "=== Region Growing V2 Benchmark ==="
    )


    print(
        "\nConfiguration:"
    )


    print(
        "k neighbours:",
        K_NEIGHBORS
    )


    print(
        "Smoothness:",
        SMOOTHNESS_THRESHOLD_DEG,
        "degrees"
    )


    print(
        "Internal region distance:",
        REGION_DISTANCE_THRESHOLD,
        "m"
    )


    print(
        "Minimum region size:",
        MIN_REGION_SIZE
    )


    print(
        "COMMON final distance:",
        FINAL_DISTANCE_THRESHOLD,
        "m"
    )


    sample_files = sorted(
        benchmark_dir.glob(
            "N*_O*_R*.npz"
        )
    )


    if len(
        sample_files
    ) != 200:

        raise RuntimeError(
            "Expected 200 samples, "
            f"found {len(sample_files)}"
        )


    rows = []


    for sample_number, npz_file in enumerate(
        sample_files,
        start=1
    ):

        sample_id = (
            npz_file.stem
        )


        # --------------------------------------------------
        # I/O outside timing
        # --------------------------------------------------

        points, true_labels = load_sample(
            npz_file
        )


        metadata = load_metadata(
            npz_file
        )


        noise_sigma_m = float(
            metadata_value(
                metadata,
                [
                    "noise_sigma_m",
                    "noise_sigma",
                    "noise"
                ]
            )
        )


        outlier_ratio = float(
            metadata_value(
                metadata,
                [
                    "outlier_ratio",
                    "outlier_fraction"
                ]
            )
        )


        repeat_id = int(
            metadata_value(
                metadata,
                [
                    "repeat_id",
                    "repeat"
                ]
            )
        )


        num_points = len(
            points
        )


        num_true_inliers = int(
            np.sum(
                true_labels
                == 1
            )
        )


        num_true_outliers = int(
            np.sum(
                true_labels
                == 0
            )
        )


        # --------------------------------------------------
        # Algorithm runtime
        # --------------------------------------------------

        t0 = time.perf_counter()


        result, failure_reason = (
            detect_plane_region_growing(
                points
            )
        )


        t1 = time.perf_counter()


        runtime_ms = (
            (
                t1
                - t0
            )
            * 1000.0
        )


        # --------------------------------------------------
        # Failure
        # --------------------------------------------------

        if result is None:

            rows.append(
                {
                    "method": (
                        "Region Growing V2"
                    ),

                    "sample_id": (
                        sample_id
                    ),

                    "noise_sigma_m": (
                        noise_sigma_m
                    ),

                    "outlier_ratio": (
                        outlier_ratio
                    ),

                    "repeat_id": (
                        repeat_id
                    ),

                    "num_points": (
                        num_points
                    ),

                    "num_true_inliers": (
                        num_true_inliers
                    ),

                    "num_true_outliers": (
                        num_true_outliers
                    ),

                    "detected_inlier_count": 0,

                    "normal_error_deg": "",
                    "offset_error_m": "",

                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "tn": 0,

                    "precision": "",
                    "recall": "",
                    "f1": "",

                    "native_region_precision": "",
                    "native_region_recall": "",
                    "native_region_f1": "",

                    "runtime_ms": (
                        runtime_ms
                    ),

                    "num_detected_regions": 0,
                    "native_region_size": 0,

                    "failure_reason": (
                        failure_reason
                    ),

                    "success": 0
                }
            )


            print(
                f"[{sample_number:3d}/200] "
                f"{sample_id} FAILED "
                f"{failure_reason} "
                f"runtime={runtime_ms:.2f} ms"
            )


            continue


        # --------------------------------------------------
        # Common final mask
        # --------------------------------------------------

        final_mask = (
            result[
                "final_mask"
            ]
        )


        native_mask = (
            result[
                "native_region_mask"
            ]
        )


        estimated_normal = (
            result[
                "normal"
            ].copy()
        )


        estimated_d = (
            result[
                "d"
            ]
        )


        # --------------------------------------------------
        # GT only used for evaluation
        # --------------------------------------------------

        gt_normal = np.array(
            [
                0.0,
                0.0,
                1.0
            ],
            dtype=np.float64
        )


        gt_d = -2.0


        if np.dot(
            estimated_normal,
            gt_normal
        ) < 0:

            estimated_normal = (
                -estimated_normal
            )

            estimated_d = (
                -estimated_d
            )


        normal_error = normal_angle_deg(
            estimated_normal,
            gt_normal
        )


        offset_error = abs(
            estimated_d
            - gt_d
        )


        # --------------------------------------------------
        # COMMON benchmark metrics
        # --------------------------------------------------

        (
            tp,
            fp,
            fn,
            tn
        ) = calculate_confusion_matrix(
            true_labels,
            final_mask
        )


        (
            precision,
            recall,
            f1
        ) = calculate_precision_recall_f1(
            tp,
            fp,
            fn
        )


        # --------------------------------------------------
        # Native region diagnostics
        #
        # NOT used for formal cross-method comparison.
        # --------------------------------------------------

        (
            native_tp,
            native_fp,
            native_fn,
            native_tn
        ) = calculate_confusion_matrix(
            true_labels,
            native_mask
        )


        (
            native_precision,
            native_recall,
            native_f1
        ) = calculate_precision_recall_f1(
            native_tp,
            native_fp,
            native_fn
        )


        rows.append(
            {
                "method": (
                    "Region Growing V2"
                ),

                "sample_id": (
                    sample_id
                ),

                "noise_sigma_m": (
                    noise_sigma_m
                ),

                "outlier_ratio": (
                    outlier_ratio
                ),

                "repeat_id": (
                    repeat_id
                ),

                "num_points": (
                    num_points
                ),

                "num_true_inliers": (
                    num_true_inliers
                ),

                "num_true_outliers": (
                    num_true_outliers
                ),

                "detected_inlier_count": int(
                    np.sum(
                        final_mask
                    )
                ),

                "normal_error_deg": (
                    normal_error
                ),

                "offset_error_m": (
                    offset_error
                ),

                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,

                "precision": (
                    precision
                ),

                "recall": (
                    recall
                ),

                "f1": (
                    f1
                ),

                "native_region_precision": (
                    native_precision
                ),

                "native_region_recall": (
                    native_recall
                ),

                "native_region_f1": (
                    native_f1
                ),

                "runtime_ms": (
                    runtime_ms
                ),

                "num_detected_regions": (
                    result[
                        "num_detected_regions"
                    ]
                ),

                "native_region_size": (
                    result[
                        "native_region_size"
                    ]
                ),

                "failure_reason": "",

                "success": 1
            }
        )


        print(
            f"[{sample_number:3d}/200] "
            f"{sample_id} "
            f"Common F1={f1:.4f} "
            f"Native F1={native_f1:.4f} "
            f"runtime={runtime_ms:.2f} ms"
        )


    # ======================================================
    # Save CSV
    # ======================================================

    fieldnames = [
        "method",

        "sample_id",

        "noise_sigma_m",
        "outlier_ratio",
        "repeat_id",

        "num_points",
        "num_true_inliers",
        "num_true_outliers",

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

        "native_region_precision",
        "native_region_recall",
        "native_region_f1",

        "runtime_ms",

        "num_detected_regions",
        "native_region_size",

        "failure_reason",
        "success"
    ]


    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )


        writer.writeheader()


        writer.writerows(
            rows
        )


    print(
        "\nResults saved to:"
    )


    print(
        output_file
    )


if __name__ == "__main__":

    main()