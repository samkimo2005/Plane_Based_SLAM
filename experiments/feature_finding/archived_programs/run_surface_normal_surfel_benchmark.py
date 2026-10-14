import csv
import json
import time
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
    / "surface_normal_surfel_results.csv"
)


# ==========================================================
# Fixed algorithm parameters
# ==========================================================

# Number of neighbours used to estimate each local
# surface normal.

K_NEIGHBORS = 30


# Local curvature threshold.
#
# Lower curvature means the local neighbourhood is
# more plane-like.

MAX_CURVATURE = 0.05


# Local surfel normal must be within this angle of
# the dominant surface orientation.

NORMAL_THRESHOLD_DEG = 15.0


# Width used when grouping surfels by plane offset.

OFFSET_BIN_M = 0.04


# Candidate surfels must lie within this distance
# of the dominant offset group.

OFFSET_GROUP_THRESHOLD_M = 0.04


# Final common plane-support criterion.
#
# Same physical threshold used in the other benchmark
# implementations.

DISTANCE_THRESHOLD = 0.02


# Minimum number of surfels required before a plane
# is considered valid.

MIN_SURFEL_SUPPORT = 50


# ==========================================================
# SVD plane fitting
# ==========================================================

def fit_plane_svd(
    points
):
    """
    Fit a plane:

        n^T p + d = 0

    using all supplied points.
    """

    if len(
        points
    ) < 3:

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
# Generic plane sign convention
# ==========================================================

def canonicalise_plane(
    normal,
    d
):
    """
    Resolve plane sign ambiguity without assuming
    that the true plane is horizontal.

    The component with the largest absolute magnitude
    is forced to be positive.
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
# Normal-angle calculation
# ==========================================================

def normal_angle_deg(
    normal_a,
    normal_b
):
    """
    Sign-invariant angle between two plane normals.
    """

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
# Estimate local surfels
# ==========================================================

def estimate_local_surfels(
    points
):
    """
    Estimate one local surface element (surfel)
    for every point.

    Each surfel contains:

        position
        normal
        curvature

    Normal and curvature are estimated from the PCA
    eigenvalues/eigenvectors of the local kNN covariance.
    """

    num_points = len(
        points
    )


    if num_points <= K_NEIGHBORS:

        return None


    # ------------------------------------------------------
    # k-nearest neighbours
    # ------------------------------------------------------

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


    # First neighbour is the point itself.

    neighbour_indices = (
        neighbour_indices[
            :,
            1:
        ]
    )


    neighbours = points[
        neighbour_indices
    ]


    # ------------------------------------------------------
    # Local centroids
    # ------------------------------------------------------

    local_centroids = np.mean(
        neighbours,
        axis=1
    )


    centred = (
        neighbours
        - local_centroids[
            :,
            None,
            :
        ]
    )


    # ------------------------------------------------------
    # Local covariance matrices
    # ------------------------------------------------------

    covariance = np.einsum(
        "nki,nkj->nij",
        centred,
        centred
    )


    covariance /= float(
        K_NEIGHBORS
    )


    # ------------------------------------------------------
    # PCA
    #
    # np.linalg.eigh returns eigenvalues in ascending order.
    #
    # Smallest eigenvector = surface normal.
    # ------------------------------------------------------

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


    curvature = np.zeros(
        num_points,
        dtype=np.float64
    )


    valid_sum = (
        eigenvalue_sum
        > 1e-12
    )


    curvature[
        valid_sum
    ] = (
        eigenvalues[
            valid_sum,
            0
        ]
        / eigenvalue_sum[
            valid_sum
        ]
    )


    curvature[
        ~valid_sum
    ] = 1.0


    return (
        normals,
        curvature
    )


# ==========================================================
# Estimate dominant surface orientation
# ==========================================================

def estimate_dominant_normal(
    normals,
    curvature,
    planar_mask
):
    """
    Estimate the dominant plane orientation from
    planar surfels.

    The outer-product accumulation:

        n n^T

    is sign invariant, so local normal sign ambiguity
    does not affect the result.
    """

    selected_normals = normals[
        planar_mask
    ]


    selected_curvature = curvature[
        planar_mask
    ]


    if len(
        selected_normals
    ) < MIN_SURFEL_SUPPORT:

        return None


    # ------------------------------------------------------
    # Give lower-curvature surfels slightly greater weight.
    # ------------------------------------------------------

    weights = (
        1.0
        - (
            selected_curvature
            / MAX_CURVATURE
        )
    )


    weights = np.clip(
        weights,
        0.0,
        1.0
    )


    if np.sum(
        weights
    ) < 1e-12:

        weights = np.ones(
            len(
                selected_normals
            ),
            dtype=np.float64
        )


    # ------------------------------------------------------
    # Sign-invariant normal scatter matrix
    # ------------------------------------------------------

    normal_matrix = np.einsum(
        "n,ni,nj->ij",
        weights,
        selected_normals,
        selected_normals
    )


    eigenvalues, eigenvectors = np.linalg.eigh(
        normal_matrix
    )


    dominant_normal = (
        eigenvectors[
            :,
            -1
        ]
    )


    dominant_normal /= np.linalg.norm(
        dominant_normal
    )


    # Generic sign convention

    dominant_normal, _ = canonicalise_plane(
        dominant_normal,
        0.0
    )


    return dominant_normal


# ==========================================================
# Find dominant offset group
# ==========================================================

def find_dominant_offset(
    points,
    candidate_mask,
    dominant_normal
):
    """
    Group orientation-compatible surfels by plane offset.

    For a fixed normal:

        n^T p + d = 0

    therefore each point gives:

        d_i = -n^T p_i

    The densest offset bin is treated as the dominant
    coplanar surfel group.
    """

    candidate_points = points[
        candidate_mask
    ]


    if len(
        candidate_points
    ) < MIN_SURFEL_SUPPORT:

        return None


    offsets = -(
        candidate_points
        @ dominant_normal
    )


    minimum_offset = float(
        np.min(
            offsets
        )
    )


    maximum_offset = float(
        np.max(
            offsets
        )
    )


    # ------------------------------------------------------
    # Degenerate case: all offsets are effectively identical.
    # ------------------------------------------------------

    if (
        maximum_offset
        - minimum_offset
    ) < 1e-12:

        dominant_d = float(
            np.mean(
                offsets
            )
        )


        return (
            dominant_d,
            len(
                offsets
            )
        )


    # ------------------------------------------------------
    # Histogram grouping
    # ------------------------------------------------------

    bin_edges = np.arange(
        minimum_offset,
        maximum_offset
        + OFFSET_BIN_M,
        OFFSET_BIN_M
    )


    if len(
        bin_edges
    ) < 2:

        bin_edges = np.array(
            [
                minimum_offset,
                minimum_offset
                + OFFSET_BIN_M
            ],
            dtype=np.float64
        )


    histogram, edges = np.histogram(
        offsets,
        bins=bin_edges
    )


    strongest_bin = int(
        np.argmax(
            histogram
        )
    )


    lower_edge = edges[
        strongest_bin
    ]


    upper_edge = edges[
        strongest_bin
        + 1
    ]


    bin_mask = (
        (
            offsets
            >= lower_edge
        )
        &
        (
            offsets
            < upper_edge
        )
    )


    # Handle the final right-most histogram boundary.

    if (
        strongest_bin
        == len(
            histogram
        )
        - 1
    ):

        bin_mask = (
            (
                offsets
                >= lower_edge
            )
            &
            (
                offsets
                <= upper_edge
            )
        )


    dominant_offsets = offsets[
        bin_mask
    ]


    if len(
        dominant_offsets
    ) == 0:

        return None


    # Median is more resistant to occasional incorrect
    # surfels than the arithmetic mean.

    dominant_d = float(
        np.median(
            dominant_offsets
        )
    )


    return (
        dominant_d,
        len(
            dominant_offsets
        )
    )


# ==========================================================
# Surface-normal / surfel grouping detector
# ==========================================================

def detect_plane_surface_normal(
    points
):
    """
    Surface-normal / surfel grouping baseline.

    Pipeline:

        local kNN neighbourhoods
                ↓
        PCA surface normals + curvature
                ↓
        planar surfel filtering
                ↓
        dominant normal estimation
                ↓
        normal-consistent surfel selection
                ↓
        dominant offset grouping
                ↓
        SVD refinement
                ↓
        common 0.02 m final plane mask
    """

    surfel_result = estimate_local_surfels(
        points
    )


    if surfel_result is None:

        return None, (
            "insufficient_points"
        )


    normals, curvature = surfel_result


    # ------------------------------------------------------
    # Step 1: planar surfels
    # ------------------------------------------------------

    finite_mask = (
        np.all(
            np.isfinite(
                normals
            ),
            axis=1
        )
        &
        np.isfinite(
            curvature
        )
    )


    planar_mask = (
        finite_mask
        &
        (
            curvature
            <= MAX_CURVATURE
        )
    )


    num_planar_surfels = int(
        np.sum(
            planar_mask
        )
    )


    if (
        num_planar_surfels
        < MIN_SURFEL_SUPPORT
    ):

        return None, (
            "insufficient_planar_surfels"
        )


    # ------------------------------------------------------
    # Step 2: dominant orientation
    # ------------------------------------------------------

    dominant_normal = (
        estimate_dominant_normal(
            normals,
            curvature,
            planar_mask
        )
    )


    if dominant_normal is None:

        return None, (
            "dominant_normal_failed"
        )


    # ------------------------------------------------------
    # Step 3: orientation-consistent surfels
    # ------------------------------------------------------

    dot_values = np.abs(
        normals
        @ dominant_normal
    )


    dot_values = np.clip(
        dot_values,
        0.0,
        1.0
    )


    angles_deg = np.degrees(
        np.arccos(
            dot_values
        )
    )


    orientation_mask = (
        planar_mask
        &
        (
            angles_deg
            <= NORMAL_THRESHOLD_DEG
        )
    )


    num_orientation_support = int(
        np.sum(
            orientation_mask
        )
    )


    if (
        num_orientation_support
        < MIN_SURFEL_SUPPORT
    ):

        return None, (
            "insufficient_orientation_support"
        )


    # ------------------------------------------------------
    # Step 4: offset grouping
    # ------------------------------------------------------

    offset_result = find_dominant_offset(
        points,
        orientation_mask,
        dominant_normal
    )


    if offset_result is None:

        return None, (
            "offset_grouping_failed"
        )


    dominant_d, peak_bin_count = (
        offset_result
    )


    # ------------------------------------------------------
    # Step 5: initial coplanar support
    # ------------------------------------------------------

    point_offsets = -(
        points
        @ dominant_normal
    )


    initial_mask = (
        orientation_mask
        &
        (
            np.abs(
                point_offsets
                - dominant_d
            )
            <= OFFSET_GROUP_THRESHOLD_M
        )
    )


    initial_support = int(
        np.sum(
            initial_mask
        )
    )


    if (
        initial_support
        < MIN_SURFEL_SUPPORT
    ):

        return None, (
            "insufficient_offset_support"
        )


    # ------------------------------------------------------
    # Step 6: SVD refinement
    # ------------------------------------------------------

    initial_points = points[
        initial_mask
    ]


    refined = fit_plane_svd(
        initial_points
    )


    if refined is None:

        return None, (
            "svd_refinement_failed"
        )


    normal, d = refined


    normal, d = canonicalise_plane(
        normal,
        d
    )


    # ------------------------------------------------------
    # Step 7: common final plane-support threshold
    #
    # This is the same physical 0.02 m criterion used
    # for the other benchmark detectors.
    # ------------------------------------------------------

    final_distances = np.abs(
        points
        @ normal
        + d
    )


    final_mask = (
        final_distances
        < DISTANCE_THRESHOLD
    )


    final_support = int(
        np.sum(
            final_mask
        )
    )


    if (
        final_support
        < MIN_SURFEL_SUPPORT
    ):

        return None, (
            "insufficient_final_support"
        )


    return {
        "normal": (
            normal
        ),

        "d": (
            d
        ),

        "inlier_mask": (
            final_mask
        ),

        "support": (
            final_support
        ),

        "num_planar_surfels": (
            num_planar_surfels
        ),

        "num_orientation_support": (
            num_orientation_support
        ),

        "offset_peak_count": int(
            peak_bin_count
        ),

        "initial_support": (
            initial_support
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


    if not metadata_file.exists():

        raise FileNotFoundError(
            f"Cannot find metadata: "
            f"{metadata_file}"
        )


    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(
            f
        )


# ==========================================================
# Metadata helper
# ==========================================================

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
        "Cannot find metadata key. "
        f"Tried: {possible_keys}"
    )


# ==========================================================
# Main benchmark
# ==========================================================

def main():

    print(
        "=== Surface-Normal / Surfel Benchmark ==="
    )


    print(
        "\nFixed configuration:"
    )

    print(
        "k neighbours:",
        K_NEIGHBORS
    )

    print(
        "Maximum curvature:",
        MAX_CURVATURE
    )

    print(
        "Normal threshold:",
        NORMAL_THRESHOLD_DEG,
        "degrees"
    )

    print(
        "Offset bin width:",
        OFFSET_BIN_M,
        "m"
    )

    print(
        "Offset group threshold:",
        OFFSET_GROUP_THRESHOLD_M,
        "m"
    )

    print(
        "Final distance threshold:",
        DISTANCE_THRESHOLD,
        "m"
    )

    print(
        "Minimum surfel support:",
        MIN_SURFEL_SUPPORT
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
            "Expected 200 benchmark samples, "
            f"but found {len(sample_files)}"
        )


    print(
        "\nSamples found:",
        len(
            sample_files
        )
    )


    result_rows = []


    # ======================================================
    # Run all samples
    # ======================================================

    for sample_number, npz_file in enumerate(
        sample_files,
        start=1
    ):

        sample_id = (
            npz_file.stem
        )


        # --------------------------------------------------
        # File I/O excluded from runtime
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
        #
        # Includes:
        #   kNN search
        #   covariance calculation
        #   local PCA
        #   surfel filtering
        #   normal grouping
        #   offset grouping
        #   SVD refinement
        #
        # Excludes:
        #   file I/O
        # --------------------------------------------------

        start_time = (
            time.perf_counter()
        )


        result, failure_reason = (
            detect_plane_surface_normal(
                points
            )
        )


        end_time = (
            time.perf_counter()
        )


        runtime_ms = (
            (
                end_time
                - start_time
            )
            * 1000.0
        )


        # --------------------------------------------------
        # Failure case
        # --------------------------------------------------

        if result is None:

            result_rows.append(
                {
                    "method": (
                        "Surface-Normal / Surfel"
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

                    "runtime_ms": (
                        runtime_ms
                    ),

                    "num_planar_surfels": 0,

                    "num_orientation_support": 0,

                    "offset_peak_count": 0,

                    "initial_support": 0,

                    "failure_reason": (
                        failure_reason
                    ),

                    "success": 0
                }
            )


            print(
                f"[{sample_number:3d}/200] "
                f"{sample_id} "
                f"FAILED "
                f"({failure_reason}) "
                f"runtime={runtime_ms:.2f} ms"
            )


            continue


        # --------------------------------------------------
        # Estimated plane
        # --------------------------------------------------

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


        predicted_mask = (
            result[
                "inlier_mask"
            ]
        )


        # --------------------------------------------------
        # GT plane:
        #
        #       z = 2
        #
        #       n = [0, 0, 1]
        #       d = -2
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


        # Align sign with GT before offset comparison.

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


        # --------------------------------------------------
        # Plane parameter errors
        # --------------------------------------------------

        normal_error = normal_angle_deg(
            estimated_normal,
            gt_normal
        )


        offset_error = abs(
            estimated_d
            - gt_d
        )


        # --------------------------------------------------
        # Segmentation metrics
        # --------------------------------------------------

        (
            tp,
            fp,
            fn,
            tn
        ) = calculate_confusion_matrix(
            true_labels,
            predicted_mask
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


        detected_inlier_count = int(
            np.sum(
                predicted_mask
            )
        )


        # --------------------------------------------------
        # Save result
        # --------------------------------------------------

        result_rows.append(
            {
                "method": (
                    "Surface-Normal / Surfel"
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

                "detected_inlier_count": (
                    detected_inlier_count
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

                "runtime_ms": (
                    runtime_ms
                ),

                "num_planar_surfels": (
                    result[
                        "num_planar_surfels"
                    ]
                ),

                "num_orientation_support": (
                    result[
                        "num_orientation_support"
                    ]
                ),

                "offset_peak_count": (
                    result[
                        "offset_peak_count"
                    ]
                ),

                "initial_support": (
                    result[
                        "initial_support"
                    ]
                ),

                "failure_reason": "",

                "success": 1
            }
        )


        print(
            f"[{sample_number:3d}/200] "
            f"{sample_id} "
            f"normal={normal_error:.4f} deg "
            f"offset={offset_error:.6f} m "
            f"F1={f1:.4f} "
            f"surfels={result['num_planar_surfels']} "
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

        "runtime_ms",

        "num_planar_surfels",
        "num_orientation_support",
        "offset_peak_count",
        "initial_support",

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
            result_rows
        )


    # ======================================================
    # Quick summary
    # ======================================================

    successful_rows = [
        row
        for row in result_rows
        if row[
            "success"
        ] == 1
    ]


    failed_rows = [
        row
        for row in result_rows
        if row[
            "success"
        ] == 0
    ]


    runtimes = np.array(
        [
            row[
                "runtime_ms"
            ]
            for row in result_rows
        ],
        dtype=np.float64
    )


    print(
        "\n=== Surface-Normal / Surfel Benchmark Complete ==="
    )


    print(
        "Successful:",
        len(
            successful_rows
        ),
        "/",
        len(
            result_rows
        )
    )


    print(
        "Failed:",
        len(
            failed_rows
        )
    )


    print(
        "Runtime mean:",
        f"{np.mean(runtimes):.3f}",
        "ms"
    )


    print(
        "Runtime median:",
        f"{np.median(runtimes):.3f}",
        "ms"
    )


    print(
        "Runtime std:",
        f"{np.std(runtimes, ddof=1):.3f}",
        "ms"
    )


    print(
        "Runtime minimum:",
        f"{np.min(runtimes):.3f}",
        "ms"
    )


    print(
        "Runtime maximum:",
        f"{np.max(runtimes):.3f}",
        "ms"
    )


    print(
        "Runtime 95th percentile:",
        f"{np.percentile(runtimes, 95):.3f}",
        "ms"
    )


    if len(
        successful_rows
    ) > 0:

        normal_errors = np.array(
            [
                row[
                    "normal_error_deg"
                ]
                for row in successful_rows
            ],
            dtype=np.float64
        )


        offset_errors = np.array(
            [
                row[
                    "offset_error_m"
                ]
                for row in successful_rows
            ],
            dtype=np.float64
        )


        f1_values = np.array(
            [
                row[
                    "f1"
                ]
                for row in successful_rows
            ],
            dtype=np.float64
        )


        print(
            "Mean normal error:",
            f"{np.mean(normal_errors):.6f}",
            "degrees"
        )


        print(
            "Mean offset error:",
            f"{np.mean(offset_errors):.6f}",
            "m"
        )


        print(
            "Mean F1:",
            f"{np.mean(f1_values):.6f}"
        )


    if len(
        failed_rows
    ) > 0:

        failure_counts = {}


        for row in failed_rows:

            reason = row[
                "failure_reason"
            ]


            failure_counts[
                reason
            ] = (
                failure_counts.get(
                    reason,
                    0
                )
                + 1
            )


        print(
            "\nFailure reasons:"
        )


        for reason, count in failure_counts.items():

            print(
                reason,
                ":",
                count
            )


    print(
        "\nResults saved to:"
    )


    print(
        output_file
    )


if __name__ == "__main__":

    main()