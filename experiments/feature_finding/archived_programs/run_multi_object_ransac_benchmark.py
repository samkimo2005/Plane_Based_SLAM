
import csv
import json
import time
from pathlib import Path

import numpy as np


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
    / "multi_object_ransac_results.csv"
)


# ==========================================================
# Fixed parameters
# ==========================================================

DISTANCE_THRESHOLD = 0.02

NUM_HYPOTHESES = 1000

MIN_HYPOTHESIS_SUPPORT = 50


# Hypothesis clustering thresholds

MODEL_NORMAL_THRESHOLD_DEG = 5.0

MODEL_OFFSET_THRESHOLD_M = 0.03


# Fixed random seed

RANSAC_SEED = 42


# ==========================================================
# Fit plane from three points
# ==========================================================

def fit_plane_from_three_points(
    p1,
    p2,
    p3
):
    """
    Fit a plane from three non-collinear points.

    Plane equation:

        n^T p + d = 0

    Returns:
        normal, d

    Returns None if the points are nearly collinear.
    """

    v1 = (
        p2
        - p1
    )

    v2 = (
        p3
        - p1
    )


    normal = np.cross(
        v1,
        v2
    )


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
        p1
    )


    return (
        normal,
        d
    )


# ==========================================================
# SVD refinement
# ==========================================================

def fit_plane_svd(
    points
):
    """
    Fit a plane to all supplied points using SVD.
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
# Plane orientation
# ==========================================================

def orient_plane(
    normal,
    d
):
    """
    Remove the sign ambiguity between:

        (n, d)
        (-n, -d)

    For this synthetic benchmark the GT normal points
    approximately towards +z.
    """

    if normal[2] < 0:

        normal = (
            -normal
        )

        d = (
            -d
        )


    return (
        normal,
        d
    )


# ==========================================================
# Plane normal angle
# ==========================================================

def normal_angle_deg(
    normal_a,
    normal_b
):
    """
    Smallest angle between two plane normals.
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


    angle_rad = np.arccos(
        dot_value
    )


    return np.degrees(
        angle_rad
    )


# ==========================================================
# Align candidate sign
# ==========================================================

def align_plane_sign(
    reference_normal,
    candidate_normal,
    candidate_d
):
    """
    Align candidate normal direction with reference normal.
    """

    if np.dot(
        reference_normal,
        candidate_normal
    ) < 0:

        candidate_normal = (
            -candidate_normal
        )

        candidate_d = (
            -candidate_d
        )


    return (
        candidate_normal,
        candidate_d
    )


# ==========================================================
# Generate RANSAC hypotheses
# ==========================================================

def generate_plane_hypotheses(
    points,
    rng
):
    """
    Generate many plane hypotheses.

    Unlike ordinary RANSAC, multiple sufficiently
    supported hypotheses are retained instead of only
    keeping the current best one.
    """

    num_points = len(
        points
    )


    hypotheses = []


    for _ in range(
        NUM_HYPOTHESES
    ):

        sample_indices = rng.choice(
            num_points,
            size=3,
            replace=False
        )


        p1 = points[
            sample_indices[0]
        ]

        p2 = points[
            sample_indices[1]
        ]

        p3 = points[
            sample_indices[2]
        ]


        result = (
            fit_plane_from_three_points(
                p1,
                p2,
                p3
            )
        )


        if result is None:

            continue


        normal, d = result


        normal, d = orient_plane(
            normal,
            d
        )


        distances = np.abs(
            points @ normal
            + d
        )


        inlier_mask = (
            distances
            < DISTANCE_THRESHOLD
        )


        support = int(
            np.sum(
                inlier_mask
            )
        )


        if (
            support
            < MIN_HYPOTHESIS_SUPPORT
        ):

            continue


        hypotheses.append(
            {
                "normal": (
                    normal
                ),

                "d": (
                    d
                ),

                "inlier_mask": (
                    inlier_mask
                ),

                "support": (
                    support
                )
            }
        )


    return hypotheses


# ==========================================================
# Cluster similar hypotheses
# ==========================================================

def cluster_plane_hypotheses(
    hypotheses
):
    """
    Group hypotheses that likely represent the same plane.

    Two hypotheses are treated as compatible when:

        normal angle < MODEL_NORMAL_THRESHOLD_DEG

    and

        offset difference < MODEL_OFFSET_THRESHOLD_M
    """

    if len(
        hypotheses
    ) == 0:

        return []


    hypotheses = sorted(
        hypotheses,
        key=lambda h: (
            h[
                "support"
            ]
        ),
        reverse=True
    )


    clusters = []


    for hypothesis in hypotheses:

        assigned = False


        for cluster in clusters:

            reference = (
                cluster[
                    "representative"
                ]
            )


            reference_normal = (
                reference[
                    "normal"
                ]
            )


            reference_d = (
                reference[
                    "d"
                ]
            )


            candidate_normal = (
                hypothesis[
                    "normal"
                ].copy()
            )


            candidate_d = (
                hypothesis[
                    "d"
                ]
            )


            (
                candidate_normal,
                candidate_d
            ) = align_plane_sign(
                reference_normal,
                candidate_normal,
                candidate_d
            )


            angle_difference = (
                normal_angle_deg(
                    reference_normal,
                    candidate_normal
                )
            )


            offset_difference = abs(
                reference_d
                - candidate_d
            )


            if (
                angle_difference
                < MODEL_NORMAL_THRESHOLD_DEG
                and
                offset_difference
                < MODEL_OFFSET_THRESHOLD_M
            ):

                cluster[
                    "hypotheses"
                ].append(
                    hypothesis
                )


                assigned = True

                break


        if not assigned:

            clusters.append(
                {
                    "representative": (
                        hypothesis
                    ),

                    "hypotheses": [
                        hypothesis
                    ]
                }
            )


    return clusters


# ==========================================================
# Refine one hypothesis cluster
# ==========================================================

def evaluate_cluster(
    points,
    cluster
):
    """
    Convert one cluster into one final plane model.

    IMPORTANT:
    For fair comparison with ordinary RANSAC,
    we DO NOT union all masks in the cluster.

    Instead:

        1. Select the strongest hypothesis.
        2. Use only its inliers for initial SVD refinement.
        3. Recalculate the final mask on all points using
           the same 0.02 m distance threshold.

    This avoids artificially increasing the support by
    unioning many slightly different hypotheses.
    """

    best_hypothesis = max(
        cluster[
            "hypotheses"
        ],
        key=lambda h: (
            h[
                "support"
            ]
        )
    )


    initial_mask = (
        best_hypothesis[
            "inlier_mask"
        ]
    )


    initial_points = points[
        initial_mask
    ]


    if len(
        initial_points
    ) < 3:

        return None


    refined = fit_plane_svd(
        initial_points
    )


    if refined is None:

        return None


    normal, d = refined


    normal, d = orient_plane(
        normal,
        d
    )


    # ------------------------------------------------------
    # Final common inlier test
    # ------------------------------------------------------

    distances = np.abs(
        points @ normal
        + d
    )


    final_mask = (
        distances
        < DISTANCE_THRESHOLD
    )


    final_support = int(
        np.sum(
            final_mask
        )
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

        "num_hypotheses": (
            len(
                cluster[
                    "hypotheses"
                ]
            )
        )
    }


# ==========================================================
# Multi-Object / Multi-Model RANSAC
# ==========================================================

def multi_object_ransac(
    points,
    seed
):
    """
    Multi-model RANSAC-inspired detector.

    Pipeline:

        RANSAC hypothesis generation
                ↓
        retain supported hypotheses
                ↓
        cluster hypotheses in plane-parameter space
                ↓
        refine one model per cluster
                ↓
        return all detected models
    """

    rng = np.random.default_rng(
        seed
    )


    hypotheses = (
        generate_plane_hypotheses(
            points,
            rng
        )
    )


    if len(
        hypotheses
    ) == 0:

        return [], (
            "no_valid_hypotheses"
        )


    clusters = (
        cluster_plane_hypotheses(
            hypotheses
        )
    )


    models = []


    for cluster in clusters:

        model = evaluate_cluster(
            points,
            cluster
        )


        if model is None:

            continue


        models.append(
            model
        )


    if len(
        models
    ) == 0:

        return [], (
            "no_valid_models"
        )


    # Strongest final model first

    models.sort(
        key=lambda model: (
            model[
                "support"
            ]
        ),
        reverse=True
    )


    return (
        models,
        ""
    )


# ==========================================================
# Confusion matrix
# ==========================================================

def calculate_confusion_matrix(
    true_labels,
    predicted_mask
):
    """
    GT convention:

        1 = true plane
        0 = outlier
    """

    true_plane = (
        true_labels
        == 1
    )


    predicted_plane = (
        predicted_mask
    )


    tp = int(
        np.sum(
            true_plane
            &
            predicted_plane
        )
    )


    fp = int(
        np.sum(
            (~true_plane)
            &
            predicted_plane
        )
    )


    fn = int(
        np.sum(
            true_plane
            &
            (~predicted_plane)
        )
    )


    tn = int(
        np.sum(
            (~true_plane)
            &
            (~predicted_plane)
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
        precision
        + recall
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
            f"Cannot find metadata file: "
            f"{metadata_file}"
        )


    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(
            f
        )


    return metadata


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
        "=== Multi-Object RANSAC Benchmark ==="
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
        "Samples found:",
        len(
            sample_files
        )
    )


    result_rows = []


    # ======================================================
    # Process all samples
    # ======================================================

    for sample_number, npz_file in enumerate(
        sample_files,
        start=1
    ):

        sample_id = (
            npz_file.stem
        )


        # --------------------------------------------------
        # File I/O is outside runtime measurement
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
        # Run algorithm
        # --------------------------------------------------

        start_time = (
            time.perf_counter()
        )


        models, failure_reason = (
            multi_object_ransac(
                points,
                seed=RANSAC_SEED
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
        # Failure
        # --------------------------------------------------

        if len(
            models
        ) == 0:

            result_rows.append(
                {
                    "method": (
                        "Multi-Object RANSAC"
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

                    "num_detected_models": 0,

                    "winning_cluster_hypotheses": 0,

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
                f"({failure_reason})"
            )


            continue


        # --------------------------------------------------
        # Current benchmark has one GT plane.
        #
        # Use strongest detected model for the common
        # evaluation.
        # --------------------------------------------------

        best_model = (
            models[0]
        )


        estimated_normal = (
            best_model[
                "normal"
            ].copy()
        )


        estimated_d = (
            best_model[
                "d"
            ]
        )


        predicted_mask = (
            best_model[
                "inlier_mask"
            ]
        )


        # --------------------------------------------------
        # Ground-truth plane
        #
        # z = 2
        #
        # n^T p + d = 0
        #
        # n = [0, 0, 1]
        # d = -2
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


        # Align sign before offset comparison

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
        # Plane errors
        # --------------------------------------------------

        normal_error = (
            normal_angle_deg(
                estimated_normal,
                gt_normal
            )
        )


        offset_error = abs(
            estimated_d
            - gt_d
        )


        # --------------------------------------------------
        # Classification metrics
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
        # Store result
        # --------------------------------------------------

        result_rows.append(
            {
                "method": (
                    "Multi-Object RANSAC"
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

                "num_detected_models": (
                    len(
                        models
                    )
                ),

                "winning_cluster_hypotheses": (
                    best_model[
                        "num_hypotheses"
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
            f"models={len(models)} "
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

        "num_detected_models",

        "winning_cluster_hypotheses",

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
    # Quick overall summary
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


    print(
        "\n=== Benchmark Complete ==="
    )


    print(
        "Total samples:",
        len(
            result_rows
        )
    )


    print(
        "Successful:",
        len(
            successful_rows
        )
    )


    print(
        "Failed:",
        len(
            failed_rows
        )
    )


    if len(
        successful_rows
    ) > 0:

        mean_f1 = np.mean(
            [
                row[
                    "f1"
                ]
                for row in successful_rows
            ]
        )


        mean_normal_error = np.mean(
            [
                row[
                    "normal_error_deg"
                ]
                for row in successful_rows
            ]
        )


        mean_offset_error = np.mean(
            [
                row[
                    "offset_error_m"
                ]
                for row in successful_rows
            ]
        )


        mean_models = np.mean(
            [
                row[
                    "num_detected_models"
                ]
                for row in successful_rows
            ]
        )


        print(
            "Mean F1:",
            f"{mean_f1:.6f}"
        )


        print(
            "Mean normal error:",
            f"{mean_normal_error:.6f}",
            "degrees"
        )


        print(
            "Mean offset error:",
            f"{mean_offset_error:.6f}",
            "m"
        )


        print(
            "Mean detected models:",
            f"{mean_models:.3f}"
        )


    mean_runtime = np.mean(
        [
            row[
                "runtime_ms"
            ]
            for row in result_rows
        ]
    )


    print(
        "Mean runtime:",
        f"{mean_runtime:.3f}",
        "ms"
    )


    print(
        "\nResults saved to:"
    )


    print(
        output_file
    )


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":

    main()