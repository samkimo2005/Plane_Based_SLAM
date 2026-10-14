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
    / "multi_object_ransac_results_v2.csv"
)


# ==========================================================
# Fixed benchmark parameters
# ==========================================================

DISTANCE_THRESHOLD = 0.02

NUM_HYPOTHESES = 1000

MIN_HYPOTHESIS_SUPPORT = 50


# Hypothesis clustering thresholds

MODEL_NORMAL_THRESHOLD_DEG = 5.0

MODEL_OFFSET_THRESHOLD_M = 0.03


# A model cluster must be supported by at least this
# number of similar RANSAC hypotheses before expensive
# SVD refinement is performed.

MIN_CLUSTER_HYPOTHESES = 5


# Number of hypotheses evaluated together.
# This controls memory use only, not the algorithmic
# hypothesis budget.

HYPOTHESIS_BATCH_SIZE = 100


RANSAC_SEED = 42


# ==========================================================
# SVD plane fitting
# ==========================================================

def fit_plane_svd(
    points
):
    """
    Fit:

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
# Plane orientation
# ==========================================================

def orient_plane(
    normal,
    d
):
    """
    Remove sign ambiguity.

    For the present benchmark the ground-truth normal
    points approximately towards +z.
    """

    if normal[2] < 0:

        normal = -normal
        d = -d


    return (
        normal,
        d
    )


# ==========================================================
# Normal angle
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
# Generate 3-point samples
# ==========================================================

def generate_random_triplets(
    num_points,
    rng
):
    """
    Generate NUM_HYPOTHESES independent 3-point samples.

    Sampling remains identical in principle to ordinary
    RANSAC: three different points per hypothesis.
    """

    samples = np.empty(
        (
            NUM_HYPOTHESES,
            3
        ),
        dtype=np.int64
    )


    for i in range(
        NUM_HYPOTHESES
    ):

        samples[i] = rng.choice(
            num_points,
            size=3,
            replace=False
        )


    return samples


# ==========================================================
# Generate plane hypotheses
# ==========================================================

def generate_plane_hypotheses(
    points,
    rng
):
    """
    V2 optimisation:

    1. Generate all three-point hypotheses.
    2. Fit their plane parameters using vectorised NumPy.
    3. Compute support in batches.
    4. Store only:
           normal
           d
           support

    No 10,000-element Boolean mask is stored for every
    hypothesis.
    """

    num_points = len(
        points
    )


    samples = generate_random_triplets(
        num_points,
        rng
    )


    p1 = points[
        samples[:, 0]
    ]

    p2 = points[
        samples[:, 1]
    ]

    p3 = points[
        samples[:, 2]
    ]


    v1 = (
        p2
        - p1
    )

    v2 = (
        p3
        - p1
    )


    normals = np.cross(
        v1,
        v2
    )


    normal_norms = np.linalg.norm(
        normals,
        axis=1
    )


    valid_mask = (
        normal_norms
        > 1e-10
    )


    normals = normals[
        valid_mask
    ]

    p1 = p1[
        valid_mask
    ]

    normal_norms = normal_norms[
        valid_mask
    ]


    if len(
        normals
    ) == 0:

        return (
            np.empty(
                (0, 3),
                dtype=np.float64
            ),
            np.empty(
                0,
                dtype=np.float64
            ),
            np.empty(
                0,
                dtype=np.int32
            )
        )


    normals = (
        normals
        / normal_norms[:, None]
    )


    ds = -np.sum(
        normals
        * p1,
        axis=1
    )


    # ------------------------------------------------------
    # Consistent orientation
    # ------------------------------------------------------

    flip_mask = (
        normals[:, 2]
        < 0
    )


    normals[
        flip_mask
    ] *= -1.0


    ds[
        flip_mask
    ] *= -1.0


    # ------------------------------------------------------
    # Compute support counts in batches
    #
    # For 10,000 points and batch size 100:
    #
    #     10,000 x 100
    #
    # rather than allocating one 10,000 x 1000 matrix.
    # ------------------------------------------------------

    num_valid = len(
        normals
    )


    supports = np.zeros(
        num_valid,
        dtype=np.int32
    )


    for start in range(
        0,
        num_valid,
        HYPOTHESIS_BATCH_SIZE
    ):

        end = min(
            start
            + HYPOTHESIS_BATCH_SIZE,
            num_valid
        )


        normal_batch = normals[
            start:end
        ]


        d_batch = ds[
            start:end
        ]


        distances = np.abs(
            points
            @ normal_batch.T
            + d_batch[None, :]
        )


        supports[
            start:end
        ] = np.count_nonzero(
            distances
            < DISTANCE_THRESHOLD,
            axis=0
        )


    # ------------------------------------------------------
    # Retain supported hypotheses only
    # ------------------------------------------------------

    supported_mask = (
        supports
        >= MIN_HYPOTHESIS_SUPPORT
    )


    return (
        normals[
            supported_mask
        ],

        ds[
            supported_mask
        ],

        supports[
            supported_mask
        ]
    )


# ==========================================================
# Cluster plane hypotheses
# ==========================================================

def cluster_plane_hypotheses(
    normals,
    ds,
    supports
):
    """
    Greedy clustering, but compatibility checks against
    existing cluster representatives are vectorised.

    The hypotheses are processed from strongest to weakest.

    The first hypothesis in a cluster remains its
    representative, matching the V1 logic.
    """

    num_hypotheses = len(
        normals
    )


    if num_hypotheses == 0:

        return {
            "normals": np.empty(
                (0, 3),
                dtype=np.float64
            ),

            "ds": np.empty(
                0,
                dtype=np.float64
            ),

            "representative_supports": np.empty(
                0,
                dtype=np.int32
            ),

            "vote_counts": np.empty(
                0,
                dtype=np.int32
            )
        }


    order = np.argsort(
        supports
    )[::-1]


    normals = normals[
        order
    ]

    ds = ds[
        order
    ]

    supports = supports[
        order
    ]


    # At most one cluster per hypothesis.

    cluster_normals = np.empty(
        (
            num_hypotheses,
            3
        ),
        dtype=np.float64
    )


    cluster_ds = np.empty(
        num_hypotheses,
        dtype=np.float64
    )


    cluster_supports = np.empty(
        num_hypotheses,
        dtype=np.int32
    )


    cluster_votes = np.zeros(
        num_hypotheses,
        dtype=np.int32
    )


    num_clusters = 0


    for i in range(
        num_hypotheses
    ):

        candidate_normal = normals[
            i
        ]

        candidate_d = ds[
            i
        ]


        if num_clusters == 0:

            cluster_normals[0] = (
                candidate_normal
            )

            cluster_ds[0] = (
                candidate_d
            )

            cluster_supports[0] = (
                supports[i]
            )

            cluster_votes[0] = 1

            num_clusters = 1

            continue


        representatives = cluster_normals[
            :num_clusters
        ]


        representative_ds = cluster_ds[
            :num_clusters
        ]


        # --------------------------------------------------
        # Vectorised normal comparison
        # --------------------------------------------------

        raw_dots = (
            representatives
            @ candidate_normal
        )


        abs_dots = np.clip(
            np.abs(
                raw_dots
            ),
            0.0,
            1.0
        )


        angles = np.degrees(
            np.arccos(
                abs_dots
            )
        )


        # --------------------------------------------------
        # Align candidate d separately relative to
        # every cluster representative.
        # --------------------------------------------------

        signs = np.where(
            raw_dots < 0,
            -1.0,
            1.0
        )


        aligned_candidate_ds = (
            candidate_d
            * signs
        )


        offset_differences = np.abs(
            representative_ds
            - aligned_candidate_ds
        )


        compatible = (
            (
                angles
                < MODEL_NORMAL_THRESHOLD_DEG
            )
            &
            (
                offset_differences
                < MODEL_OFFSET_THRESHOLD_M
            )
        )


        compatible_indices = np.flatnonzero(
            compatible
        )


        if len(
            compatible_indices
        ) > 0:

            # Same behaviour as V1:
            # assign to first compatible cluster.

            cluster_index = int(
                compatible_indices[0]
            )


            cluster_votes[
                cluster_index
            ] += 1


        else:

            cluster_normals[
                num_clusters
            ] = candidate_normal


            cluster_ds[
                num_clusters
            ] = candidate_d


            cluster_supports[
                num_clusters
            ] = supports[i]


            cluster_votes[
                num_clusters
            ] = 1


            num_clusters += 1


    return {
        "normals": (
            cluster_normals[
                :num_clusters
            ]
        ),

        "ds": (
            cluster_ds[
                :num_clusters
            ]
        ),

        "representative_supports": (
            cluster_supports[
                :num_clusters
            ]
        ),

        "vote_counts": (
            cluster_votes[
                :num_clusters
            ]
        )
    }


# ==========================================================
# Refine accepted clusters
# ==========================================================

def refine_plane_clusters(
    points,
    clusters
):
    """
    Only repeatedly supported model clusters are refined.

    A cluster supported by fewer than
    MIN_CLUSTER_HYPOTHESES is treated as an unstable
    one-off candidate and is not promoted to a detected
    object/model.

    Refinement remains:

        representative hypothesis
              ↓
        common 0.02 m inlier mask
              ↓
        SVD refinement
              ↓
        common 0.02 m final mask
    """

    cluster_normals = clusters[
        "normals"
    ]


    cluster_ds = clusters[
        "ds"
    ]


    cluster_votes = clusters[
        "vote_counts"
    ]


    cluster_supports = clusters[
        "representative_supports"
    ]


    accepted_indices = np.flatnonzero(
        cluster_votes
        >= MIN_CLUSTER_HYPOTHESES
    )


    models = []


    for cluster_index in accepted_indices:

        initial_normal = (
            cluster_normals[
                cluster_index
            ]
        )


        initial_d = (
            cluster_ds[
                cluster_index
            ]
        )


        # --------------------------------------------------
        # Recreate mask only for accepted models
        # --------------------------------------------------

        initial_distances = np.abs(
            points
            @ initial_normal
            + initial_d
        )


        initial_mask = (
            initial_distances
            < DISTANCE_THRESHOLD
        )


        initial_points = points[
            initial_mask
        ]


        if len(
            initial_points
        ) < 3:

            continue


        refined = fit_plane_svd(
            initial_points
        )


        if refined is None:

            continue


        normal, d = refined


        normal, d = orient_plane(
            normal,
            d
        )


        # --------------------------------------------------
        # Final common inlier criterion
        # --------------------------------------------------

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


        models.append(
            {
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

                "cluster_votes": int(
                    cluster_votes[
                        cluster_index
                    ]
                ),

                "representative_support": int(
                    cluster_supports[
                        cluster_index
                    ]
                )
            }
        )


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
        len(
            accepted_indices
        )
    )


# ==========================================================
# Multi-Object RANSAC V2
# ==========================================================

def multi_object_ransac_v2(
    points,
    seed
):
    """
    Return:
        models
        diagnostics
        failure_reason
    """

    rng = np.random.default_rng(
        seed
    )


    # ------------------------------------------------------
    # Stage 1: hypothesis generation
    # ------------------------------------------------------

    t0 = time.perf_counter()


    (
        hypothesis_normals,
        hypothesis_ds,
        hypothesis_supports
    ) = generate_plane_hypotheses(
        points,
        rng
    )


    t1 = time.perf_counter()


    num_valid_hypotheses = len(
        hypothesis_normals
    )


    if num_valid_hypotheses == 0:

        diagnostics = {
            "num_valid_hypotheses": 0,
            "num_raw_clusters": 0,
            "num_accepted_clusters": 0,

            "hypothesis_runtime_ms": (
                (t1 - t0)
                * 1000.0
            ),

            "clustering_runtime_ms": 0.0,
            "refinement_runtime_ms": 0.0
        }


        return (
            [],
            diagnostics,
            "no_valid_hypotheses"
        )


    # ------------------------------------------------------
    # Stage 2: clustering
    # ------------------------------------------------------

    clusters = cluster_plane_hypotheses(
        hypothesis_normals,
        hypothesis_ds,
        hypothesis_supports
    )


    t2 = time.perf_counter()


    num_raw_clusters = len(
        clusters[
            "normals"
        ]
    )


    # ------------------------------------------------------
    # Stage 3: model acceptance and refinement
    # ------------------------------------------------------

    (
        models,
        num_accepted_clusters
    ) = refine_plane_clusters(
        points,
        clusters
    )


    t3 = time.perf_counter()


    diagnostics = {
        "num_valid_hypotheses": (
            num_valid_hypotheses
        ),

        "num_raw_clusters": (
            num_raw_clusters
        ),

        "num_accepted_clusters": (
            num_accepted_clusters
        ),

        "hypothesis_runtime_ms": (
            (t1 - t0)
            * 1000.0
        ),

        "clustering_runtime_ms": (
            (t2 - t1)
            * 1000.0
        ),

        "refinement_runtime_ms": (
            (t3 - t2)
            * 1000.0
        )
    }


    if len(
        models
    ) == 0:

        return (
            [],
            diagnostics,
            "no_cluster_with_sufficient_support"
        )


    return (
        models,
        diagnostics,
        ""
    )


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
        "=== Multi-Object RANSAC V2 Benchmark ==="
    )


    print(
        "\nFixed configuration:"
    )

    print(
        "Distance threshold:",
        DISTANCE_THRESHOLD,
        "m"
    )

    print(
        "Hypotheses:",
        NUM_HYPOTHESES
    )

    print(
        "Minimum hypothesis support:",
        MIN_HYPOTHESIS_SUPPORT
    )

    print(
        "Model normal threshold:",
        MODEL_NORMAL_THRESHOLD_DEG,
        "degrees"
    )

    print(
        "Model offset threshold:",
        MODEL_OFFSET_THRESHOLD_M,
        "m"
    )

    print(
        "Minimum cluster hypotheses:",
        MIN_CLUSTER_HYPOTHESES
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


    result_rows = []


    for sample_number, npz_file in enumerate(
        sample_files,
        start=1
    ):

        sample_id = (
            npz_file.stem
        )


        # --------------------------------------------------
        # File I/O outside algorithm runtime
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
        # Common algorithm runtime
        # --------------------------------------------------

        start_time = (
            time.perf_counter()
        )


        (
            models,
            diagnostics,
            failure_reason
        ) = multi_object_ransac_v2(
            points,
            seed=RANSAC_SEED
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

        if len(
            models
        ) == 0:

            result_rows.append(
                {
                    "method": (
                        "Multi-Object RANSAC V2"
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

                    "num_valid_hypotheses": (
                        diagnostics[
                            "num_valid_hypotheses"
                        ]
                    ),

                    "num_raw_clusters": (
                        diagnostics[
                            "num_raw_clusters"
                        ]
                    ),

                    "num_accepted_clusters": (
                        diagnostics[
                            "num_accepted_clusters"
                        ]
                    ),

                    "hypothesis_runtime_ms": (
                        diagnostics[
                            "hypothesis_runtime_ms"
                        ]
                    ),

                    "clustering_runtime_ms": (
                        diagnostics[
                            "clustering_runtime_ms"
                        ]
                    ),

                    "refinement_runtime_ms": (
                        diagnostics[
                            "refinement_runtime_ms"
                        ]
                    ),

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
                f"hyp={diagnostics['num_valid_hypotheses']} "
                f"clusters={diagnostics['num_raw_clusters']} "
                f"runtime={runtime_ms:.2f} ms"
            )


            continue


        # --------------------------------------------------
        # Single-plane common benchmark:
        # strongest final model is compared with GT.
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
        # Ground truth
        #
        # z = 2
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


        result_rows.append(
            {
                "method": (
                    "Multi-Object RANSAC V2"
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
                        "cluster_votes"
                    ]
                ),

                "num_valid_hypotheses": (
                    diagnostics[
                        "num_valid_hypotheses"
                    ]
                ),

                "num_raw_clusters": (
                    diagnostics[
                        "num_raw_clusters"
                    ]
                ),

                "num_accepted_clusters": (
                    diagnostics[
                        "num_accepted_clusters"
                    ]
                ),

                "hypothesis_runtime_ms": (
                    diagnostics[
                        "hypothesis_runtime_ms"
                    ]
                ),

                "clustering_runtime_ms": (
                    diagnostics[
                        "clustering_runtime_ms"
                    ]
                ),

                "refinement_runtime_ms": (
                    diagnostics[
                        "refinement_runtime_ms"
                    ]
                ),

                "failure_reason": "",

                "success": 1
            }
        )


        print(
            f"[{sample_number:3d}/200] "
            f"{sample_id} "
            f"F1={f1:.4f} "
            f"hyp={diagnostics['num_valid_hypotheses']} "
            f"clusters={diagnostics['num_raw_clusters']} "
            f"accepted={diagnostics['num_accepted_clusters']} "
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

        "num_valid_hypotheses",
        "num_raw_clusters",
        "num_accepted_clusters",

        "hypothesis_runtime_ms",
        "clustering_runtime_ms",
        "refinement_runtime_ms",

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
        "\n=== Multi-Object RANSAC V2 Complete ==="
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

        f1_values = np.array(
            [
                row[
                    "f1"
                ]
                for row in successful_rows
            ]
        )


        print(
            "Mean F1:",
            f"{np.mean(f1_values):.6f}"
        )


    print(
        "\nResults saved to:"
    )

    print(
        output_file
    )


if __name__ == "__main__":

    main()