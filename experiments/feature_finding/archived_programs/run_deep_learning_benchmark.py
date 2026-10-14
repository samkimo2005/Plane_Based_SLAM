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

model_dir = (
    project_root
    / "results"
    / "feature_finding"
    / "models"
)

results_dir.mkdir(
    parents=True,
    exist_ok=True
)

model_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_csv = (
    results_dir
    / "deep_learning_results.csv"
)

model_file = (
    model_dir
    / "plane_mlp_model.npz"
)


# ==================================================
# Parameters
# ==================================================

K_NEIGHBORS = 20

CLASSIFICATION_THRESHOLD = 0.5

MIN_PREDICTED_INLIERS = 50

TRAINING_CLOUDS = 60

TRAINING_POINTS_PER_CLOUD = 2500

TRAIN_SAMPLES_PER_CLASS_PER_CLOUD = 400

HIDDEN_1 = 32
HIDDEN_2 = 16

EPOCHS = 40

LEARNING_RATE = 0.005

BATCH_SIZE = 1024

TRAINING_SEED = 20260917


# ==================================================
# Local geometric features
# ==================================================

def extract_features(
    points,
    k_neighbors
):
    """
    Extract local geometric features for each point.

    Features:
        curvature
        linearity
        planarity
        scattering
        roughness
        mean neighbour distance
        standard deviation of neighbour distance
    """

    tree = cKDTree(
        points
    )

    distances, indices = tree.query(
        points,
        k=k_neighbors + 1
    )

    neighbour_indices = indices[
        :,
        1:
    ]

    neighbour_distances = distances[
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

    lambda_0 = eigenvalues[:, 0]
    lambda_1 = eigenvalues[:, 1]
    lambda_2 = eigenvalues[:, 2]

    total = (
        lambda_0
        + lambda_1
        + lambda_2
        + 1e-12
    )

    curvature = (
        lambda_0
        / total
    )

    linearity = (
        lambda_2
        - lambda_1
    ) / (
        lambda_2
        + 1e-12
    )

    planarity = (
        lambda_1
        - lambda_0
    ) / (
        lambda_2
        + 1e-12
    )

    scattering = (
        lambda_0
        / (
            lambda_2
            + 1e-12
        )
    )

    normals = eigenvectors[
        :,
        :,
        0
    ]

    point_from_centroid = (
        points
        - centroids
    )

    roughness = np.abs(
        np.sum(
            point_from_centroid
            * normals,
            axis=1
        )
    )

    mean_distance = np.mean(
        neighbour_distances,
        axis=1
    )

    std_distance = np.std(
        neighbour_distances,
        axis=1
    )

    features = np.column_stack(
        (
            curvature,
            linearity,
            planarity,
            scattering,
            roughness,
            mean_distance,
            std_distance
        )
    )

    return features


# ==================================================
# Generate separate neural-network training data
# ==================================================

def random_unit_vector(
    rng
):
    vector = rng.normal(
        size=3
    )

    vector = (
        vector
        / np.linalg.norm(
            vector
        )
    )

    return vector


def create_plane_basis(
    normal
):
    """
    Create two orthogonal vectors lying
    in the plane.
    """

    if abs(
        normal[2]
    ) < 0.9:

        reference = np.array(
            [
                0.0,
                0.0,
                1.0
            ]
        )

    else:

        reference = np.array(
            [
                1.0,
                0.0,
                0.0
            ]
        )

    basis_1 = np.cross(
        normal,
        reference
    )

    basis_1 = (
        basis_1
        / np.linalg.norm(
            basis_1
        )
    )

    basis_2 = np.cross(
        normal,
        basis_1
    )

    basis_2 = (
        basis_2
        / np.linalg.norm(
            basis_2
        )
    )

    return (
        basis_1,
        basis_2
    )


def generate_training_cloud(
    rng,
    num_points,
    noise_sigma,
    outlier_ratio
):
    """
    Generate a training point cloud.

    Plane orientation and position are randomised
    so the neural network does not simply memorise
    z = 2.
    """

    num_outliers = int(
        round(
            num_points
            * outlier_ratio
        )
    )

    num_plane = (
        num_points
        - num_outliers
    )

    normal = random_unit_vector(
        rng
    )

    d = rng.uniform(
        -1.5,
        1.5
    )

    plane_origin = (
        -d
        * normal
    )

    basis_1, basis_2 = create_plane_basis(
        normal
    )

    u = rng.uniform(
        -2.0,
        2.0,
        num_plane
    )

    v = rng.uniform(
        -2.0,
        2.0,
        num_plane
    )

    plane_points = (
        plane_origin
        + u[:, None] * basis_1
        + v[:, None] * basis_2
    )

    plane_noise = rng.normal(
        0.0,
        noise_sigma,
        num_plane
    )

    plane_points = (
        plane_points
        + plane_noise[:, None]
        * normal
    )


    # ----------------------------------------------
    # Generate outliers away from true plane
    # ----------------------------------------------

    outliers = []

    while len(
        outliers
    ) < num_outliers:

        candidate = (
            plane_origin
            + rng.uniform(
                -3.0,
                3.0,
                3
            )
        )

        distance = abs(
            np.dot(
                normal,
                candidate
            )
            + d
        )

        if distance > 0.25:

            outliers.append(
                candidate
            )


    if num_outliers > 0:

        outliers = np.array(
            outliers,
            dtype=float
        )

    else:

        outliers = np.empty(
            (
                0,
                3
            ),
            dtype=float
        )


    points = np.vstack(
        (
            plane_points,
            outliers
        )
    )

    labels = np.concatenate(
        (
            np.ones(
                num_plane,
                dtype=int
            ),

            np.zeros(
                num_outliers,
                dtype=int
            )
        )
    )

    permutation = rng.permutation(
        len(points)
    )

    points = points[
        permutation
    ]

    labels = labels[
        permutation
    ]

    return (
        points,
        labels
    )


# ==================================================
# Prepare training dataset
# ==================================================

def create_training_dataset():
    """
    Generate independent training clouds.

    These are NOT the 200 benchmark samples.
    """

    print(
        "\nGenerating neural-network training data..."
    )

    rng = np.random.default_rng(
        TRAINING_SEED
    )

    feature_list = []
    label_list = []

    noise_levels = [
        0.000,
        0.005,
        0.010,
        0.020
    ]

    outlier_levels = [
        0.10,
        0.20,
        0.30,
        0.40
    ]


    for cloud_index in range(
        TRAINING_CLOUDS
    ):

        noise = rng.choice(
            noise_levels
        )

        outlier_ratio = rng.choice(
            outlier_levels
        )

        points, labels = generate_training_cloud(
            rng,
            TRAINING_POINTS_PER_CLOUD,
            noise,
            outlier_ratio
        )

        features = extract_features(
            points,
            K_NEIGHBORS
        )


        plane_indices = np.where(
            labels == 1
        )[0]

        outlier_indices = np.where(
            labels == 0
        )[0]


        num_plane_samples = min(
            TRAIN_SAMPLES_PER_CLASS_PER_CLOUD,
            len(
                plane_indices
            )
        )

        num_outlier_samples = min(
            TRAIN_SAMPLES_PER_CLASS_PER_CLOUD,
            len(
                outlier_indices
            )
        )


        selected_plane = rng.choice(
            plane_indices,
            size=num_plane_samples,
            replace=False
        )

        selected_outliers = rng.choice(
            outlier_indices,
            size=num_outlier_samples,
            replace=False
        )


        selected = np.concatenate(
            (
                selected_plane,
                selected_outliers
            )
        )

        feature_list.append(
            features[
                selected
            ]
        )

        label_list.append(
            labels[
                selected
            ]
        )


        print(
            f"Training cloud "
            f"{cloud_index + 1:02d}/"
            f"{TRAINING_CLOUDS}"
        )


    X = np.vstack(
        feature_list
    )

    y = np.concatenate(
        label_list
    ).astype(
        float
    )


    # Shuffle
    permutation = rng.permutation(
        len(X)
    )

    X = X[
        permutation
    ]

    y = y[
        permutation
    ]


    print(
        "\nTraining samples:",
        len(X)
    )

    print(
        "Plane samples:",
        np.count_nonzero(
            y == 1
        )
    )

    print(
        "Outlier samples:",
        np.count_nonzero(
            y == 0
        )
    )


    return (
        X,
        y
    )


# ==================================================
# MLP helper functions
# ==================================================

def relu(
    x
):
    return np.maximum(
        x,
        0.0
    )


def relu_derivative(
    x
):
    return (
        x > 0
    ).astype(
        float
    )


def sigmoid(
    x
):
    x = np.clip(
        x,
        -50,
        50
    )

    return (
        1.0
        / (
            1.0
            + np.exp(
                -x
            )
        )
    )


# ==================================================
# Initialise neural network
# ==================================================

def initialise_model(
    input_size,
    rng
):
    """
    MLP structure:

    input
      ->
    32 ReLU
      ->
    16 ReLU
      ->
    1 sigmoid
    """

    W1 = rng.normal(
        0,
        np.sqrt(
            2.0
            / input_size
        ),
        (
            input_size,
            HIDDEN_1
        )
    )

    b1 = np.zeros(
        HIDDEN_1
    )

    W2 = rng.normal(
        0,
        np.sqrt(
            2.0
            / HIDDEN_1
        ),
        (
            HIDDEN_1,
            HIDDEN_2
        )
    )

    b2 = np.zeros(
        HIDDEN_2
    )

    W3 = rng.normal(
        0,
        np.sqrt(
            1.0
            / HIDDEN_2
        ),
        (
            HIDDEN_2,
            1
        )
    )

    b3 = np.zeros(
        1
    )


    return {
        "W1": W1,
        "b1": b1,

        "W2": W2,
        "b2": b2,

        "W3": W3,
        "b3": b3
    }


# ==================================================
# Forward pass
# ==================================================

def forward(
    X,
    model
):
    z1 = (
        X @ model["W1"]
        + model["b1"]
    )

    a1 = relu(
        z1
    )


    z2 = (
        a1 @ model["W2"]
        + model["b2"]
    )

    a2 = relu(
        z2
    )


    z3 = (
        a2 @ model["W3"]
        + model["b3"]
    )


    probabilities = sigmoid(
        z3
    ).reshape(
        -1
    )


    cache = {
        "z1": z1,
        "a1": a1,

        "z2": z2,
        "a2": a2,

        "probabilities": probabilities
    }


    return (
        probabilities,
        cache
    )


# ==================================================
# Train MLP
# ==================================================

def train_model(
    X,
    y
):
    rng = np.random.default_rng(
        TRAINING_SEED
    )


    # ----------------------------------------------
    # Standardise features
    # ----------------------------------------------

    feature_mean = np.mean(
        X,
        axis=0
    )

    feature_std = np.std(
        X,
        axis=0
    )

    feature_std[
        feature_std < 1e-12
    ] = 1.0


    X = (
        X
        - feature_mean
    ) / feature_std


    model = initialise_model(
        X.shape[1],
        rng
    )


    num_samples = len(
        X
    )


    print(
        "\nTraining neural network..."
    )


    for epoch in range(
        EPOCHS
    ):

        permutation = rng.permutation(
            num_samples
        )


        epoch_loss = 0.0

        batch_count = 0


        for start in range(
            0,
            num_samples,
            BATCH_SIZE
        ):

            batch_indices = permutation[
                start:
                start + BATCH_SIZE
            ]

            X_batch = X[
                batch_indices
            ]

            y_batch = y[
                batch_indices
            ]


            probabilities, cache = forward(
                X_batch,
                model
            )


            epsilon = 1e-8


            loss = -np.mean(
                y_batch
                * np.log(
                    probabilities
                    + epsilon
                )
                +
                (
                    1.0
                    - y_batch
                )
                * np.log(
                    1.0
                    - probabilities
                    + epsilon
                )
            )


            epoch_loss += loss
            batch_count += 1


            # ------------------------------------------
            # Back propagation
            # ------------------------------------------

            batch_size = len(
                X_batch
            )


            dz3 = (
                probabilities
                - y_batch
            ).reshape(
                -1,
                1
            ) / batch_size


            dW3 = (
                cache["a2"].T
                @ dz3
            )

            db3 = np.sum(
                dz3,
                axis=0
            )


            da2 = (
                dz3
                @ model["W3"].T
            )


            dz2 = (
                da2
                * relu_derivative(
                    cache["z2"]
                )
            )


            dW2 = (
                cache["a1"].T
                @ dz2
            )

            db2 = np.sum(
                dz2,
                axis=0
            )


            da1 = (
                dz2
                @ model["W2"].T
            )


            dz1 = (
                da1
                * relu_derivative(
                    cache["z1"]
                )
            )


            dW1 = (
                X_batch.T
                @ dz1
            )

            db1 = np.sum(
                dz1,
                axis=0
            )


            # ------------------------------------------
            # Gradient descent update
            # ------------------------------------------

            model["W1"] -= (
                LEARNING_RATE
                * dW1
            )

            model["b1"] -= (
                LEARNING_RATE
                * db1
            )

            model["W2"] -= (
                LEARNING_RATE
                * dW2
            )

            model["b2"] -= (
                LEARNING_RATE
                * db2
            )

            model["W3"] -= (
                LEARNING_RATE
                * dW3
            )

            model["b3"] -= (
                LEARNING_RATE
                * db3
            )


        average_loss = (
            epoch_loss
            / batch_count
        )


        print(
            f"Epoch "
            f"{epoch + 1:02d}/"
            f"{EPOCHS}"
            f" | loss = "
            f"{average_loss:.6f}"
        )


    model[
        "feature_mean"
    ] = feature_mean

    model[
        "feature_std"
    ] = feature_std


    return model


# ==================================================
# Save / load model
# ==================================================

def save_model(
    model,
    filename
):
    np.savez(
        filename,

        W1=model["W1"],
        b1=model["b1"],

        W2=model["W2"],
        b2=model["b2"],

        W3=model["W3"],
        b3=model["b3"],

        feature_mean=model[
            "feature_mean"
        ],

        feature_std=model[
            "feature_std"
        ]
    )


def load_model(
    filename
):
    data = np.load(
        filename
    )

    return {
        "W1": data["W1"],
        "b1": data["b1"],

        "W2": data["W2"],
        "b2": data["b2"],

        "W3": data["W3"],
        "b3": data["b3"],

        "feature_mean": data[
            "feature_mean"
        ],

        "feature_std": data[
            "feature_std"
        ]
    }


# ==================================================
# Plane fit
# ==================================================

def fit_plane_svd(
    points
):
    if len(
        points
    ) < 3:

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


    normal = (
        normal
        / np.linalg.norm(
            normal
        )
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
# Neural-network plane detection
# ==================================================

def neural_plane_detection(
    points,
    model
):
    features = extract_features(
        points,
        K_NEIGHBORS
    )


    features = (
        features
        - model[
            "feature_mean"
        ]
    ) / model[
        "feature_std"
    ]


    probabilities, _ = forward(
        features,
        model
    )


    predicted_inliers = (
        probabilities
        >= CLASSIFICATION_THRESHOLD
    )


    predicted_count = np.count_nonzero(
        predicted_inliers
    )


    if predicted_count < MIN_PREDICTED_INLIERS:

        return {
            "success": False,

            "failure_reason": (
                "TOO_FEW_PREDICTED_INLIERS"
            ),

            "predicted_count": int(
                predicted_count
            )
        }


    plane_result = fit_plane_svd(
        points[
            predicted_inliers
        ]
    )


    if plane_result is None:

        return {
            "success": False,

            "failure_reason": (
                "PLANE_FIT_FAILED"
            ),

            "predicted_count": int(
                predicted_count
            )
        }


    normal, d = plane_result


    return {
        "success": True,

        "failure_reason": "",

        "normal": normal,

        "d": float(
            d
        ),

        "inlier_mask": (
            predicted_inliers
        ),

        "probabilities": (
            probabilities
        )
    }


# ==================================================
# Evaluation functions
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

    return float(
        np.degrees(
            np.arccos(
                cosine
            )
        )
    )


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
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),

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
# Main
# ==================================================

if __name__ == "__main__":

    print(
        "=== Neural Network Plane Detection Benchmark ==="
    )


    # ==================================================
    # Train model only if model does not already exist
    # ==================================================

    if model_file.exists():

        print(
            "\nExisting neural network model found."
        )

        print(
            "Loading:"
        )

        print(
            model_file
        )

        model = load_model(
            model_file
        )

    else:

        X_train, y_train = (
            create_training_dataset()
        )

        model = train_model(
            X_train,
            y_train
        )

        save_model(
            model,
            model_file
        )

        print(
            "\nModel saved to:"
        )

        print(
            model_file
        )


    # ==================================================
    # Benchmark samples
    # ==================================================

    sample_files = sorted(
        data_dir.glob(
            "N*_O*_R*.npz"
        )
    )


    print(
        "\nNumber of benchmark samples:",
        len(
            sample_files
        )
    )


    assert len(
        sample_files
    ) == 200, (
        f"Expected 200 samples, "
        f"but found {len(sample_files)}"
    )


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
        "classification_threshold",

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

        "failure_reason",

        "success"
    ]


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


            data = np.load(
                sample_file
            )

            points = data[
                "points"
            ]

            labels = data[
                "labels"
            ]


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


            # ------------------------------------------
            # Run neural network
            # ------------------------------------------

            start_time = perf_counter()


            result = neural_plane_detection(
                points,
                model
            )


            end_time = perf_counter()


            runtime_ms = (
                end_time
                - start_time
            ) * 1000.0


            # ------------------------------------------
            # Failure
            # ------------------------------------------

            if not result[
                "success"
            ]:

                writer.writerow(
                    {
                        "method": (
                            "Neural Network"
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

                        "classification_threshold": (
                            CLASSIFICATION_THRESHOLD
                        ),

                        "detected_inlier_count": (
                            result.get(
                                "predicted_count",
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


            # ------------------------------------------
            # Successful result
            # ------------------------------------------

            normal = result[
                "normal"
            ]

            d = result[
                "d"
            ]

            predicted_inliers = result[
                "inlier_mask"
            ]


            normal, d = align_plane_direction(
                normal,
                d,
                gt_normal
            )


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


            writer.writerow(
                {
                    "method": (
                        "Neural Network"
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

                    "classification_threshold": (
                        CLASSIFICATION_THRESHOLD
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

                "| runtime =",
                f"{runtime_ms:.2f}",
                "ms"
            )


    print(
        "\n==========================================="
    )

    print(
        "=== Neural Network Benchmark Complete ==="
    )

    print(
        "==========================================="
    )


    print(
        "\nResults saved to:"
    )

    print(
        output_csv
    )