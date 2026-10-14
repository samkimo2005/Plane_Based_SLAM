from __future__ import annotations

from pathlib import Path
import json
import csv

import numpy as np


# ============================================================
# Configuration
# ============================================================

# Different from DEV seed: FINAL must remain independent.
FINAL_MASTER_SEED = 20261001

BAD_RATIOS = [
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
]

NORMAL_NOISE_DEG = [
    0.0,
    2.0,
    5.0,
]

MATCH_COUNTS = [
    6,
    10,
    15,
]

GEOMETRIES = [
    "well_conditioned",
    "mostly_parallel",
    "weakly_constrained",
]

REPEATS = 3

OFFSET_NOISE_M = 0.02
CENTROID_NOISE_M = 0.02


# ============================================================
# Paths
# ============================================================

HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent

OUTPUT_DIR = (
    EXPERIMENTS_DIR
    / "data"
    / "bad_match_rejection"
    / "final"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Basic geometry
# ============================================================

def normalise(v):
    v = np.asarray(v, dtype=float)

    norm = np.linalg.norm(v)

    if norm < 1e-12:
        raise ValueError("Zero-length vector.")

    return v / norm


def axis_angle_rotation(axis, angle_rad):
    axis = normalise(axis)

    x, y, z = axis

    K = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])

    I = np.eye(3)

    return (
        I
        + np.sin(angle_rad) * K
        + (1.0 - np.cos(angle_rad))
        * (K @ K)
    )


def random_pose(rng):
    """
    Moderate inter-frame motion.
    """

    axis = normalise(
        rng.normal(size=3)
    )

    angle_deg = rng.uniform(
        -15.0,
        15.0,
    )

    R = axis_angle_rotation(
        axis,
        np.deg2rad(angle_deg),
    )

    t = rng.uniform(
        -0.30,
        0.30,
        size=3,
    )

    return R, t, angle_deg


def transform_plane(plane, R, t):
    """
    Plane:
        n^T x + d = 0

    Point transform:
        x_B = R x_A + t
    """

    n_a = plane[:3]
    d_a = float(plane[3])

    n_b = R @ n_a

    d_b = (
        d_a
        - n_b @ t
    )

    return np.hstack(
        [n_b, d_b]
    )


# ============================================================
# Plane generation
# ============================================================

def random_tangent(normal, rng):
    v = rng.normal(size=3)

    v = (
        v
        - np.dot(v, normal) * normal
    )

    if np.linalg.norm(v) < 1e-8:
        return random_tangent(
            normal,
            rng,
        )

    return normalise(v)


def make_centroid(normal, d, rng):
    """
    Generate a point lying on the plane.
    """

    base = -d * normal

    tangent_1 = random_tangent(
        normal,
        rng,
    )

    tangent_2 = normalise(
        np.cross(
            normal,
            tangent_1,
        )
    )

    a = rng.uniform(-1.5, 1.5)
    b = rng.uniform(-1.5, 1.5)

    return (
        base
        + a * tangent_1
        + b * tangent_2
    )


def generate_normals(
    count,
    geometry,
    rng,
):
    normals = []

    if geometry == "well_conditioned":

        # Broadly distributed directions.
        while len(normals) < count:

            n = normalise(
                rng.normal(size=3)
            )

            normals.append(n)

    elif geometry == "mostly_parallel":

        # Most planes have approximately the same normal.
        base = normalise(
            np.array([0.0, 0.0, 1.0])
            + 0.05 * rng.normal(size=3)
        )

        for _ in range(count):

            n = normalise(
                base
                + 0.08 * rng.normal(size=3)
            )

            normals.append(n)

    elif geometry == "weakly_constrained":

        # Normals mainly lie close to the x-z plane,
        # giving weak constraint along one direction.
        for _ in range(count):

            angle = rng.uniform(
                0.0,
                2.0 * np.pi,
            )

            n = np.array([
                np.cos(angle),
                rng.normal(0.0, 0.03),
                np.sin(angle),
            ])

            normals.append(
                normalise(n)
            )

    else:
        raise ValueError(
            f"Unknown geometry: {geometry}"
        )

    return np.asarray(normals)


def generate_clean_planes(
    count,
    geometry,
    rng,
):
    normals = generate_normals(
        count,
        geometry,
        rng,
    )

    planes = []
    centroids = []

    for normal in normals:

        distance = rng.uniform(
            1.0,
            4.0,
        )

        d = -distance

        plane = np.hstack(
            [normal, d]
        )

        centroid = make_centroid(
            normal,
            d,
            rng,
        )

        planes.append(plane)
        centroids.append(centroid)

    return (
        np.asarray(planes),
        np.asarray(centroids),
    )


# ============================================================
# Measurement noise
# ============================================================

def perturb_normal(
    normal,
    sigma_deg,
    rng,
):
    if sigma_deg <= 0.0:
        return normal.copy()

    tangent = random_tangent(
        normal,
        rng,
    )

    angle = rng.normal(
        0.0,
        np.deg2rad(sigma_deg),
    )

    perturbed = (
        np.cos(angle) * normal
        + np.sin(angle) * tangent
    )

    return normalise(perturbed)


def add_measurement_noise(
    planes,
    centroids,
    normal_sigma_deg,
    rng,
):
    noisy_planes = []
    noisy_centroids = []

    for plane, centroid in zip(
        planes,
        centroids,
    ):
        n = perturb_normal(
            plane[:3],
            normal_sigma_deg,
            rng,
        )

        d = (
            float(plane[3])
            + rng.normal(
                0.0,
                OFFSET_NOISE_M,
            )
        )

        c = (
            centroid
            + rng.normal(
                0.0,
                CENTROID_NOISE_M,
                size=3,
            )
        )

        noisy_planes.append(
            np.hstack([n, d])
        )

        noisy_centroids.append(c)

    return (
        np.asarray(noisy_planes),
        np.asarray(noisy_centroids),
    )


# ============================================================
# Correspondence corruption
# ============================================================

def generate_assignment(
    count,
    bad_ratio,
    rng,
):
    """
    Start from the correct one-to-one assignment i -> i.

    A selected subset is cyclically permuted to create
    wrong correspondences while retaining one-to-one matching.
    """

    permutation = np.arange(
        count,
        dtype=int,
    )

    requested_bad = int(
        round(
            bad_ratio * count
        )
    )

    if requested_bad == 0:

        gt_labels = np.ones(
            count,
            dtype=bool,
        )

        return (
            np.column_stack([
                np.arange(count),
                permutation,
            ]),
            gt_labels,
        )

    # A non-identity one-to-one permutation cannot contain
    # exactly one displaced element.
    bad_count = max(
        2,
        requested_bad,
    )

    bad_count = min(
        bad_count,
        count,
    )

    bad_indices = rng.choice(
        count,
        size=bad_count,
        replace=False,
    )

    shifted = np.roll(
        bad_indices,
        -1,
    )

    permutation[
        bad_indices
    ] = shifted

    assigned_pairs = np.column_stack([
        np.arange(count),
        permutation,
    ])

    gt_labels = (
        assigned_pairs[:, 0]
        == assigned_pairs[:, 1]
    )

    return (
        assigned_pairs,
        gt_labels,
    )


# ============================================================
# One synthetic sample
# ============================================================

def generate_sample(
    match_count,
    bad_ratio,
    normal_noise_deg,
    geometry,
    rng,
):
    # Clean physical planes in frame A.
    clean_planes_a, clean_centroids_a = (
        generate_clean_planes(
            match_count,
            geometry,
            rng,
        )
    )

    # Ground-truth relative camera motion.
    R_gt, t_gt, angle_deg = (
        random_pose(rng)
    )

    # Same physical planes expressed in frame B.
    clean_planes_b = np.asarray([
        transform_plane(
            plane,
            R_gt,
            t_gt,
        )
        for plane in clean_planes_a
    ])

    clean_centroids_b = (
        clean_centroids_a
        @ R_gt.T
        + t_gt
    )

    # Independent measurement noise in A and B.
    planes_a, centroids_a = (
        add_measurement_noise(
            clean_planes_a,
            clean_centroids_a,
            normal_noise_deg,
            rng,
        )
    )

    planes_b, centroids_b = (
        add_measurement_noise(
            clean_planes_b,
            clean_centroids_b,
            normal_noise_deg,
            rng,
        )
    )

    # Inject bad assigned correspondences.
    (
        assigned_pairs,
        gt_match_labels,
    ) = generate_assignment(
        match_count,
        bad_ratio,
        rng,
    )

    return {
        "planes_a": planes_a,
        "planes_b": planes_b,
        "centroids_a": centroids_a,
        "centroids_b": centroids_b,

        "assigned_pairs":
            assigned_pairs,

        # True = correct match
        # False = bad match
        "gt_match_labels":
            gt_match_labels,

        "R_gt": R_gt,
        "t_gt": t_gt,

        "rotation_angle_deg":
            angle_deg,
    }


# ============================================================
# Main generation
# ============================================================

def main():

    # Prevent accidental mixing with an older FINAL set.
    old_files = list(
        OUTPUT_DIR.glob("sample_*.npz")
    )

    if old_files:
        raise RuntimeError(
            "FINAL directory already contains sample files.\n"
            f"Please clear or rename:\n{OUTPUT_DIR}"
        )

    master_rng = np.random.default_rng(
        FINAL_MASTER_SEED
    )

    manifest_rows = []

    sample_index = 0

    for bad_ratio in BAD_RATIOS:
        for normal_noise_deg in NORMAL_NOISE_DEG:
            for match_count in MATCH_COUNTS:
                for geometry in GEOMETRIES:
                    for repeat in range(REPEATS):

                        # Independent deterministic seed
                        # for every FINAL sample.
                        sample_seed = int(
                            master_rng.integers(
                                0,
                                2**32 - 1,
                            )
                        )

                        rng = (
                            np.random.default_rng(
                                sample_seed
                            )
                        )

                        sample = generate_sample(
                            match_count=match_count,
                            bad_ratio=bad_ratio,
                            normal_noise_deg=
                                normal_noise_deg,
                            geometry=geometry,
                            rng=rng,
                        )

                        filename = (
                            f"sample_"
                            f"{sample_index:04d}.npz"
                        )

                        np.savez_compressed(
                            OUTPUT_DIR
                            / filename,

                            planes_a=
                                sample["planes_a"],

                            planes_b=
                                sample["planes_b"],

                            centroids_a=
                                sample["centroids_a"],

                            centroids_b=
                                sample["centroids_b"],

                            assigned_pairs=
                                sample[
                                    "assigned_pairs"
                                ],

                            gt_match_labels=
                                sample[
                                    "gt_match_labels"
                                ],

                            R_gt=
                                sample["R_gt"],

                            t_gt=
                                sample["t_gt"],
                        )

                        actual_bad_count = int(
                            np.sum(
                                ~sample[
                                    "gt_match_labels"
                                ]
                            )
                        )

                        actual_bad_ratio = (
                            actual_bad_count
                            / match_count
                        )

                        manifest_rows.append({
                            "sample":
                                filename,

                            "sample_index":
                                sample_index,

                            "seed":
                                sample_seed,

                            "repeat":
                                repeat,

                            "match_count":
                                match_count,

                            "requested_bad_ratio":
                                bad_ratio,

                            "actual_bad_count":
                                actual_bad_count,

                            "actual_bad_ratio":
                                actual_bad_ratio,

                            "normal_noise_deg":
                                normal_noise_deg,

                            "offset_noise_m":
                                OFFSET_NOISE_M,

                            "centroid_noise_m":
                                CENTROID_NOISE_M,

                            "geometry":
                                geometry,

                            "rotation_angle_deg":
                                sample[
                                    "rotation_angle_deg"
                                ],

                            "translation_norm_m":
                                float(
                                    np.linalg.norm(
                                        sample["t_gt"]
                                    )
                                ),
                        })

                        sample_index += 1

    # ========================================================
    # Manifest
    # ========================================================

    manifest_file = (
        OUTPUT_DIR
        / "manifest.csv"
    )

    with open(
        manifest_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
                manifest_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(
            manifest_rows
        )

    # ========================================================
    # Generation configuration
    # ========================================================

    config = {
        "dataset":
            "bad_match_rejection_final",

        "master_seed":
            FINAL_MASTER_SEED,

        "bad_ratios":
            BAD_RATIOS,

        "normal_noise_deg":
            NORMAL_NOISE_DEG,

        "match_counts":
            MATCH_COUNTS,

        "geometries":
            GEOMETRIES,

        "repeats":
            REPEATS,

        "offset_noise_m":
            OFFSET_NOISE_M,

        "centroid_noise_m":
            CENTROID_NOISE_M,

        "number_of_samples":
            sample_index,

        "purpose":
            (
                "Independent synthetic FINAL benchmark. "
                "Not used for parameter tuning."
            ),
    }

    with open(
        OUTPUT_DIR
        / "generation_config.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )

    print()
    print("=" * 65)
    print("BAD-MATCH REJECTION FINAL DATASET GENERATED")
    print("=" * 65)
    print(
        f"Samples generated: {sample_index}"
    )
    print(
        f"Master seed:       {FINAL_MASTER_SEED}"
    )
    print(
        f"Output directory:  {OUTPUT_DIR}"
    )
    print()
    print(
        "This FINAL dataset must not be used "
        "for further parameter tuning."
    )


if __name__ == "__main__":
    main()