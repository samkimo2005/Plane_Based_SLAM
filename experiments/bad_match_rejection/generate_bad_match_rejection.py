import csv
import json
from pathlib import Path

import numpy as np

from bad_match_rejection_common import (
    normalise_plane,
    transform_plane,
)


# ============================================================
# Configuration
# ============================================================

MASTER_SEED = 20260919

# Controlled DEV conditions
BAD_MATCH_RATIOS = [
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

GEOMETRY_TYPES = [
    "well_conditioned",
    "mostly_parallel",
    "weakly_constrained",
]

REPEATS = 3

# Plane offset measurement noise
OFFSET_NOISE_STD_M = 0.02

# Plane centroid measurement noise
CENTROID_NOISE_STD_M = 0.02

# Relative camera motion used to generate Frame B
ROTATION_RANGE_DEG = (5.0, 18.0)
TRANSLATION_RANGE_M = (0.05, 0.30)


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "bad_match_rejection"
    / "dev"
)

MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"
CONFIG_PATH = OUTPUT_DIR / "generation_config.json"


# ============================================================
# Basic geometry helpers
# ============================================================

def random_unit_vector(rng):
    """
    Generate a random 3D unit vector.
    """

    v = rng.normal(size=3)

    norm = np.linalg.norm(v)

    if norm < 1e-12:
        return random_unit_vector(rng)

    return v / norm


def rotation_matrix_from_axis_angle(axis, angle_rad):
    """
    Rodrigues' rotation formula.
    """

    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    x, y, z = axis

    K = np.array([
        [0.0, -z,   y],
        [z,    0.0, -x],
        [-y,   x,   0.0],
    ])

    I = np.eye(3)

    R = (
        I
        + np.sin(angle_rad) * K
        + (1.0 - np.cos(angle_rad)) * (K @ K)
    )

    return R


def generate_random_pose(rng):
    """
    Generate a moderate relative camera motion from Frame A to Frame B.
    """

    axis = random_unit_vector(rng)

    angle_deg = rng.uniform(
        ROTATION_RANGE_DEG[0],
        ROTATION_RANGE_DEG[1],
    )

    angle_rad = np.deg2rad(angle_deg)

    R = rotation_matrix_from_axis_angle(
        axis,
        angle_rad,
    )

    direction = random_unit_vector(rng)

    magnitude = rng.uniform(
        TRANSLATION_RANGE_M[0],
        TRANSLATION_RANGE_M[1],
    )

    t = direction * magnitude

    return R, t


def orthogonal_unit_vector(normal, rng):
    """
    Generate a random unit vector perpendicular to normal.
    """

    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    candidate = random_unit_vector(rng)

    tangent = (
        candidate
        - np.dot(candidate, normal) * normal
    )

    norm = np.linalg.norm(tangent)

    if norm < 1e-8:
        return orthogonal_unit_vector(normal, rng)

    return tangent / norm


def perturb_normal(normal, noise_deg, rng):
    """
    Perturb a plane normal by a random angular error.

    noise_deg is interpreted as one-sigma angular noise.
    """

    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    if noise_deg <= 0.0:
        return normal.copy()

    angle_deg = rng.normal(
        loc=0.0,
        scale=noise_deg,
    )

    angle_rad = np.deg2rad(angle_deg)

    axis = orthogonal_unit_vector(
        normal,
        rng,
    )

    R_noise = rotation_matrix_from_axis_angle(
        axis,
        angle_rad,
    )

    perturbed = R_noise @ normal

    return perturbed / np.linalg.norm(perturbed)


def project_point_to_plane(point, normal, d):
    """
    Project a 3D point onto:

        n^T x + d = 0

    with unit normal n.
    """

    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    point = np.asarray(point, dtype=float)

    signed_distance = (
        np.dot(normal, point) + d
    )

    return point - signed_distance * normal


# ============================================================
# Plane generation
# ============================================================

def generate_normal_for_geometry(
    geometry_type,
    index,
    rng,
):
    """
    Generate plane normals for different geometric conditions.

    well_conditioned:
        normals distributed over 3D directions.

    mostly_parallel:
        most normals concentrated around one dominant direction.

    weakly_constrained:
        normals are mostly close to a 2D span, giving weak
        observability in one translation direction.
    """

    if geometry_type == "well_conditioned":

        # Cycle around three major directions, then add variation.
        basis = np.eye(3)

        base = basis[index % 3]

        perturbation = (
            0.35 * rng.normal(size=3)
        )

        normal = base + perturbation


    elif geometry_type == "mostly_parallel":

        # Most planes are approximately parallel to Z.
        base = np.array([
            0.0,
            0.0,
            1.0,
        ])

        perturbation = np.array([
            rng.normal(scale=0.08),
            rng.normal(scale=0.08),
            rng.normal(scale=0.02),
        ])

        normal = base + perturbation

        # Add occasional non-parallel planes so the entire
        # set is not always exactly rank deficient.
        if index % 5 == 4:

            normal = np.array([
                1.0,
                0.15 * rng.normal(),
                0.15 * rng.normal(),
            ])


    elif geometry_type == "weakly_constrained":

        # Normals lie mostly in the XY plane.
        angle = rng.uniform(
            0.0,
            2.0 * np.pi,
        )

        normal = np.array([
            np.cos(angle),
            np.sin(angle),
            rng.normal(scale=0.05),
        ])


    else:
        raise ValueError(
            f"Unknown geometry type: {geometry_type}"
        )

    norm = np.linalg.norm(normal)

    if norm < 1e-12:
        return generate_normal_for_geometry(
            geometry_type,
            index,
            rng,
        )

    return normal / norm


def generate_plane_set(
    n_planes,
    geometry_type,
    rng,
):
    """
    Generate clean planes and representative plane centroids
    in Frame A.

    Plane representation:

        [nx, ny, nz, d]

    with:

        n^T x + d = 0
    """

    planes = []
    centroids = []

    for i in range(n_planes):

        normal = generate_normal_for_geometry(
            geometry_type,
            i,
            rng,
        )

        # Signed plane distance from camera origin.
        distance = rng.uniform(
            1.0,
            4.0,
        )

        d = -distance

        normal, d = normalise_plane(
            normal,
            d,
        )

        # Closest point on plane to camera origin.
        closest_point = -d * normal

        # Move along two tangent directions to create
        # a representative visible centroid.
        tangent_1 = orthogonal_unit_vector(
            normal,
            rng,
        )

        tangent_2 = np.cross(
            normal,
            tangent_1,
        )

        tangent_2 /= np.linalg.norm(
            tangent_2
        )

        centroid = (
            closest_point
            + rng.uniform(-1.0, 1.0) * tangent_1
            + rng.uniform(-1.0, 1.0) * tangent_2
        )

        planes.append([
            normal[0],
            normal[1],
            normal[2],
            d,
        ])

        centroids.append(
            centroid
        )

    return (
        np.asarray(planes, dtype=float),
        np.asarray(centroids, dtype=float),
    )


# ============================================================
# Measurement noise
# ============================================================

def add_plane_measurement_noise(
    planes,
    centroids,
    normal_noise_deg,
    rng,
):
    """
    Add synthetic plane detection / measurement noise.
    """

    noisy_planes = []
    noisy_centroids = []

    for plane, centroid in zip(
        planes,
        centroids,
    ):

        normal = plane[:3]
        d = plane[3]

        noisy_normal = perturb_normal(
            normal,
            normal_noise_deg,
            rng,
        )

        noisy_d = (
            d
            + rng.normal(
                scale=OFFSET_NOISE_STD_M
            )
        )

        noisy_normal, noisy_d = normalise_plane(
            noisy_normal,
            noisy_d,
        )

        noisy_centroid = (
            centroid
            + rng.normal(
                scale=CENTROID_NOISE_STD_M,
                size=3,
            )
        )

        # Make centroid geometrically consistent with the
        # noisy detected plane.
        noisy_centroid = project_point_to_plane(
            noisy_centroid,
            noisy_normal,
            noisy_d,
        )

        noisy_planes.append([
            noisy_normal[0],
            noisy_normal[1],
            noisy_normal[2],
            noisy_d,
        ])

        noisy_centroids.append(
            noisy_centroid
        )

    return (
        np.asarray(noisy_planes, dtype=float),
        np.asarray(noisy_centroids, dtype=float),
    )


# ============================================================
# Frame transformation
# ============================================================

def transform_plane_set(
    planes_a,
    centroids_a,
    R,
    t,
):
    """
    Transform clean Frame-A planes and centroids into Frame B.
    """

    planes_b = []
    centroids_b = []

    for plane, centroid in zip(
        planes_a,
        centroids_a,
    ):

        normal_b, d_b = transform_plane(
            plane[:3],
            plane[3],
            R,
            t,
        )

        centroid_b = (
            R @ centroid + t
        )

        planes_b.append([
            normal_b[0],
            normal_b[1],
            normal_b[2],
            d_b,
        ])

        centroids_b.append(
            centroid_b
        )

    return (
        np.asarray(planes_b, dtype=float),
        np.asarray(centroids_b, dtype=float),
    )


# ============================================================
# Bad-match injection
# ============================================================

def choose_bad_match_count(
    n_matches,
    requested_ratio,
):
    """
    Convert requested bad-match ratio to an integer number of
    incorrect correspondences.

    One isolated wrong assignment cannot be created while keeping
    a strict one-to-one permutation, so a non-zero value is forced
    to at least two bad matches.
    """

    if requested_ratio <= 0.0:
        return 0

    n_bad = int(
        round(
            requested_ratio
            * n_matches
        )
    )

    n_bad = max(
        2,
        n_bad,
    )

    n_bad = min(
        n_bad,
        n_matches,
    )

    return n_bad


def create_assigned_pairs(
    n_matches,
    requested_bad_ratio,
    rng,
):
    """
    Create one-to-one assigned correspondences.

    Ground-truth mapping is:

        A_i <-> B_i

    Bad matches are introduced by cyclically permuting B indices
    inside a selected subset.

    This preserves strict one-to-one assignment while guaranteeing
    that selected correspondences are incorrect.
    """

    assigned_b = np.arange(
        n_matches,
        dtype=int,
    )

    n_bad = choose_bad_match_count(
        n_matches,
        requested_bad_ratio,
    )

    bad_indices = np.array(
        [],
        dtype=int,
    )

    if n_bad > 0:

        bad_indices = rng.choice(
            n_matches,
            size=n_bad,
            replace=False,
        )

        shuffled = bad_indices.copy()

        rng.shuffle(
            shuffled
        )

        # A cyclic shift guarantees no selected index
        # maps to itself.
        shifted = np.roll(
            shuffled,
            1,
        )

        for source_idx, target_idx in zip(
            shuffled,
            shifted,
        ):
            assigned_b[source_idx] = target_idx

    assigned_pairs = np.column_stack([
        np.arange(
            n_matches,
            dtype=int,
        ),
        assigned_b,
    ])

    gt_match_labels = (
        assigned_pairs[:, 0]
        == assigned_pairs[:, 1]
    )

    actual_bad_ratio = (
        1.0
        - np.mean(
            gt_match_labels
        )
    )

    return (
        assigned_pairs,
        gt_match_labels,
        bad_indices,
        float(actual_bad_ratio),
    )


# ============================================================
# Single sample generation
# ============================================================

def generate_sample(
    sample_id,
    n_matches,
    requested_bad_ratio,
    normal_noise_deg,
    geometry_type,
    repeat_index,
    seed,
):
    """
    Generate one controlled bad-match rejection sample.
    """

    rng = np.random.default_rng(
        seed
    )

    # --------------------------------------------------------
    # 1. Clean Frame-A geometry
    # --------------------------------------------------------

    clean_planes_a, clean_centroids_a = (
        generate_plane_set(
            n_matches,
            geometry_type,
            rng,
        )
    )

    # --------------------------------------------------------
    # 2. Ground-truth relative pose
    # --------------------------------------------------------

    R_gt, t_gt = generate_random_pose(
        rng
    )

    # --------------------------------------------------------
    # 3. Transform clean geometry into Frame B
    # --------------------------------------------------------

    clean_planes_b, clean_centroids_b = (
        transform_plane_set(
            clean_planes_a,
            clean_centroids_a,
            R_gt,
            t_gt,
        )
    )

    # --------------------------------------------------------
    # 4. Add independent measurement noise to A and B
    # --------------------------------------------------------

    planes_a, centroids_a = (
        add_plane_measurement_noise(
            clean_planes_a,
            clean_centroids_a,
            normal_noise_deg,
            rng,
        )
    )

    planes_b, centroids_b = (
        add_plane_measurement_noise(
            clean_planes_b,
            clean_centroids_b,
            normal_noise_deg,
            rng,
        )
    )

    # --------------------------------------------------------
    # 5. Simulate output from Assignment
    # --------------------------------------------------------

    (
        assigned_pairs,
        gt_match_labels,
        bad_indices,
        actual_bad_ratio,
    ) = create_assigned_pairs(
        n_matches,
        requested_bad_ratio,
        rng,
    )

    # --------------------------------------------------------
    # 6. Metadata
    # --------------------------------------------------------

    metadata = {
        "sample_id": sample_id,
        "seed": int(seed),
        "n_matches": int(n_matches),
        "requested_bad_ratio": float(
            requested_bad_ratio
        ),
        "actual_bad_ratio": float(
            actual_bad_ratio
        ),
        "normal_noise_deg": float(
            normal_noise_deg
        ),
        "offset_noise_std_m": float(
            OFFSET_NOISE_STD_M
        ),
        "centroid_noise_std_m": float(
            CENTROID_NOISE_STD_M
        ),
        "geometry_type": geometry_type,
        "repeat": int(repeat_index),
        "n_bad_matches": int(
            np.sum(
                ~gt_match_labels
            )
        ),
        "n_good_matches": int(
            np.sum(
                gt_match_labels
            )
        ),
    }

    return {
        "planes_a": planes_a,
        "planes_b": planes_b,
        "centroids_a": centroids_a,
        "centroids_b": centroids_b,
        "assigned_pairs": assigned_pairs,
        "gt_match_labels": gt_match_labels,
        "R_gt": R_gt,
        "t_gt": t_gt,
        "bad_indices": bad_indices,
        "metadata": metadata,
    }


# ============================================================
# Validation
# ============================================================

def validate_sample(sample):
    """
    Sanity-check generated data before saving.
    """

    planes_a = sample["planes_a"]
    planes_b = sample["planes_b"]

    assigned_pairs = sample[
        "assigned_pairs"
    ]

    labels = sample[
        "gt_match_labels"
    ]

    n_matches = len(
        assigned_pairs
    )

    # Plane normal length check
    norms_a = np.linalg.norm(
        planes_a[:, :3],
        axis=1,
    )

    norms_b = np.linalg.norm(
        planes_b[:, :3],
        axis=1,
    )

    if not np.allclose(
        norms_a,
        1.0,
        atol=1e-8,
    ):
        raise RuntimeError(
            "Frame-A plane normals are not unit length."
        )

    if not np.allclose(
        norms_b,
        1.0,
        atol=1e-8,
    ):
        raise RuntimeError(
            "Frame-B plane normals are not unit length."
        )

    # One-to-one source assignment
    if len(
        np.unique(
            assigned_pairs[:, 0]
        )
    ) != n_matches:
        raise RuntimeError(
            "Source plane assignments are not one-to-one."
        )

    # One-to-one target assignment
    if len(
        np.unique(
            assigned_pairs[:, 1]
        )
    ) != n_matches:
        raise RuntimeError(
            "Target plane assignments are not one-to-one."
        )

    # Label consistency
    expected_labels = (
        assigned_pairs[:, 0]
        == assigned_pairs[:, 1]
    )

    if not np.array_equal(
        labels,
        expected_labels,
    ):
        raise RuntimeError(
            "Ground-truth labels are inconsistent."
        )

    # Rotation sanity check
    R_gt = sample["R_gt"]

    if not np.allclose(
        R_gt.T @ R_gt,
        np.eye(3),
        atol=1e-8,
    ):
        raise RuntimeError(
            "Ground-truth rotation is not orthonormal."
        )

    if not np.isclose(
        np.linalg.det(R_gt),
        1.0,
        atol=1e-8,
    ):
        raise RuntimeError(
            "Ground-truth rotation determinant is not +1."
        )


# ============================================================
# Saving
# ============================================================

def save_sample(
    sample,
    filepath,
):
    """
    Save one sample as a compressed NumPy archive.
    """

    metadata_json = json.dumps(
        sample["metadata"]
    )

    np.savez_compressed(
        filepath,
        planes_a=sample["planes_a"],
        planes_b=sample["planes_b"],
        centroids_a=sample[
            "centroids_a"
        ],
        centroids_b=sample[
            "centroids_b"
        ],
        assigned_pairs=sample[
            "assigned_pairs"
        ],
        gt_match_labels=sample[
            "gt_match_labels"
        ],
        R_gt=sample["R_gt"],
        t_gt=sample["t_gt"],
        bad_indices=sample[
            "bad_indices"
        ],
        metadata_json=np.array(
            metadata_json
        ),
    )


def save_generation_config():
    """
    Save the exact DEV generation settings for reproducibility.
    """

    config = {
        "master_seed": MASTER_SEED,
        "bad_match_ratios": BAD_MATCH_RATIOS,
        "normal_noise_deg": NORMAL_NOISE_DEG,
        "match_counts": MATCH_COUNTS,
        "geometry_types": GEOMETRY_TYPES,
        "repeats": REPEATS,
        "offset_noise_std_m": OFFSET_NOISE_STD_M,
        "centroid_noise_std_m": CENTROID_NOISE_STD_M,
        "rotation_range_deg": ROTATION_RANGE_DEG,
        "translation_range_m": TRANSLATION_RANGE_M,
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=4,
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 68)
    print("GENERATING BAD-MATCH REJECTION DEV DATASET")
    print("=" * 68)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Independent deterministic seed stream.
    master_rng = np.random.default_rng(
        MASTER_SEED
    )

    manifest_rows = []

    sample_number = 0

    total_samples = (
        len(BAD_MATCH_RATIOS)
        * len(NORMAL_NOISE_DEG)
        * len(MATCH_COUNTS)
        * len(GEOMETRY_TYPES)
        * REPEATS
    )

    for geometry_type in GEOMETRY_TYPES:

        for n_matches in MATCH_COUNTS:

            for normal_noise_deg in NORMAL_NOISE_DEG:

                for bad_ratio in BAD_MATCH_RATIOS:

                    for repeat in range(REPEATS):

                        sample_number += 1

                        sample_id = (
                            f"dev_{sample_number:04d}"
                        )

                        seed = int(
                            master_rng.integers(
                                0,
                                2**32 - 1,
                            )
                        )

                        sample = generate_sample(
                            sample_id=sample_id,
                            n_matches=n_matches,
                            requested_bad_ratio=bad_ratio,
                            normal_noise_deg=normal_noise_deg,
                            geometry_type=geometry_type,
                            repeat_index=repeat,
                            seed=seed,
                        )

                        validate_sample(
                            sample
                        )

                        filename = (
                            f"{sample_id}.npz"
                        )

                        filepath = (
                            OUTPUT_DIR
                            / filename
                        )

                        save_sample(
                            sample,
                            filepath,
                        )

                        row = (
                            sample[
                                "metadata"
                            ].copy()
                        )

                        row[
                            "filename"
                        ] = filename

                        manifest_rows.append(
                            row
                        )

                        print(
                            f"[{sample_number:03d}/"
                            f"{total_samples}] "
                            f"{sample_id} | "
                            f"N={n_matches:2d} | "
                            f"bad={sample['metadata']['actual_bad_ratio']:.2f} | "
                            f"noise={normal_noise_deg:.1f} deg | "
                            f"{geometry_type}"
                        )

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    fieldnames = [
        "sample_id",
        "filename",
        "seed",
        "n_matches",
        "requested_bad_ratio",
        "actual_bad_ratio",
        "normal_noise_deg",
        "offset_noise_std_m",
        "centroid_noise_std_m",
        "geometry_type",
        "repeat",
        "n_bad_matches",
        "n_good_matches",
    ]

    with open(
        MANIFEST_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            manifest_rows
        )

    save_generation_config()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 68)
    print("DEV DATASET GENERATION COMPLETE")
    print("=" * 68)

    print(
        f"Samples generated : {sample_number}"
    )

    print(
        f"Output directory  : {OUTPUT_DIR}"
    )

    print(
        f"Manifest          : {MANIFEST_PATH}"
    )

    print(
        f"Configuration     : {CONFIG_PATH}"
    )

    print("=" * 68)


if __name__ == "__main__":
    main()