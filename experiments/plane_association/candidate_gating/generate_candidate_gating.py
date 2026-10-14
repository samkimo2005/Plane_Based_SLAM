"""
generate_candidate_gating.py

Generate controlled synthetic datasets for the Candidate Gating
stage of Plane Association.

Direct execution:
    Simply press Run in the IDE or execute this file.

No command-line arguments are required.

Output:
    data/candidate_gating/dev/
or:
    data/candidate_gating/final/

Plane representation:
    n^T x + d = 0

Pose convention:
    x_target = R @ x_source + t
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ============================================================
# USER SETTINGS
# ============================================================

# Use "dev" during development / tuning.
# Change to "final" only after parameters have been frozen.
SPLIT = "final"

# Number of random repetitions for each condition.
DEV_SAMPLES_PER_CONDITION = 10
FINAL_SAMPLES_PER_CONDITION = 20

# Different seeds ensure DEV and FINAL data are independent.
DEV_BASE_SEED = 12345
FINAL_BASE_SEED = 54321

# If True, an existing dataset for the selected split
# will be deleted and regenerated.
OVERWRITE = True

# Number of sampled contour points on each rectangle edge.
CONTOUR_POINTS_PER_EDGE = 8


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "candidate_gating"
    / SPLIT
)


# ============================================================
# Data structures
# ============================================================

@dataclass
class PlanePatch:
    """
    Bounded planar observation.

    Plane equation:
        n^T x + d = 0
    """

    normal: np.ndarray
    d: float

    centroid: np.ndarray
    area: float

    contour: np.ndarray

    width: float
    height: float


@dataclass
class Condition:
    """
    One controlled synthetic benchmark condition.
    """

    name: str

    n_matches: int
    n_extra_source: int
    n_extra_target: int

    normal_noise_deg: float
    centroid_noise_m: float
    offset_noise_m: float

    pose_rotation_error_deg: float
    pose_translation_error_m: float


# ============================================================
# Experimental conditions
# ============================================================

CONDITIONS: List[Condition] = [

    Condition(
        name="clean",
        n_matches=4,
        n_extra_source=1,
        n_extra_target=1,

        normal_noise_deg=0.2,
        centroid_noise_m=0.005,
        offset_noise_m=0.003,

        pose_rotation_error_deg=0.2,
        pose_translation_error_m=0.005,
    ),

    Condition(
        name="low_measurement_noise",
        n_matches=4,
        n_extra_source=1,
        n_extra_target=1,

        normal_noise_deg=1.0,
        centroid_noise_m=0.010,
        offset_noise_m=0.010,

        pose_rotation_error_deg=0.5,
        pose_translation_error_m=0.010,
    ),

    Condition(
        name="medium_measurement_noise",
        n_matches=4,
        n_extra_source=2,
        n_extra_target=2,

        normal_noise_deg=3.0,
        centroid_noise_m=0.030,
        offset_noise_m=0.020,

        pose_rotation_error_deg=1.0,
        pose_translation_error_m=0.020,
    ),

    Condition(
        name="pose_rotation_error",
        n_matches=4,
        n_extra_source=2,
        n_extra_target=2,

        normal_noise_deg=1.0,
        centroid_noise_m=0.015,
        offset_noise_m=0.010,

        pose_rotation_error_deg=5.0,
        pose_translation_error_m=0.020,
    ),

    Condition(
        name="pose_translation_error",
        n_matches=4,
        n_extra_source=2,
        n_extra_target=2,

        normal_noise_deg=1.0,
        centroid_noise_m=0.015,
        offset_noise_m=0.010,

        pose_rotation_error_deg=1.0,
        pose_translation_error_m=0.100,
    ),

    Condition(
        name="clutter",
        n_matches=4,
        n_extra_source=4,
        n_extra_target=4,

        normal_noise_deg=1.5,
        centroid_noise_m=0.020,
        offset_noise_m=0.015,

        pose_rotation_error_deg=2.0,
        pose_translation_error_m=0.030,
    ),

    Condition(
        name="few_matches",
        n_matches=2,
        n_extra_source=3,
        n_extra_target=3,

        normal_noise_deg=1.5,
        centroid_noise_m=0.020,
        offset_noise_m=0.015,

        pose_rotation_error_deg=2.0,
        pose_translation_error_m=0.030,
    ),

    Condition(
        name="hard",
        n_matches=5,
        n_extra_source=4,
        n_extra_target=4,

        normal_noise_deg=5.0,
        centroid_noise_m=0.050,
        offset_noise_m=0.030,

        pose_rotation_error_deg=7.0,
        pose_translation_error_m=0.120,
    ),
]


# ============================================================
# Basic geometry
# ============================================================

def normalize(
    vector: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Return a unit vector.
    """

    vector = np.asarray(
        vector,
        dtype=float,
    )

    magnitude = float(
        np.linalg.norm(vector)
    )

    if magnitude < eps:
        raise ValueError(
            "Cannot normalize a near-zero vector."
        )

    return vector / magnitude


def skew(
    vector: np.ndarray,
) -> np.ndarray:
    """
    Return the skew-symmetric matrix of a 3D vector.
    """

    x, y, z = vector

    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def axis_angle_to_rotation(
    axis: np.ndarray,
    angle_rad: float,
) -> np.ndarray:
    """
    Rodrigues' rotation formula.
    """

    axis = normalize(
        axis
    )

    K = skew(
        axis
    )

    I = np.eye(3)

    return (
        I
        + math.sin(angle_rad) * K
        + (1.0 - math.cos(angle_rad))
        * (K @ K)
    )


def random_rotation_noise(
    rng: np.random.Generator,
    std_deg: float,
) -> np.ndarray:
    """
    Generate a random small rotation.
    """

    if std_deg <= 0:
        return np.eye(3)

    axis = rng.normal(
        size=3
    )

    while np.linalg.norm(axis) < 1e-10:
        axis = rng.normal(
            size=3
        )

    angle_rad = math.radians(
        rng.normal(
            loc=0.0,
            scale=std_deg,
        )
    )

    return axis_angle_to_rotation(
        axis,
        angle_rad,
    )


def rotation_angle_deg(
    R: np.ndarray,
) -> float:
    """
    Magnitude of rotation represented by R.
    """

    value = (
        np.trace(R) - 1.0
    ) / 2.0

    value = np.clip(
        value,
        -1.0,
        1.0,
    )

    return math.degrees(
        math.acos(value)
    )


# ============================================================
# Plane generation
# ============================================================

def plane_basis(
    normal: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Generate two orthonormal vectors lying in a plane.
    """

    normal = normalize(
        normal
    )

    if abs(normal[2]) < 0.9:

        reference = np.array(
            [0.0, 0.0, 1.0]
        )

    else:

        reference = np.array(
            [1.0, 0.0, 0.0]
        )

    u = normalize(
        np.cross(
            normal,
            reference,
        )
    )

    v = normalize(
        np.cross(
            normal,
            u,
        )
    )

    return u, v


def generate_rectangle_contour(
    centroid: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    width: float,
    height: float,
) -> np.ndarray:
    """
    Sample contour points along the four edges
    of a rectangular bounded plane.
    """

    half_width = (
        width / 2.0
    )

    half_height = (
        height / 2.0
    )

    corners = np.array(
        [
            centroid
            - half_width * u
            - half_height * v,

            centroid
            + half_width * u
            - half_height * v,

            centroid
            + half_width * u
            + half_height * v,

            centroid
            - half_width * u
            + half_height * v,
        ]
    )

    contour = []

    for edge_idx in range(4):

        start = corners[
            edge_idx
        ]

        end = corners[
            (edge_idx + 1) % 4
        ]

        for alpha in np.linspace(
            0.0,
            1.0,
            CONTOUR_POINTS_PER_EDGE,
            endpoint=False,
        ):

            point = (
                (1.0 - alpha) * start
                + alpha * end
            )

            contour.append(
                point
            )

    return np.asarray(
        contour,
        dtype=float,
    )


def generate_random_plane(
    rng: np.random.Generator,
) -> PlanePatch:
    """
    Generate one random bounded plane.
    """

    normal = normalize(
        rng.normal(
            size=3
        )
    )

    u, v = plane_basis(
        normal
    )

    # Distance of plane from origin.
    rho = rng.uniform(
        1.5,
        4.0,
    )

    # Random displacement inside plane.
    shift_u = rng.uniform(
        -1.0,
        1.0,
    )

    shift_v = rng.uniform(
        -1.0,
        1.0,
    )

    centroid = (
        rho * normal
        + shift_u * u
        + shift_v * v
    )

    width = rng.uniform(
        0.8,
        2.8,
    )

    height = rng.uniform(
        0.8,
        2.8,
    )

    area = (
        width * height
    )

    contour = (
        generate_rectangle_contour(
            centroid=centroid,
            u=u,
            v=v,
            width=width,
            height=height,
        )
    )

    d = -float(
        normal @ centroid
    )

    return PlanePatch(
        normal=normal,
        d=d,

        centroid=centroid,
        area=area,

        contour=contour,

        width=width,
        height=height,
    )


# ============================================================
# Plane transformation
# ============================================================

def transform_plane(
    plane: PlanePatch,
    R: np.ndarray,
    t: np.ndarray,
) -> PlanePatch:
    """
    Transform a plane from source frame to target frame.

    x_target = R x_source + t
    """

    normal_target = normalize(
        R @ plane.normal
    )

    centroid_target = (
        R @ plane.centroid
        + t
    )

    contour_target = (
        (R @ plane.contour.T).T
        + t
    )

    d_target = -float(
        normal_target
        @ centroid_target
    )

    return PlanePatch(
        normal=normal_target,
        d=d_target,

        centroid=centroid_target,
        area=plane.area,

        contour=contour_target,

        width=plane.width,
        height=plane.height,
    )


# ============================================================
# Plane measurement noise
# ============================================================

def perturb_plane(
    plane: PlanePatch,
    rng: np.random.Generator,
    normal_noise_deg: float,
    centroid_noise_m: float,
    offset_noise_m: float,
) -> PlanePatch:
    """
    Add controlled measurement noise to a plane observation.
    """

    # --------------------------------------------------------
    # Normal orientation noise
    # --------------------------------------------------------

    R_noise = random_rotation_noise(
        rng,
        normal_noise_deg,
    )

    normal_new = normalize(
        R_noise @ plane.normal
    )

    # Rotate bounded contour around its centroid.
    local_contour = (
        plane.contour
        - plane.centroid
    )

    contour_new = (
        plane.centroid
        + (
            R_noise
            @ local_contour.T
        ).T
    )

    # --------------------------------------------------------
    # Centroid tangential noise
    # --------------------------------------------------------

    raw_centroid_noise = (
        rng.normal(
            loc=0.0,
            scale=centroid_noise_m,
            size=3,
        )
    )

    tangential_noise = (
        raw_centroid_noise
        - normal_new
        * float(
            raw_centroid_noise
            @ normal_new
        )
    )

    # --------------------------------------------------------
    # Offset / normal-direction noise
    # --------------------------------------------------------

    normal_shift = (
        rng.normal(
            loc=0.0,
            scale=offset_noise_m,
        )
        * normal_new
    )

    total_translation = (
        tangential_noise
        + normal_shift
    )

    centroid_new = (
        plane.centroid
        + total_translation
    )

    contour_new = (
        contour_new
        + total_translation
    )

    d_new = -float(
        normal_new
        @ centroid_new
    )

    return PlanePatch(
        normal=normal_new,
        d=d_new,

        centroid=centroid_new,
        area=plane.area,

        contour=contour_new,

        width=plane.width,
        height=plane.height,
    )


# ============================================================
# Pose generation
# ============================================================

def generate_true_pose(
    rng: np.random.Generator,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Generate a small realistic inter-frame motion.
    """

    axis = rng.normal(
        size=3
    )

    while np.linalg.norm(axis) < 1e-10:

        axis = rng.normal(
            size=3
        )

    angle_deg = rng.uniform(
        -8.0,
        8.0,
    )

    R_true = (
        axis_angle_to_rotation(
            axis,
            math.radians(
                angle_deg
            ),
        )
    )

    t_true = rng.uniform(
        low=-0.15,
        high=0.15,
        size=3,
    )

    return (
        R_true,
        t_true,
    )


def generate_pose_prior(
    R_true: np.ndarray,
    t_true: np.ndarray,
    rng: np.random.Generator,
    rotation_error_deg: float,
    translation_error_m: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    """
    Generate an imperfect pose prior from the true pose.
    """

    R_error = (
        random_rotation_noise(
            rng,
            rotation_error_deg,
        )
    )

    translation_error = (
        rng.normal(
            loc=0.0,
            scale=translation_error_m,
            size=3,
        )
    )

    R_prior = (
        R_error @ R_true
    )

    t_prior = (
        t_true
        + translation_error
    )

    realised_rotation_error = (
        rotation_angle_deg(
            R_error
        )
    )

    realised_translation_error = float(
        np.linalg.norm(
            translation_error
        )
    )

    return (
        R_prior,
        t_prior,

        realised_rotation_error,
        realised_translation_error,
    )


# ============================================================
# Data conversion
# ============================================================

def pack_planes(
    planes: List[PlanePatch],
) -> dict:
    """
    Convert PlanePatch list into NumPy arrays.
    """

    return {

        "normals":
            np.stack(
                [
                    plane.normal
                    for plane
                    in planes
                ],
                axis=0,
            ),

        "offsets":
            np.asarray(
                [
                    plane.d
                    for plane
                    in planes
                ],
                dtype=float,
            ),

        "centroids":
            np.stack(
                [
                    plane.centroid
                    for plane
                    in planes
                ],
                axis=0,
            ),

        "areas":
            np.asarray(
                [
                    plane.area
                    for plane
                    in planes
                ],
                dtype=float,
            ),

        "contours":
            np.stack(
                [
                    plane.contour
                    for plane
                    in planes
                ],
                axis=0,
            ),

        "widths":
            np.asarray(
                [
                    plane.width
                    for plane
                    in planes
                ],
                dtype=float,
            ),

        "heights":
            np.asarray(
                [
                    plane.height
                    for plane
                    in planes
                ],
                dtype=float,
            ),
    }


def build_gt_matrix(
    n_source: int,
    n_target: int,
    gt_pairs: np.ndarray,
) -> np.ndarray:
    """
    Construct binary GT correspondence matrix.
    """

    matrix = np.zeros(
        (
            n_source,
            n_target,
        ),
        dtype=np.uint8,
    )

    for (
        source_idx,
        target_idx,
    ) in gt_pairs:

        matrix[
            source_idx,
            target_idx
        ] = 1

    return matrix


# ============================================================
# Generate one sample
# ============================================================

def generate_sample(
    condition: Condition,
    seed: int,
) -> Tuple[
    dict,
    dict,
]:

    rng = np.random.default_rng(
        seed
    )

    # --------------------------------------------------------
    # Ground-truth relative pose
    # --------------------------------------------------------

    (
        R_true,
        t_true,
    ) = generate_true_pose(
        rng
    )

    (
        R_prior,
        t_prior,
        realised_pose_rot_error,
        realised_pose_trans_error,
    ) = generate_pose_prior(
        R_true=R_true,
        t_true=t_true,

        rng=rng,

        rotation_error_deg=(
            condition
            .pose_rotation_error_deg
        ),

        translation_error_m=(
            condition
            .pose_translation_error_m
        ),
    )

    # --------------------------------------------------------
    # Generate matched physical planes
    # --------------------------------------------------------

    source_entries = []
    target_entries = []

    for pair_id in range(
        condition.n_matches
    ):

        clean_source = (
            generate_random_plane(
                rng
            )
        )

        clean_target = (
            transform_plane(
                clean_source,
                R_true,
                t_true,
            )
        )

        source_observation = (
            perturb_plane(
                plane=clean_source,

                rng=rng,

                normal_noise_deg=(
                    condition
                    .normal_noise_deg
                ),

                centroid_noise_m=(
                    condition
                    .centroid_noise_m
                ),

                offset_noise_m=(
                    condition
                    .offset_noise_m
                ),
            )
        )

        target_observation = (
            perturb_plane(
                plane=clean_target,

                rng=rng,

                normal_noise_deg=(
                    condition
                    .normal_noise_deg
                ),

                centroid_noise_m=(
                    condition
                    .centroid_noise_m
                ),

                offset_noise_m=(
                    condition
                    .offset_noise_m
                ),
            )
        )

        source_entries.append(
            (
                source_observation,
                pair_id,
            )
        )

        target_entries.append(
            (
                target_observation,
                pair_id,
            )
        )

    # --------------------------------------------------------
    # Add unmatched source planes
    # --------------------------------------------------------

    for _ in range(
        condition.n_extra_source
    ):

        plane = (
            generate_random_plane(
                rng
            )
        )

        plane = perturb_plane(
            plane=plane,

            rng=rng,

            normal_noise_deg=(
                condition
                .normal_noise_deg
            ),

            centroid_noise_m=(
                condition
                .centroid_noise_m
            ),

            offset_noise_m=(
                condition
                .offset_noise_m
            ),
        )

        source_entries.append(
            (
                plane,
                None,
            )
        )

    # --------------------------------------------------------
    # Add unmatched target planes
    # --------------------------------------------------------

    for _ in range(
        condition.n_extra_target
    ):

        plane = (
            generate_random_plane(
                rng
            )
        )

        plane = perturb_plane(
            plane=plane,

            rng=rng,

            normal_noise_deg=(
                condition
                .normal_noise_deg
            ),

            centroid_noise_m=(
                condition
                .centroid_noise_m
            ),

            offset_noise_m=(
                condition
                .offset_noise_m
            ),
        )

        target_entries.append(
            (
                plane,
                None,
            )
        )

    # --------------------------------------------------------
    # Shuffle source and target plane order independently
    # --------------------------------------------------------

    rng.shuffle(
        source_entries
    )

    rng.shuffle(
        target_entries
    )

    source_planes = [
        item[0]
        for item
        in source_entries
    ]

    target_planes = [
        item[0]
        for item
        in target_entries
    ]

    source_pair_ids = [
        item[1]
        for item
        in source_entries
    ]

    target_pair_ids = [
        item[1]
        for item
        in target_entries
    ]

    # --------------------------------------------------------
    # Recover GT correspondence indices
    # --------------------------------------------------------

    gt_pairs = []

    for pair_id in range(
        condition.n_matches
    ):

        source_idx = (
            source_pair_ids.index(
                pair_id
            )
        )

        target_idx = (
            target_pair_ids.index(
                pair_id
            )
        )

        gt_pairs.append(
            [
                source_idx,
                target_idx,
            ]
        )

    gt_pairs = np.asarray(
        gt_pairs,
        dtype=np.int32,
    )

    source_data = pack_planes(
        source_planes
    )

    target_data = pack_planes(
        target_planes
    )

    gt_matrix = build_gt_matrix(
        n_source=len(
            source_planes
        ),

        n_target=len(
            target_planes
        ),

        gt_pairs=gt_pairs,
    )

    # --------------------------------------------------------
    # Numerical data
    # --------------------------------------------------------

    arrays = {

        "source_normals":
            source_data["normals"],

        "source_offsets":
            source_data["offsets"],

        "source_centroids":
            source_data["centroids"],

        "source_areas":
            source_data["areas"],

        "source_contours":
            source_data["contours"],

        "source_widths":
            source_data["widths"],

        "source_heights":
            source_data["heights"],


        "target_normals":
            target_data["normals"],

        "target_offsets":
            target_data["offsets"],

        "target_centroids":
            target_data["centroids"],

        "target_areas":
            target_data["areas"],

        "target_contours":
            target_data["contours"],

        "target_widths":
            target_data["widths"],

        "target_heights":
            target_data["heights"],


        "gt_pairs":
            gt_pairs,

        "gt_match_matrix":
            gt_matrix,


        "true_R":
            R_true,

        "true_t":
            t_true,


        "prior_R":
            R_prior,

        "prior_t":
            t_prior,
    }

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "seed":
            int(seed),

        "condition":
            asdict(
                condition
            ),

        "source_plane_count":
            len(
                source_planes
            ),

        "target_plane_count":
            len(
                target_planes
            ),

        "true_match_count":
            int(
                condition.n_matches
            ),

        "all_possible_pair_count":
            (
                len(source_planes)
                * len(target_planes)
            ),

        "true_motion_rotation_deg":
            rotation_angle_deg(
                R_true
            ),

        "true_motion_translation_m":
            float(
                np.linalg.norm(
                    t_true
                )
            ),

        "realised_pose_prior_rotation_error_deg":
            realised_pose_rot_error,

        "realised_pose_prior_translation_error_m":
            realised_pose_trans_error,

        "plane_equation":
            "n^T x + d = 0",

        "pose_convention":
            "x_target = R @ x_source + t",
    }

    return (
        arrays,
        metadata,
    )


# ============================================================
# Output directory
# ============================================================

def prepare_output_directory() -> None:
    """
    Create or replace selected output directory.
    """

    if OUTPUT_DIR.exists():

        existing_files = list(
            OUTPUT_DIR.iterdir()
        )

        if existing_files:

            if not OVERWRITE:

                raise FileExistsError(
                    "\nDataset already exists:\n"
                    f"{OUTPUT_DIR}\n\n"
                    "Set OVERWRITE = True if you "
                    "want to regenerate it."
                )

            shutil.rmtree(
                OUTPUT_DIR
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Generate complete dataset
# ============================================================

def generate_dataset() -> None:

    if SPLIT not in {
        "dev",
        "final",
    }:

        raise ValueError(
            'SPLIT must be either "dev" or "final".'
        )

    if SPLIT == "dev":

        samples_per_condition = (
            DEV_SAMPLES_PER_CONDITION
        )

        base_seed = (
            DEV_BASE_SEED
        )

    else:

        samples_per_condition = (
            FINAL_SAMPLES_PER_CONDITION
        )

        base_seed = (
            FINAL_BASE_SEED
        )

    prepare_output_directory()

    total_samples = (
        len(CONDITIONS)
        * samples_per_condition
    )

    print()
    print("=" * 70)
    print("Candidate Gating Dataset Generator")
    print("=" * 70)

    print(
        f"Split               : {SPLIT.upper()}"
    )

    print(
        f"Output folder       : {OUTPUT_DIR}"
    )

    print(
        f"Conditions          : {len(CONDITIONS)}"
    )

    print(
        f"Samples / condition : {samples_per_condition}"
    )

    print(
        f"Total samples       : {total_samples}"
    )

    print(
        f"Base random seed    : {base_seed}"
    )

    print("=" * 70)
    print()

    manifest_rows = []

    sample_counter = 0

    for (
        condition_index,
        condition,
    ) in enumerate(
        CONDITIONS
    ):

        print(
            f"[{condition_index + 1}/"
            f"{len(CONDITIONS)}] "
            f"{condition.name}"
        )

        for repetition in range(
            samples_per_condition
        ):

            sample_seed = (
                base_seed
                + condition_index * 100000
                + repetition
            )

            (
                arrays,
                metadata,
            ) = generate_sample(
                condition=condition,
                seed=sample_seed,
            )

            sample_name = (
                f"sample_"
                f"{sample_counter:04d}"
            )

            npz_file = (
                OUTPUT_DIR
                / f"{sample_name}.npz"
            )

            json_file = (
                OUTPUT_DIR
                / f"{sample_name}.json"
            )

            np.savez_compressed(
                npz_file,
                **arrays,
            )

            with open(
                json_file,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    metadata,
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

            manifest_rows.append(
                {
                    "sample_id":
                        sample_counter,

                    "condition":
                        condition.name,

                    "repetition":
                        repetition,

                    "seed":
                        sample_seed,

                    "npz_file":
                        npz_file.name,

                    "metadata_file":
                        json_file.name,

                    "source_planes":
                        metadata[
                            "source_plane_count"
                        ],

                    "target_planes":
                        metadata[
                            "target_plane_count"
                        ],

                    "true_matches":
                        metadata[
                            "true_match_count"
                        ],

                    "possible_pairs":
                        metadata[
                            "all_possible_pair_count"
                        ],

                    "pose_rot_error_deg":
                        metadata[
                            "realised_pose_prior_rotation_error_deg"
                        ],

                    "pose_trans_error_m":
                        metadata[
                            "realised_pose_prior_translation_error_m"
                        ],
                }
            )

            sample_counter += 1

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

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
            fieldnames=[
                "sample_id",
                "condition",
                "repetition",
                "seed",
                "npz_file",
                "metadata_file",
                "source_planes",
                "target_planes",
                "true_matches",
                "possible_pairs",
                "pose_rot_error_deg",
                "pose_trans_error_m",
            ],
        )

        writer.writeheader()

        writer.writerows(
            manifest_rows
        )

    print()
    print("=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Generated samples : {sample_counter}"
    )

    print(
        f"Dataset folder    : {OUTPUT_DIR}"
    )

    print(
        f"Manifest          : {manifest_file}"
    )

    print("=" * 70)
    print()


# ============================================================
# Direct execution
# ============================================================

def main() -> None:

    generate_dataset()


if __name__ == "__main__":
    main()