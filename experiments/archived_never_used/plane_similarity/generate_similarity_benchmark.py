from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ============================================================
# 1. Project paths
# ============================================================

# This file should be located at:
# D:\PlaneBasedSLAM\experiments\plane_similarity\
#
# parents[0] -> plane_similarity
# parents[1] -> experiments
# parents[2] -> PlaneBasedSLAM

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "synthetic_similarity_benchmark"
)

DEV_DIR = OUTPUT_ROOT / "dev"
FINAL_DIR = OUTPUT_ROOT / "final"

# If True, old JSON files in dev/final will be removed
# before regeneration.
CLEAN_OUTPUT = True


# ============================================================
# 2. Dataset size
# ============================================================

# Development set:
# 3 scene types × 3 pose-prior levels × 3 repeats = 27 cases
DEV_REPEATS_PER_CONDITION = 3

# Final benchmark:
# 3 scene types × 3 pose-prior levels × 10 repeats = 90 cases
FINAL_REPEATS_PER_CONDITION = 10

BASE_SEED = 20260918


# ============================================================
# 3. Plane-measurement noise
# ============================================================

# These are controlled engineering starting values.
# They are not claimed to be literature-optimal constants.

NORMAL_NOISE_STD_DEG = 1.0
OFFSET_NOISE_STD_M = 0.01
CENTROID_NOISE_STD_M = 0.02
SIZE_RELATIVE_NOISE_STD = 0.05

# Plane equation sign ambiguity:
#
#     (n, d)
#
# and
#
#     (-n, -d)
#
# describe the same plane.
#
# Some observations are deliberately sign-flipped so that
# later similarity methods must handle this correctly.
NORMAL_SIGN_FLIP_PROB = 0.25


# ============================================================
# 4. Pose-prior conditions
# ============================================================

PRIOR_LEVELS = {
    "exact": {
        "rotation_error_deg": 0.0,
        "translation_error_m": 0.0,
    },

    "mild": {
        "rotation_error_deg": 2.0,
        "translation_error_m": 0.03,
    },

    "moderate": {
        "rotation_error_deg": 5.0,
        "translation_error_m": 0.10,
    },
}


SCENE_TYPES = (
    "distinct",
    "repetitive",
    "partial_visibility",
)


# ============================================================
# 5. Plane structure
# ============================================================

@dataclass
class WorldPlane:
    plane_id: str

    # Representative centre of the finite plane patch
    # in world coordinates.
    centroid_w: np.ndarray

    # Unit plane normal in world coordinates.
    normal_w: np.ndarray

    width_m: float
    height_m: float

    @property
    def area_m2(self) -> float:
        return self.width_m * self.height_m

    @property
    def d_w(self) -> float:
        """
        Plane equation:

            n^T x + d = 0

        Since centroid_w lies on the plane:

            d = -n^T centroid
        """

        return -float(
            np.dot(
                self.normal_w,
                self.centroid_w,
            )
        )


# ============================================================
# 6. Geometry helper functions
# ============================================================

def normalize(v: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(v)

    if length < 1e-12:
        raise ValueError(
            "Cannot normalise a near-zero vector."
        )

    return v / length


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v

    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def rotation_matrix_axis_angle(
    axis: np.ndarray,
    angle_rad: float,
) -> np.ndarray:
    """
    Rodrigues' rotation formula.
    """

    axis = normalize(axis)

    K = skew(axis)

    I = np.eye(3)

    return (
        I
        + math.sin(angle_rad) * K
        + (1.0 - math.cos(angle_rad)) * (K @ K)
    )


def rotation_matrix_xyz(
    roll_rad: float,
    pitch_rad: float,
    yaw_rad: float,
) -> np.ndarray:
    """
    Build a rotation matrix using:

        R = Rz @ Ry @ Rx
    """

    cr = math.cos(roll_rad)
    sr = math.sin(roll_rad)

    cp = math.cos(pitch_rad)
    sp = math.sin(pitch_rad)

    cy = math.cos(yaw_rad)
    sy = math.sin(yaw_rad)

    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=float,
    )

    Ry = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=float,
    )

    Rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    return Rz @ Ry @ Rx


def random_unit_vector(
    rng: np.random.Generator,
) -> np.ndarray:
    return normalize(
        rng.normal(size=3)
    )


def perturb_normal(
    normal: np.ndarray,
    std_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply a small random angular perturbation
    to a unit normal.
    """

    if std_deg <= 0.0:
        return normal.copy()

    angle_deg = rng.normal(
        0.0,
        std_deg,
    )

    angle_rad = math.radians(
        angle_deg
    )

    # Generate an axis perpendicular to the normal.
    for _ in range(20):

        candidate = rng.normal(
            size=3
        )

        axis = (
            candidate
            - np.dot(candidate, normal) * normal
        )

        if np.linalg.norm(axis) > 1e-8:

            axis = normalize(axis)

            R = rotation_matrix_axis_angle(
                axis,
                angle_rad,
            )

            return normalize(
                R @ normal
            )

    raise RuntimeError(
        "Failed to generate a normal perturbation."
    )


def angle_between_normals_deg(
    n1: np.ndarray,
    n2: np.ndarray,
) -> float:
    """
    Plane normals are sign ambiguous.

    Therefore n and -n must be treated as equivalent.
    """

    dot = abs(
        float(
            np.dot(
                normalize(n1),
                normalize(n2),
            )
        )
    )

    dot = np.clip(
        dot,
        0.0,
        1.0,
    )

    return math.degrees(
        math.acos(dot)
    )


# ============================================================
# 7. Plane constructor
# ============================================================

def make_plane(
    plane_id: str,
    centroid,
    normal,
    width_m: float,
    height_m: float,
) -> WorldPlane:

    return WorldPlane(
        plane_id=plane_id,

        centroid_w=np.array(
            centroid,
            dtype=float,
        ),

        normal_w=normalize(
            np.array(
                normal,
                dtype=float,
            )
        ),

        width_m=float(width_m),
        height_m=float(height_m),
    )


# ============================================================
# 8. Synthetic scene definitions
# ============================================================

def build_scene(scene_type: str):
    """
    Returns:

        plane_dict
        visible_in_frame_A
        visible_in_frame_B
    """

    # --------------------------------------------------------
    # 8.1 Distinct geometry
    # --------------------------------------------------------

    if scene_type == "distinct":

        planes = [

            make_plane(
                "floor",
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                6.0,
                6.0,
            ),

            make_plane(
                "wall_x",
                (-3.0, 0.0, 1.5),
                (1.0, 0.0, 0.0),
                6.0,
                3.0,
            ),

            make_plane(
                "wall_y",
                (0.0, 3.0, 1.5),
                (0.0, -1.0, 0.0),
                6.0,
                3.0,
            ),

            make_plane(
                "tilted_1",
                (1.0, -1.0, 1.8),
                (0.55, 0.25, 0.80),
                2.2,
                1.6,
            ),

            make_plane(
                "tilted_2",
                (-1.2, 1.0, 2.0),
                (-0.20, 0.90, 0.38),
                1.8,
                1.4,
            ),

            make_plane(
                "table",
                (0.4, 0.3, 0.85),
                (0.10, -0.18, 0.978),
                1.6,
                0.9,
            ),
        ]

        visible_a = [
            p.plane_id
            for p in planes
        ]

        visible_b = [
            p.plane_id
            for p in planes
        ]

    # --------------------------------------------------------
    # 8.2 Repetitive / ambiguous geometry
    # --------------------------------------------------------

    elif scene_type == "repetitive":

        planes = [

            make_plane(
                "floor",
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                8.0,
                8.0,
            ),

            make_plane(
                "ceiling",
                (0.0, 0.0, 3.0),
                (0.0, 0.0, -1.0),
                8.0,
                8.0,
            ),

            # Four parallel walls with identical orientation.
            make_plane(
                "wall_x_m3",
                (-3.0, 0.0, 1.5),
                (1.0, 0.0, 0.0),
                6.0,
                3.0,
            ),

            make_plane(
                "wall_x_m1",
                (-1.0, 0.0, 1.5),
                (1.0, 0.0, 0.0),
                6.0,
                3.0,
            ),

            make_plane(
                "wall_x_p1",
                (1.0, 0.0, 1.5),
                (1.0, 0.0, 0.0),
                6.0,
                3.0,
            ),

            make_plane(
                "wall_x_p3",
                (3.0, 0.0, 1.5),
                (1.0, 0.0, 0.0),
                6.0,
                3.0,
            ),

            make_plane(
                "wall_y_m3",
                (0.0, -3.0, 1.5),
                (0.0, 1.0, 0.0),
                6.0,
                3.0,
            ),

            make_plane(
                "wall_y_p3",
                (0.0, 3.0, 1.5),
                (0.0, 1.0, 0.0),
                6.0,
                3.0,
            ),
        ]

        visible_a = [
            p.plane_id
            for p in planes
        ]

        visible_b = [
            p.plane_id
            for p in planes
        ]

    # --------------------------------------------------------
    # 8.3 Partial visibility
    # --------------------------------------------------------

    elif scene_type == "partial_visibility":

        planes = [

            make_plane(
                "floor",
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                7.0,
                7.0,
            ),

            make_plane(
                "ceiling",
                (0.0, 0.0, 3.0),
                (0.0, 0.0, -1.0),
                7.0,
                7.0,
            ),

            make_plane(
                "left_wall",
                (-3.0, 0.0, 1.5),
                (1.0, 0.0, 0.0),
                6.0,
                3.0,
            ),

            make_plane(
                "right_wall",
                (3.0, 0.0, 1.5),
                (-1.0, 0.0, 0.0),
                6.0,
                3.0,
            ),

            make_plane(
                "front_wall",
                (0.0, 3.0, 1.5),
                (0.0, -1.0, 0.0),
                6.0,
                3.0,
            ),

            make_plane(
                "back_wall",
                (0.0, -3.0, 1.5),
                (0.0, 1.0, 0.0),
                6.0,
                3.0,
            ),

            make_plane(
                "table",
                (0.2, 0.2, 0.85),
                (0.0, 0.0, 1.0),
                1.8,
                1.0,
            ),

            make_plane(
                "divider",
                (1.2, -0.8, 1.4),
                (0.70, 0.10, 0.707),
                2.0,
                1.6,
            ),
        ]

        # Frame A and Frame B deliberately observe
        # different plane sets.
        visible_a = [
            "floor",
            "ceiling",
            "left_wall",
            "front_wall",
            "right_wall",
            "table",
        ]

        visible_b = [
            "floor",
            "ceiling",
            "left_wall",
            "front_wall",
            "back_wall",
            "divider",
        ]

    else:

        raise ValueError(
            f"Unknown scene type: {scene_type}"
        )

    plane_dict = {
        plane.plane_id: plane
        for plane in planes
    }

    return (
        plane_dict,
        visible_a,
        visible_b,
    )


# ============================================================
# 9. Generate true neighbouring-frame motion
# ============================================================

def generate_true_relative_pose(
    rng: np.random.Generator,
):
    """
    Frame A is treated as the reference/world frame.

    Convention:

        x_A = R_ab x_B + t_ab

    Typical neighbouring-frame motion:
        rotation = several degrees
        translation = 0.1–0.4 m
    """

    roll = math.radians(
        rng.uniform(-3.0, 3.0)
    )

    pitch = math.radians(
        rng.uniform(-3.0, 3.0)
    )

    yaw = math.radians(
        rng.uniform(-8.0, 8.0)
    )

    R_ab = rotation_matrix_xyz(
        roll,
        pitch,
        yaw,
    )

    direction = random_unit_vector(
        rng
    )

    distance = rng.uniform(
        0.10,
        0.40,
    )

    t_ab = direction * distance

    return R_ab, t_ab


# ============================================================
# 10. World plane -> camera-frame plane
# ============================================================

def transform_world_plane_to_camera(
    plane: WorldPlane,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
):
    """
    Camera pose:

        x_w = R_wc x_c + t_wc

    World plane:

        n_w^T x_w + d_w = 0

    Camera-frame plane:

        n_c = R_wc^T n_w

        d_c = d_w + n_w^T t_wc
    """

    n_c = normalize(
        R_wc.T @ plane.normal_w
    )

    d_c = float(
        plane.d_w
        + np.dot(
            plane.normal_w,
            t_wc,
        )
    )

    centroid_c = (
        R_wc.T
        @ (
            plane.centroid_w
            - t_wc
        )
    )

    return {
        "normal": n_c,
        "offset": d_c,
        "centroid": centroid_c,
        "width_m": plane.width_m,
        "height_m": plane.height_m,
        "area_m2": plane.area_m2,
    }


# ============================================================
# 11. Add synthetic plane measurement error
# ============================================================

def apply_measurement_noise(
    true_observation,
    rng: np.random.Generator,
    visibility_scale: float = 1.0,
):
    n_true = np.asarray(
        true_observation["normal"],
        dtype=float,
    )

    d_true = float(
        true_observation["offset"]
    )

    centroid_true = np.asarray(
        true_observation["centroid"],
        dtype=float,
    )

    # --------------------------------------------------------
    # Normal noise
    # --------------------------------------------------------

    n_measured = perturb_normal(
        n_true,
        NORMAL_NOISE_STD_DEG,
        rng,
    )

    # --------------------------------------------------------
    # Offset noise
    # --------------------------------------------------------

    d_measured = (
        d_true
        + rng.normal(
            0.0,
            OFFSET_NOISE_STD_M,
        )
    )

    # --------------------------------------------------------
    # Centroid noise
    # --------------------------------------------------------

    centroid_measured = (
        centroid_true
        + rng.normal(
            0.0,
            CENTROID_NOISE_STD_M,
            size=3,
        )
    )

    # Project noisy centroid back onto the measured plane.
    signed_distance = float(
        np.dot(
            n_measured,
            centroid_measured,
        )
        + d_measured
    )

    centroid_measured = (
        centroid_measured
        - signed_distance * n_measured
    )

    # --------------------------------------------------------
    # Visible size / area noise
    # --------------------------------------------------------

    scale_sqrt = math.sqrt(
        visibility_scale
    )

    width_true = (
        float(
            true_observation["width_m"]
        )
        * scale_sqrt
    )

    height_true = (
        float(
            true_observation["height_m"]
        )
        * scale_sqrt
    )

    width_measured = (
        width_true
        * max(
            0.20,
            1.0
            + rng.normal(
                0.0,
                SIZE_RELATIVE_NOISE_STD,
            ),
        )
    )

    height_measured = (
        height_true
        * max(
            0.20,
            1.0
            + rng.normal(
                0.0,
                SIZE_RELATIVE_NOISE_STD,
            ),
        )
    )

    area_measured = (
        width_measured
        * height_measured
    )

    # --------------------------------------------------------
    # Random sign flip
    # --------------------------------------------------------

    if (
        rng.random()
        < NORMAL_SIGN_FLIP_PROB
    ):

        n_measured = -n_measured
        d_measured = -d_measured

    return {
        "normal": n_measured,
        "offset": float(
            d_measured
        ),
        "centroid": centroid_measured,
        "width_m": float(
            width_measured
        ),
        "height_m": float(
            height_measured
        ),
        "area_m2": float(
            area_measured
        ),
    }


# ============================================================
# 12. Generate imperfect pose prior
# ============================================================

def generate_pose_prior(
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    prior_level: str,
    rng: np.random.Generator,
):
    settings = PRIOR_LEVELS[
        prior_level
    ]

    rotation_error_deg = float(
        settings[
            "rotation_error_deg"
        ]
    )

    translation_error_m = float(
        settings[
            "translation_error_m"
        ]
    )

    # --------------------------------------------------------
    # Rotation error
    # --------------------------------------------------------

    if rotation_error_deg > 0.0:

        error_axis = random_unit_vector(
            rng
        )

        R_error = (
            rotation_matrix_axis_angle(
                error_axis,
                math.radians(
                    rotation_error_deg
                ),
            )
        )

        R_prior = (
            R_error
            @ R_gt
        )

    else:

        R_prior = R_gt.copy()

    # --------------------------------------------------------
    # Translation error
    # --------------------------------------------------------

    if translation_error_m > 0.0:

        error_direction = (
            random_unit_vector(
                rng
            )
        )

        translation_error = (
            error_direction
            * translation_error_m
        )

        t_prior = (
            t_gt
            + translation_error
        )

    else:

        t_prior = t_gt.copy()

    return R_prior, t_prior


# ============================================================
# 13. Generate observations for one frame
# ============================================================

def make_frame_observations(
    scene_planes,
    visible_ids,
    R_wc,
    t_wc,
    scene_type,
    frame_name,
    rng,
):
    observations = []

    for plane_id in visible_ids:

        plane = scene_planes[
            plane_id
        ]

        true_observation = (
            transform_world_plane_to_camera(
                plane,
                R_wc,
                t_wc,
            )
        )

        visibility_scale = 1.0

        # In partial-visibility scenes,
        # the visible part of a physical plane
        # may change between frames.
        if (
            scene_type
            == "partial_visibility"
            and frame_name == "B"
        ):

            visibility_scale = rng.uniform(
                0.55,
                1.0,
            )

        measured = apply_measurement_noise(
            true_observation,
            rng,
            visibility_scale,
        )

        observations.append(
            {
                "obs_id": "",

                # Hidden ground truth.
                # Similarity algorithms must NOT use this.
                "gt_plane_id": plane_id,

                "normal": np.asarray(
                    measured["normal"]
                ).tolist(),

                "offset": float(
                    measured["offset"]
                ),

                "centroid": np.asarray(
                    measured["centroid"]
                ).tolist(),

                "width_m": float(
                    measured["width_m"]
                ),

                "height_m": float(
                    measured["height_m"]
                ),

                "area_m2": float(
                    measured["area_m2"]
                ),
            }
        )

    # Shuffle ordering so matching cannot be solved
    # using list indices.
    permutation = rng.permutation(
        len(observations)
    )

    observations = [
        observations[i]
        for i in permutation
    ]

    for index, observation in enumerate(
        observations
    ):

        observation["obs_id"] = (
            f"{frame_name}{index}"
        )

    return observations


# ============================================================
# 14. Ground-truth correspondence list
# ============================================================

def build_gt_matches(
    frame_a,
    frame_b,
):
    b_by_plane_id = {
        plane["gt_plane_id"]:
            plane
        for plane in frame_b
    }

    matches = []

    for plane_a in frame_a:

        gt_id = plane_a[
            "gt_plane_id"
        ]

        if gt_id in b_by_plane_id:

            plane_b = b_by_plane_id[
                gt_id
            ]

            matches.append(
                {
                    "gt_plane_id":
                        gt_id,

                    "a_obs_id":
                        plane_a[
                            "obs_id"
                        ],

                    "b_obs_id":
                        plane_b[
                            "obs_id"
                        ],
                }
            )

    return matches


# ============================================================
# 15. Plane transform B -> A
# ============================================================

def transform_plane_b_to_a(
    normal_b,
    offset_b,
    centroid_b,
    R_ab,
    t_ab,
):
    """
    Transform plane parameters from Frame B
    into Frame A.
    """

    normal_a = normalize(
        R_ab @ normal_b
    )

    offset_a = float(
        offset_b
        - np.dot(
            normal_a,
            t_ab,
        )
    )

    centroid_a = (
        R_ab @ centroid_b
        + t_ab
    )

    return (
        normal_a,
        offset_a,
        centroid_a,
    )


# ============================================================
# 16. Coordinate transformation sanity check
# ============================================================

def run_transform_sanity_check(
    scene_planes,
    visible_a,
    visible_b,
    R_ab,
    t_ab,
):
    """
    Before measurement noise is added, the same physical
    plane observed in B and transformed back into A must
    match the A-frame plane almost exactly.
    """

    common_ids = sorted(
        set(visible_a)
        & set(visible_b)
    )

    # Frame A is world/reference.
    R_wa = np.eye(3)
    t_wa = np.zeros(3)

    # Frame B pose in world coordinates.
    R_wb = R_ab
    t_wb = t_ab

    for plane_id in common_ids:

        plane = scene_planes[
            plane_id
        ]

        obs_a = (
            transform_world_plane_to_camera(
                plane,
                R_wa,
                t_wa,
            )
        )

        obs_b = (
            transform_world_plane_to_camera(
                plane,
                R_wb,
                t_wb,
            )
        )

        (
            normal_back,
            offset_back,
            centroid_back,
        ) = transform_plane_b_to_a(
            np.asarray(
                obs_b["normal"]
            ),
            float(
                obs_b["offset"]
            ),
            np.asarray(
                obs_b["centroid"]
            ),
            R_ab,
            t_ab,
        )

        normal_a = np.asarray(
            obs_a["normal"]
        )

        offset_a = float(
            obs_a["offset"]
        )

        centroid_a = np.asarray(
            obs_a["centroid"]
        )

        # Resolve sign ambiguity.
        if (
            np.dot(
                normal_a,
                normal_back,
            )
            < 0.0
        ):

            normal_back = (
                -normal_back
            )

            offset_back = (
                -offset_back
            )

        angle_error = (
            angle_between_normals_deg(
                normal_a,
                normal_back,
            )
        )

        offset_error = abs(
            offset_a
            - offset_back
        )

        centroid_error = (
            np.linalg.norm(
                centroid_a
                - centroid_back
            )
        )

        if (
            angle_error > 1e-6
            or offset_error > 1e-8
            or centroid_error > 1e-8
        ):

            raise RuntimeError(
                "Plane transformation sanity check failed.\n"
                f"Plane: {plane_id}\n"
                f"Normal error: {angle_error}\n"
                f"Offset error: {offset_error}\n"
                f"Centroid error: {centroid_error}"
            )


# ============================================================
# 17. JSON transform helper
# ============================================================

def transform_to_json(
    R,
    t,
):
    return {
        "R": np.asarray(
            R,
            dtype=float,
        ).tolist(),

        "t": np.asarray(
            t,
            dtype=float,
        ).tolist(),
    }


# ============================================================
# 18. Generate one benchmark case
# ============================================================

def generate_case(
    split,
    scene_type,
    prior_level,
    repetition,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    (
        scene_planes,
        visible_a,
        visible_b,
    ) = build_scene(
        scene_type
    )

    # True transformation B -> A.
    R_gt, t_gt = (
        generate_true_relative_pose(
            rng
        )
    )

    # Verify plane-coordinate mathematics.
    run_transform_sanity_check(
        scene_planes,
        visible_a,
        visible_b,
        R_gt,
        t_gt,
    )

    # --------------------------------------------------------
    # Frame A
    # --------------------------------------------------------

    R_wa = np.eye(3)
    t_wa = np.zeros(3)

    # --------------------------------------------------------
    # Frame B
    # --------------------------------------------------------

    R_wb = R_gt
    t_wb = t_gt

    frame_a = (
        make_frame_observations(
            scene_planes,
            visible_a,
            R_wa,
            t_wa,
            scene_type,
            "A",
            rng,
        )
    )

    frame_b = (
        make_frame_observations(
            scene_planes,
            visible_b,
            R_wb,
            t_wb,
            scene_type,
            "B",
            rng,
        )
    )

    gt_matches = build_gt_matches(
        frame_a,
        frame_b,
    )

    # The similarity algorithm receives this imperfect prior.
    R_prior, t_prior = (
        generate_pose_prior(
            R_gt,
            t_gt,
            prior_level,
            rng,
        )
    )

    sample_id = (
        f"{split.upper()}_"
        f"{scene_type.upper()}_"
        f"{prior_level.upper()}_"
        f"R{repetition:02d}"
    )

    ids_a = {
        p["gt_plane_id"]
        for p in frame_a
    }

    ids_b = {
        p["gt_plane_id"]
        for p in frame_b
    }

    unmatched_a = [
        p["obs_id"]
        for p in frame_a
        if p["gt_plane_id"]
        not in ids_b
    ]

    unmatched_b = [
        p["obs_id"]
        for p in frame_b
        if p["gt_plane_id"]
        not in ids_a
    ]

    return {
        "sample_id": sample_id,

        "split": split,
        "scene_type": scene_type,
        "prior_level": prior_level,
        "repetition": repetition,
        "seed": seed,

        "measurement_noise": {
            "normal_noise_std_deg":
                NORMAL_NOISE_STD_DEG,

            "offset_noise_std_m":
                OFFSET_NOISE_STD_M,

            "centroid_noise_std_m":
                CENTROID_NOISE_STD_M,

            "size_relative_noise_std":
                SIZE_RELATIVE_NOISE_STD,

            "normal_sign_flip_probability":
                NORMAL_SIGN_FLIP_PROB,
        },

        "pose_prior_target_error":
            PRIOR_LEVELS[
                prior_level
            ],

        # Ground truth only.
        "gt_transform_b_to_a":
            transform_to_json(
                R_gt,
                t_gt,
            ),

        # Similarity algorithms must use this.
        "prior_transform_b_to_a":
            transform_to_json(
                R_prior,
                t_prior,
            ),

        "frame_a": {
            "num_planes":
                len(frame_a),

            "planes":
                frame_a,
        },

        "frame_b": {
            "num_planes":
                len(frame_b),

            "planes":
                frame_b,
        },

        # Ground truth only.
        "gt_matches":
            gt_matches,

        "num_gt_matches":
            len(gt_matches),

        "unmatched_a_obs_ids":
            unmatched_a,

        "unmatched_b_obs_ids":
            unmatched_b,
    }


# ============================================================
# 19. Generate one split
# ============================================================

def generate_split(
    split,
    output_dir,
    repeats_per_condition,
    seed_offset,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if CLEAN_OUTPUT:

        for old_file in output_dir.glob(
            "*.json"
        ):
            old_file.unlink()

    rows = []

    total_cases = (
        len(SCENE_TYPES)
        * len(PRIOR_LEVELS)
        * repeats_per_condition
    )

    counter = 0

    for scene_type in SCENE_TYPES:

        for prior_level in PRIOR_LEVELS:

            for repetition in range(
                1,
                repeats_per_condition + 1,
            ):

                seed = (
                    BASE_SEED
                    + seed_offset
                    + counter
                )

                payload = generate_case(
                    split=split,
                    scene_type=scene_type,
                    prior_level=prior_level,
                    repetition=repetition,
                    seed=seed,
                )

                filename = (
                    f"{payload['sample_id']}.json"
                )

                filepath = (
                    output_dir
                    / filename
                )

                with filepath.open(
                    "w",
                    encoding="utf-8",
                ) as file:

                    json.dump(
                        payload,
                        file,
                        indent=2,
                        ensure_ascii=False,
                    )

                rows.append(
                    {
                        "sample_id":
                            payload[
                                "sample_id"
                            ],

                        "split":
                            split,

                        "scene_type":
                            scene_type,

                        "prior_level":
                            prior_level,

                        "repetition":
                            repetition,

                        "seed":
                            seed,

                        "num_planes_a":
                            payload[
                                "frame_a"
                            ][
                                "num_planes"
                            ],

                        "num_planes_b":
                            payload[
                                "frame_b"
                            ][
                                "num_planes"
                            ],

                        "num_gt_matches":
                            payload[
                                "num_gt_matches"
                            ],

                        "num_unmatched_a":
                            len(
                                payload[
                                    "unmatched_a_obs_ids"
                                ]
                            ),

                        "num_unmatched_b":
                            len(
                                payload[
                                    "unmatched_b_obs_ids"
                                ]
                            ),

                        "filename":
                            filename,
                    }
                )

                counter += 1

                print(
                    f"[{split}] "
                    f"{counter:03d}/"
                    f"{total_cases:03d} "
                    f"{payload['sample_id']}"
                )

    return rows


# ============================================================
# 20. Write manifest
# ============================================================

def write_manifest(
    rows,
    path,
):
    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 21. Main
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print(
        "Generating synthetic "
        "plane-similarity benchmark"
    )

    print(
        f"Output: {OUTPUT_ROOT}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Development set
    # --------------------------------------------------------

    dev_rows = generate_split(
        split="dev",

        output_dir=DEV_DIR,

        repeats_per_condition=
            DEV_REPEATS_PER_CONDITION,

        seed_offset=0,
    )

    # --------------------------------------------------------
    # Final benchmark
    # --------------------------------------------------------

    final_rows = generate_split(
        split="final",

        output_dir=FINAL_DIR,

        repeats_per_condition=
            FINAL_REPEATS_PER_CONDITION,

        seed_offset=10000,
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    all_rows = (
        dev_rows
        + final_rows
    )

    manifest_path = (
        OUTPUT_ROOT
        / "manifest.csv"
    )

    write_manifest(
        all_rows,
        manifest_path,
    )

    expected_dev = (
        len(SCENE_TYPES)
        * len(PRIOR_LEVELS)
        * DEV_REPEATS_PER_CONDITION
    )

    expected_final = (
        len(SCENE_TYPES)
        * len(PRIOR_LEVELS)
        * FINAL_REPEATS_PER_CONDITION
    )

    assert (
        len(dev_rows)
        == expected_dev
    )

    assert (
        len(final_rows)
        == expected_final
    )

    print()
    print("=" * 70)
    print("Generation complete")

    print(
        f"Development cases : "
        f"{len(dev_rows)}"
    )

    print(
        f"Final cases       : "
        f"{len(final_rows)}"
    )

    print(
        f"Manifest          : "
        f"{manifest_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()