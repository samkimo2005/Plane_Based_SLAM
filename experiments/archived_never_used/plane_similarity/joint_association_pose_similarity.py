from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scipy.optimize import (
    least_squares,
    linear_sum_assignment,
)

from scipy.spatial.transform import Rotation

from similarity_common import (
    load_similarity_case,
)

from similarity_evaluation import (
    evaluate_predicted_pairs,
)


# ============================================================
# 1. Parameters
# ============================================================

@dataclass
class JointAssociationPoseParameters:
    """
    Project-specific simplified Joint Association
    + Pose Refinement baseline.

    This is NOT an exact reproduction of CPA-SLAM.

    Procedure:

        current pose prior
              ↓
        plane-pair cost matrix
              ↓
        one-to-one association
              ↓
        pose refinement
              ↓
        rebuild cost matrix
              ↓
        repeat until stable

    These are initial DEV parameters only.
    """

    # --------------------------------------------------------
    # Association cost weights
    # --------------------------------------------------------

    normal_weight: float = 0.45
    offset_weight: float = 0.35
    centroid_weight: float = 0.15
    area_weight: float = 0.05

    # --------------------------------------------------------
    # Association normalisation scales
    # --------------------------------------------------------

    normal_scale_deg: float = 15.0

    offset_scale_m: float = 0.20

    centroid_scale_m: float = 1.00

    area_scale_ratio: float = 0.50

    # --------------------------------------------------------
    # Cost of leaving an A-plane unmatched
    # --------------------------------------------------------

    unmatched_cost: float = 1.00

    # --------------------------------------------------------
    # Pose refinement scales
    # --------------------------------------------------------

    pose_normal_scale_deg: float = 5.0

    pose_offset_scale_m: float = 0.05

    pose_centroid_scale_m: float = 0.25

    # Centroids are less reliable under partial visibility,
    # so centroid residuals receive less influence.
    pose_centroid_weight: float = 0.25

    # --------------------------------------------------------
    # Robust nonlinear optimisation
    # --------------------------------------------------------

    robust_loss: str = "soft_l1"

    robust_f_scale: float = 1.0

    # --------------------------------------------------------
    # Joint iterations
    # --------------------------------------------------------

    max_joint_iterations: int = 8

    min_matches_for_pose: int = 3

    # --------------------------------------------------------
    # Convergence criteria
    # --------------------------------------------------------

    rotation_convergence_deg: float = 0.01

    translation_convergence_m: float = 0.0005

    # --------------------------------------------------------
    # Safety bounds for ONE pose refinement step
    # --------------------------------------------------------

    max_rotation_update_deg: float = 10.0

    max_translation_update_m: float = 0.25


# ============================================================
# 2. Basic mathematical helpers
# ============================================================

def normalize_vector(
    vector,
):
    vector = np.asarray(
        vector,
        dtype=float,
    )

    norm = np.linalg.norm(
        vector
    )

    if norm < 1e-12:

        raise ValueError(
            "Cannot normalise a near-zero vector."
        )

    return (
        vector
        / norm
    )


def safe_acos_deg(
    value,
):
    value = np.clip(
        value,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(
                value
            )
        )
    )


# ============================================================
# 3. Plane-data access helpers
#
# IMPORTANT:
#
# These helpers prevent the Joint method from assuming
# that the benchmark always uses one exact key name.
# ============================================================

def get_plane_normal(
    plane,
):
    """
    Read a plane normal.
    """

    possible_keys = [
        "normal",
        "normal_vector",
        "normal_xyz",
    ]

    for key in possible_keys:

        if key in plane:

            return normalize_vector(
                plane[
                    key
                ]
            )

    raise KeyError(
        "Cannot find plane normal.\n"
        f"Available keys: {list(plane.keys())}"
    )


def get_plane_offset(
    plane,
):
    """
    Read plane offset from:

        n^T x + d = 0

    Different earlier scripts may call this:

        offset_m
        offset
        d
    """

    possible_keys = [
        "offset_m",
        "offset",
        "d",
    ]

    for key in possible_keys:

        if key in plane:

            return float(
                plane[
                    key
                ]
            )

    raise KeyError(
        "Cannot find plane offset.\n"
        f"Available keys: {list(plane.keys())}"
    )


def get_plane_centroid(
    plane,
):
    """
    Read plane centroid.
    """

    possible_keys = [
        "centroid",
        "centroid_m",
        "centre",
        "center",
    ]

    for key in possible_keys:

        if key in plane:

            centroid = np.asarray(
                plane[
                    key
                ],
                dtype=float,
            )

            return centroid.reshape(
                3,
            )

    raise KeyError(
        "Cannot find plane centroid.\n"
        f"Available keys: {list(plane.keys())}"
    )


def get_plane_area(
    plane,
):
    """
    Read visible plane area.

    If area_m2 is not explicitly stored but width and
    height are available, calculate area from them.
    """

    possible_keys = [
        "area_m2",
        "area",
    ]

    for key in possible_keys:

        if key in plane:

            return max(
                float(
                    plane[
                        key
                    ]
                ),
                1e-12,
            )

    if (
        "width_m" in plane
        and "height_m" in plane
    ):

        return max(
            float(
                plane[
                    "width_m"
                ]
            )
            * float(
                plane[
                    "height_m"
                ]
            ),
            1e-12,
        )

    raise KeyError(
        "Cannot find plane area.\n"
        f"Available keys: {list(plane.keys())}"
    )


def get_plane_obs_id(
    plane,
):
    """
    Read observation ID used by the common evaluator.
    """

    possible_keys = [
        "obs_id",
        "observation_id",
        "id",
    ]

    for key in possible_keys:

        if key in plane:

            return plane[
                key
            ]

    raise KeyError(
        "Cannot find plane observation ID.\n"
        f"Available keys: {list(plane.keys())}"
    )


# ============================================================
# 4. Transform-data access
# ============================================================

def parse_transform(
    transform_data,
):
    """
    Convert stored transform data into:

        R : 3 x 3 rotation matrix
        t : 3-vector translation

    Several common representations are accepted.
    """

    # --------------------------------------------------------
    # Direct 4x4 matrix
    # --------------------------------------------------------

    if isinstance(
        transform_data,
        (
            list,
            tuple,
            np.ndarray,
        ),
    ):

        matrix = np.asarray(
            transform_data,
            dtype=float,
        )

        if matrix.shape == (
            4,
            4,
        ):

            return (
                matrix[
                    :3,
                    :3
                ],
                matrix[
                    :3,
                    3
                ],
            )

    # --------------------------------------------------------
    # Dictionary representation
    # --------------------------------------------------------

    if isinstance(
        transform_data,
        dict,
    ):

        # ----------------------------------------------
        # Full 4x4 matrix stored under a key
        # ----------------------------------------------

        for key in [
            "matrix",
            "transform",
            "T",
            "T_b_to_a",
        ]:

            if key in transform_data:

                matrix = np.asarray(
                    transform_data[
                        key
                    ],
                    dtype=float,
                )

                if matrix.shape == (
                    4,
                    4,
                ):

                    return (
                        matrix[
                            :3,
                            :3
                        ],
                        matrix[
                            :3,
                            3
                        ],
                    )

        # ----------------------------------------------
        # Rotation + translation
        # ----------------------------------------------

        rotation = None
        translation = None

        rotation_keys = [
            "rotation",
            "rotation_matrix",
            "R",
        ]

        translation_keys = [
            "translation",
            "translation_m",
            "t",
        ]

        for key in rotation_keys:

            if key in transform_data:

                rotation = np.asarray(
                    transform_data[
                        key
                    ],
                    dtype=float,
                )

                break

        for key in translation_keys:

            if key in transform_data:

                translation = np.asarray(
                    transform_data[
                        key
                    ],
                    dtype=float,
                )

                break

        if (
            rotation is not None
            and translation is not None
        ):

            return (
                rotation.reshape(
                    3,
                    3,
                ),

                translation.reshape(
                    3,
                ),
            )

    raise ValueError(
        "Unsupported transform format.\n"
        f"Transform data: {transform_data}"
    )


def make_transform_matrix(
    rotation,
    translation,
):
    """
    Build a conventional 4x4 rigid transformation matrix.
    """

    matrix = np.eye(
        4,
        dtype=float,
    )

    matrix[
        :3,
        :3
    ] = rotation

    matrix[
        :3,
        3
    ] = translation

    return matrix


# ============================================================
# 5. Transform one Frame-B plane into Frame A
# ============================================================

def transform_plane_b_to_a(
    plane_b,
    rotation_b_to_a,
    translation_b_to_a,
):
    """
    Point transformation:

        x_A = R x_B + t

    Plane equation in B:

        n_B^T x_B + d_B = 0

    transforms to:

        n_A = R n_B

        d_A = d_B - n_A^T t
    """

    normal_b = get_plane_normal(
        plane_b
    )

    offset_b = get_plane_offset(
        plane_b
    )

    centroid_b = get_plane_centroid(
        plane_b
    )

    area_b = get_plane_area(
        plane_b
    )

    normal_a = (
        rotation_b_to_a
        @ normal_b
    )

    normal_a = normalize_vector(
        normal_a
    )

    offset_a = (
        offset_b
        - float(
            np.dot(
                normal_a,
                translation_b_to_a,
            )
        )
    )

    centroid_a = (
        rotation_b_to_a
        @ centroid_b
        + translation_b_to_a
    )

    return {
        "normal":
            normal_a,

        # Standard internal representation for THIS file.
        "offset_m":
            float(
                offset_a
            ),

        "centroid":
            centroid_a,

        "area_m2":
            area_b,
    }


# ============================================================
# 6. Plane sign alignment
# ============================================================

def align_plane_sign(
    reference_plane,
    transformed_plane,
):
    """
    A plane represented by:

        (n, d)

    is identical to:

        (-n, -d)

    Therefore align transformed plane B with plane A
    before comparing offset values.
    """

    normal_a = get_plane_normal(
        reference_plane
    )

    normal_b = get_plane_normal(
        transformed_plane
    )

    offset_b = get_plane_offset(
        transformed_plane
    )

    if np.dot(
        normal_a,
        normal_b,
    ) < 0:

        normal_b = (
            -normal_b
        )

        offset_b = (
            -offset_b
        )

    return {
        "normal":
            normal_b,

        "offset_m":
            float(
                offset_b
            ),

        "centroid":
            get_plane_centroid(
                transformed_plane
            ),

        "area_m2":
            get_plane_area(
                transformed_plane
            ),
    }


# ============================================================
# 7. Pairwise geometry under current pose
# ============================================================

def calculate_pair_geometry(
    plane_a,
    plane_b,
    rotation_b_to_a,
    translation_b_to_a,
):
    transformed_b = (
        transform_plane_b_to_a(
            plane_b,
            rotation_b_to_a,
            translation_b_to_a,
        )
    )

    aligned_b = (
        align_plane_sign(
            plane_a,
            transformed_b,
        )
    )

    # --------------------------------------------------------
    # Normal angle
    # --------------------------------------------------------

    normal_a = get_plane_normal(
        plane_a
    )

    normal_b = get_plane_normal(
        aligned_b
    )

    dot_value = np.clip(
        abs(
            float(
                np.dot(
                    normal_a,
                    normal_b,
                )
            )
        ),
        0.0,
        1.0,
    )

    normal_angle_deg = (
        safe_acos_deg(
            dot_value
        )
    )

    # --------------------------------------------------------
    # Plane offset
    # --------------------------------------------------------

    offset_difference_m = abs(
        get_plane_offset(
            plane_a
        )
        - get_plane_offset(
            aligned_b
        )
    )

    # --------------------------------------------------------
    # Centroid
    # --------------------------------------------------------

    centroid_distance_m = float(
        np.linalg.norm(
            get_plane_centroid(
                plane_a
            )
            - get_plane_centroid(
                aligned_b
            )
        )
    )

    # --------------------------------------------------------
    # Visible area
    # --------------------------------------------------------

    area_a = get_plane_area(
        plane_a
    )

    area_b = get_plane_area(
        aligned_b
    )

    area_difference_ratio = (
        abs(
            area_a
            - area_b
        )
        / max(
            area_a,
            area_b,
        )
    )

    return {
        "normal_angle_deg":
            float(
                normal_angle_deg
            ),

        "offset_difference_m":
            float(
                offset_difference_m
            ),

        "centroid_distance_m":
            float(
                centroid_distance_m
            ),

        "area_difference_ratio":
            float(
                area_difference_ratio
            ),
    }


# ============================================================
# 8. Association cost for one plane pair
# ============================================================

def calculate_joint_pair_cost(
    plane_a,
    plane_b,
    rotation_b_to_a,
    translation_b_to_a,
    params,
):
    geometry = (
        calculate_pair_geometry(
            plane_a,
            plane_b,
            rotation_b_to_a,
            translation_b_to_a,
        )
    )

    normal_term = (
        geometry[
            "normal_angle_deg"
        ]
        / params.normal_scale_deg
    )

    offset_term = (
        geometry[
            "offset_difference_m"
        ]
        / params.offset_scale_m
    )

    centroid_term = (
        geometry[
            "centroid_distance_m"
        ]
        / params.centroid_scale_m
    )

    area_term = (
        geometry[
            "area_difference_ratio"
        ]
        / params.area_scale_ratio
    )

    cost = (
        params.normal_weight
        * normal_term

        + params.offset_weight
        * offset_term

        + params.centroid_weight
        * centroid_term

        + params.area_weight
        * area_term
    )

    return {
        **geometry,

        "joint_cost":
            float(
                cost
            ),
    }


# ============================================================
# 9. Build complete cost matrix
# ============================================================

def build_joint_cost_matrix(
    frame_a_planes,
    frame_b_planes,
    rotation_b_to_a,
    translation_b_to_a,
    params,
):
    num_a = len(
        frame_a_planes
    )

    num_b = len(
        frame_b_planes
    )

    cost_matrix = np.zeros(
        (
            num_a,
            num_b,
        ),
        dtype=float,
    )

    pair_details = {}

    for i, plane_a in enumerate(
        frame_a_planes
    ):

        for j, plane_b in enumerate(
            frame_b_planes
        ):

            result = (
                calculate_joint_pair_cost(
                    plane_a,
                    plane_b,
                    rotation_b_to_a,
                    translation_b_to_a,
                    params,
                )
            )

            cost_matrix[
                i,
                j
            ] = result[
                "joint_cost"
            ]

            pair_details[
                (
                    i,
                    j
                )
            ] = result

    return (
        cost_matrix,
        pair_details,
    )


# ============================================================
# 10. One-to-one association
# ============================================================

def solve_assignment(
    frame_a_planes,
    frame_b_planes,
    cost_matrix,
    pair_details,
    params,
):
    """
    Hungarian assignment is used internally by this
    JOINT method.

    Dummy columns allow A-planes to remain unmatched.

    This Hungarian step is NOT the later independent
    Assignment-module comparison.
    """

    num_a = len(
        frame_a_planes
    )

    num_b = len(
        frame_b_planes
    )

    # --------------------------------------------------------
    # One dummy unmatched column per A-plane
    # --------------------------------------------------------

    dummy_matrix = np.full(
        (
            num_a,
            num_a,
        ),
        params.unmatched_cost,
        dtype=float,
    )

    augmented_matrix = np.hstack(
        [
            cost_matrix,
            dummy_matrix,
        ]
    )

    row_indices, column_indices = (
        linear_sum_assignment(
            augmented_matrix
        )
    )

    matches = []

    for row, column in zip(
        row_indices,
        column_indices,
    ):

        # Dummy assignment = unmatched
        if column >= num_b:
            continue

        pair_cost = float(
            cost_matrix[
                row,
                column
            ]
        )

        # Matching must be preferable to leaving
        # the A-plane unmatched.
        if (
            pair_cost
            >= params.unmatched_cost
        ):
            continue

        plane_a = (
            frame_a_planes[
                row
            ]
        )

        plane_b = (
            frame_b_planes[
                column
            ]
        )

        details = (
            pair_details[
                (
                    row,
                    column
                )
            ]
        )

        matches.append(
            {
                "a_index":
                    int(
                        row
                    ),

                "b_index":
                    int(
                        column
                    ),

                "a_obs_id":
                    get_plane_obs_id(
                        plane_a
                    ),

                "b_obs_id":
                    get_plane_obs_id(
                        plane_b
                    ),

                **details,
            }
        )

    return matches


# ============================================================
# 11. Pose-update composition
# ============================================================

def compose_pose_increment(
    base_rotation,
    base_translation,
    increment,
):
    """
    Apply a small left-multiplicative pose correction:

        T_new = Delta_T * T_current
    """

    rotation_vector = np.asarray(
        increment[
            :3
        ],
        dtype=float,
    )

    translation_increment = np.asarray(
        increment[
            3:
        ],
        dtype=float,
    )

    rotation_increment = (
        Rotation.from_rotvec(
            rotation_vector
        )
        .as_matrix()
    )

    new_rotation = (
        rotation_increment
        @ base_rotation
    )

    new_translation = (
        rotation_increment
        @ base_translation
        + translation_increment
    )

    return (
        new_rotation,
        new_translation,
    )


# ============================================================
# 12. Pose optimisation residuals
# ============================================================

def pose_residual_vector(
    increment,
    frame_a_planes,
    frame_b_planes,
    matches,
    base_rotation,
    base_translation,
    params,
):
    (
        rotation,
        translation,
    ) = compose_pose_increment(
        base_rotation,
        base_translation,
        increment,
    )

    residuals = []

    normal_scale_rad = max(
        np.deg2rad(
            params.pose_normal_scale_deg
        ),
        1e-12,
    )

    for match in matches:

        plane_a = (
            frame_a_planes[
                match[
                    "a_index"
                ]
            ]
        )

        plane_b = (
            frame_b_planes[
                match[
                    "b_index"
                ]
            ]
        )

        transformed_b = (
            transform_plane_b_to_a(
                plane_b,
                rotation,
                translation,
            )
        )

        aligned_b = (
            align_plane_sign(
                plane_a,
                transformed_b,
            )
        )

        # ----------------------------------------------------
        # A. Normal residual
        # ----------------------------------------------------

        normal_a = get_plane_normal(
            plane_a
        )

        normal_b = get_plane_normal(
            aligned_b
        )

        normal_residual = (
            normal_a
            - normal_b
        ) / normal_scale_rad

        residuals.extend(
            normal_residual.tolist()
        )

        # ----------------------------------------------------
        # B. Offset residual
        # ----------------------------------------------------

        offset_residual = (
            get_plane_offset(
                plane_a
            )
            - get_plane_offset(
                aligned_b
            )
        ) / params.pose_offset_scale_m

        residuals.append(
            float(
                offset_residual
            )
        )

        # ----------------------------------------------------
        # C. Centroid residual
        #
        # Lower weight because partial visibility may move
        # the observed centroid even for the same plane.
        # ----------------------------------------------------

        centroid_a = (
            get_plane_centroid(
                plane_a
            )
        )

        centroid_b = (
            get_plane_centroid(
                aligned_b
            )
        )

        centroid_residual = (
            (
                centroid_a
                - centroid_b
            )
            / params.pose_centroid_scale_m
        )

        centroid_residual *= np.sqrt(
            params.pose_centroid_weight
        )

        residuals.extend(
            centroid_residual.tolist()
        )

    return np.asarray(
        residuals,
        dtype=float,
    )


# ============================================================
# 13. Refine current pose
# ============================================================

def refine_pose_from_matches(
    frame_a_planes,
    frame_b_planes,
    matches,
    current_rotation,
    current_translation,
    params,
):
    if len(
        matches
    ) < params.min_matches_for_pose:

        return {
            "success":
                False,

            "reason":
                "insufficient_matches",

            "rotation":
                current_rotation,

            "translation":
                current_translation,

            "rotation_update_deg":
                0.0,

            "translation_update_m":
                0.0,

            "optimisation_cost":
                None,
        }

    initial_increment = np.zeros(
        6,
        dtype=float,
    )

    max_rotation_rad = np.deg2rad(
        params.max_rotation_update_deg
    )

    lower_bounds = np.array(
        [
            -max_rotation_rad,
            -max_rotation_rad,
            -max_rotation_rad,

            -params.max_translation_update_m,
            -params.max_translation_update_m,
            -params.max_translation_update_m,
        ],
        dtype=float,
    )

    upper_bounds = np.array(
        [
            max_rotation_rad,
            max_rotation_rad,
            max_rotation_rad,

            params.max_translation_update_m,
            params.max_translation_update_m,
            params.max_translation_update_m,
        ],
        dtype=float,
    )

    result = least_squares(
        pose_residual_vector,
        initial_increment,

        bounds=(
            lower_bounds,
            upper_bounds,
        ),

        args=(
            frame_a_planes,
            frame_b_planes,
            matches,
            current_rotation,
            current_translation,
            params,
        ),

        loss=params.robust_loss,

        f_scale=params.robust_f_scale,
    )

    (
        refined_rotation,
        refined_translation,
    ) = compose_pose_increment(
        current_rotation,
        current_translation,
        result.x,
    )

    rotation_update_deg = float(
        np.degrees(
            np.linalg.norm(
                result.x[
                    :3
                ]
            )
        )
    )

    translation_update_m = float(
        np.linalg.norm(
            result.x[
                3:
            ]
        )
    )

    return {
        "success":
            bool(
                result.success
            ),

        "reason":
            "",

        "rotation":
            refined_rotation,

        "translation":
            refined_translation,

        "rotation_update_deg":
            rotation_update_deg,

        "translation_update_m":
            translation_update_m,

        "optimisation_cost":
            float(
                result.cost
            ),
    }


# ============================================================
# 14. Match-set helper
# ============================================================

def build_match_set(
    matches,
):
    return {
        (
            match[
                "a_obs_id"
            ],
            match[
                "b_obs_id"
            ],
        )
        for match in matches
    }


# ============================================================
# 15. Joint association + pose refinement
# ============================================================

def joint_association_pose_refinement(
    frame_a_planes,
    frame_b_planes,
    prior_transform,
    params,
):
    """
    Alternating optimisation:

        Pose 0
          ↓
        association
          ↓
        pose refinement
          ↓
        re-association
          ↓
        pose refinement
          ↓
        ...

    The final output is a set of one-to-one
    correspondences and a refined B->A pose.
    """

    (
        current_rotation,
        current_translation,
    ) = parse_transform(
        prior_transform
    )

    previous_match_set = None

    iteration_history = []

    for iteration in range(
        1,
        params.max_joint_iterations + 1,
    ):

        # ----------------------------------------------------
        # A. Build association costs using current pose
        # ----------------------------------------------------

        (
            cost_matrix,
            pair_details,
        ) = build_joint_cost_matrix(
            frame_a_planes,
            frame_b_planes,
            current_rotation,
            current_translation,
            params,
        )

        # ----------------------------------------------------
        # B. One-to-one association
        # ----------------------------------------------------

        matches = solve_assignment(
            frame_a_planes,
            frame_b_planes,
            cost_matrix,
            pair_details,
            params,
        )

        current_match_set = (
            build_match_set(
                matches
            )
        )

        # ----------------------------------------------------
        # C. Refine pose using current correspondences
        # ----------------------------------------------------

        pose_result = (
            refine_pose_from_matches(
                frame_a_planes,
                frame_b_planes,
                matches,
                current_rotation,
                current_translation,
                params,
            )
        )

        iteration_history.append(
            {
                "iteration":
                    int(
                        iteration
                    ),

                "num_matches":
                    int(
                        len(
                            matches
                        )
                    ),

                "match_pairs":
                    sorted(
                        list(
                            current_match_set
                        )
                    ),

                "rotation_update_deg":
                    float(
                        pose_result[
                            "rotation_update_deg"
                        ]
                    ),

                "translation_update_m":
                    float(
                        pose_result[
                            "translation_update_m"
                        ]
                    ),

                "pose_optimisation_success":
                    bool(
                        pose_result[
                            "success"
                        ]
                    ),

                "optimisation_cost":
                    pose_result[
                        "optimisation_cost"
                    ],
            }
        )

        # ----------------------------------------------------
        # Cannot refine pose
        # ----------------------------------------------------

        if not pose_result[
            "success"
        ]:

            break

        # ----------------------------------------------------
        # Update pose
        # ----------------------------------------------------

        current_rotation = (
            pose_result[
                "rotation"
            ]
        )

        current_translation = (
            pose_result[
                "translation"
            ]
        )

        # ----------------------------------------------------
        # Convergence
        # ----------------------------------------------------

        correspondences_stable = (
            previous_match_set
            is not None
            and current_match_set
            == previous_match_set
        )

        pose_stable = (
            pose_result[
                "rotation_update_deg"
            ]
            <= params.rotation_convergence_deg

            and

            pose_result[
                "translation_update_m"
            ]
            <= params.translation_convergence_m
        )

        if (
            correspondences_stable
            and pose_stable
        ):

            break

        previous_match_set = (
            current_match_set
        )

    # ========================================================
    # FINAL association using final refined pose
    # ========================================================

    (
        final_cost_matrix,
        final_pair_details,
    ) = build_joint_cost_matrix(
        frame_a_planes,
        frame_b_planes,
        current_rotation,
        current_translation,
        params,
    )

    final_matches = solve_assignment(
        frame_a_planes,
        frame_b_planes,
        final_cost_matrix,
        final_pair_details,
        params,
    )

    return {
        "predicted_pairs":
            final_matches,

        "rotation_b_to_a":
            current_rotation,

        "translation_b_to_a":
            current_translation,

        "transform_b_to_a":
            make_transform_matrix(
                current_rotation,
                current_translation,
            ),

        "cost_matrix":
            final_cost_matrix,

        "iteration_history":
            iteration_history,

        "num_iterations":
            int(
                len(
                    iteration_history
                )
            ),
    }


# ============================================================
# 16. Run one benchmark case
# ============================================================

def run_joint_association_pose_case(
    case_data,
    params,
):
    frame_a_planes = (
        case_data[
            "frame_a"
        ][
            "planes"
        ]
    )

    frame_b_planes = (
        case_data[
            "frame_b"
        ][
            "planes"
        ]
    )

    prior_transform = (
        case_data[
            "prior_transform_b_to_a"
        ]
    )

    result = (
        joint_association_pose_refinement(
            frame_a_planes,
            frame_b_planes,
            prior_transform,
            params,
        )
    )

    return {
        "sample_id":
            case_data[
                "sample_id"
            ],

        "scene_type":
            case_data[
                "scene_type"
            ],

        "prior_level":
            case_data[
                "prior_level"
            ],

        **result,
    }


# ============================================================
# 17. Pose-evaluation helpers
# ============================================================

def rotation_error_deg(
    estimated_rotation,
    ground_truth_rotation,
):
    relative_rotation = (
        estimated_rotation
        @ ground_truth_rotation.T
    )

    cosine = (
        np.trace(
            relative_rotation
        )
        - 1.0
    ) / 2.0

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(
                cosine
            )
        )
    )


def translation_error_m(
    estimated_translation,
    ground_truth_translation,
):
    return float(
        np.linalg.norm(
            estimated_translation
            - ground_truth_translation
        )
    )


# ============================================================
# 18. Pretty-print cost matrix
# ============================================================

def print_cost_matrix(
    frame_a_planes,
    frame_b_planes,
    matrix,
):
    a_ids = [
        get_plane_obs_id(
            plane
        )
        for plane in frame_a_planes
    ]

    b_ids = [
        get_plane_obs_id(
            plane
        )
        for plane in frame_b_planes
    ]

    print(
        " " * 12,
        end="",
    )

    for b_id in b_ids:

        print(
            f"{str(b_id):>10}",
            end="",
        )

    print()

    for i, a_id in enumerate(
        a_ids
    ):

        print(
            f"{str(a_id):<12}",
            end="",
        )

        for j in range(
            len(
                b_ids
            )
        ):

            print(
                f"{matrix[i, j]:10.3f}",
                end="",
            )

        print()


# ============================================================
# 19. Standalone DEV test
# ============================================================

def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    dev_dir = (
        project_root
        / "data"
        / "synthetic_similarity_benchmark"
        / "dev"
    )

    json_files = sorted(
        dev_dir.glob(
            "*.json"
        )
    )

    if not json_files:

        raise FileNotFoundError(
            "No DEV benchmark JSON files found.\n"
            "Run generate_similarity_benchmark.py first."
        )

    test_file = (
        json_files[
            0
        ]
    )

    print(
        "=" * 76
    )

    print(
        "Joint Association + Pose Refinement"
    )

    print(
        "=" * 76
    )

    print(
        f"Input file: {test_file.name}"
    )

    # --------------------------------------------------------
    # Load case
    # --------------------------------------------------------

    case_data = (
        load_similarity_case(
            test_file
        )
    )

    frame_a_planes = (
        case_data[
            "frame_a"
        ][
            "planes"
        ]
    )

    frame_b_planes = (
        case_data[
            "frame_b"
        ][
            "planes"
        ]
    )

    # --------------------------------------------------------
    # Print schema once so that the actual benchmark
    # representation is visible.
    # --------------------------------------------------------

    print()
    print(
        "Frame-A plane keys:"
    )

    print(
        list(
            frame_a_planes[
                0
            ].keys()
        )
    )

    print()
    print(
        "Frame-B plane keys:"
    )

    print(
        list(
            frame_b_planes[
                0
            ].keys()
        )
    )

    # --------------------------------------------------------
    # Initial parameters
    # --------------------------------------------------------

    params = (
        JointAssociationPoseParameters()
    )

    # --------------------------------------------------------
    # Run joint method
    # --------------------------------------------------------

    result = (
        run_joint_association_pose_case(
            case_data,
            params,
        )
    )

    # --------------------------------------------------------
    # Common final-pair evaluator
    # --------------------------------------------------------

    evaluation = (
        evaluate_predicted_pairs(
            case_data,
            result[
                "predicted_pairs"
            ],
        )
    )

    print()
    print(
        f"Sample ID  : "
        f"{result['sample_id']}"
    )

    print(
        f"Scene      : "
        f"{result['scene_type']}"
    )

    print(
        f"Pose prior : "
        f"{result['prior_level']}"
    )

    print(
        f"Iterations : "
        f"{result['num_iterations']}"
    )

    # --------------------------------------------------------
    # Iteration history
    # --------------------------------------------------------

    print()
    print(
        "=" * 76
    )

    print(
        "Iteration History"
    )

    print(
        "=" * 76
    )

    for row in result[
        "iteration_history"
    ]:

        print(
            f"Iteration "
            f"{row['iteration']} "
            f"| matches = "
            f"{row['num_matches']} "
            f"| rotation update = "
            f"{row['rotation_update_deg']:.4f} deg "
            f"| translation update = "
            f"{row['translation_update_m']:.5f} m"
        )

    # --------------------------------------------------------
    # Final cost matrix
    # --------------------------------------------------------

    print()
    print(
        "=" * 76
    )

    print(
        "Final Joint Cost Matrix"
    )

    print(
        "=" * 76
    )

    print()

    print_cost_matrix(
        frame_a_planes,
        frame_b_planes,
        result[
            "cost_matrix"
        ],
    )

    # --------------------------------------------------------
    # Final correspondences
    # --------------------------------------------------------

    print()
    print(
        "=" * 76
    )

    print(
        "Final Predicted Correspondences"
    )

    print(
        "=" * 76
    )

    print()

    if not result[
        "predicted_pairs"
    ]:

        print(
            "No final correspondences."
        )

    else:

        sorted_matches = sorted(
            result[
                "predicted_pairs"
            ],
            key=lambda row: row[
                "joint_cost"
            ],
        )

        for match in sorted_matches:

            print(
                f"{match['a_obs_id']} "
                f"<-> "
                f"{match['b_obs_id']} "
                f"| cost = "
                f"{match['joint_cost']:.3f} "
                f"| angle = "
                f"{match['normal_angle_deg']:.2f} deg "
                f"| offset = "
                f"{match['offset_difference_m']:.3f} m "
                f"| centroid = "
                f"{match['centroid_distance_m']:.3f} m "
                f"| area = "
                f"{match['area_difference_ratio']:.3f}"
            )

    # --------------------------------------------------------
    # GT correspondence evaluation
    # --------------------------------------------------------

    print()
    print(
        "=" * 76
    )

    print(
        "Correspondence Evaluation"
    )

    print(
        "=" * 76
    )

    print(
        f"GT matches : "
        f"{evaluation['num_gt_matches']}"
    )

    print(
        f"Predicted  : "
        f"{evaluation['num_predicted_pairs']}"
    )

    print(
        f"TP         : "
        f"{evaluation['tp']}"
    )

    print(
        f"FP         : "
        f"{evaluation['fp']}"
    )

    print(
        f"FN         : "
        f"{evaluation['fn']}"
    )

    print(
        f"Precision  : "
        f"{evaluation['precision']:.3f}"
    )

    print(
        f"Recall     : "
        f"{evaluation['recall']:.3f}"
    )

    print(
        f"F1         : "
        f"{evaluation['f1']:.3f}"
    )

    # --------------------------------------------------------
    # Pose diagnostic
    #
    # GT is used ONLY for evaluation.
    # The algorithm itself never reads GT correspondences.
    # --------------------------------------------------------

    if (
        "gt_transform_b_to_a"
        in case_data
    ):

        (
            gt_rotation,
            gt_translation,
        ) = parse_transform(
            case_data[
                "gt_transform_b_to_a"
            ]
        )

        (
            prior_rotation,
            prior_translation,
        ) = parse_transform(
            case_data[
                "prior_transform_b_to_a"
            ]
        )

        prior_rotation_error = (
            rotation_error_deg(
                prior_rotation,
                gt_rotation,
            )
        )

        final_rotation_error = (
            rotation_error_deg(
                result[
                    "rotation_b_to_a"
                ],
                gt_rotation,
            )
        )

        prior_translation_error = (
            translation_error_m(
                prior_translation,
                gt_translation,
            )
        )

        final_translation_error = (
            translation_error_m(
                result[
                    "translation_b_to_a"
                ],
                gt_translation,
            )
        )

        print()
        print(
            "=" * 76
        )

        print(
            "Pose Diagnostic"
        )

        print(
            "=" * 76
        )

        print(
            f"Prior rotation error    : "
            f"{prior_rotation_error:.4f} deg"
        )

        print(
            f"Final rotation error    : "
            f"{final_rotation_error:.4f} deg"
        )

        print(
            f"Prior translation error : "
            f"{prior_translation_error:.5f} m"
        )

        print(
            f"Final translation error : "
            f"{final_translation_error:.5f} m"
        )

    print()
    print(
        "=" * 76
    )

    print(
        "Joint method test complete."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":
    main()