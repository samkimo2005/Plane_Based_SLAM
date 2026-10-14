from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


# ============================================================
# 1. Basic vector utilities
# ============================================================

def normalize_vector(
    vector: np.ndarray,
) -> np.ndarray:
    """
    Convert a vector to unit length.
    """

    vector = np.asarray(
        vector,
        dtype=float,
    )

    length = np.linalg.norm(
        vector
    )

    if length < 1e-12:
        raise ValueError(
            "Cannot normalise a near-zero vector."
        )

    return vector / length


def safe_acos_deg(
    value: float,
) -> float:
    """
    acos with numerical clipping.

    Returns the angle in degrees.
    """

    value = float(
        np.clip(
            value,
            -1.0,
            1.0,
        )
    )

    return math.degrees(
        math.acos(value)
    )


# ============================================================
# 2. Load one similarity benchmark case
# ============================================================

def load_similarity_case(
    filepath: Path | str,
) -> dict:
    """
    Load one JSON benchmark case.
    """

    filepath = Path(
        filepath
    )

    with filepath.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    return data


# ============================================================
# 3. Load rigid transformation
# ============================================================

def load_transform(
    transform_dict: dict,
):
    """
    Expected JSON format:

        {
            "R": [[...], [...], [...]],
            "t": [...]
        }

    Returns:

        R, t
    """

    R = np.asarray(
        transform_dict["R"],
        dtype=float,
    )

    t = np.asarray(
        transform_dict["t"],
        dtype=float,
    )

    if R.shape != (3, 3):
        raise ValueError(
            f"Rotation matrix must be 3x3, "
            f"got {R.shape}"
        )

    if t.shape != (3,):
        raise ValueError(
            f"Translation vector must have shape (3,), "
            f"got {t.shape}"
        )

    return R, t


# ============================================================
# 4. Transform one Frame-B plane into Frame A
# ============================================================

def transform_plane_b_to_a(
    plane_b: dict,
    R_ab: np.ndarray,
    t_ab: np.ndarray,
) -> dict:
    """
    Coordinate convention:

        x_A = R_ab x_B + t_ab

    Plane in B:

        n_B^T x_B + d_B = 0

    Plane transformed into A:

        n_A = R_ab n_B

        d_A = d_B - n_A^T t_ab

    Centroid:

        c_A = R_ab c_B + t_ab
    """

    normal_b = normalize_vector(
        np.asarray(
            plane_b["normal"],
            dtype=float,
        )
    )

    offset_b = float(
        plane_b["offset"]
    )

    centroid_b = np.asarray(
        plane_b["centroid"],
        dtype=float,
    )

    normal_a = normalize_vector(
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

    return {
        "obs_id":
            plane_b["obs_id"],

        "normal":
            normal_a,

        "offset":
            offset_a,

        "centroid":
            centroid_a,

        "width_m":
            float(
                plane_b["width_m"]
            ),

        "height_m":
            float(
                plane_b["height_m"]
            ),

        "area_m2":
            float(
                plane_b["area_m2"]
            ),
    }


# ============================================================
# 5. Transform every Frame-B plane into Frame A
# ============================================================

def transform_frame_b_to_a(
    frame_b_planes: list,
    prior_transform: dict,
) -> list:
    """
    Transform all Frame-B plane observations into Frame A
    using the supplied pose prior.
    """

    R_ab, t_ab = load_transform(
        prior_transform
    )

    transformed_planes = []

    for plane_b in frame_b_planes:

        transformed_plane = (
            transform_plane_b_to_a(
                plane_b,
                R_ab,
                t_ab,
            )
        )

        transformed_planes.append(
            transformed_plane
        )

    return transformed_planes


# ============================================================
# 6. Resolve plane-normal sign ambiguity
# ============================================================

def align_plane_sign(
    reference_plane: dict,
    candidate_plane: dict,
) -> dict:
    """
    The two representations

        (n, d)

    and

        (-n, -d)

    describe the same infinite plane.

    The candidate plane is therefore flipped if its normal
    points in the opposite direction to the reference normal.
    """

    normal_ref = normalize_vector(
        np.asarray(
            reference_plane["normal"],
            dtype=float,
        )
    )

    normal_candidate = normalize_vector(
        np.asarray(
            candidate_plane["normal"],
            dtype=float,
        )
    )

    offset_candidate = float(
        candidate_plane["offset"]
    )

    if (
        np.dot(
            normal_ref,
            normal_candidate,
        )
        < 0.0
    ):

        normal_candidate = (
            -normal_candidate
        )

        offset_candidate = (
            -offset_candidate
        )

    aligned_plane = dict(
        candidate_plane
    )

    aligned_plane["normal"] = (
        normal_candidate
    )

    aligned_plane["offset"] = (
        offset_candidate
    )

    return aligned_plane


# ============================================================
# 7. Calculate common pairwise geometric quantities
# ============================================================

def calculate_pair_geometry(
    plane_a: dict,
    plane_b_in_a: dict,
) -> dict:
    """
    Calculate geometric differences between two planes.

    IMPORTANT:

    Both plane observations must already be expressed
    in Frame A.

    Returned quantities:

        normal_angle_deg
        offset_difference_m
        centroid_distance_m
        area_difference_ratio
    """

    # --------------------------------------------------------
    # Resolve normal direction ambiguity first
    # --------------------------------------------------------

    plane_b_in_a = align_plane_sign(
        plane_a,
        plane_b_in_a,
    )

    # --------------------------------------------------------
    # Normal angle
    # --------------------------------------------------------

    normal_a = normalize_vector(
        np.asarray(
            plane_a["normal"],
            dtype=float,
        )
    )

    normal_b = normalize_vector(
        np.asarray(
            plane_b_in_a["normal"],
            dtype=float,
        )
    )

    normal_dot = float(
        np.dot(
            normal_a,
            normal_b,
        )
    )

    normal_angle_deg = (
        safe_acos_deg(
            normal_dot
        )
    )

    # --------------------------------------------------------
    # Plane offset difference
    # --------------------------------------------------------

    offset_difference_m = abs(
        float(
            plane_a["offset"]
        )
        -
        float(
            plane_b_in_a["offset"]
        )
    )

    # --------------------------------------------------------
    # Centroid distance
    # --------------------------------------------------------

    centroid_a = np.asarray(
        plane_a["centroid"],
        dtype=float,
    )

    centroid_b = np.asarray(
        plane_b_in_a["centroid"],
        dtype=float,
    )

    centroid_distance_m = float(
        np.linalg.norm(
            centroid_a
            - centroid_b
        )
    )

    # --------------------------------------------------------
    # Relative area difference
    # --------------------------------------------------------

    area_a = float(
        plane_a["area_m2"]
    )

    area_b = float(
        plane_b_in_a["area_m2"]
    )

    denominator = max(
        area_a,
        area_b,
        1e-12,
    )

    area_difference_ratio = (
        abs(
            area_a
            - area_b
        )
        / denominator
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
# 8. Ground-truth utilities
# ============================================================

def build_gt_match_set(
    case_data: dict,
) -> set:
    """
    Ground truth is for evaluation only.

    Example result:

        {
            ("A0", "B2"),
            ("A1", "B0"),
            ...
        }
    """

    return {
        (
            match["a_obs_id"],
            match["b_obs_id"],
        )
        for match
        in case_data["gt_matches"]
    }


def is_gt_match(
    a_obs_id: str,
    b_obs_id: str,
    gt_match_set: set,
) -> bool:
    """
    Return True if this A-B pair is a real physical
    correspondence according to ground truth.
    """

    return (
        a_obs_id,
        b_obs_id,
    ) in gt_match_set