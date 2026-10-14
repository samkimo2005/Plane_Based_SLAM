"""
generate_similarity_synthetic.py

Generate controlled synthetic plane pairs for the
Pairwise Plane Similarity experiment.

The same samples are used by all five similarity methods.
"""

from pathlib import Path
import numpy as np

from similarity_common import (
    PlaneObservation,
    SimilaritySample,
    assign_split,
    save_samples_jsonl,
    validate_samples,
)


# ============================================================
# Configuration
# ============================================================

OUTPUT_FILE = Path(
    "data/similarity/synthetic_samples.jsonl"
)

SEED = 20260924
RNG = np.random.default_rng(SEED)

REPEATS_PER_CONDITION = 50

CONDITIONS = {
    "clean": {
        "pos_angle_deg": 0.5,
        "pos_shift_m": 0.005,
        "pos_shape_change": 0.02,

        "neg_angle_deg": (25.0, 55.0),
        "neg_normal_shift_m": (0.35, 0.70),
        "neg_tangent_shift_m": (0.30, 0.80),
        "neg_shape_change": (0.20, 0.45),
    },

    "moderate": {
        "pos_angle_deg": 1.5,
        "pos_shift_m": 0.020,
        "pos_shape_change": 0.04,

        "neg_angle_deg": (10.0, 30.0),
        "neg_normal_shift_m": (0.15, 0.40),
        "neg_tangent_shift_m": (0.15, 0.50),
        "neg_shape_change": (0.10, 0.30),
    },

    "hard": {
        "pos_angle_deg": 3.0,
        "pos_shift_m": 0.050,
        "pos_shape_change": 0.08,

        "neg_angle_deg": (2.0, 10.0),
        "neg_normal_shift_m": (0.03, 0.15),
        "neg_tangent_shift_m": (0.03, 0.20),
        "neg_shape_change": (0.02, 0.12),
    },
}


# ============================================================
# Basic geometry
# ============================================================

def normalise(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def random_unit_vector():
    return normalise(
        RNG.normal(size=3)
    )


def plane_basis(normal):
    """Return two unit vectors lying on the plane."""
    normal = normalise(normal)

    helper = np.array(
        [1.0, 0.0, 0.0]
    )

    if abs(np.dot(helper, normal)) > 0.9:
        helper = np.array(
            [0.0, 1.0, 0.0]
        )

    u = normalise(
        np.cross(normal, helper)
    )

    v = normalise(
        np.cross(normal, u)
    )

    return u, v


def rotate_normal(normal, angle_deg):
    """Rotate a normal by a controlled angular amount."""
    normal = normalise(normal)

    axis = random_unit_vector()

    # Force rotation axis to lie in tangent space.
    axis = (
        axis
        - np.dot(axis, normal) * normal
    )

    if np.linalg.norm(axis) < 1e-8:
        axis = plane_basis(normal)[0]

    axis = normalise(axis)

    angle = np.radians(angle_deg)

    rotated = (
        normal * np.cos(angle)
        + np.cross(axis, normal)
        * np.sin(angle)
        + axis
        * np.dot(axis, normal)
        * (1.0 - np.cos(angle))
    )

    return normalise(rotated)


# ============================================================
# Ordered bounded-plane geometry
# ============================================================

def ordered_contour(
    centroid,
    normal,
    width,
    height,
    points_per_edge=8,
):
    """
    Generate contour points in true cyclic boundary order:

        bottom: left -> right
        right:  bottom -> top
        top:    right -> left
        left:   top -> bottom

    This ordering is required by contour-based descriptors.
    """
    u, v = plane_basis(normal)

    t = np.linspace(
        -0.5,
        0.5,
        points_per_edge,
        endpoint=False,
    )

    contour = []

    # Bottom
    for x in t:
        contour.append(
            centroid
            + x * width * u
            - 0.5 * height * v
        )

    # Right
    for y in t:
        contour.append(
            centroid
            + 0.5 * width * u
            + y * height * v
        )

    # Top
    for x in t:
        contour.append(
            centroid
            - x * width * u
            + 0.5 * height * v
        )

    # Left
    for y in t:
        contour.append(
            centroid
            - 0.5 * width * u
            - y * height * v
        )

    return np.asarray(
        contour,
        dtype=float,
    )


def sample_patch(
    centroid,
    normal,
    width,
    height,
    num_points=150,
):
    """Sample points uniformly from the bounded plane."""
    u, v = plane_basis(normal)

    uv = RNG.uniform(
        -0.5,
        0.5,
        size=(num_points, 2),
    )

    return (
        centroid
        + uv[:, 0, None] * width * u
        + uv[:, 1, None] * height * v
    )


def make_plane(
    plane_id,
    normal,
    centroid,
    width,
    height,
):
    normal = normalise(normal)

    centroid = np.asarray(
        centroid,
        dtype=float,
    )

    width = float(width)
    height = float(height)

    offset = -float(
        np.dot(normal, centroid)
    )

    contour = ordered_contour(
        centroid,
        normal,
        width,
        height,
    )

    patch = sample_patch(
        centroid,
        normal,
        width,
        height,
    )

    return PlaneObservation(
        plane_id=plane_id,
        normal=normal,
        offset=offset,
        centroid=centroid,
        area=width * height,
        support_size=len(patch),

        metadata={
            "width": width,
            "height": height,
            "contour": contour,
            "patch": patch,
        },
    )


# ============================================================
# Base physical plane
# ============================================================

def generate_base_plane(index):

    normal = random_unit_vector()

    centroid = RNG.uniform(
        [-1.5, -1.5, 1.0],
        [1.5, 1.5, 3.0],
    )

    width = RNG.uniform(
        0.8,
        2.2,
    )

    height = RNG.uniform(
        0.6,
        1.8,
    )

    return {
        "id": f"physical_{index:04d}",
        "normal": normal,
        "centroid": centroid,
        "width": width,
        "height": height,
    }


# ============================================================
# Positive pair
# ============================================================

def make_positive_pair(
    index,
    cfg,
):
    """
    Two noisy observations of the same physical plane.
    """

    base = generate_base_plane(index)

    reference = make_plane(
        plane_id=f"ref_{index:04d}",
        normal=base["normal"],
        centroid=base["centroid"],
        width=base["width"],
        height=base["height"],
    )

    angle = RNG.normal(
        0.0,
        cfg["pos_angle_deg"],
    )

    current_normal = rotate_normal(
        base["normal"],
        angle,
    )

    current_centroid = (
        base["centroid"]
        + RNG.normal(
            0.0,
            cfg["pos_shift_m"],
            size=3,
        )
    )

    shape_noise = cfg[
        "pos_shape_change"
    ]

    current_width = (
        base["width"]
        * (
            1.0
            + RNG.uniform(
                -shape_noise,
                shape_noise,
            )
        )
    )

    current_height = (
        base["height"]
        * (
            1.0
            + RNG.uniform(
                -shape_noise,
                shape_noise,
            )
        )
    )

    current = make_plane(
        plane_id=f"cur_{index:04d}",
        normal=current_normal,
        centroid=current_centroid,
        width=current_width,
        height=current_height,
    )

    return reference, current


# ============================================================
# Negative pair
# ============================================================

def make_negative_pair(
    index,
    cfg,
):
    """
    Two different physical planes.

    Negative pairs become progressively more similar from
    clean -> moderate -> hard.
    """

    base = generate_base_plane(index)

    reference = make_plane(
        plane_id=f"ref_{index:04d}",
        normal=base["normal"],
        centroid=base["centroid"],
        width=base["width"],
        height=base["height"],
    )

    # --------------------------------------------------------
    # Orientation difference
    # --------------------------------------------------------

    angle = RNG.uniform(
        *cfg["neg_angle_deg"]
    )

    current_normal = rotate_normal(
        base["normal"],
        angle,
    )

    # --------------------------------------------------------
    # Spatial / offset difference
    # --------------------------------------------------------

    u, v = plane_basis(
        base["normal"]
    )

    normal_shift = RNG.uniform(
        *cfg["neg_normal_shift_m"]
    )

    tangent_shift = RNG.uniform(
        *cfg["neg_tangent_shift_m"]
    )

    tangent_angle = RNG.uniform(
        0.0,
        2.0 * np.pi,
    )

    tangent_direction = (
        np.cos(tangent_angle) * u
        + np.sin(tangent_angle) * v
    )

    current_centroid = (
        base["centroid"]
        + normal_shift * base["normal"]
        + tangent_shift * tangent_direction
    )

    # --------------------------------------------------------
    # Finite-plane shape difference
    # --------------------------------------------------------

    shape_min, shape_max = (
        cfg["neg_shape_change"]
    )

    width_change = RNG.uniform(
        shape_min,
        shape_max,
    )

    height_change = RNG.uniform(
        shape_min,
        shape_max,
    )

    # Change aspect ratio rather than only uniform scale.
    if RNG.random() < 0.5:
        width_scale = 1.0 + width_change
        height_scale = 1.0 - height_change
    else:
        width_scale = 1.0 - width_change
        height_scale = 1.0 + height_change

    current_width = max(
        0.4,
        base["width"] * width_scale,
    )

    current_height = max(
        0.4,
        base["height"] * height_scale,
    )

    current = make_plane(
        plane_id=f"neg_{index:04d}",
        normal=current_normal,
        centroid=current_centroid,
        width=current_width,
        height=current_height,
    )

    return reference, current


# ============================================================
# Dataset generation
# ============================================================

def main():

    samples = []

    index = 0

    for condition, cfg in CONDITIONS.items():

        for repeat in range(
            REPEATS_PER_CONDITION
        ):

            group_id = (
                f"{condition}_{repeat:03d}"
            )

            # Positive and negative samples from the same
            # experimental group always remain in the same split.
            split = assign_split(
                group_id,
                dev_fraction=0.20,
            )

            # =================================================
            # Positive correspondence
            # =================================================

            reference, current = (
                make_positive_pair(
                    index,
                    cfg,
                )
            )

            samples.append(
                SimilaritySample(
                    sample_id=(
                        f"{condition}_"
                        f"{repeat:03d}_pos"
                    ),

                    current_plane=current,
                    reference_plane=reference,

                    gt_match=True,

                    data_source="synthetic",
                    split=split,
                    condition=condition,

                    sequence_id=group_id,

                    metadata={
                        "label":
                            "same_physical_plane"
                    },
                )
            )

            index += 1

            # =================================================
            # Negative correspondence
            # =================================================

            reference, current = (
                make_negative_pair(
                    index,
                    cfg,
                )
            )

            samples.append(
                SimilaritySample(
                    sample_id=(
                        f"{condition}_"
                        f"{repeat:03d}_neg"
                    ),

                    current_plane=current,
                    reference_plane=reference,

                    gt_match=False,

                    data_source="synthetic",
                    split=split,
                    condition=condition,

                    sequence_id=group_id,

                    metadata={
                        "label":
                            "different_physical_plane"
                    },
                )
            )

            index += 1

    # ========================================================
    # Sanity check
    # ========================================================

    summary = validate_samples(
        samples
    )

    expected = (
        len(CONDITIONS)
        * REPEATS_PER_CONDITION
        * 2
    )

    if summary["total"] != expected:
        raise RuntimeError(
            f"Expected {expected} samples, "
            f"got {summary['total']}."
        )

    if summary["positive"] != (
        expected // 2
    ):
        raise RuntimeError(
            "Positive/negative balance is incorrect."
        )

    # ========================================================
    # Save
    # ========================================================

    save_samples_jsonl(
        OUTPUT_FILE,
        samples,
    )

    print(
        "\n=== Synthetic Similarity Dataset ==="
    )

    print(
        f"Total samples : {summary['total']}"
    )

    print(
        f"Positive      : {summary['positive']}"
    )

    print(
        f"Negative      : {summary['negative']}"
    )

    print(
        f"DEV           : {summary['dev']}"
    )

    print(
        f"FINAL         : {summary['final']}"
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()