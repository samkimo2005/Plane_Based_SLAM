"""
candidate_gating_common.py

Common geometry and data-loading functions for the Candidate Gating
stage of Plane Association.

Shared by:

    Motion-model-based distance constraint
    Contour-point plane data association
    Normal-based plane data association
    PCPM-based coarse plane matching

Plane representation:
    n^T x + d = 0

Pose convention:
    x_target = R @ x_source + t

This file can be run directly.

Default test input:
    data/candidate_gating/dev/sample_0000.npz
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

DEV_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "candidate_gating"
    / "dev"
)

DEFAULT_SAMPLE = (
    DEV_DATA_DIR
    / "sample_0000.npz"
)


# ============================================================
# Constants
# ============================================================

EPS = 1e-12


# ============================================================
# Data structures
# ============================================================

@dataclass
class PlaneSet:
    """
    Collection of detected / synthetic planes.

    normals:
        shape (N, 3)

    offsets:
        shape (N,)

    centroids:
        shape (N, 3)

    areas:
        shape (N,)

    contours:
        shape (N, K, 3)

    widths:
        shape (N,)

    heights:
        shape (N,)
    """

    normals: np.ndarray
    offsets: np.ndarray
    centroids: np.ndarray
    areas: np.ndarray
    contours: np.ndarray
    widths: np.ndarray
    heights: np.ndarray

    def __len__(self) -> int:
        return int(self.normals.shape[0])


@dataclass
class CandidateSample:
    """
    Complete Candidate Gating benchmark sample.
    """

    source: PlaneSet
    target: PlaneSet

    gt_pairs: np.ndarray
    gt_match_matrix: np.ndarray

    true_R: np.ndarray
    true_t: np.ndarray

    prior_R: np.ndarray
    prior_t: np.ndarray

    metadata: Optional[Dict] = None


@dataclass
class PairGeometry:
    """
    Common geometric quantities for one source-target plane pair.

    The source plane must already be transformed into
    the target coordinate frame.
    """

    source_idx: int
    target_idx: int

    normal_angle_deg: float
    offset_difference_m: float
    centroid_distance_m: float

    contour_min_distance_m: float
    contour_mean_distance_m: float

    area_ratio: float


# ============================================================
# Basic numerical utilities
# ============================================================

def normalize_vector(
    vector: np.ndarray,
) -> np.ndarray:
    """
    Normalize one 3D vector.
    """

    vector = np.asarray(
        vector,
        dtype=float,
    )

    magnitude = float(
        np.linalg.norm(vector)
    )

    if magnitude < EPS:
        raise ValueError(
            "Cannot normalize a near-zero vector."
        )

    return vector / magnitude


def normalize_rows(
    vectors: np.ndarray,
) -> np.ndarray:
    """
    Normalize every row of an N x 3 matrix.
    """

    vectors = np.asarray(
        vectors,
        dtype=float,
    )

    magnitudes = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    if np.any(magnitudes < EPS):
        raise ValueError(
            "At least one vector has near-zero magnitude."
        )

    return vectors / magnitudes


def clamp_dot(
    value: float,
) -> float:
    """
    Avoid numerical errors before acos().
    """

    return float(
        np.clip(
            value,
            -1.0,
            1.0,
        )
    )


# ============================================================
# Plane normal sign handling
# ============================================================

def align_plane_sign(
    reference_normal: np.ndarray,
    normal: np.ndarray,
    offset: float,
) -> Tuple[np.ndarray, float]:
    """
    Resolve the plane-normal sign ambiguity.

    (n, d) and (-n, -d) describe the same plane.

    The second plane is flipped if necessary so that its
    normal points approximately in the same direction as
    the reference normal.
    """

    reference_normal = normalize_vector(
        reference_normal
    )

    normal = normalize_vector(
        normal
    )

    if float(
        reference_normal @ normal
    ) < 0.0:

        normal = -normal
        offset = -float(offset)

    return normal, float(offset)


# ============================================================
# Plane geometry
# ============================================================

def normal_angle_rad(
    normal_a: np.ndarray,
    normal_b: np.ndarray,
) -> float:
    """
    Sign-invariant angle between two plane normals.

    angle = acos(|n_a^T n_b|)
    """

    normal_a = normalize_vector(
        normal_a
    )

    normal_b = normalize_vector(
        normal_b
    )

    dot_value = abs(
        float(
            normal_a @ normal_b
        )
    )

    dot_value = clamp_dot(
        dot_value
    )

    return math.acos(
        dot_value
    )


def normal_angle_deg(
    normal_a: np.ndarray,
    normal_b: np.ndarray,
) -> float:
    """
    Sign-invariant plane-normal angle in degrees.
    """

    return math.degrees(
        normal_angle_rad(
            normal_a,
            normal_b,
        )
    )


def aligned_offset_difference(
    normal_a: np.ndarray,
    offset_a: float,
    normal_b: np.ndarray,
    offset_b: float,
) -> float:
    """
    Absolute plane-offset difference after resolving
    normal-direction ambiguity.

    Both planes must already be represented in the
    same coordinate frame.
    """

    normal_a = normalize_vector(
        normal_a
    )

    _, aligned_offset_b = (
        align_plane_sign(
            reference_normal=normal_a,
            normal=normal_b,
            offset=offset_b,
        )
    )

    return abs(
        float(offset_a)
        - aligned_offset_b
    )


def centroid_distance(
    centroid_a: np.ndarray,
    centroid_b: np.ndarray,
) -> float:
    """
    Euclidean distance between two plane centroids.
    """

    centroid_a = np.asarray(
        centroid_a,
        dtype=float,
    )

    centroid_b = np.asarray(
        centroid_b,
        dtype=float,
    )

    return float(
        np.linalg.norm(
            centroid_a
            - centroid_b
        )
    )


def point_to_plane_signed_distance(
    points: np.ndarray,
    normal: np.ndarray,
    offset: float,
) -> np.ndarray:
    """
    Signed perpendicular point-to-plane distance.

    Plane:
        n^T x + d = 0

    Assumes n is normalized.
    """

    points = np.asarray(
        points,
        dtype=float,
    )

    normal = normalize_vector(
        normal
    )

    return (
        points @ normal
        + float(offset)
    )


def point_to_plane_distance(
    points: np.ndarray,
    normal: np.ndarray,
    offset: float,
) -> np.ndarray:
    """
    Absolute perpendicular point-to-plane distance.
    """

    return np.abs(
        point_to_plane_signed_distance(
            points=points,
            normal=normal,
            offset=offset,
        )
    )


def contour_to_plane_distances(
    contour: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
) -> np.ndarray:
    """
    Calculate the distance from every contour point
    to a plane.
    """

    contour = np.asarray(
        contour,
        dtype=float,
    )

    if (
        contour.ndim != 2
        or contour.shape[1] != 3
    ):
        raise ValueError(
            "Contour must have shape (K, 3)."
        )

    return point_to_plane_distance(
        points=contour,
        normal=plane_normal,
        offset=plane_offset,
    )


def contour_to_plane_min_distance(
    contour: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
) -> float:
    """
    Minimum contour-point-to-plane distance.
    """

    distances = (
        contour_to_plane_distances(
            contour=contour,
            plane_normal=plane_normal,
            plane_offset=plane_offset,
        )
    )

    return float(
        np.min(distances)
    )


def contour_to_plane_mean_distance(
    contour: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
) -> float:
    """
    Mean contour-point-to-plane distance.
    """

    distances = (
        contour_to_plane_distances(
            contour=contour,
            plane_normal=plane_normal,
            plane_offset=plane_offset,
        )
    )

    return float(
        np.mean(distances)
    )


def symmetric_contour_plane_distance(
    contour_a: np.ndarray,
    normal_a: np.ndarray,
    offset_a: float,
    contour_b: np.ndarray,
    normal_b: np.ndarray,
    offset_b: float,
) -> float:
    """
    Symmetric contour-to-plane proximity.

    Computes:

        contour A -> plane B

    and:

        contour B -> plane A

    then averages the two minimum distances.
    """

    distance_ab = (
        contour_to_plane_min_distance(
            contour=contour_a,
            plane_normal=normal_b,
            plane_offset=offset_b,
        )
    )

    distance_ba = (
        contour_to_plane_min_distance(
            contour=contour_b,
            plane_normal=normal_a,
            plane_offset=offset_a,
        )
    )

    return 0.5 * (
        distance_ab
        + distance_ba
    )


def plane_area_ratio(
    area_a: float,
    area_b: float,
) -> float:
    """
    Ratio between the smaller and larger plane area.

    Output range:
        0 <= ratio <= 1

    1 means equal areas.
    """

    area_a = abs(
        float(area_a)
    )

    area_b = abs(
        float(area_b)
    )

    largest = max(
        area_a,
        area_b,
    )

    if largest < EPS:
        return 0.0

    return (
        min(area_a, area_b)
        / largest
    )


# ============================================================
# Coordinate transformations
# ============================================================

def transform_points(
    points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Transform 3D points:

        x_target = R @ x_source + t
    """

    points = np.asarray(
        points,
        dtype=float,
    )

    R = np.asarray(
        R,
        dtype=float,
    )

    t = np.asarray(
        t,
        dtype=float,
    ).reshape(3)

    return (
        (R @ points.T).T
        + t
    )


def transform_plane_parameters(
    normal: np.ndarray,
    offset: float,
    R: np.ndarray,
    t: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Transform one plane from source frame to target frame.

    Point transform:

        x_t = R x_s + t

    Plane transform:

        n_t = R n_s

        d_t = d_s - n_t^T t
    """

    normal = normalize_vector(
        normal
    )

    R = np.asarray(
        R,
        dtype=float,
    )

    t = np.asarray(
        t,
        dtype=float,
    ).reshape(3)

    normal_target = normalize_vector(
        R @ normal
    )

    offset_target = (
        float(offset)
        - float(
            normal_target @ t
        )
    )

    return (
        normal_target,
        offset_target,
    )


def transform_plane_set(
    planes: PlaneSet,
    R: np.ndarray,
    t: np.ndarray,
) -> PlaneSet:
    """
    Transform a complete PlaneSet into another frame.
    """

    transformed_normals = []
    transformed_offsets = []

    for normal, offset in zip(
        planes.normals,
        planes.offsets,
    ):

        normal_target, offset_target = (
            transform_plane_parameters(
                normal=normal,
                offset=offset,
                R=R,
                t=t,
            )
        )

        transformed_normals.append(
            normal_target
        )

        transformed_offsets.append(
            offset_target
        )

    transformed_normals = np.asarray(
        transformed_normals,
        dtype=float,
    )

    transformed_offsets = np.asarray(
        transformed_offsets,
        dtype=float,
    )

    transformed_centroids = (
        transform_points(
            points=planes.centroids,
            R=R,
            t=t,
        )
    )

    number_of_planes = (
        planes.contours.shape[0]
    )

    contour_points_per_plane = (
        planes.contours.shape[1]
    )

    flat_contours = (
        planes.contours.reshape(
            -1,
            3,
        )
    )

    transformed_flat_contours = (
        transform_points(
            points=flat_contours,
            R=R,
            t=t,
        )
    )

    transformed_contours = (
        transformed_flat_contours.reshape(
            number_of_planes,
            contour_points_per_plane,
            3,
        )
    )

    return PlaneSet(
        normals=transformed_normals,
        offsets=transformed_offsets,
        centroids=transformed_centroids,
        areas=planes.areas.copy(),
        contours=transformed_contours,
        widths=planes.widths.copy(),
        heights=planes.heights.copy(),
    )


# ============================================================
# Load generated dataset
# ============================================================

def plane_set_from_npz(
    data: np.lib.npyio.NpzFile,
    prefix: str,
) -> PlaneSet:
    """
    Load either source_* or target_* plane data.
    """

    return PlaneSet(

        normals=normalize_rows(
            data[
                f"{prefix}_normals"
            ]
        ),

        offsets=np.asarray(
            data[
                f"{prefix}_offsets"
            ],
            dtype=float,
        ),

        centroids=np.asarray(
            data[
                f"{prefix}_centroids"
            ],
            dtype=float,
        ),

        areas=np.asarray(
            data[
                f"{prefix}_areas"
            ],
            dtype=float,
        ),

        contours=np.asarray(
            data[
                f"{prefix}_contours"
            ],
            dtype=float,
        ),

        widths=np.asarray(
            data[
                f"{prefix}_widths"
            ],
            dtype=float,
        ),

        heights=np.asarray(
            data[
                f"{prefix}_heights"
            ],
            dtype=float,
        ),
    )


def load_candidate_sample(
    npz_path: Path | str,
) -> CandidateSample:
    """
    Load one generated Candidate Gating sample.

    The matching .json metadata file is loaded
    automatically if it exists.
    """

    npz_path = Path(
        npz_path
    ).resolve()

    if not npz_path.exists():

        raise FileNotFoundError(
            "\nCandidate Gating sample not found:\n"
            f"{npz_path}"
        )

    with np.load(
        npz_path,
        allow_pickle=False,
    ) as data:

        source = plane_set_from_npz(
            data,
            "source",
        )

        target = plane_set_from_npz(
            data,
            "target",
        )

        gt_pairs = np.asarray(
            data["gt_pairs"],
            dtype=np.int32,
        )

        gt_match_matrix = np.asarray(
            data["gt_match_matrix"],
            dtype=np.uint8,
        )

        true_R = np.asarray(
            data["true_R"],
            dtype=float,
        )

        true_t = np.asarray(
            data["true_t"],
            dtype=float,
        ).reshape(3)

        prior_R = np.asarray(
            data["prior_R"],
            dtype=float,
        )

        prior_t = np.asarray(
            data["prior_t"],
            dtype=float,
        ).reshape(3)

    # Automatically load matching JSON metadata.
    metadata_path = (
        npz_path.with_suffix(
            ".json"
        )
    )

    metadata = None

    if metadata_path.exists():

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(
                file
            )

    sample = CandidateSample(
        source=source,
        target=target,

        gt_pairs=gt_pairs,
        gt_match_matrix=gt_match_matrix,

        true_R=true_R,
        true_t=true_t,

        prior_R=prior_R,
        prior_t=prior_t,

        metadata=metadata,
    )

    validate_sample(
        sample
    )

    return sample


# ============================================================
# Candidate pair generation
# ============================================================

def all_candidate_pairs(
    number_source_planes: int,
    number_target_planes: int,
) -> Iterator[Tuple[int, int]]:
    """
    Generate all source-target plane combinations.
    """

    for source_idx in range(
        number_source_planes
    ):

        for target_idx in range(
            number_target_planes
        ):

            yield (
                source_idx,
                target_idx,
            )


def candidate_pair_array(
    number_source_planes: int,
    number_target_planes: int,
) -> np.ndarray:
    """
    Return all possible candidate pairs as:

        [[source_idx, target_idx],
         ...]

    shape:
        (N_pairs, 2)
    """

    pairs = list(
        all_candidate_pairs(
            number_source_planes,
            number_target_planes,
        )
    )

    if len(pairs) == 0:

        return np.empty(
            (0, 2),
            dtype=np.int32,
        )

    return np.asarray(
        pairs,
        dtype=np.int32,
    )


# ============================================================
# Common pairwise geometry
# ============================================================

def compute_pair_geometry(
    source_planes_target_frame: PlaneSet,
    target_planes: PlaneSet,
    source_idx: int,
    target_idx: int,
) -> PairGeometry:
    """
    Calculate the common geometric quantities for one
    source-target plane pair.

    IMPORTANT:
        source_planes_target_frame must already have been
        transformed into the target coordinate frame.
    """

    source_normal = (
        source_planes_target_frame
        .normals[source_idx]
    )

    source_offset = float(
        source_planes_target_frame
        .offsets[source_idx]
    )

    source_centroid = (
        source_planes_target_frame
        .centroids[source_idx]
    )

    source_contour = (
        source_planes_target_frame
        .contours[source_idx]
    )

    source_area = float(
        source_planes_target_frame
        .areas[source_idx]
    )


    target_normal = (
        target_planes
        .normals[target_idx]
    )

    target_offset = float(
        target_planes
        .offsets[target_idx]
    )

    target_centroid = (
        target_planes
        .centroids[target_idx]
    )

    target_area = float(
        target_planes
        .areas[target_idx]
    )


    angle = normal_angle_deg(
        source_normal,
        target_normal,
    )

    offset_difference = (
        aligned_offset_difference(
            normal_a=source_normal,
            offset_a=source_offset,
            normal_b=target_normal,
            offset_b=target_offset,
        )
    )

    centre_distance = (
        centroid_distance(
            source_centroid,
            target_centroid,
        )
    )

    contour_distances = (
        contour_to_plane_distances(
            contour=source_contour,
            plane_normal=target_normal,
            plane_offset=target_offset,
        )
    )

    contour_min = float(
        np.min(
            contour_distances
        )
    )

    contour_mean = float(
        np.mean(
            contour_distances
        )
    )

    area_similarity = (
        plane_area_ratio(
            source_area,
            target_area,
        )
    )

    return PairGeometry(

        source_idx=source_idx,
        target_idx=target_idx,

        normal_angle_deg=angle,

        offset_difference_m=(
            offset_difference
        ),

        centroid_distance_m=(
            centre_distance
        ),

        contour_min_distance_m=(
            contour_min
        ),

        contour_mean_distance_m=(
            contour_mean
        ),

        area_ratio=area_similarity,
    )


def compute_all_pair_geometry(
    source_planes_target_frame: PlaneSet,
    target_planes: PlaneSet,
) -> Dict[
    Tuple[int, int],
    PairGeometry,
]:
    """
    Calculate common geometry for every possible
    source-target plane pair.
    """

    results = {}

    for (
        source_idx,
        target_idx,
    ) in all_candidate_pairs(
        len(
            source_planes_target_frame
        ),
        len(
            target_planes
        ),
    ):

        geometry = (
            compute_pair_geometry(
                source_planes_target_frame=(
                    source_planes_target_frame
                ),
                target_planes=target_planes,
                source_idx=source_idx,
                target_idx=target_idx,
            )
        )

        results[
            (
                source_idx,
                target_idx,
            )
        ] = geometry

    return results


# ============================================================
# Ground-truth helper functions
# ============================================================

def is_ground_truth_match(
    sample: CandidateSample,
    source_idx: int,
    target_idx: int,
) -> bool:
    """
    Return True if this pair is a known GT match.
    """

    return bool(
        sample.gt_match_matrix[
            source_idx,
            target_idx,
        ]
    )


def ground_truth_pair_set(
    sample: CandidateSample,
) -> set[Tuple[int, int]]:
    """
    Return GT correspondences as a Python set.
    """

    return {

        (
            int(source_idx),
            int(target_idx),
        )

        for (
            source_idx,
            target_idx
        ) in sample.gt_pairs
    }


# ============================================================
# Validation
# ============================================================

def validate_rotation_matrix(
    R: np.ndarray,
    tolerance: float = 1e-6,
) -> None:
    """
    Verify that R is approximately a valid rotation matrix.
    """

    R = np.asarray(
        R,
        dtype=float,
    )

    if R.shape != (3, 3):

        raise ValueError(
            "Rotation matrix must have shape (3, 3)."
        )

    orthogonality_error = float(
        np.linalg.norm(
            R.T @ R
            - np.eye(3)
        )
    )

    determinant = float(
        np.linalg.det(R)
    )

    if (
        orthogonality_error
        > tolerance
    ):

        raise ValueError(
            "Rotation matrix is not orthonormal. "
            f"Error = {orthogonality_error:.3e}"
        )

    if abs(
        determinant - 1.0
    ) > tolerance:

        raise ValueError(
            "Rotation matrix determinant is not +1. "
            f"det(R) = {determinant:.8f}"
        )


def validate_plane_set(
    planes: PlaneSet,
    tolerance: float = 1e-6,
) -> None:
    """
    Verify one plane set.
    """

    number_of_planes = len(
        planes
    )

    if planes.normals.shape != (
        number_of_planes,
        3,
    ):
        raise ValueError(
            "Invalid normals shape."
        )

    if planes.offsets.shape != (
        number_of_planes,
    ):
        raise ValueError(
            "Invalid offsets shape."
        )

    if planes.centroids.shape != (
        number_of_planes,
        3,
    ):
        raise ValueError(
            "Invalid centroids shape."
        )

    if planes.areas.shape != (
        number_of_planes,
    ):
        raise ValueError(
            "Invalid areas shape."
        )

    if (
        planes.contours.ndim != 3
        or planes.contours.shape[0]
        != number_of_planes
        or planes.contours.shape[2]
        != 3
    ):
        raise ValueError(
            "Contours must have shape (N, K, 3)."
        )

    # --------------------------------------------------------
    # Check unit normals
    # --------------------------------------------------------

    normal_lengths = np.linalg.norm(
        planes.normals,
        axis=1,
    )

    max_normal_error = float(
        np.max(
            np.abs(
                normal_lengths - 1.0
            )
        )
    )

    if (
        max_normal_error
        > tolerance
    ):

        raise ValueError(
            "Plane normals are not unit vectors."
        )

    # --------------------------------------------------------
    # Check centroid lies on plane
    # --------------------------------------------------------

    residuals = np.abs(

        np.sum(
            planes.normals
            * planes.centroids,
            axis=1,
        )

        + planes.offsets
    )

    max_residual = float(
        np.max(
            residuals
        )
    )

    if max_residual > tolerance:

        raise ValueError(
            "Plane centroid does not lie on its plane. "
            f"Maximum residual = {max_residual:.3e}"
        )


def validate_sample(
    sample: CandidateSample,
) -> None:
    """
    Verify one complete benchmark sample.
    """

    validate_plane_set(
        sample.source
    )

    validate_plane_set(
        sample.target
    )

    validate_rotation_matrix(
        sample.true_R
    )

    validate_rotation_matrix(
        sample.prior_R
    )

    expected_matrix_shape = (
        len(sample.source),
        len(sample.target),
    )

    if (
        sample.gt_match_matrix.shape
        != expected_matrix_shape
    ):

        raise ValueError(
            "GT match matrix has incorrect shape."
        )

    for (
        source_idx,
        target_idx,
    ) in sample.gt_pairs:

        source_idx = int(
            source_idx
        )

        target_idx = int(
            target_idx
        )

        if not (
            0
            <= source_idx
            < len(sample.source)
        ):

            raise ValueError(
                "Invalid source GT index."
            )

        if not (
            0
            <= target_idx
            < len(sample.target)
        ):

            raise ValueError(
                "Invalid target GT index."
            )

        if (
            sample.gt_match_matrix[
                source_idx,
                target_idx,
            ]
            != 1
        ):

            raise ValueError(
                "GT pair list and GT matrix disagree."
            )


# ============================================================
# Prepare data for gating algorithms
# ============================================================

def prepare_sample_for_gating(
    sample: CandidateSample,
    use_true_pose: bool = False,
) -> Tuple[
    PlaneSet,
    PlaneSet,
]:
    """
    Transform source planes into the target coordinate frame.

    Default:
        uses noisy pose prior

    Debug mode:
        use_true_pose=True

        uses exact ground-truth pose
    """

    if use_true_pose:

        R = sample.true_R
        t = sample.true_t

    else:

        R = sample.prior_R
        t = sample.prior_t

    source_in_target_frame = (
        transform_plane_set(
            planes=sample.source,
            R=R,
            t=t,
        )
    )

    return (
        source_in_target_frame,
        sample.target,
    )


# ============================================================
# Direct-run sanity test
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print("Candidate Gating Common Geometry Test")
    print("=" * 70)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Input sample : {DEFAULT_SAMPLE}"
    )

    print()

    # --------------------------------------------------------
    # Load sample
    # --------------------------------------------------------

    sample = load_candidate_sample(
        DEFAULT_SAMPLE
    )

    print(
        f"Source planes  : {len(sample.source)}"
    )

    print(
        f"Target planes  : {len(sample.target)}"
    )

    print(
        f"GT matches     : {len(sample.gt_pairs)}"
    )

    possible_pairs = (
        len(sample.source)
        * len(sample.target)
    )

    print(
        f"Possible pairs : {possible_pairs}"
    )

    print()

    print(
        "Ground-truth pairs:"
    )

    print(
        sample.gt_pairs
    )

    # --------------------------------------------------------
    # Transform source planes using noisy pose prior
    # --------------------------------------------------------

    (
        source_in_target_frame,
        target_planes,
    ) = prepare_sample_for_gating(
        sample=sample,
        use_true_pose=False,
    )

    # --------------------------------------------------------
    # Compute all pairwise geometry
    # --------------------------------------------------------

    pair_geometry = (
        compute_all_pair_geometry(
            source_planes_target_frame=(
                source_in_target_frame
            ),
            target_planes=target_planes,
        )
    )

    print()
    print(
        "Pairwise geometry calculated successfully."
    )

    print(
        f"Number of pairs : {len(pair_geometry)}"
    )

    # --------------------------------------------------------
    # Print first pair as example
    # --------------------------------------------------------

    first_pair = next(
        iter(pair_geometry)
    )

    geometry = (
        pair_geometry[
            first_pair
        ]
    )

    print()
    print(
        "Example pair geometry"
    )

    print(
        "-" * 70
    )

    print(
        f"Pair              : "
        f"S{geometry.source_idx} "
        f"-> T{geometry.target_idx}"
    )

    print(
        f"Normal angle      : "
        f"{geometry.normal_angle_deg:.4f} deg"
    )

    print(
        f"Offset difference : "
        f"{geometry.offset_difference_m:.6f} m"
    )

    print(
        f"Centroid distance : "
        f"{geometry.centroid_distance_m:.6f} m"
    )

    print(
        f"Contour min dist  : "
        f"{geometry.contour_min_distance_m:.6f} m"
    )

    print(
        f"Contour mean dist : "
        f"{geometry.contour_mean_distance_m:.6f} m"
    )

    print(
        f"Area ratio        : "
        f"{geometry.area_ratio:.4f}"
    )

    print(
        "-" * 70
    )

    # --------------------------------------------------------
    # Also inspect GT-pair geometry
    # --------------------------------------------------------

    print()
    print(
        "Ground-truth pair geometry:"
    )

    print(
        "-" * 70
    )

    for (
        source_idx,
        target_idx
    ) in sample.gt_pairs:

        key = (
            int(source_idx),
            int(target_idx),
        )

        g = pair_geometry[
            key
        ]

        print(
            f"S{g.source_idx} -> T{g.target_idx} | "
            f"angle={g.normal_angle_deg:.3f} deg | "
            f"offset={g.offset_difference_m:.4f} m | "
            f"centroid={g.centroid_distance_m:.4f} m"
        )

    print(
        "-" * 70
    )

    print()
    print(
        "Sample validation : PASSED"
    )

    print(
        "Common geometry   : PASSED"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()