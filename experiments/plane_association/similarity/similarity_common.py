"""
similarity_common.py

Common data structures and utilities for controlled experiments on
pairwise plane similarity methods.

This module deliberately contains no method-specific similarity logic.
All four similarity algorithms should consume the same SimilaritySample
objects so that the underlying test cases and ground-truth labels remain
identical across methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json

import numpy as np


# ============================================================
# 1. Numerical constants
# ============================================================

EPS = 1e-12


# ============================================================
# 2. Basic geometry helpers
# ============================================================

def normalise_vector(v: np.ndarray) -> np.ndarray:
    """
    Return a unit-length copy of vector v.

    Raises
    ------
    ValueError
        If the vector norm is effectively zero.
    """
    v = np.asarray(v, dtype=float).reshape(3)
    norm = np.linalg.norm(v)

    if norm < EPS:
        raise ValueError("Cannot normalise a zero-length vector.")

    return v / norm


def plane_angle_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    """
    Return the unsigned angle between two plane normals in degrees.

    Plane normals have a sign ambiguity, therefore |n1^T n2| is used.

    Returns
    -------
    float
        Angle in the range [0, 90] degrees.
    """
    n1 = normalise_vector(n1)
    n2 = normalise_vector(n2)

    cosine = np.clip(abs(np.dot(n1, n2)), 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def plane_offset_difference(d1: float, d2: float) -> float:
    """
    Absolute difference between two signed plane offsets.

    This function assumes that both planes are already represented in
    the same coordinate frame and use a consistent normal orientation.
    """
    return float(abs(d1 - d2))


def transform_point(
    point: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """
    Transform a 3D point using:

        p_target = R @ p_source + t
    """
    point = np.asarray(point, dtype=float).reshape(3)
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).reshape(3)

    return R @ point + t


def transform_plane(
    normal: np.ndarray,
    offset: float,
    R: np.ndarray,
    t: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Transform a plane between coordinate frames.

    Plane convention:

        n^T p + d = 0

    Point transform:

        p_target = R @ p_source + t

    Therefore:

        n_target = R @ n_source
        d_target = d_source - n_target^T t
    """
    n_source = normalise_vector(normal)
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).reshape(3)

    n_target = normalise_vector(R @ n_source)
    d_target = float(offset - np.dot(n_target, t))

    return n_target, d_target


# ============================================================
# 3. Plane representation
# ============================================================

@dataclass
class PlaneObservation:
    """
    Generic representation of one detected plane.

    All similarity algorithms should receive planes using this common
    representation. Method-specific descriptors should not replace
    these basic geometric attributes.
    """

    plane_id: str

    normal: np.ndarray
    offset: float
    centroid: np.ndarray

    area: float | None = None
    support_size: int | None = None

    # Optional generic information retained for methods that require
    # additional geometric information.
    covariance: np.ndarray | None = None

    # Additional method-independent information may be stored here
    # without changing the common interface.
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.normal = normalise_vector(self.normal)
        self.centroid = np.asarray(
            self.centroid,
            dtype=float
        ).reshape(3)

        self.offset = float(self.offset)

        if self.area is not None:
            self.area = float(self.area)

        if self.support_size is not None:
            self.support_size = int(self.support_size)

        if self.covariance is not None:
            self.covariance = np.asarray(
                self.covariance,
                dtype=float
            )

    def validate(self) -> None:
        """
        Check whether the plane contains valid numerical information.
        """
        if not np.all(np.isfinite(self.normal)):
            raise ValueError(
                f"Plane {self.plane_id}: normal contains invalid values."
            )

        if not np.isfinite(self.offset):
            raise ValueError(
                f"Plane {self.plane_id}: offset is invalid."
            )

        if not np.all(np.isfinite(self.centroid)):
            raise ValueError(
                f"Plane {self.plane_id}: centroid contains invalid values."
            )

        if self.area is not None and self.area < 0:
            raise ValueError(
                f"Plane {self.plane_id}: area cannot be negative."
            )

        if self.support_size is not None and self.support_size < 0:
            raise ValueError(
                f"Plane {self.plane_id}: support_size cannot be negative."
            )


# ============================================================
# 4. Common similarity experiment sample
# ============================================================

@dataclass
class SimilaritySample:
    """
    One pairwise plane-similarity test case.

    IMPORTANT:
    The same SimilaritySample should be supplied to every similarity
    algorithm in the controlled experiment.
    """

    sample_id: str

    current_plane: PlaneObservation
    reference_plane: PlaneObservation

    # Ground truth:
    # True  -> same physical plane
    # False -> different physical planes
    gt_match: bool

    # "synthetic" or "real"
    data_source: str

    # "dev" or "final"
    split: str

    # Optional frame / sequence information.
    sequence_id: str | None = None
    current_frame_id: str | None = None
    reference_frame_id: str | None = None

    # Experimental condition:
    # e.g. clean, noise_01, hard, repetitive_geometry
    condition: str = "default"

    # Optional common-frame pose prior:
    # p_ref = R_ref_cur @ p_cur + t_ref_cur
    R_ref_cur: np.ndarray | None = None
    t_ref_cur: np.ndarray | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.gt_match = bool(self.gt_match)
        self.data_source = self.data_source.lower()
        self.split = self.split.lower()

        if self.data_source not in {"synthetic", "real"}:
            raise ValueError(
                f"Unknown data_source: {self.data_source}"
            )

        if self.split not in {"dev", "final"}:
            raise ValueError(
                f"Unknown split: {self.split}"
            )

        if self.R_ref_cur is not None:
            self.R_ref_cur = np.asarray(
                self.R_ref_cur,
                dtype=float
            ).reshape(3, 3)

        if self.t_ref_cur is not None:
            self.t_ref_cur = np.asarray(
                self.t_ref_cur,
                dtype=float
            ).reshape(3)

        self.current_plane.validate()
        self.reference_plane.validate()


# ============================================================
# 5. Common result representation
# ============================================================

@dataclass
class SimilarityResult:
    """
    Standard output record written by run_similarity.py.

    The algorithms may use different mathematical formulations, but
    their results are stored in one common format for evaluation.
    """

    sample_id: str
    method: str

    score: float

    # True if a larger score indicates a stronger match.
    higher_is_better: bool

    predicted_match: bool

    gt_match: bool

    runtime_ms: float

    data_source: str
    split: str
    condition: str

    sequence_id: str | None = None
    current_frame_id: str | None = None
    reference_frame_id: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# 6. DEV / FINAL handling
# ============================================================

def deterministic_fraction(key: str) -> float:
    """
    Convert a string into a deterministic pseudo-random number in [0, 1).

    This is useful for reproducible DEV/FINAL splitting without relying
    on a changing random seed.
    """
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    integer = int(digest[:16], 16)

    return integer / float(16 ** 16)


def assign_split(
    group_id: str,
    dev_fraction: float = 0.20
) -> str:
    """
    Deterministically assign an experimental group to DEV or FINAL.

    All samples belonging to the same group_id should use the same
    split so that closely related examples do not leak between tuning
    and final evaluation.
    """
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError(
            "dev_fraction must lie strictly between 0 and 1."
        )

    value = deterministic_fraction(group_id)

    return "dev" if value < dev_fraction else "final"


def filter_split(
    samples: Iterable[SimilaritySample],
    split: str
) -> list[SimilaritySample]:
    """
    Return samples belonging to a requested split.
    """
    split = split.lower()

    if split not in {"dev", "final"}:
        raise ValueError("split must be 'dev' or 'final'.")

    return [
        sample
        for sample in samples
        if sample.split == split
    ]


def filter_source(
    samples: Iterable[SimilaritySample],
    data_source: str
) -> list[SimilaritySample]:
    """
    Return only synthetic or real samples.
    """
    data_source = data_source.lower()

    if data_source not in {"synthetic", "real"}:
        raise ValueError(
            "data_source must be 'synthetic' or 'real'."
        )

    return [
        sample
        for sample in samples
        if sample.data_source == data_source
    ]


# ============================================================
# 7. Common-frame plane preparation
# ============================================================

def current_plane_in_reference_frame(
    sample: SimilaritySample
) -> PlaneObservation:
    """
    Transform the current-frame plane into the reference frame.

    If no pose prior is supplied, the original current-plane
    representation is returned.

    This helper ensures that algorithms requiring common-frame geometry
    use exactly the same transformation convention.
    """
    plane = sample.current_plane

    if sample.R_ref_cur is None or sample.t_ref_cur is None:
        return plane

    normal_ref, offset_ref = transform_plane(
        plane.normal,
        plane.offset,
        sample.R_ref_cur,
        sample.t_ref_cur,
    )

    centroid_ref = transform_point(
        plane.centroid,
        sample.R_ref_cur,
        sample.t_ref_cur,
    )

    return PlaneObservation(
        plane_id=plane.plane_id,
        normal=normal_ref,
        offset=offset_ref,
        centroid=centroid_ref,
        area=plane.area,
        support_size=plane.support_size,
        covariance=plane.covariance,
        metadata=dict(plane.metadata),
    )


# ============================================================
# 8. Common geometric quantities
# ============================================================

def common_geometric_features(
    sample: SimilaritySample
) -> dict[str, float]:
    """
    Calculate a small set of method-independent geometric quantities.

    These values are not themselves a similarity algorithm. They simply
    provide consistent geometry where several methods require the same
    quantities.
    """
    current = current_plane_in_reference_frame(sample)
    reference = sample.reference_plane

    normal_angle = plane_angle_deg(
        current.normal,
        reference.normal,
    )

    offset_difference = plane_offset_difference(
        current.offset,
        reference.offset,
    )

    centroid_distance = float(
        np.linalg.norm(
            current.centroid - reference.centroid
        )
    )

    features = {
        "normal_angle_deg": normal_angle,
        "offset_difference_m": offset_difference,
        "centroid_distance_m": centroid_distance,
    }

    if current.area is not None and reference.area is not None:
        area_max = max(
            current.area,
            reference.area,
            EPS,
        )

        features["relative_area_difference"] = (
            abs(current.area - reference.area) / area_max
        )

    return features


# ============================================================
# 9. Threshold helper
# ============================================================

def classify_score(
    score: float,
    threshold: float,
    higher_is_better: bool
) -> bool:
    """
    Convert a continuous similarity/cost score into a binary match.

    Examples
    --------
    Similarity:
        higher_is_better = True
        match if score >= threshold

    Cost:
        higher_is_better = False
        match if score <= threshold
    """
    if higher_is_better:
        return bool(score >= threshold)

    return bool(score <= threshold)


# ============================================================
# 10. JSON serialisation helpers
# ============================================================

def _numpy_to_serialisable(value: Any) -> Any:
    """
    Convert numpy-based objects recursively into JSON-compatible data.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            key: _numpy_to_serialisable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _numpy_to_serialisable(item)
            for item in value
        ]

    return value


def plane_to_dict(plane: PlaneObservation) -> dict[str, Any]:
    data = asdict(plane)
    return _numpy_to_serialisable(data)


def sample_to_dict(sample: SimilaritySample) -> dict[str, Any]:
    data = asdict(sample)
    return _numpy_to_serialisable(data)


def result_to_dict(result: SimilarityResult) -> dict[str, Any]:
    data = asdict(result)
    return _numpy_to_serialisable(data)


def plane_from_dict(data: dict[str, Any]) -> PlaneObservation:
    return PlaneObservation(
        plane_id=data["plane_id"],
        normal=np.asarray(data["normal"], dtype=float),
        offset=float(data["offset"]),
        centroid=np.asarray(data["centroid"], dtype=float),
        area=data.get("area"),
        support_size=data.get("support_size"),
        covariance=(
            None
            if data.get("covariance") is None
            else np.asarray(data["covariance"], dtype=float)
        ),
        metadata=data.get("metadata", {}),
    )


def sample_from_dict(data: dict[str, Any]) -> SimilaritySample:
    return SimilaritySample(
        sample_id=data["sample_id"],
        current_plane=plane_from_dict(
            data["current_plane"]
        ),
        reference_plane=plane_from_dict(
            data["reference_plane"]
        ),
        gt_match=bool(data["gt_match"]),
        data_source=data["data_source"],
        split=data["split"],
        sequence_id=data.get("sequence_id"),
        current_frame_id=data.get("current_frame_id"),
        reference_frame_id=data.get("reference_frame_id"),
        condition=data.get("condition", "default"),
        R_ref_cur=(
            None
            if data.get("R_ref_cur") is None
            else np.asarray(data["R_ref_cur"], dtype=float)
        ),
        t_ref_cur=(
            None
            if data.get("t_ref_cur") is None
            else np.asarray(data["t_ref_cur"], dtype=float)
        ),
        metadata=data.get("metadata", {}),
    )


def save_samples_jsonl(
    path: str | Path,
    samples: Iterable[SimilaritySample]
) -> None:
    """
    Save samples as JSON Lines.

    One line corresponds to one common similarity test case.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8"
    ) as file:
        for sample in samples:
            file.write(
                json.dumps(
                    sample_to_dict(sample),
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_samples_jsonl(
    path: str | Path
) -> list[SimilaritySample]:
    """
    Load common similarity samples from JSON Lines.
    """
    path = Path(path)

    samples: list[SimilaritySample] = []

    with path.open(
        "r",
        encoding="utf-8"
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            data = json.loads(line)
            samples.append(
                sample_from_dict(data)
            )

    return samples


# ============================================================
# 11. Dataset sanity checks
# ============================================================

def validate_samples(
    samples: Iterable[SimilaritySample]
) -> dict[str, int]:
    """
    Validate a similarity dataset and return a short summary.

    Also checks that sample IDs are unique.
    """
    samples = list(samples)

    seen_ids: set[str] = set()

    summary = {
        "total": 0,
        "positive": 0,
        "negative": 0,
        "synthetic": 0,
        "real": 0,
        "dev": 0,
        "final": 0,
    }

    for sample in samples:
        if sample.sample_id in seen_ids:
            raise ValueError(
                f"Duplicate sample_id: {sample.sample_id}"
            )

        seen_ids.add(sample.sample_id)

        sample.current_plane.validate()
        sample.reference_plane.validate()

        summary["total"] += 1
        summary[
            "positive" if sample.gt_match else "negative"
        ] += 1

        summary[sample.data_source] += 1
        summary[sample.split] += 1

    return summary