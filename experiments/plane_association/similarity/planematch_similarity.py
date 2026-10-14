"""
planematch_similarity.py

Simplified PlaneMatch-inspired coplanarity similarity.
"""

import numpy as np

from similarity_common import (
    SimilaritySample,
    current_plane_in_reference_frame,
)


def compute_score(sample: SimilaritySample, params=None) -> float:
    params = params or {}
    scale = params.get("residual_scale", 0.05)

    p1 = current_plane_in_reference_frame(sample)
    p2 = sample.reference_plane

    patch1 = np.asarray(
        sample.current_plane.metadata["patch"],
        dtype=float,
    )

    patch2 = np.asarray(
        sample.reference_plane.metadata["patch"],
        dtype=float,
    )

    # Plane 1 -> Plane 2 residual
    r12 = np.mean(
        np.abs(
            patch1 @ p2.normal + p2.offset
        )
    )

    # Plane 2 -> Plane 1 residual
    r21 = np.mean(
        np.abs(
            patch2 @ p1.normal + p1.offset
        )
    )

    residual = 0.5 * (r12 + r21)

    return float(
        np.exp(-residual / scale)
    )