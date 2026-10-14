import numpy as np

from similarity_common import (
    SimilaritySample,
    current_plane_in_reference_frame,
)


def compute_score(sample: SimilaritySample, params=None) -> float:
    params = params or {}
    pos_scale = params.get("pos_scale", 0.5)

    p1 = current_plane_in_reference_frame(sample)
    p2 = sample.reference_plane

    n2 = p2.normal.copy()
    if np.dot(p1.normal, n2) < 0:
        n2 = -n2

    d1 = np.hstack([
        p1.centroid / pos_scale,
        p1.normal,
    ])

    d2 = np.hstack([
        p2.centroid / pos_scale,
        n2,
    ])

    distance = np.linalg.norm(d1 - d2)

    return float(np.exp(-distance))