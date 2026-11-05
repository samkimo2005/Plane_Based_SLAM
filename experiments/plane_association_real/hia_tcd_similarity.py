import numpy as np

from similarity_common import SimilaritySample


def _descriptor(contour, bins=12):
    contour = np.asarray(contour, dtype=float)

    centre = contour.mean(axis=0)
    v = contour - centre

    r = np.linalg.norm(v, axis=1)
    r /= max(r.max(), 1e-12)

    h_r, _ = np.histogram(
        r, bins=bins, range=(0, 1), density=True
    )

    v1 = np.roll(v, 1, axis=0)
    v2 = np.roll(v, -1, axis=0)

    cos_a = np.sum(v1 * v2, axis=1)
    cos_a /= (
        np.linalg.norm(v1, axis=1)
        * np.linalg.norm(v2, axis=1)
        + 1e-12
    )

    angles = np.arccos(np.clip(cos_a, -1, 1))

    h_a, _ = np.histogram(
        angles, bins=bins, range=(0, np.pi), density=True
    )

    return np.hstack([h_r, h_a])


def compute_score(sample: SimilaritySample, params=None) -> float:
    c1 = sample.current_plane.metadata["contour"]
    c2 = sample.reference_plane.metadata["contour"]

    d1 = _descriptor(c1)
    d2 = _descriptor(c2)

    distance = np.linalg.norm(d1 - d2)

    return float(np.exp(-distance))