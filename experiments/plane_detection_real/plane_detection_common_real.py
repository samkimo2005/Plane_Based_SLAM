"""Real-scene common geometry and evaluation utilities for plane detectors.

Drop-in API for the four legacy detector modules, with configurable acceptance
rules and EXPLICIT disjoint point assignment for fair point-level metrics.

Use this as ``plane_detection_common.py`` ONLY in a separate real-data folder,
or update the four real-data detector imports to:
    import plane_detection_common_real as common
Never overwrite the original synthetic/TUM common module.

Units: XYZ and d in metres; thresholds in metres; runtime excludes/ includes
normal estimation depending on where callers begin their timer.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial import cKDTree

# Lab Book 9-style default acceptance; not automatically an optimal setting.
FINAL_DISTANCE_THRESHOLD = 0.008
MIN_PLANE_SUPPORT = 80
MIN_PLANE_FRACTION = 0.025
MAX_PLANES = 10
K_NEIGHBORS = 30


@dataclass(frozen=True)
class DetectionSettings:
    distance_threshold: float = 0.008
    min_support: int = 80
    min_fraction: float = 0.025
    max_planes: int = 10
    k_neighbors: int = 30


def configure(settings: DetectionSettings) -> None:
    """Configure before processing a cloud; do not change while running threads."""
    global FINAL_DISTANCE_THRESHOLD, MIN_PLANE_SUPPORT, MIN_PLANE_FRACTION
    global MAX_PLANES, K_NEIGHBORS
    if not (settings.distance_threshold > 0 and settings.min_support >= 3 and
            0 <= settings.min_fraction <= 1 and settings.max_planes >= 1 and
            settings.k_neighbors >= 3):
        raise ValueError('Invalid detection settings')
    FINAL_DISTANCE_THRESHOLD = float(settings.distance_threshold)
    MIN_PLANE_SUPPORT = int(settings.min_support)
    MIN_PLANE_FRACTION = float(settings.min_fraction)
    MAX_PLANES = int(settings.max_planes)
    K_NEIGHBORS = int(settings.k_neighbors)


def minimum_support(n_points: int) -> int:
    return max(MIN_PLANE_SUPPORT, math.ceil(MIN_PLANE_FRACTION * n_points))


def _check_points(points):
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('points must be a N x 3 NumPy array')
    if not np.isfinite(pts).all():
        raise ValueError('points must contain only finite XYZ coordinates')
    return pts


def canonicalise_plane(normal, d):
    n = np.asarray(normal, dtype=np.float64).reshape(3).copy()
    length = float(np.linalg.norm(n))
    if not np.isfinite(length) or length <= 1e-12:
        raise ValueError('Invalid plane normal')
    n /= length
    d = float(d) / length
    pivot = int(np.argmax(np.abs(n)))
    if n[pivot] < 0:
        n, d = -n, -d
    return n, d


def fit_plane_three(p1, p2, p3):
    p1, p2, p3 = (np.asarray(p, dtype=np.float64) for p in (p1, p2, p3))
    n = np.cross(p2 - p1, p3 - p1)
    if np.linalg.norm(n) < 1e-10:
        return None
    return canonicalise_plane(n, -np.dot(n, p1))


def fit_plane_svd(points):
    pts = _check_points(points)
    if len(pts) < 3:
        return None
    centre = pts.mean(axis=0)
    _, singular, vh = np.linalg.svd(pts - centre, full_matrices=False)
    if len(singular) < 3 or singular[1] < 1e-10:
        return None
    normal = vh[-1]
    return canonicalise_plane(normal, -np.dot(normal, centre))


def plane_angle_deg(normal_a, normal_b):
    a, _ = canonicalise_plane(normal_a, 0)
    b, _ = canonicalise_plane(normal_b, 0)
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), 0, 1))))


def align_plane_to_reference(normal, d, ref_normal):
    n, d = canonicalise_plane(normal, d)
    ref = np.asarray(ref_normal, dtype=np.float64)
    if np.dot(n, ref) < 0:
        return -n, -d
    return n, d


def build_model(points, normal, d, source_support=None):
    """Preserve legacy return structure; support is NOT a disjoint label count."""
    pts = _check_points(points)
    n, d = canonicalise_plane(normal, d)
    mask = np.abs(pts @ n + d) < FINAL_DISTANCE_THRESHOLD
    count = int(mask.sum())
    if count < minimum_support(len(pts)):
        return None
    return {'normal': n, 'd': d, 'mask': mask, 'support': count,
            'source_support': source_support}


def merge_duplicate_models(models, normal_threshold_deg=5.0, offset_threshold_m=0.05):
    """Legacy geometric deduplication; spatially separate coplanar patches can merge."""
    ordered = sorted((m for m in models if m is not None),
                     key=lambda x: x['support'], reverse=True)
    kept = []
    for m in ordered:
        duplicate = False
        for existing in kept:
            n, d = align_plane_to_reference(m['normal'], m['d'], existing['normal'])
            if (plane_angle_deg(existing['normal'], n) <= normal_threshold_deg and
                    abs(existing['d'] - d) <= offset_threshold_m):
                duplicate = True
                break
        if not duplicate:
            kept.append(m)
        if len(kept) == MAX_PLANES:
            break
    return kept


def estimate_local_geometry(points):
    pts = _check_points(points)
    if len(pts) <= K_NEIGHBORS:
        return None
    neighbours = cKDTree(pts).query(pts, k=K_NEIGHBORS + 1)[1][:, 1:]
    samples = pts[neighbours]
    delta = samples - samples.mean(axis=1, keepdims=True)
    covariance = np.einsum('nki,nkj->nij', delta, delta) / K_NEIGHBORS
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    pivot = np.argmax(np.abs(normals), axis=1)
    signs = normals[np.arange(len(normals)), pivot] < 0
    normals[signs] *= -1
    denominators = eigenvalues.sum(axis=1)
    curvature = np.ones(len(pts), dtype=np.float64)
    valid = denominators > 1e-12
    curvature[valid] = eigenvalues[valid, 0] / denominators[valid]
    return normals, curvature, neighbours


def assign_unique_labels(points, models, *, threshold=None, min_support=None,
                         min_fraction=None, refit=False):
    """Assign each point to ONE closest accepted plane, or -1.

    Returns (labels, accepted_models, summary). Labels refer to accepted_models
    in order. This is a common geometric reassignment, NOT each detector's
    original region segmentation. For Region Growing preserve native labels
    separately if spatial connectivity must be evaluated.

    The final membership check and minimum support are iterated because
    removing undersupported planes can change nearest-plane assignments.
    """
    pts = _check_points(points)
    t = FINAL_DISTANCE_THRESHOLD if threshold is None else float(threshold)
    if t <= 0:
        raise ValueError('threshold must be positive')
    lower = max((MIN_PLANE_SUPPORT if min_support is None else int(min_support)),
                math.ceil((MIN_PLANE_FRACTION if min_fraction is None else float(min_fraction)) * len(pts)))
    active = [dict(m) for m in models if m is not None][:MAX_PLANES]
    if not active or len(pts) == 0:
        empty = np.full(len(pts), -1, dtype=np.int32)
        return empty, [], _summary(pts, empty, [], t)
    while active:
        residual = np.column_stack([np.abs(pts @ canonicalise_plane(m['normal'], m['d'])[0] +
                                          canonicalise_plane(m['normal'], m['d'])[1]) for m in active])
        labels = np.argmin(residual, axis=1).astype(np.int32)
        labels[np.min(residual, axis=1) >= t] = -1
        counts = np.bincount(labels[labels >= 0], minlength=len(active))
        keep = [i for i, c in enumerate(counts) if c >= lower]
        if len(keep) == len(active):
            break
        active = [active[i] for i in keep]
    if not active:
        labels = np.full(len(pts), -1, dtype=np.int32)
    if refit and active:
        # Opt-in: after unique assignment, fit on assigned points and reassign.
        new_models = []
        for i, m in enumerate(active):
            fitted = fit_plane_svd(pts[labels == i])
            item = dict(m)
            if fitted is not None:
                item['normal'], item['d'] = fitted
            new_models.append(item)
        return assign_unique_labels(pts, new_models, threshold=t,
                                    min_support=lower, min_fraction=0, refit=False)
    accepted = []
    for i, m in enumerate(active):
        members = labels == i
        n, d = canonicalise_plane(m['normal'], m['d'])
        rr = np.abs(pts[members] @ n + d)
        result = dict(m)
        result.update(normal=n, d=d, mask=members, support=int(members.sum()),
                      fit_rms_mm=float(1000 * np.sqrt(np.mean(rr ** 2))),
                      fit_p90_mm=float(1000 * np.percentile(rr, 90)))
        accepted.append(result)
    return labels, accepted, _summary(pts, labels, accepted, t)


def _summary(pts, labels, models, threshold):
    assigned = int(np.count_nonzero(labels >= 0))
    return {'input_points': int(len(pts)), 'detected_planes': len(models),
            'assigned_points': assigned, 'unassigned_points': int(len(pts) - assigned),
            'coverage_percent': 100 * assigned / len(pts) if len(pts) else 0.,
            'evaluation_distance_threshold_m': threshold}
