from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


# ==========================================================
# Common benchmark / detector rules
# ==========================================================

FINAL_DISTANCE_THRESHOLD = 0.02
MIN_PLANE_SUPPORT = 120
MAX_PLANES = 6
K_NEIGHBORS = 30


# ==========================================================
# Generic geometry
# ==========================================================

def canonicalise_plane(normal, d):
    normal = np.asarray(normal, dtype=np.float64)

    index = int(np.argmax(np.abs(normal)))

    if normal[index] < 0:
        normal = -normal
        d = -d

    return normal, float(d)


def fit_plane_three(p1, p2, p3):
    normal = np.cross(p2 - p1, p3 - p1)
    norm = np.linalg.norm(normal)

    if norm < 1e-10:
        return None

    normal = normal / norm
    d = -float(np.dot(normal, p1))

    return canonicalise_plane(normal, d)


def fit_plane_svd(points):
    if len(points) < 3:
        return None

    centroid = np.mean(points, axis=0)
    centred = points - centroid

    _, _, vh = np.linalg.svd(
        centred,
        full_matrices=False,
    )

    normal = vh[-1]
    norm = np.linalg.norm(normal)

    if norm < 1e-10:
        return None

    normal = normal / norm
    d = -float(np.dot(normal, centroid))

    return canonicalise_plane(normal, d)


def plane_angle_deg(normal_a, normal_b):
    dot_value = float(np.dot(normal_a, normal_b))
    dot_value = np.clip(abs(dot_value), 0.0, 1.0)

    return float(
        np.degrees(
            np.arccos(dot_value)
        )
    )


def align_plane_to_reference(normal, d, ref_normal):
    if np.dot(normal, ref_normal) < 0:
        return -normal, -d

    return normal, d


def build_model(points, normal, d, source_support=None):
    """
    Common final adapter used by ALL methods.

    IMPORTANT:
    The common support definition is deliberately kept here
    so that every detector is evaluated using the same
    FINAL_DISTANCE_THRESHOLD.
    """
    normal, d = canonicalise_plane(
        normal,
        d,
    )

    distances = np.abs(
        points @ normal + d
    )

    mask = (
        distances
        < FINAL_DISTANCE_THRESHOLD
    )

    support = int(
        np.sum(mask)
    )

    if support < MIN_PLANE_SUPPORT:
        return None

    return {
        "normal": normal,
        "d": float(d),
        "mask": mask,
        "support": support,
        "source_support": source_support,
    }


def merge_duplicate_models(
    models,
    normal_threshold_deg=5.0,
    offset_threshold_m=0.05,
):
    """
    Remove duplicate detections of the same geometric plane.
    Keep the stronger model.
    """
    models = sorted(
        models,
        key=lambda model: model["support"],
        reverse=True,
    )

    accepted = []

    for model in models:
        duplicate = False

        for existing in accepted:
            n2, d2 = align_plane_to_reference(
                model["normal"].copy(),
                model["d"],
                existing["normal"],
            )

            if (
                plane_angle_deg(
                    existing["normal"],
                    n2,
                )
                <= normal_threshold_deg
                and abs(
                    existing["d"] - d2
                )
                <= offset_threshold_m
            ):
                duplicate = True
                break

        if not duplicate:
            accepted.append(model)

        if len(accepted) >= MAX_PLANES:
            break

    return accepted


# ==========================================================
# Local normals / curvature
# ==========================================================

def estimate_local_geometry(points):
    if len(points) <= K_NEIGHBORS:
        return None

    tree = cKDTree(points)

    _, neighbour_indices = tree.query(
        points,
        k=K_NEIGHBORS + 1,
    )

    neighbour_indices = neighbour_indices[:, 1:]

    neighbours = points[neighbour_indices]
    centroids = np.mean(neighbours, axis=1)

    centred = (
        neighbours
        - centroids[:, None, :]
    )

    covariance = np.einsum(
        "nki,nkj->nij",
        centred,
        centred,
    )

    covariance /= float(
        K_NEIGHBORS
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    normals = eigenvectors[:, :, 0]

    eigenvalue_sum = np.sum(
        eigenvalues,
        axis=1,
    )

    curvature = np.ones(
        len(points),
        dtype=np.float64,
    )

    valid = (
        eigenvalue_sum
        > 1e-12
    )

    curvature[valid] = (
        eigenvalues[valid, 0]
        / eigenvalue_sum[valid]
    )

    for i in range(len(normals)):
        n, _ = canonicalise_plane(
            normals[i],
            0.0,
        )

        normals[i] = n

    return (
        normals,
        curvature,
        neighbour_indices,
    )
