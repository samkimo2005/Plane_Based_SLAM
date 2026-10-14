import numpy as np


# ============================================================
# TUM depth format
# ============================================================

DEPTH_FACTOR = 5000.0


# ============================================================
# Depth -> point cloud
# ============================================================

def depth_to_point_cloud(
    depth,
    intrinsics,
    min_depth=0.3,
    max_depth=5.0,
):
    """
    Convert a TUM depth image into an N x 3 point cloud.

    Plane-detection convention:

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth / 5000

    Parameters
    ----------
    depth : ndarray
        Raw TUM 16-bit depth image.

    intrinsics : dict
        Must contain:
            fx, fy, cx, cy

    min_depth : float
        Minimum accepted depth in metres.

    max_depth : float
        Maximum accepted depth in metres.

    Returns
    -------
    points : ndarray, shape (N, 3)
        Valid 3D points.

    pixels : ndarray, shape (N, 2)
        Corresponding image coordinates [u, v].

        These are preserved because they may later be useful
        for TUM-derived plane-correspondence ground truth.
    """

    if intrinsics is None:
        raise ValueError(
            "Camera intrinsics must be provided."
        )

    required_keys = {
        "fx",
        "fy",
        "cx",
        "cy",
    }

    missing = (
        required_keys
        - set(intrinsics.keys())
    )

    if missing:
        raise ValueError(
            f"Missing intrinsic parameters: {missing}"
        )

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    depth = np.asarray(depth)

    if depth.ndim != 2:
        raise ValueError(
            "Depth image must be a 2D array."
        )

    # TUM PNG depth values -> metres
    depth_m = (
        depth.astype(np.float64)
        / DEPTH_FACTOR
    )

    height, width = depth.shape

    u, v = np.meshgrid(
        np.arange(width),
        np.arange(height),
    )

    valid = (
        (depth > 0)
        & np.isfinite(depth_m)
        & (depth_m >= min_depth)
        & (depth_m <= max_depth)
    )

    z = depth_m[valid]

    x = (
        (u[valid] - cx)
        * z
        / fx
    )

    y = (
        (v[valid] - cy)
        * z
        / fy
    )

    points = np.column_stack(
        (
            x,
            y,
            z,
        )
    )

    pixels = np.column_stack(
        (
            u[valid],
            v[valid],
        )
    ).astype(np.int32)

    return points, pixels
