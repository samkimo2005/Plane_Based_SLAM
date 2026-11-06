import numpy as np


EPS = 1e-12


# ============================================================
# Basic utilities
# ============================================================

def _as_vec3(x):
    x = np.asarray(x, dtype=float).reshape(3)
    return x


def _normalise_vector(v):
    v = _as_vec3(v)
    norm = np.linalg.norm(v)

    if norm < EPS:
        raise ValueError("Zero-length vector cannot be normalised.")

    return v / norm


# ============================================================
# Plane representation
# Plane equation:
#
#       n^T x + d = 0
#
# where ||n|| = 1
# ============================================================

def normalise_plane(normal, d):
    """
    Normalise plane parameters so that ||normal|| = 1.
    """

    normal = _as_vec3(normal)
    d = float(d)

    norm = np.linalg.norm(normal)

    if norm < EPS:
        raise ValueError("Plane normal must not be zero.")

    return normal / norm, d / norm


def canonicalise_plane(normal, d):
    """
    Give a plane a deterministic sign for storage.

    Note:
        (n, d) and (-n, -d) describe the same physical plane.

    The component of n with the largest absolute value is forced
    to be positive.

    This is mainly useful for storage / printing.
    Cross-frame comparison should still use sign-aware functions.
    """

    normal, d = normalise_plane(normal, d)

    idx = np.argmax(np.abs(normal))

    if normal[idx] < 0:
        normal = -normal
        d = -d

    return normal, d


def align_plane_sign(reference_normal, normal, d):
    """
    Flip (normal, d) if necessary so that its normal points in
    approximately the same direction as reference_normal.
    """

    reference_normal = _normalise_vector(reference_normal)
    normal, d = normalise_plane(normal, d)

    if np.dot(reference_normal, normal) < 0:
        normal = -normal
        d = -d

    return normal, d


# ============================================================
# Plane / point transformation
# ============================================================

def transform_point(point, R, t):
    """
    Transform a 3D point from frame A to frame B:

        x_B = R x_A + t
    """

    point = _as_vec3(point)
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = _as_vec3(t)

    return R @ point + t


def transform_plane(normal, d, R, t):
    """
    Transform plane from frame A to frame B.

    Plane in A:
        n_A^T x_A + d_A = 0

    Point transformation:
        x_B = R x_A + t

    Result:
        n_B = R n_A
        d_B = d_A - n_B^T t
    """

    normal, d = normalise_plane(normal, d)

    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = _as_vec3(t)

    normal_b = R @ normal
    d_b = d - np.dot(normal_b, t)

    return normalise_plane(normal_b, d_b)


# ============================================================
# Plane residuals
# ============================================================

def normal_angle_deg(normal_a, normal_b):
    """
    Sign-invariant angular difference between two plane normals.

    Because n and -n describe the same plane:

        angle = acos(|n_a^T n_b|)
    """

    normal_a = _normalise_vector(normal_a)
    normal_b = _normalise_vector(normal_b)

    cos_angle = abs(np.dot(normal_a, normal_b))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


def plane_offset_residual(normal_a, d_a, normal_b, d_b):
    """
    Absolute offset difference between two planes after resolving
    normal-sign ambiguity.
    """

    normal_a, d_a = normalise_plane(normal_a, d_a)
    normal_b, d_b = normalise_plane(normal_b, d_b)

    normal_b, d_b = align_plane_sign(
        reference_normal=normal_a,
        normal=normal_b,
        d=d_b
    )

    return abs(d_a - d_b)


def centroid_distance(centroid_a, centroid_b):
    """
    Euclidean distance between two plane centroids.
    """

    centroid_a = _as_vec3(centroid_a)
    centroid_b = _as_vec3(centroid_b)

    return float(np.linalg.norm(centroid_a - centroid_b))


# ============================================================
# Rotation estimation from matched plane normals
# ============================================================

def estimate_rotation_from_plane_normals(normals_a, normals_b):
    """
    Estimate rotation R such that:

        n_B ≈ R n_A

    using SVD / orthogonal Procrustes.

    normals_a and normals_b:
        arrays of shape (N, 3)

    Important:
        This function resolves sign ambiguity by aligning each
        n_B with n_A before SVD.

        This is suitable for normal adjacent-frame SLAM motion,
        where inter-frame rotation is not extreme.
    """

    normals_a = np.asarray(normals_a, dtype=float)
    normals_b = np.asarray(normals_b, dtype=float)

    if normals_a.ndim != 2 or normals_a.shape[1] != 3:
        raise ValueError("normals_a must have shape (N, 3).")

    if normals_b.shape != normals_a.shape:
        raise ValueError(
            "normals_a and normals_b must have the same shape."
        )

    if len(normals_a) < 2:
        raise ValueError(
            "At least two plane-normal correspondences are required."
        )

    A = np.zeros_like(normals_a)
    B = np.zeros_like(normals_b)

    for i in range(len(normals_a)):
        na = _normalise_vector(normals_a[i])
        nb = _normalise_vector(normals_b[i])

        # Resolve plane normal sign ambiguity.
        if np.dot(na, nb) < 0:
            nb = -nb

        A[i] = na
        B[i] = nb

    H = A.T @ B

    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # Prevent reflection.
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


# ============================================================
# Translation estimation from plane constraints
# ============================================================

def estimate_translation_from_planes(
    planes_a,
    planes_b,
    R
):
    """
    Estimate translation t after rotation R is known.

    Plane transformation gives:

        d_B = d_A - n_B^T t

    therefore:

        n_B^T t = d_A - d_B

    Stack all matched planes:

        A t = b

    and solve by least squares.

    Plane arrays must have shape:

        (N, 4)

    with rows:

        [nx, ny, nz, d]
    """

    planes_a = np.asarray(planes_a, dtype=float)
    planes_b = np.asarray(planes_b, dtype=float)
    R = np.asarray(R, dtype=float).reshape(3, 3)

    if planes_a.ndim != 2 or planes_a.shape[1] != 4:
        raise ValueError("planes_a must have shape (N, 4).")

    if planes_b.shape != planes_a.shape:
        raise ValueError(
            "planes_a and planes_b must have the same shape."
        )

    if len(planes_a) < 3:
        raise ValueError(
            "At least three plane correspondences are required "
            "to constrain 3D translation."
        )

    A_rows = []
    b_values = []

    for pa, pb in zip(planes_a, planes_b):

        na, da = normalise_plane(pa[:3], pa[3])
        nb, db = normalise_plane(pb[:3], pb[3])

        predicted_nb = R @ na

        # Align observed B-plane sign with predicted B normal.
        nb, db = align_plane_sign(
            reference_normal=predicted_nb,
            normal=nb,
            d=db
        )

        A_rows.append(nb)
        b_values.append(da - db)

    A = np.asarray(A_rows)
    b = np.asarray(b_values)

    t, _, rank, singular_values = np.linalg.lstsq(
        A,
        b,
        rcond=None
    )

    if len(singular_values) > 0 and singular_values[-1] > EPS:
        condition_number = (
            singular_values[0] / singular_values[-1]
        )
    else:
        condition_number = np.inf

    diagnostics = {
        "rank": int(rank),
        "singular_values": singular_values,
        "condition_number": float(condition_number),
        "degenerate": bool(rank < 3),
    }

    return t, diagnostics


# ============================================================
# Full plane-based relative pose estimation
# ============================================================

def estimate_pose_from_plane_matches(planes_a, planes_b):
    """
    Estimate relative pose from matched planes.

    Returns transformation from frame A to frame B:

        x_B = R x_A + t
    """

    planes_a = np.asarray(planes_a, dtype=float)
    planes_b = np.asarray(planes_b, dtype=float)

    if planes_a.shape != planes_b.shape:
        raise ValueError(
            "planes_a and planes_b must have the same shape."
        )

    if planes_a.ndim != 2 or planes_a.shape[1] != 4:
        raise ValueError(
            "Plane arrays must have shape (N, 4)."
        )

    if len(planes_a) < 3:
        raise ValueError(
            "At least three matched planes are required."
        )

    normals_a = planes_a[:, :3]
    normals_b = planes_b[:, :3]

    R = estimate_rotation_from_plane_normals(
        normals_a,
        normals_b
    )

    t, translation_diagnostics = (
        estimate_translation_from_planes(
            planes_a,
            planes_b,
            R
        )
    )

    return {
        "R": R,
        "t": t,
        "translation_diagnostics": translation_diagnostics,
    }


# ============================================================
# Pose error utilities
# ============================================================

def rotation_error_deg(R_est, R_gt):
    """
    Angular difference between two rotation matrices.
    """

    R_est = np.asarray(R_est, dtype=float).reshape(3, 3)
    R_gt = np.asarray(R_gt, dtype=float).reshape(3, 3)

    R_error = R_gt.T @ R_est

    value = (np.trace(R_error) - 1.0) / 2.0
    value = np.clip(value, -1.0, 1.0)

    return float(np.degrees(np.arccos(value)))


def translation_error(t_est, t_gt):
    """
    Euclidean translation error.
    """

    t_est = _as_vec3(t_est)
    t_gt = _as_vec3(t_gt)

    return float(np.linalg.norm(t_est - t_gt))


# ============================================================
# Built-in verification test
# ============================================================

def _self_test():

    print("=" * 60)
    print("BAD-MATCH REJECTION COMMON GEOMETRY TEST")
    print("=" * 60)

    # Four non-degenerate reference planes in Frame A.
    normals_a = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ])

    normals_a = normals_a / np.linalg.norm(
        normals_a,
        axis=1,
        keepdims=True
    )

    offsets_a = np.array([
        -1.0,
        -2.0,
        -3.0,
        -1.5
    ])

    planes_a = np.column_stack(
        [normals_a, offsets_a]
    )

    # Known ground-truth rotation: 12 degrees around Z.
    theta = np.deg2rad(12.0)

    R_gt = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ])

    # Known translation.
    t_gt = np.array([
        0.20,
        -0.10,
        0.05
    ])

    # Generate Frame-B planes from the known transformation.
    planes_b = []

    for plane in planes_a:

        n_b, d_b = transform_plane(
            plane[:3],
            plane[3],
            R_gt,
            t_gt
        )

        planes_b.append([
            n_b[0],
            n_b[1],
            n_b[2],
            d_b
        ])

    planes_b = np.asarray(planes_b)

    # Deliberately flip one plane sign to verify sign handling.
    planes_b[1] *= -1.0

    result = estimate_pose_from_plane_matches(
        planes_a,
        planes_b
    )

    R_est = result["R"]
    t_est = result["t"]

    rot_error = rotation_error_deg(
        R_est,
        R_gt
    )

    trans_error = translation_error(
        t_est,
        t_gt
    )

    print("\nGround-truth rotation:")
    print(R_gt)

    print("\nEstimated rotation:")
    print(R_est)

    print("\nGround-truth translation:")
    print(t_gt)

    print("\nEstimated translation:")
    print(t_est)

    print("\nRotation error (deg):")
    print(rot_error)

    print("\nTranslation error (m):")
    print(trans_error)

    print("\nGeometry diagnostics:")
    print(
        result["translation_diagnostics"]
    )

    print("\n" + "-" * 60)

    if (
        rot_error < 1e-6
        and trans_error < 1e-6
        and not result[
            "translation_diagnostics"
        ]["degenerate"]
    ):
        print("PASS: common geometry functions are working.")
    else:
        print("FAIL: common geometry test did not pass.")

    print("-" * 60)

def estimate_pose_from_assigned_pairs(
    planes_a,
    planes_b,
    assigned_pairs,
):
    """
    Convert index correspondences into matched plane arrays,
    then use the existing pose estimator.
    """

    pairs = np.asarray(assigned_pairs, dtype=int)

    matched_a = np.asarray([
        planes_a[ia]
        for ia, _ in pairs
    ])

    matched_b = np.asarray([
        planes_b[ib]
        for _, ib in pairs
    ])

    return estimate_pose_from_plane_matches(
        matched_a,
        matched_b,
    )

def main():
    _self_test()


if __name__ == "__main__":
    main()