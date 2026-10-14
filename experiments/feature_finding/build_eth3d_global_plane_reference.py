from __future__ import annotations

"""
build_eth3d_global_plane_reference.py

Build a global planar reference from the ETH3D TUM-derived mesh.

Place in:
    D:/PlaneBasedSLAM/experiments/feature_finding/
        build_eth3d_global_plane_reference.py

Input:
    data/eth3d/extracted/rgbd_dataset_freiburg1_xyz.ply

Output:
    data/derived_gt/fr1_xyz/global_plane_reference/
        global_planes.json
        global_planes.npz
        plane_component_labels.npy
        simplified_mesh.ply
        plane_reference_points.ply

Method:
    ETH3D triangle mesh
        -> optional mesh simplification
        -> triangle normals / centroids / areas
        -> shared-edge face adjacency
        -> local coplanarity test
        -> connected planar components
        -> SVD refinement per component
        -> filter by physical area / support

This does NOT use any of the five project detector implementations.
It is an offline mesh-topology-derived reference builder.

Important:
    The result is a derived reference, not official TUM plane annotation.
    It should be visually sanity-checked once globally before being used
    for per-frame GT generation.
"""

import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


# ============================================================
# USER SETTINGS
# ============================================================

SEQUENCE_NAME = "rgbd_dataset_freiburg1_xyz"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MESH_FILE = (
    PROJECT_ROOT
    / "data"
    / "eth3d"
    / "extracted"
    / f"{SEQUENCE_NAME}.ply"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "derived_gt"
    / "fr1_xyz"
    / "global_plane_reference"
)

# The original mesh has ~1.35M triangles. Simplifying first makes
# topology segmentation much faster while preserving large planes.
SIMPLIFY_MESH = True
TARGET_TRIANGLES = 250_000

# Adjacent triangles are considered locally coplanar if:
LOCAL_NORMAL_ANGLE_DEG = 6.0
LOCAL_PLANE_DISTANCE_M = 0.025

# Keep only components large enough to be useful SLAM planes.
MIN_COMPONENT_TRIANGLES = 80
MIN_PLANE_AREA_M2 = 0.035

# Refined plane quality.
MAX_REFINED_RMS_M = 0.030

# To avoid huge SVDs.
MAX_FIT_VERTICES = 25_000

# Debug/reference point cloud sampled from accepted plane triangles.
REFERENCE_POINTS_PER_M2 = 8_000
MIN_REFERENCE_POINTS_PER_PLANE = 1_000
MAX_REFERENCE_POINTS_PER_PLANE = 30_000

RANDOM_SEED = 20260919
SHOW_RESULT = True

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def canonicalise_plane(normal, d):
    normal = np.asarray(normal, dtype=np.float64)
    norm = np.linalg.norm(normal)

    if norm < 1e-12:
        raise ValueError("Cannot canonicalise zero normal.")

    normal = normal / norm
    d = float(d) / norm

    idx = int(np.argmax(np.abs(normal)))

    if normal[idx] < 0:
        normal = -normal
        d = -d

    return normal, d


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

    if norm < 1e-12:
        return None

    normal = normal / norm
    d = -float(np.dot(normal, centroid))
    normal, d = canonicalise_plane(normal, d)

    distances = np.abs(points @ normal + d)
    rms = float(np.sqrt(np.mean(distances ** 2)))

    return normal, d, centroid, rms


def triangle_geometry(vertices, triangles):
    a = vertices[triangles[:, 0]]
    b = vertices[triangles[:, 1]]
    c = vertices[triangles[:, 2]]

    cross = np.cross(b - a, c - a)
    double_area = np.linalg.norm(cross, axis=1)

    valid = double_area > 1e-12

    normals = np.zeros_like(cross)
    normals[valid] = cross[valid] / double_area[valid, None]

    areas = 0.5 * double_area
    centroids = (a + b + c) / 3.0

    return normals, centroids, areas, valid


# ============================================================
# FACE ADJACENCY
# ============================================================

def build_shared_edge_adjacency(triangles):
    """
    Build face adjacency using shared undirected mesh edges.

    Returns:
        face_a, face_b
    where each pair shares an edge.
    """

    n_faces = len(triangles)

    face_ids = np.repeat(
        np.arange(n_faces, dtype=np.int32),
        3,
    )

    edges = np.concatenate(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ],
        axis=0,
    ).astype(np.int32, copy=False)

    # IMPORTANT: face_ids must follow the same block order as edges.
    face_ids = np.concatenate(
        [
            np.arange(n_faces, dtype=np.int32),
            np.arange(n_faces, dtype=np.int32),
            np.arange(n_faces, dtype=np.int32),
        ]
    )

    edges.sort(axis=1)

    # Lexicographic sort by (v0, v1).
    order = np.lexsort(
        (edges[:, 1], edges[:, 0])
    )

    edges_sorted = edges[order]
    faces_sorted = face_ids[order]

    same_as_previous = np.zeros(
        len(edges_sorted),
        dtype=bool,
    )

    same_as_previous[1:] = np.all(
        edges_sorted[1:] == edges_sorted[:-1],
        axis=1,
    )

    # Standard manifold mesh: a shared edge appears exactly twice.
    idx_second = np.flatnonzero(same_as_previous)

    face_a = faces_sorted[idx_second - 1]
    face_b = faces_sorted[idx_second]

    keep = face_a != face_b

    return (
        face_a[keep].astype(np.int32),
        face_b[keep].astype(np.int32),
    )


# ============================================================
# LOCAL PLANAR CONNECTIVITY
# ============================================================

def planar_adjacency(
    face_a,
    face_b,
    normals,
    centroids,
):
    n1 = normals[face_a]
    n2 = normals[face_b]

    dot = np.einsum(
        "ij,ij->i",
        n1,
        n2,
    )

    # Triangle winding may differ locally. Plane normal sign is irrelevant.
    abs_dot = np.clip(
        np.abs(dot),
        0.0,
        1.0,
    )

    cos_threshold = math.cos(
        math.radians(
            LOCAL_NORMAL_ANGLE_DEG
        )
    )

    normal_ok = abs_dot >= cos_threshold

    delta = centroids[face_b] - centroids[face_a]

    # Symmetric point-to-plane separation.
    dist_1 = np.abs(
        np.einsum(
            "ij,ij->i",
            n1,
            delta,
        )
    )

    dist_2 = np.abs(
        np.einsum(
            "ij,ij->i",
            n2,
            delta,
        )
    )

    distance_ok = (
        (dist_1 <= LOCAL_PLANE_DISTANCE_M)
        & (dist_2 <= LOCAL_PLANE_DISTANCE_M)
    )

    return normal_ok & distance_ok


# ============================================================
# COMPONENT REFINEMENT
# ============================================================

def refine_component(
    component_id,
    face_indices,
    vertices,
    triangles,
    triangle_areas,
):
    if len(face_indices) < MIN_COMPONENT_TRIANGLES:
        return None

    area = float(
        np.sum(
            triangle_areas[face_indices]
        )
    )

    if area < MIN_PLANE_AREA_M2:
        return None

    component_triangles = triangles[
        face_indices
    ]

    vertex_indices = np.unique(
        component_triangles.reshape(-1)
    )

    component_vertices = vertices[
        vertex_indices
    ]

    if len(component_vertices) > MAX_FIT_VERTICES:
        chosen = rng.choice(
            len(component_vertices),
            size=MAX_FIT_VERTICES,
            replace=False,
        )

        fit_vertices = component_vertices[
            chosen
        ]

    else:
        fit_vertices = component_vertices

    fitted = fit_plane_svd(
        fit_vertices
    )

    if fitted is None:
        return None

    normal, d, centroid, rms = fitted

    if rms > MAX_REFINED_RMS_M:
        return None

    # Compute simple plane-aligned extents.
    axis_helper = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float64,
    )

    if abs(np.dot(axis_helper, normal)) > 0.9:
        axis_helper = np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        )

    u = np.cross(
        normal,
        axis_helper,
    )
    u /= np.linalg.norm(u)

    v = np.cross(
        normal,
        u,
    )
    v /= np.linalg.norm(v)

    relative = component_vertices - centroid

    coord_u = relative @ u
    coord_v = relative @ v

    extent_u = float(
        np.max(coord_u) - np.min(coord_u)
    )

    extent_v = float(
        np.max(coord_v) - np.min(coord_v)
    )

    return {
        "component_id": int(component_id),
        "face_indices": face_indices.astype(np.int32),
        "vertex_indices": vertex_indices.astype(np.int32),
        "normal": normal,
        "offset": float(d),
        "centroid": centroid,
        "area_m2": area,
        "rms_m": rms,
        "extent_u_m": extent_u,
        "extent_v_m": extent_v,
        "triangle_count": int(len(face_indices)),
        "vertex_count": int(len(vertex_indices)),
    }


# ============================================================
# REFERENCE POINT SAMPLING
# ============================================================

def sample_plane_points(
    vertices,
    triangles,
    face_indices,
    triangle_areas,
    number_points,
):
    """
    Uniformly sample points over the accepted triangles.
    """

    local_areas = triangle_areas[
        face_indices
    ]

    total_area = float(
        np.sum(local_areas)
    )

    probabilities = (
        local_areas / total_area
    )

    chosen_local = rng.choice(
        len(face_indices),
        size=number_points,
        replace=True,
        p=probabilities,
    )

    chosen_faces = face_indices[
        chosen_local
    ]

    tri = triangles[
        chosen_faces
    ]

    a = vertices[
        tri[:, 0]
    ]

    b = vertices[
        tri[:, 1]
    ]

    c = vertices[
        tri[:, 2]
    ]

    r1 = np.sqrt(
        rng.random(
            number_points
        )
    )

    r2 = rng.random(
        number_points
    )

    points = (
        (1.0 - r1)[:, None] * a
        + (
            r1 * (1.0 - r2)
        )[:, None] * b
        + (
            r1 * r2
        )[:, None] * c
    )

    return points


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MESH_FILE.exists():
        raise FileNotFoundError(
            f"ETH3D mesh not found:\n{MESH_FILE}"
        )

    print()
    print("=" * 92)
    print("ETH3D GLOBAL PLANE REFERENCE BUILDER")
    print("=" * 92)
    print(f"Input mesh : {MESH_FILE}")
    print()

    mesh = o3d.io.read_triangle_mesh(
        str(MESH_FILE)
    )

    if (
        len(mesh.vertices) == 0
        or len(mesh.triangles) == 0
    ):
        raise RuntimeError(
            "The input PLY did not load as a triangle mesh."
        )

    print(
        f"Original mesh: "
        f"{len(mesh.vertices):,} vertices, "
        f"{len(mesh.triangles):,} triangles"
    )

    if SIMPLIFY_MESH and len(mesh.triangles) > TARGET_TRIANGLES:
        print()
        print(
            f"Simplifying mesh to approximately "
            f"{TARGET_TRIANGLES:,} triangles..."
        )

        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=
                TARGET_TRIANGLES
        )

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()

    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()

    print(
        f"Working mesh : "
        f"{len(mesh.vertices):,} vertices, "
        f"{len(mesh.triangles):,} triangles"
    )

    simplified_path = (
        OUTPUT_DIR
        / "simplified_mesh.ply"
    )

    o3d.io.write_triangle_mesh(
        str(simplified_path),
        mesh,
        write_ascii=False,
        compressed=True,
    )

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float64,
    )

    triangles = np.asarray(
        mesh.triangles,
        dtype=np.int32,
    )

    normals, centroids, areas, valid_faces = (
        triangle_geometry(
            vertices,
            triangles,
        )
    )

    print()
    print("Building shared-edge face adjacency...")

    face_a, face_b = (
        build_shared_edge_adjacency(
            triangles
        )
    )

    print(
        f"Shared-edge pairs: "
        f"{len(face_a):,}"
    )

    print()
    print(
        "Applying local coplanarity test "
        f"(angle <= {LOCAL_NORMAL_ANGLE_DEG:.1f} deg, "
        f"distance <= {LOCAL_PLANE_DISTANCE_M*1000:.1f} mm)..."
    )

    coplanar = planar_adjacency(
        face_a,
        face_b,
        normals,
        centroids,
    )

    face_a_good = face_a[coplanar]
    face_b_good = face_b[coplanar]

    print(
        f"Coplanar adjacency pairs: "
        f"{len(face_a_good):,}"
    )

    n_faces = len(triangles)

    rows = np.concatenate(
        [
            face_a_good,
            face_b_good,
        ]
    )

    cols = np.concatenate(
        [
            face_b_good,
            face_a_good,
        ]
    )

    data = np.ones(
        len(rows),
        dtype=np.uint8,
    )

    graph = coo_matrix(
        (
            data,
            (
                rows,
                cols,
            ),
        ),
        shape=(
            n_faces,
            n_faces,
        ),
    ).tocsr()

    print()
    print("Finding connected planar components...")

    n_components, labels = (
        connected_components(
            graph,
            directed=False,
            return_labels=True,
        )
    )

    print(
        f"Raw connected components: "
        f"{n_components:,}"
    )

    np.save(
        OUTPUT_DIR
        / "plane_component_labels.npy",
        labels,
    )

    counts = np.bincount(
        labels,
        minlength=n_components,
    )

    candidate_component_ids = np.flatnonzero(
        counts >= MIN_COMPONENT_TRIANGLES
    )

    print(
        f"Components with >= "
        f"{MIN_COMPONENT_TRIANGLES} triangles: "
        f"{len(candidate_component_ids):,}"
    )

    # Sort by triangle count first for deterministic IDs.
    candidate_component_ids = sorted(
        candidate_component_ids.tolist(),
        key=lambda cid: int(counts[cid]),
        reverse=True,
    )

    accepted = []

    print()
    print("Refining candidate planes...")

    for idx, component_id in enumerate(
        candidate_component_ids,
        start=1,
    ):
        face_indices = np.flatnonzero(
            labels == component_id
        ).astype(np.int32)

        plane = refine_component(
            component_id,
            face_indices,
            vertices,
            triangles,
            areas,
        )

        if plane is None:
            continue

        accepted.append(
            plane
        )

        print(
            f"  accepted provisional plane "
            f"{len(accepted)-1:02d}: "
            f"area={plane['area_m2']:.3f} m^2, "
            f"faces={plane['triangle_count']:,}, "
            f"RMS={plane['rms_m']*1000:.2f} mm"
        )

    # Final deterministic order by physical area.
    accepted.sort(
        key=lambda p: p["area_m2"],
        reverse=True,
    )

    print()
    print(
        f"Accepted global planes: "
        f"{len(accepted)}"
    )

    # --------------------------------------------------------
    # Save compact arrays
    # --------------------------------------------------------

    plane_ids = np.arange(
        len(accepted),
        dtype=np.int32,
    )

    if accepted:
        plane_normals = np.stack(
            [
                p["normal"]
                for p in accepted
            ],
            axis=0,
        )

        plane_offsets = np.asarray(
            [
                p["offset"]
                for p in accepted
            ],
            dtype=np.float64,
        )

        plane_centroids = np.stack(
            [
                p["centroid"]
                for p in accepted
            ],
            axis=0,
        )

        plane_areas = np.asarray(
            [
                p["area_m2"]
                for p in accepted
            ],
            dtype=np.float64,
        )

        plane_rms = np.asarray(
            [
                p["rms_m"]
                for p in accepted
            ],
            dtype=np.float64,
        )

        plane_triangle_counts = np.asarray(
            [
                p["triangle_count"]
                for p in accepted
            ],
            dtype=np.int32,
        )

        plane_component_ids = np.asarray(
            [
                p["component_id"]
                for p in accepted
            ],
            dtype=np.int32,
        )

    else:
        plane_normals = np.empty(
            (0, 3),
            dtype=np.float64,
        )
        plane_offsets = np.empty(
            0,
            dtype=np.float64,
        )
        plane_centroids = np.empty(
            (0, 3),
            dtype=np.float64,
        )
        plane_areas = np.empty(
            0,
            dtype=np.float64,
        )
        plane_rms = np.empty(
            0,
            dtype=np.float64,
        )
        plane_triangle_counts = np.empty(
            0,
            dtype=np.int32,
        )
        plane_component_ids = np.empty(
            0,
            dtype=np.int32,
        )

    np.savez_compressed(
        OUTPUT_DIR
        / "global_planes.npz",
        plane_ids=plane_ids,
        normals_world=plane_normals,
        offsets_world=plane_offsets,
        centroids_world=plane_centroids,
        areas_m2=plane_areas,
        rms_m=plane_rms,
        triangle_counts=plane_triangle_counts,
        component_ids=plane_component_ids,
    )

    # --------------------------------------------------------
    # Save JSON metadata
    # --------------------------------------------------------

    json_planes = []

    for plane_id, p in enumerate(
        accepted
    ):
        json_planes.append(
            {
                "plane_id":
                    int(plane_id),

                "component_id":
                    int(
                        p["component_id"]
                    ),

                "normal_world":
                    [
                        float(x)
                        for x in p["normal"]
                    ],

                "offset_world":
                    float(
                        p["offset"]
                    ),

                "centroid_world":
                    [
                        float(x)
                        for x in p["centroid"]
                    ],

                "area_m2":
                    float(
                        p["area_m2"]
                    ),

                "rms_m":
                    float(
                        p["rms_m"]
                    ),

                "extent_u_m":
                    float(
                        p["extent_u_m"]
                    ),

                "extent_v_m":
                    float(
                        p["extent_v_m"]
                    ),

                "triangle_count":
                    int(
                        p["triangle_count"]
                    ),

                "vertex_count":
                    int(
                        p["vertex_count"]
                    ),
            }
        )

    metadata = {
        "source":
            "ETH3D TUM-derived reconstruction mesh",

        "sequence":
            SEQUENCE_NAME,

        "note":
            (
                "Derived global planar reference; "
                "not official TUM plane annotation."
            ),

        "settings": {
            "simplify_mesh":
                SIMPLIFY_MESH,

            "target_triangles":
                TARGET_TRIANGLES,

            "local_normal_angle_deg":
                LOCAL_NORMAL_ANGLE_DEG,

            "local_plane_distance_m":
                LOCAL_PLANE_DISTANCE_M,

            "min_component_triangles":
                MIN_COMPONENT_TRIANGLES,

            "min_plane_area_m2":
                MIN_PLANE_AREA_M2,

            "max_refined_rms_m":
                MAX_REFINED_RMS_M,
        },

        "number_of_planes":
            len(json_planes),

        "planes":
            json_planes,
    }

    with open(
        OUTPUT_DIR
        / "global_planes.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Sample accepted plane surfaces into one debug cloud.
    # --------------------------------------------------------

    reference_points = []
    reference_colors = []

    for plane_id, p in enumerate(
        accepted
    ):
        n_points = int(
            np.clip(
                p["area_m2"]
                * REFERENCE_POINTS_PER_M2,
                MIN_REFERENCE_POINTS_PER_PLANE,
                MAX_REFERENCE_POINTS_PER_PLANE,
            )
        )

        points = sample_plane_points(
            vertices,
            triangles,
            p["face_indices"],
            areas,
            n_points,
        )

        reference_points.append(
            points
        )

        # Deterministic visually distinct RGB values.
        local_rng = np.random.default_rng(
            RANDOM_SEED + plane_id
        )

        colour = local_rng.uniform(
            0.15,
            0.95,
            size=3,
        )

        reference_colors.append(
            np.repeat(
                colour[None, :],
                n_points,
                axis=0,
            )
        )

    if reference_points:
        reference_points = np.vstack(
            reference_points
        )

        reference_colors = np.vstack(
            reference_colors
        )

        reference_cloud = (
            o3d.geometry.PointCloud()
        )

        reference_cloud.points = (
            o3d.utility.Vector3dVector(
                reference_points
            )
        )

        reference_cloud.colors = (
            o3d.utility.Vector3dVector(
                reference_colors
            )
        )

        reference_path = (
            OUTPUT_DIR
            / "plane_reference_points.ply"
        )

        o3d.io.write_point_cloud(
            str(reference_path),
            reference_cloud,
            write_ascii=False,
            compressed=True,
        )

    else:
        reference_cloud = None
        reference_path = None

    print()
    print("=" * 92)
    print("GLOBAL PLANE REFERENCE COMPLETE")
    print("=" * 92)
    print(
        f"Planes retained : "
        f"{len(accepted)}"
    )
    print(
        f"Metadata        : "
        f"{OUTPUT_DIR / 'global_planes.json'}"
    )
    print(
        f"Arrays          : "
        f"{OUTPUT_DIR / 'global_planes.npz'}"
    )

    if reference_path is not None:
        print(
            f"Debug cloud     : "
            f"{reference_path}"
        )

    print()
    print(
        "IMPORTANT: visually inspect the coloured plane reference once "
        "before projecting it into TUM frames."
    )

    if (
        SHOW_RESULT
        and reference_cloud is not None
    ):
        o3d.visualization.draw_geometries(
            [reference_cloud],
            window_name=
                "ETH3D-derived global plane reference",
            width=1280,
            height=800,
        )


if __name__ == "__main__":
    main()
