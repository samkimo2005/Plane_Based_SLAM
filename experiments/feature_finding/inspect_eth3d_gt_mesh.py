from __future__ import annotations

"""
inspect_eth3d_gt_mesh.py

Place this file in:
    D:\PlaneBasedSLAM\experiments\feature_finding\inspect_eth3d_gt_mesh.py

Purpose
-------
1. Automatically find the ETH3D TUM-derived Freiburg1 XYZ reconstruction.
2. Load it with Open3D.
3. Report whether it is a triangle mesh or a point cloud.
4. Print size / bounds information.
5. Save a uniformly sampled point cloud that can be used by later
   global-plane extraction code.

This script does NOT yet create plane GT.
"""

from pathlib import Path
import sys

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:
    raise ImportError(
        "Open3D is not installed in the active environment.\n"
        "Install it with:\n"
        "    pip install open3d"
    ) from exc


# ============================================================
# USER SETTINGS
# ============================================================

SEQUENCE_KEY = "freiburg1_xyz"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ETH3D_EXTRACTED_DIR = (
    PROJECT_ROOT
    / "data"
    / "eth3d"
    / "extracted"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "derived_gt"
    / "fr1_xyz"
    / "eth3d_mesh"
)

SAMPLED_POINT_COUNT = 300_000
SHOW_VIEWER = True


# ============================================================
# FILE DISCOVERY
# ============================================================

SUPPORTED_EXTENSIONS = {".ply", ".obj", ".stl", ".off"}


def find_candidate_files():
    if not ETH3D_EXTRACTED_DIR.exists():
        raise FileNotFoundError(
            f"ETH3D extracted directory does not exist:\n"
            f"{ETH3D_EXTRACTED_DIR}"
        )

    candidates = []

    for path in ETH3D_EXTRACTED_DIR.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        name = path.name.lower()

        if (
            "freiburg1_xyz" in name
            or "freiburg1-xyz" in name
            or "fr1_xyz" in name
            or "fr1-xyz" in name
        ):
            candidates.append(path)

    return sorted(candidates)


# ============================================================
# LOAD GEOMETRY
# ============================================================

def load_geometry(path: Path):
    mesh = o3d.io.read_triangle_mesh(str(path))

    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        return "triangle_mesh", mesh

    pcd = o3d.io.read_point_cloud(str(path))

    if len(pcd.points) > 0:
        return "point_cloud", pcd

    raise RuntimeError(
        f"Open3D could not load useful geometry from:\n{path}"
    )


def print_bounds(points: np.ndarray):
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    extent = maximum - minimum
    center = 0.5 * (minimum + maximum)

    print("Bounds:")
    print(f"  min    = {minimum}")
    print(f"  max    = {maximum}")
    print(f"  extent = {extent}")
    print(f"  center = {center}")


# ============================================================
# MAIN
# ============================================================

def main():
    candidates = find_candidate_files()

    print()
    print("=" * 80)
    print("ETH3D TUM-DERIVED GEOMETRY IMPORT CHECK")
    print("=" * 80)
    print(f"Search directory : {ETH3D_EXTRACTED_DIR}")
    print(f"Sequence key     : {SEQUENCE_KEY}")
    print()

    if not candidates:
        print(
            "No Freiburg1 XYZ geometry file was found automatically.\n"
            "Check the extracted filenames and move the correct file under:\n"
            f"  {ETH3D_EXTRACTED_DIR}\n"
        )
        sys.exit(1)

    if len(candidates) > 1:
        print("Multiple candidates found:")
        for i, path in enumerate(candidates):
            print(f"  [{i}] {path}")

        print()
        print(
            "Using the first candidate automatically.\n"
            "If this is not the correct reconstruction, set the filename "
            "explicitly in the script."
        )

    mesh_path = candidates[0]

    print(f"Selected file    : {mesh_path}")
    print()

    geometry_type, geometry = load_geometry(mesh_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if geometry_type == "triangle_mesh":
        mesh = geometry

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        print("Geometry type    : Triangle mesh")
        print(f"Vertices         : {len(vertices):,}")
        print(f"Triangles        : {len(triangles):,}")
        print(f"Has colors       : {mesh.has_vertex_colors()}")
        print(f"Has normals      : {mesh.has_vertex_normals()}")
        print()

        print_bounds(vertices)

        number_to_sample = min(
            SAMPLED_POINT_COUNT,
            max(10_000, len(vertices)),
        )

        sampled = mesh.sample_points_uniformly(
            number_of_points=number_to_sample
        )

        sampled_path = OUTPUT_DIR / "fr1_xyz_eth3d_sampled.ply"

        ok = o3d.io.write_point_cloud(
            str(sampled_path),
            sampled,
            write_ascii=False,
            compressed=True,
        )

        if not ok:
            raise RuntimeError(
                f"Failed to save sampled point cloud:\n{sampled_path}"
            )

        print()
        print(f"Sampled cloud    : {len(sampled.points):,} points")
        print(f"Saved to         : {sampled_path}")

        if SHOW_VIEWER:
            print()
            print("Opening Open3D viewer...")
            o3d.visualization.draw_geometries(
                [mesh],
                window_name="ETH3D Freiburg1 XYZ reconstruction",
                width=1280,
                height=800,
            )

    else:
        pcd = geometry
        points = np.asarray(pcd.points)

        print("Geometry type    : Point cloud")
        print(f"Points           : {len(points):,}")
        print(f"Has colors       : {pcd.has_colors()}")
        print(f"Has normals      : {pcd.has_normals()}")
        print()

        print_bounds(points)

        output_path = OUTPUT_DIR / "fr1_xyz_eth3d_pointcloud.ply"

        ok = o3d.io.write_point_cloud(
            str(output_path),
            pcd,
            write_ascii=False,
            compressed=True,
        )

        if not ok:
            raise RuntimeError(
                f"Failed to save point cloud:\n{output_path}"
            )

        print()
        print(f"Saved to         : {output_path}")

        if SHOW_VIEWER:
            print()
            print("Opening Open3D viewer...")
            o3d.visualization.draw_geometries(
                [pcd],
                window_name="ETH3D Freiburg1 XYZ reconstruction",
                width=1280,
                height=800,
            )

    print()
    print("=" * 80)
    print("IMPORT CHECK COMPLETE")
    print("=" * 80)
    print(
        "Next step: verify coordinate alignment with the original "
        "TUM ground-truth trajectory before using this geometry "
        "to create plane GT."
    )


if __name__ == "__main__":
    main()