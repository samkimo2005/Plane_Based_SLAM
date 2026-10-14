
import argparse
from pathlib import Path

import open3d as o3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    args = parser.parse_args()

    path = args.file.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Point cloud not found: {path}")

    pcd = o3d.io.read_point_cloud(str(path))

    if pcd.is_empty():
        raise RuntimeError("The point cloud is empty.")

    print(f"File: {path}")
    print(f"Points: {len(pcd.points):,}")
    print(f"Has colours: {pcd.has_colors()}")
    print(f"Bounds: {pcd.get_axis_aligned_bounding_box()}")

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.15,
        origin=[0, 0, 0]
    )

    o3d.visualization.draw_geometries(
        [pcd, axis],
        window_name=path.name,
        width=1280,
        height=800,
        point_show_normal=False
    )


if __name__ == "__main__":
    main()
