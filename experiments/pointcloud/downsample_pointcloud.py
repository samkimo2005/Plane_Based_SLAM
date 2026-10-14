"""Voxel-grid downsampling for Lab Book 8 (native depth geometry).

Run from PlaneBasedSLAM root:
    python experiments/pointcloud/downsample_pointcloud.py
Or specify an input:
    python experiments/pointcloud/downsample_pointcloud.py --input path/to/cloud.ply

Inputs remain unchanged. All outputs go to a new timestamped directory.
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d

DEFAULT_VOXELS_MM = (2.0, 5.0, 10.0, 15.0, 20.0)


def project_root():
    here = Path(__file__).resolve()
    candidates = [Path.cwd().resolve(), *here.parents]
    for base in candidates:
        if (base / "experiments" / "pointcloud").is_dir():
            return base
    raise FileNotFoundError("Cannot locate PlaneBasedSLAM project root.")


def discover_input(root, frame):
    result_root = root / "experiments" / "pointcloud" / "results"
    files = list(result_root.glob(f"depth_pointcloud_*/{frame}_depth.ply"))
    if not files:
        raise FileNotFoundError(
            f"No {frame}_depth.ply found in {result_root}. "
            "Use --input to select your point cloud."
        )
    # Prioritise newest result-directory name; avoid accidentally selecting a coloured PLY.
    return max(files, key=lambda p: p.parent.name)


def create_output_directory(parent):
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result = parent / f"downsampling_{stamp}"
    result.mkdir(exist_ok=False)
    return result


def finite_point_cloud(cloud):
    points = np.asarray(cloud.points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("Input point cloud is empty or has invalid XYZ shape.")
    good = np.isfinite(points).all(axis=1)
    if not good.all():
        # Keep colours/normals aligned through Open3D selection.
        cloud = cloud.select_by_index(np.flatnonzero(good).tolist())
    if cloud.is_empty():
        raise ValueError("Input has no finite points.")
    return cloud, int(np.count_nonzero(~good))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Input PLY (defaults to latest native depth PLY)")
    parser.add_argument("--frame", default="0000", help="Frame ID for auto-discovery (default 0000)")
    parser.add_argument("--voxels-mm", type=float, nargs="+", default=list(DEFAULT_VOXELS_MM),
                        help="Voxel sizes in millimetres (default 2 5 10 15 20)")
    parser.add_argument("--out-parent", type=Path, help="Parent directory for timestamped results")
    parser.add_argument("--visualize", action="store_true", help="Display original and downsampled clouds")
    args = parser.parse_args()

    if any(not np.isfinite(x) or x <= 0 for x in args.voxels_mm):
        parser.error("All voxel sizes must be positive finite numbers.")
    if len(set(args.voxels_mm)) != len(args.voxels_mm):
        parser.error("Voxel sizes must be distinct.")

    root = project_root()
    input_path = (args.input.expanduser().resolve() if args.input else
                  discover_input(root, args.frame).resolve())
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    print(f"Input: {input_path}")
    original = o3d.io.read_point_cloud(str(input_path))
    original, removed_invalid = finite_point_cloud(original)
    n_original = len(original.points)
    bounds_min = np.asarray(original.get_min_bound()).tolist()
    bounds_max = np.asarray(original.get_max_bound()).tolist()
    print(f"Original: {n_original:,} finite points; removed invalid: {removed_invalid:,}")
    print(f"Point cloud colours present: {original.has_colors()}")

    output_parent = (args.out_parent.expanduser().resolve() if args.out_parent else
                     root / "experiments" / "pointcloud" / "results")
    output_dir = create_output_directory(output_parent)
    rows = []
    visual_clouds = []

    for size_mm in args.voxels_mm:
        size_m = size_mm / 1000.0
        start = time.perf_counter()
        sampled = original.voxel_down_sample(voxel_size=size_m)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        n_sampled = len(sampled.points)
        if n_sampled == 0:
            raise RuntimeError(f"Voxel size {size_mm:g} mm produced no points")

        tag = f"{size_mm:07.2f}".replace(".", "p")
        ply_path = output_dir / f"voxel_{tag}mm.ply"
        if not o3d.io.write_point_cloud(str(ply_path), sampled, write_ascii=False):
            raise OSError(f"Unable to write: {ply_path}")

        row = {
            "voxel_size_mm": size_mm,
            "voxel_size_m": size_m,
            "original_points": n_original,
            "downsampled_points": n_sampled,
            "point_reduction_percent": round(100.0 * (1.0 - n_sampled / n_original), 4),
            "retained_percent": round(100.0 * n_sampled / n_original, 4),
            "runtime_ms": round(elapsed_ms, 3),
            "output_ply": str(ply_path),
        }
        rows.append(row)
        visual_clouds.append((size_mm, sampled))
        print(f"{size_mm:5g} mm: {n_sampled:>8,} pts | "
              f"reduction {row['point_reduction_percent']:6.2f}% | {elapsed_ms:8.2f} ms")

    csv_path = output_dir / "downsampling_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "experiment": "Lab Book 8 - voxel-grid point-cloud downsampling",
        "input_ply": str(input_path),
        "coordinate_frame": "unchanged from input (native depth optical frame for depth PLY)",
        "units": "metres",
        "original_points": n_original,
        "discarded_nonfinite_xyz_points": removed_invalid,
        "has_input_colours": original.has_colors(),
        "original_bounds_min_m": bounds_min,
        "original_bounds_max_m": bounds_max,
        "timing_note": "Voxel downsampling only; excludes reading/writing/visualisation; single run per size, not a benchmark average.",
        "results": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved results in: {output_dir}")

    if args.visualize:
        for size_mm, cloud in visual_clouds:
            print(f"Opening {size_mm:g} mm cloud. Close the viewer to continue.")
            o3d.visualization.draw_geometries(
                [cloud], window_name=f"Voxel grid {size_mm:g} mm",
                width=1100, height=760,
            )


if __name__ == "__main__":
    main()
