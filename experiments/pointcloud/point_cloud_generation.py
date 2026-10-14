"""Generate a geometry-only Open3D point cloud from a raw RealSense depth PNG.

No RGB intrinsics, RGB extrinsics, or colour mapping are used.
The output XYZ coordinates remain in the native depth optical frame (metres).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def find_project_root():
    """Find PlaneBasedSLAM root from the script location or working directory."""
    starts = (Path(__file__).resolve().parent, Path.cwd().resolve())
    for start in starts:
        for folder in (start, *start.parents):
            if (folder / "data" / "camera_calibration").is_dir():
                return folder
    raise FileNotFoundError(
        "Cannot find project root containing data/camera_calibration. "
        "Run from the project directory, or pass --session."
    )


def choose_session(project_root):
    sessions = sorted(
        (project_root / "data" / "camera_calibration").glob("alignment_session_*")
    )
    sessions = [
        p for p in sessions
        if p.is_dir() and (p / "camera_parameters.json").is_file()
    ]
    if not sessions:
        raise FileNotFoundError("No alignment_session_* with camera_parameters.json found.")
    preferred = project_root / "data" / "camera_calibration" / "alignment_session_20261005_171922_220185"
    if preferred in sessions:
        return preferred
    if len(sessions) == 1:
        return sessions[0]
    print("Multiple alignment sessions found:")
    for i, path in enumerate(sessions, 1):
        print(f"  {i}. {path}")
    selection = int(input("Select session number: "))
    if not 1 <= selection <= len(sessions):
        raise ValueError("Invalid session selection")
    return sessions[selection - 1]


def load_depth(session, frame_id):
    params_path = session / "camera_parameters.json"
    depth_path = session / "depth" / f"{frame_id}.png"
    if not params_path.is_file():
        raise FileNotFoundError(params_path)
    if not depth_path.is_file():
        raise FileNotFoundError(depth_path)

    with params_path.open("r", encoding="utf-8-sig") as f:
        params = json.load(f)

    raw = np.fromfile(str(depth_path), dtype=np.uint8)
    depth = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError(f"Expected a single-channel uint16 depth PNG: {depth_path}")

    intr = params["streams"]["depth"]
    expected_size = (int(intr["height"]), int(intr["width"]))
    if depth.shape != expected_size:
        raise ValueError(f"Depth shape {depth.shape} != metadata {expected_size}")

    model = str(intr.get("distortion_model", "")).lower()
    coeffs = np.asarray(intr.get("coeffs", []), dtype=np.float64)
    if not any(m in model for m in ("brown_conrady", "none")):
        raise ValueError(f"Unsupported depth distortion model: {model}")
    if np.any(np.abs(coeffs) > 1e-12):
        raise ValueError("Non-zero depth distortion requires undistortion before pinhole back-projection")

    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx, cy = float(intr["cx"]), float(intr["cy"])
    scale = float(params["depth_scale_m_per_unit"])
    if fx <= 0 or fy <= 0 or scale <= 0:
        raise ValueError("Invalid depth intrinsics or depth scale")

    return depth, depth_path, (fx, fy, cx, cy), scale


def deproject(depth, intrinsics, depth_scale, min_depth, max_depth):
    fx, fy, cx, cy = intrinsics
    z_all = depth.astype(np.float64) * depth_scale
    valid = (depth > 0) & np.isfinite(z_all) & (z_all >= min_depth) & (z_all <= max_depth)
    v, u = np.nonzero(valid)
    z = z_all[v, u]
    x = (u.astype(np.float64) - cx) * z / fx
    y = (v.astype(np.float64) - cy) * z / fy
    points = np.column_stack((x, y, z))
    return points, valid, z


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, help="Path to alignment_session_* directory")
    parser.add_argument("--frame", default="0000", help="Frame ID, e.g. 0000")
    parser.add_argument("--min-depth", type=float, default=0.1, help="Minimum depth in metres")
    parser.add_argument("--max-depth", type=float, default=5.0, help="Maximum depth in metres")
    parser.add_argument("--output", type=Path, help="Optional output directory")
    parser.add_argument("--no-visualisation", action="store_true", help="Save output without opening Open3D window")
    args = parser.parse_args()

    if not 0 < args.min_depth < args.max_depth:
        parser.error("Require 0 < --min-depth < --max-depth")
    if not (args.frame.isdigit() and len(args.frame) == 4):
        parser.error("--frame must be a four-digit identifier such as 0000")

    if args.session:
        session = args.session.expanduser().resolve()
    else:
        session = choose_session(find_project_root())

    depth, depth_path, intr, scale = load_depth(session, args.frame)
    points, valid_mask, valid_z = deproject(
        depth, intr, scale, args.min_depth, args.max_depth
    )
    if len(points) == 0:
        raise ValueError("No valid depth samples in the selected range")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if args.output:
        output_dir = args.output.expanduser().resolve()
    else:
        project_root = session.parent.parent.parent
        output_dir = (
            project_root / "experiments" / "pointcloud" / "results"
            / ("depth_pointcloud_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    ply_path = output_dir / f"{args.frame}_depth.ply"
    if not o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False, compressed=False):
        raise OSError(f"Failed to write point cloud: {ply_path}")

    n_pixels = int(depth.size)
    valid_count = int(np.count_nonzero(valid_mask))
    summary = {
        "source_depth_file": str(depth_path),
        "camera_parameters_file": str(session / "camera_parameters.json"),
        "frame_id": args.frame,
        "coordinate_frame": "native_depth_optical_frame",
        "units": "metres",
        "depth_scale_m_per_unit": scale,
        "intrinsics_fx_fy_cx_cy": list(intr),
        "image_width": int(depth.shape[1]),
        "image_height": int(depth.shape[0]),
        "total_pixels": n_pixels,
        "valid_points": valid_count,
        "valid_depth_coverage_percent": 100.0 * valid_count / n_pixels,
        "depth_filter_range_m": [args.min_depth, args.max_depth],
        "min_valid_depth_m": float(valid_z.min()),
        "median_valid_depth_m": float(np.median(valid_z)),
        "max_valid_depth_m": float(valid_z.max()),
        "pointcloud_file": str(ply_path),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Input: {depth_path}")
    print(f"Frame: native depth optical frame; units: metres")
    print(f"Depth scale: {scale:.12f} m/unit")
    print(f"Valid points: {valid_count:,} / {n_pixels:,} ({summary['valid_depth_coverage_percent']:.2f}%)")
    print(f"Depth min/median/max: {valid_z.min():.3f} / {np.median(valid_z):.3f} / {valid_z.max():.3f} m")
    print(f"PLY: {ply_path}")
    print(f"Summary: {output_dir / 'summary.json'}")

    if not args.no_visualisation:
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        o3d.visualization.draw_geometries(
            [pcd, coordinate_frame],
            window_name=f"Depth Point Cloud - {args.frame}",
            width=1100,
            height=750,
        )


if __name__ == "__main__":
    main()
