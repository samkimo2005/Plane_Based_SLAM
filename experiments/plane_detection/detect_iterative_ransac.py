"""Lab Book 9: batch iterative RANSAC plane extraction from native-depth PLYs.

Requires: pip install open3d numpy
Run from the project root:
    python experiments/plane_detection/detect_iterative_ransac.py
Or:
    python experiments/plane_detection/detect_iterative_ransac.py --input ".../real_scene_downsampling_..."

INPUT: a run directory containing voxel_0005p00mm and voxel_0010p00mm,
       one voxel directory, or one .ply file. No camera is needed.
OUTPUT: timestamped directory with segmented PLYs, per-plane CSV,
        per-cloud CSV, and machine-readable summary JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def locate_project_root() -> Path:
    script = Path(__file__).resolve()
    for parent in (script.parent, *script.parents):
        if (parent / "experiments").is_dir():
            return parent
    return Path.cwd()


def get_input_files(path: Path) -> tuple[Path, list[Path]]:
    path = path.expanduser().resolve()
    if path.is_file() and path.suffix.lower() == ".ply":
        return path.parent, [path]
    if not path.is_dir():
        raise ValueError(f"Not a PLY file or directory: {path}")
    # If pointing to the parent downsampling run, collect files only from voxel folders.
    voxel_folders = sorted(p for p in path.iterdir() if p.is_dir() and p.name.startswith("voxel_"))
    if voxel_folders:
        files = [f for folder in voxel_folders for f in sorted(folder.glob("*.ply"))]
    else:
        files = sorted(path.glob("*.ply"))
    if not files:
        raise ValueError(f"No .ply files found in {path}; enter a voxel folder or its parent run folder.")
    return path, files


def normalized_plane(coeff: np.ndarray) -> tuple[np.ndarray, float]:
    n = np.asarray(coeff[:3], dtype=np.float64)
    length = float(np.linalg.norm(n))
    if length < 1e-12 or not np.isfinite(length):
        raise ValueError("RANSAC returned an invalid plane normal")
    n /= length
    d = float(coeff[3]) / length
    # Deterministic orientation for reporting (not an absolute physical orientation).
    if n[int(np.argmax(np.abs(n)))] < 0:
        n, d = -n, -d
    return n, d


def make_colors(count: int, labels: np.ndarray) -> np.ndarray:
    # Muted grey for unassigned; fixed categorical palette for plane identities.
    palette = np.array([
        [0.89, 0.27, 0.29], [0.21, 0.66, 0.39], [0.23, 0.48, 0.87],
        [0.93, 0.65, 0.18], [0.61, 0.38, 0.81], [0.14, 0.72, 0.73],
        [0.93, 0.44, 0.67], [0.55, 0.60, 0.15], [0.68, 0.43, 0.25],
        [0.44, 0.63, 0.94],
    ])
    colors = np.full((count, 3), 0.42, dtype=np.float64)
    valid = labels >= 0
    colors[valid] = palette[labels[valid] % len(palette)]
    return colors


def csv_write(path: Path, records: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def process_cloud(o3d, input_file: Path, input_root: Path, output_root: Path, args) -> tuple[dict, list[dict]]:
    pc = o3d.io.read_point_cloud(str(input_file))
    raw = np.asarray(pc.points)
    if len(raw) == 0:
        raise ValueError("Empty PLY")
    finite = np.isfinite(raw).all(axis=1)
    xyz = np.asarray(raw[finite], dtype=np.float64)
    if len(xyz) < args.ransac_n:
        raise ValueError("Too few finite points")
    # Group outputs by parent voxel folder; never overwrite original files.
    group = input_file.parent.name if input_file.parent != input_root else "individual_clouds"
    dest = output_root / group
    dest.mkdir(parents=True, exist_ok=True)
    total = len(xyz)
    source_indices = np.arange(total, dtype=np.int64)
    labels = np.full(total, -1, dtype=np.int32)
    remaining = xyz
    remaining_ids = source_indices
    plane_records = []
    min_required = max(args.min_inliers, math.ceil(args.min_fraction * total))
    # Seed before each cloud, not just once for whole batch: independent of batch order.
    if args.seed is not None and hasattr(o3d.utility, "random"):
        o3d.utility.random.seed(args.seed)
    tic = time.perf_counter()
    for k in range(args.max_planes):
        if len(remaining) < max(args.ransac_n, min_required):
            break
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(remaining)
        coeff, local_inliers = cloud.segment_plane(
            distance_threshold=args.distance_threshold,
            ransac_n=args.ransac_n,
            num_iterations=args.iterations,
        )
        inliers = np.asarray(local_inliers, dtype=np.int64)
        if len(inliers) < min_required:
            break
        n, d = normalized_plane(np.asarray(coeff))
        global_ids = remaining_ids[inliers]
        pts = xyz[global_ids]
        absolute = np.abs(pts @ n + d)
        plane_id = len(plane_records)
        labels[global_ids] = plane_id
        plane_records.append({
            "plane_id": plane_id,
            "nx": float(n[0]), "ny": float(n[1]), "nz": float(n[2]), "d_m": d,
            "inlier_count": len(global_ids),
            "inlier_ratio_total_percent": 100.0 * len(global_ids) / total,
            "fit_rms_mm": 1000.0 * float(np.sqrt(np.mean(absolute ** 2))),
            "fit_p90_mm": 1000.0 * float(np.percentile(absolute, 90)),
            "inlier_indices": global_ids.tolist(),
        })
        keep = np.ones(len(remaining), dtype=bool)
        keep[inliers] = False
        remaining = remaining[keep]
        remaining_ids = remaining_ids[keep]
    detect_ms = (time.perf_counter() - tic) * 1000.0
    colored = o3d.geometry.PointCloud()
    colored.points = o3d.utility.Vector3dVector(xyz)
    colored.colors = o3d.utility.Vector3dVector(make_colors(total, labels))
    segmented_path = dest / f"{input_file.stem}_segmented.ply"
    if not o3d.io.write_point_cloud(str(segmented_path), colored, write_ascii=False):
        raise IOError(f"Failed to write {segmented_path}")
    # One NPZ per cloud maps plane IDs to the same ordering as segmented PLY.
    labels_path = dest / f"{input_file.stem}_labels.npz"
    np.savez_compressed(labels_path, labels=labels)
    assigned = int(np.count_nonzero(labels >= 0))
    relative = input_file.relative_to(input_root) if input_file != input_root else Path(input_file.name)
    summary = {
        "input_ply": str(input_file), "relative_input": str(relative),
        "group": group, "frame": input_file.stem,
        "input_points": len(raw), "valid_points": total,
        "discarded_nonfinite_points": int(len(raw) - total),
        "detected_planes": len(plane_records), "assigned_points": assigned,
        "coverage_percent": 100.0 * assigned / total,
        "unassigned_points": total - assigned,
        "detection_runtime_ms": detect_ms,
        "segmented_ply": str(segmented_path), "labels_npz": str(labels_path),
    }
    flat_records = []
    for plane in plane_records:
        row = {"group": group, "frame": input_file.stem, "input_ply": str(input_file)}
        row.update({k: v for k, v in plane.items() if k != "inlier_indices"})
        flat_records.append(row)
    if args.visualize:
        o3d.visualization.draw_geometries([colored], window_name=f"{group}/{input_file.stem} - {len(plane_records)} planes")
    return summary, flat_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Lab Book 9: iterative RANSAC plane extraction, single or batch PLY.")
    parser.add_argument("--input", type=str, help="One PLY, one voxel PLY folder, or parent folder containing voxel_* folders")
    parser.add_argument("--output", type=Path, help="Output directory (default: new timestamped directory under experiments/plane_detection/results)")
    parser.add_argument("--distance-threshold", type=float, default=0.008, help="Inlier distance in metres (default 0.008 = 8 mm)")
    parser.add_argument("--ransac-n", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--max-planes", type=int, default=10)
    parser.add_argument("--min-inliers", type=int, default=80, help="Absolute minimum plane support")
    parser.add_argument("--min-fraction", type=float, default=0.025, help="Minimum support as fraction of initial cloud (default 2.5%%)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", action="store_true", help="Open segmented PLY visualisation for each input frame")
    args = parser.parse_args()
    if not args.input:
        args.input = input("Paste a PLY file, voxel folder, or parent downsampling run folder:\nInput path: ").strip().strip('"').strip("'")
    if not args.input:
        parser.error("Input path is required")
    if not (args.distance_threshold > 0 and args.ransac_n >= 3 and args.iterations > 0 and args.max_planes > 0 and args.min_inliers >= 3 and 0 < args.min_fraction < 1):
        parser.error("Invalid RANSAC settings; check distance, iterations, and support limits")
    try:
        import open3d as o3d
    except ImportError as exc:
        parser.error("Open3D is missing. Run: python -m pip install open3d")
    try:
        root, files = get_input_files(Path(args.input))
    except ValueError as exc:
        parser.error(str(exc))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output or locate_project_root() / "experiments" / "plane_detection" / "results" / f"iterative_ransac_{timestamp}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = []
    planes = []
    failures = []
    for i, file in enumerate(files, 1):
        try:
            metric, entries = process_cloud(o3d, file, root, output, args)
            metrics.append(metric)
            planes.extend(entries)
            print(f"[{i}/{len(files)}] {metric['group']}/{metric['frame']}: {metric['detected_planes']} planes, {metric['coverage_percent']:.2f}% coverage, {metric['detection_runtime_ms']:.1f} ms")
        except Exception as exc:
            failures.append({"input_ply": str(file), "error": str(exc)})
            print(f"[{i}/{len(files)}] FAILED {file}: {exc}")
    csv_write(output / "detection_metrics.csv", metrics, ["group", "frame", "input_ply", "input_points", "valid_points", "discarded_nonfinite_points", "detected_planes", "assigned_points", "coverage_percent", "unassigned_points", "detection_runtime_ms", "segmented_ply", "labels_npz"])
    csv_write(output / "plane_parameters.csv", planes, ["group", "frame", "input_ply", "plane_id", "nx", "ny", "nz", "d_m", "inlier_count", "inlier_ratio_total_percent", "fit_rms_mm", "fit_p90_mm"])
    report = {
        "experiment": "Lab Book 9 - iterative RANSAC real-scene plane detection",
        "input_path": str(root), "output_dir": str(output),
        "parameter_settings": vars(args),
        "parameter_notes": "distance_threshold in m; min_fraction relative to initial input cloud, plus absolute min_inliers; plane normals have deterministic sign",
        "evaluation_notes": "Fitting RMS and coverage are internal, no GT-based accuracy. Detection timing excludes I/O and visualization; one run per cloud.",
        "successful_files": len(metrics), "failed_files": failures,
        "clouds": metrics,
        "planes": planes,
    }
    report["parameter_settings"]["output"] = str(args.output) if args.output else None
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults: {output}\nSuccessful: {len(metrics)}/{len(files)}; failed: {len(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
