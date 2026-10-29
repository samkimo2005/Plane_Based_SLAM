"""Real-scene five-method plane-detection evaluation.

Place alongside:
  plane_detection_common_real.py
  multi_object_ransac_real.py
  region_growing_real.py
  hough_plane_detection_real.py
  surface_normal_surfel_real.py

Run:
  python evaluate_plane_detection_real.py --input "path/to/voxel_0005p00mm/000002.ply"
  python evaluate_plane_detection_real.py --input "path/to/real_scene_downsampling_run" --methods all

Uses the same PLY data for every detector. All methods are scored with one
common nearest-plane assignment (geometric model coverage), while the Open3D
RANSAC's native sequential-removal labels are saved separately. Note that
this does not assess ground-truth detection accuracy or spatial connectivity.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import plane_detection_common_real as common

# Legacy detector copies may still import the historical module name.
# Route that import to the same real-data common instance so no detector
# files need editing; the original synthetic/TUM files are untouched.
sys.modules.setdefault("plane_detection_common", common)

METHODS = ("iterative_ransac", "multi_object_ransac", "region_growing",
           "hough_plane_detection", "surface_normal_surfel")
MODULE_NAMES = {
    "multi_object_ransac": "multi_object_ransac_real",
    "region_growing": "region_growing_real",
    "hough_plane_detection": "hough_plane_detection_real",
    "surface_normal_surfel": "surface_normal_surfel_real",
}
PALETTE = np.array([
    [0.89, 0.27, 0.29], [0.21, 0.66, 0.39],
    [0.23, 0.48, 0.87], [0.93, 0.65, 0.18],
    [0.61, 0.38, 0.81], [0.14, 0.72, 0.73],
    [0.93, 0.44, 0.67], [0.55, 0.60, 0.15],
    [0.68, 0.43, 0.25], [0.44, 0.63, 0.94],
])


def inputs(path: Path) -> tuple[Path, list[Path]]:
    path = path.expanduser().resolve()
    if path.is_file() and path.suffix.lower() == ".ply":
        return path.parent, [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    folders = sorted(p for p in path.iterdir()
                     if p.is_dir() and p.name.startswith("voxel_"))
    files = ([f for folder in folders for f in sorted(folder.glob("*.ply"))]
             if folders else sorted(path.glob("*.ply")))
    if not files:
        raise ValueError(f"No PLYs found in {path}")
    return path, files


def parse_methods(s: str) -> list[str]:
    if s.strip().lower() == "all":
        return list(METHODS)
    names = [v.strip() for v in s.split(",") if v.strip()]
    unknown = set(names).difference(METHODS)
    if unknown or not names:
        raise ValueError(f"Unknown/empty methods: {sorted(unknown)}; choose {METHODS}")
    return list(dict.fromkeys(names))


def normalise_model(coeff):
    arr = np.asarray(coeff, dtype=np.float64)
    return common.canonicalise_plane(arr[:3], arr[3])


def open3d_iterative_ransac(o3d, xyz: np.ndarray, args) -> tuple[list[dict], np.ndarray]:
    """Replicate Lab Book 9's segment_plane/stop/remove protocol."""
    n = len(xyz)
    required = max(args.min_inliers, math.ceil(args.min_fraction * n))
    remaining = xyz
    remaining_ids = np.arange(n, dtype=np.int64)
    native_labels = np.full(n, -1, dtype=np.int32)
    models = []
    if args.seed is not None and hasattr(o3d.utility, "random"):
        o3d.utility.random.seed(args.seed)
    for _ in range(args.max_planes):
        if len(remaining) < max(3, required):
            break
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(remaining)
        coeff, local_inliers = cloud.segment_plane(
            distance_threshold=args.distance_threshold,
            ransac_n=3, num_iterations=args.iterations)
        idx = np.asarray(local_inliers, dtype=np.int64)
        if len(idx) < required:
            break
        normal, d = normalise_model(coeff)
        plane_id = len(models)
        native_labels[remaining_ids[idx]] = plane_id
        models.append({"normal": normal, "d": d, "source_support": int(len(idx))})
        keep = np.ones(len(remaining), dtype=bool)
        keep[idx] = False
        remaining = remaining[keep]
        remaining_ids = remaining_ids[keep]
    return models, native_labels


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare 5 detectors on native real-scene PLY clouds")
    ap.add_argument("--input", help="PLY, voxel directory, or parent downsampling directory")
    ap.add_argument("--check", action="store_true", help="Check all detector imports/interfaces without running point clouds")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--methods", default="all", help="all or comma-separated method names")
    ap.add_argument("--distance-threshold", type=float, default=0.008)
    ap.add_argument("--min-inliers", type=int, default=80)
    ap.add_argument("--min-fraction", type=float, default=0.025)
    ap.add_argument("--max-planes", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=1000, help="Open3D RANSAC iterations")
    ap.add_argument("--k-neighbors", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-files", type=int, default=0, help="Limit for quick test; 0 = all")
    args = ap.parse_args()
    if not args.input and not args.check:
        args.input = input("Paste full path to PLY, voxel folder or downsampling run: ").strip().strip('"').strip("'")
    if not (args.distance_threshold > 0 and args.min_inliers >= 3
            and 0 <= args.min_fraction <= 1 and args.max_planes >= 1
            and args.iterations > 0 and args.k_neighbors >= 3):
        ap.error("Invalid thresholds or iteration settings")

    try:
        selected = parse_methods(args.methods)
    except ValueError as exc:
        ap.error(str(exc))

    detectors = {}
    print("Checking detector imports:", flush=True)
    for method in selected:
        if method == "iterative_ransac":
            print("  iterative_ransac: Open3D implementation (checked on actual run)", flush=True)
            continue
        module_name = MODULE_NAMES[method]
        try:
            module = importlib.import_module(module_name)
        except (ModuleNotFoundError, ImportError) as exc:
            ap.error(f"Cannot import {module_name}.py from this folder: {exc}. "
                     "Keep all *_real.py files alongside the evaluator.")
        if not callable(getattr(module, "detect_planes", None)):
            ap.error(f"{module_name}.py has no callable detect_planes()")
        detectors[method] = module
        print(f"  {module_name}.py: OK", flush=True)
    for attr in ("DetectionSettings", "configure", "assign_unique_labels",
                 "canonicalise_plane"):
        if not hasattr(common, attr):
            ap.error(f"plane_detection_common_real.py is missing {attr}")
    print("  plane_detection_common_real.py: OK", flush=True)
    if args.check:
        print("Import and interface checks passed. No PLY files were processed.")
        return 0

    try:
        import open3d as o3d
    except ImportError as exc:
        ap.error(f"Open3D is required to process PLY files: {exc}")
    try:
        root, paths = inputs(Path(args.input))
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))
    if args.max_files > 0:
        paths = paths[:args.max_files]
    common.configure(common.DetectionSettings(
        distance_threshold=args.distance_threshold,
        min_support=args.min_inliers,
        min_fraction=args.min_fraction,
        max_planes=args.max_planes,
        k_neighbors=args.k_neighbors,
    ))
    dest = (args.output or
            Path(__file__).resolve().parent / "results" /
            ("real_comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    rows, plane_rows, failed = [], [], []
    for file_idx, ply in enumerate(paths, 1):
        pcd = o3d.io.read_point_cloud(str(ply))
        raw = np.asarray(pcd.points)
        xyz = np.asarray(raw[np.isfinite(raw).all(axis=1)], dtype=np.float64)
        if len(xyz) < 3:
            failed.append({"file": str(ply), "error": "Fewer than three finite XYZ points"})
            continue
        rel = ply.relative_to(root) if ply != root else Path(ply.name)
        group = ply.parent.name if ply.parent != root else "individual_clouds"
        print(f"[{file_idx}/{len(paths)}] {rel}: {len(xyz)} points", flush=True)
        for method in selected:
            name = f"{ply.stem}_{method}"
            outdir = dest / group / method
            outdir.mkdir(parents=True, exist_ok=True)
            try:
                t0 = time.perf_counter()
                native_labels = None
                if method == "iterative_ransac":
                    models, native_labels = open3d_iterative_ransac(o3d, xyz, args)
                else:
                    rng = np.random.default_rng(args.seed)
                    models = detectors[method].detect_planes(xyz, rng)
                detection_ms = (time.perf_counter() - t0) * 1000.0

                t1 = time.perf_counter()
                # This is *model coverage* after common geometric reassignment,
                # not original Region Growing's connected-region segmentation.
                labels, accepted, stats = common.assign_unique_labels(
                    xyz, models, threshold=args.distance_threshold,
                    min_support=args.min_inliers, min_fraction=args.min_fraction)
                assignment_ms = (time.perf_counter() - t1) * 1000.0

                colors = np.full((len(xyz), 3), 0.42, dtype=float)
                keep = labels >= 0
                colors[keep] = PALETTE[labels[keep] % len(PALETTE)]
                segmented = o3d.geometry.PointCloud()
                segmented.points = o3d.utility.Vector3dVector(xyz)
                segmented.colors = o3d.utility.Vector3dVector(colors)
                output_ply = outdir / f"{name}_segmented.ply"
                if not o3d.io.write_point_cloud(str(output_ply), segmented):
                    raise IOError(f"Could not save {output_ply}")

                labels_file = outdir / f"{name}_labels.npz"
                np.savez_compressed(labels_file, labels=labels,
                                    **({"native_ransac_labels": native_labels}
                                       if native_labels is not None else {}))
                row = {
                    "input_ply": str(ply), "scene_group": group, "frame": ply.stem,
                    "algorithm": method, "valid_points": len(xyz),
                    "candidate_planes": len(models), "detected_planes": len(accepted),
                    "assigned_points": stats["assigned_points"],
                    "coverage_percent": stats["coverage_percent"],
                    "detection_runtime_ms": detection_ms,
                    "assignment_runtime_ms": assignment_ms,
                    "total_processing_ms": detection_ms + assignment_ms,
                    "segmented_ply": str(output_ply), "labels_npz": str(labels_file),
                }
                if native_labels is not None:
                    row["native_ransac_coverage_percent"] = (
                        100.0 * np.count_nonzero(native_labels >= 0) / len(xyz))
                rows.append(row)
                for j, model in enumerate(accepted):
                    plane_rows.append({
                        "input_ply": str(ply), "algorithm": method,
                        "plane_id": j, "nx": float(model["normal"][0]),
                        "ny": float(model["normal"][1]), "nz": float(model["normal"][2]),
                        "d_m": float(model["d"]), "inlier_count": model["support"],
                        "fit_rms_mm": model["fit_rms_mm"],
                        "fit_p90_mm": model["fit_p90_mm"],
                    })
                print(f"  {method}: {len(accepted)} planes, "
                      f"{stats['coverage_percent']:.2f}% coverage, "
                      f"{detection_ms:.1f} ms detection", flush=True)
            except Exception as exc:
                failed.append({"file": str(ply), "algorithm": method,
                               "error": f"{type(exc).__name__}: {exc}"})
                print(f"  FAILED {method}: {exc}", flush=True)

    metric_cols = ["input_ply", "scene_group", "frame", "algorithm",
                   "valid_points", "candidate_planes", "detected_planes",
                   "assigned_points", "coverage_percent",
                   "native_ransac_coverage_percent",
                   "detection_runtime_ms", "assignment_runtime_ms",
                   "total_processing_ms", "segmented_ply", "labels_npz"]
    plane_cols = ["input_ply", "algorithm", "plane_id", "nx", "ny", "nz",
                  "d_m", "inlier_count", "fit_rms_mm", "fit_p90_mm"]
    write_csv(dest / "detection_metrics.csv", rows, metric_cols)
    write_csv(dest / "plane_parameters.csv", plane_rows, plane_cols)
    (dest / "summary.json").write_text(json.dumps({
        "parameters": {k: str(v) if isinstance(v, Path) else v
                       for k, v in vars(args).items()},
        "methods": selected, "input_count": len(paths),
        "successes": len(rows), "failures": failed,
        "metric_notes": {
            "coverage": "unique nearest-plane geometric model coverage; NOT GT accuracy",
            "native_ransac_coverage": "Lab Book 9 sequential-removal assignment, recorded separately",
            "runtime": "detection includes each detector's local normal estimation; "
                       "assignment runtime excludes PLY I/O and output writing",
            "limitations": "legacy detectors use method-specific proposal thresholds; "
                           "connected-region boundaries not preserved by common reassignment; "
                           "metrics cannot be read as precision or recall",
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {dest}\nSuccessful runs: {len(rows)}; failures: {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    main()
