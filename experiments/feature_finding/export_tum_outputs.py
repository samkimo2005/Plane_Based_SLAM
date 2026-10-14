#!/usr/bin/env python3
"""
export_tum_plane_features_with_gt_ids.py

Purpose
-------
Create a Candidate-Gating-ready TUM plane-feature output by combining:

    1) Plane Detection per-frame output
    2) ETH3D-derived per-frame plane reference

The script DOES NOT change the detected plane geometry. It only appends an
evaluation-only field:

    global_plane_id

to each detected plane when a geometrically valid one-to-one match to an
ETH3D-derived reference plane is found.

This avoids data leakage:
- Candidate Gating may use normal / offset / centroid / contour / area / etc.
- Candidate Gating must NOT use global_plane_id.
- global_plane_id is only for producing gt_match labels during evaluation.

Supported input
---------------
Per-frame .json or .npz files.

Recommended JSON structure:

{
  "frame_id": 20,
  "timestamp": 1305031102.160407,
  "planes": [
    {
      "plane_id": 0,
      "normal": [nx, ny, nz],
      "offset": d,
      "centroid": [x, y, z],
      "area": 1.23,
      "support_size": 1000,
      "support_indices": [1, 2, 3, ...]   // optional
    }
  ]
}

Reference planes should additionally contain a persistent physical-plane ID:

{
  "plane_id": 4,
  "global_plane_id": "P4",
  "normal": [...],
  "offset": ...,
  "centroid": [...],
  "support_indices": [...]                // optional
}

If global_plane_id is absent in the ETH3D reference, plane_id is used as the
persistent ID.

Outputs
-------
1) tum_plane_features_with_gt_ids.json
2) tum_plane_features_with_gt_ids.csv
3) tum_plane_gt_export_summary.txt

Example
-------
python export_tum_plane_features_with_gt_ids.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Basic I/O
# ---------------------------------------------------------------------------

def _to_builtin(obj: Any) -> Any:
    """Convert numpy values recursively to JSON-serialisable Python values."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    return obj


def load_file(path: Path) -> Dict[str, Any]:
    """Load one per-frame JSON or NPZ file."""
    suffix = path.suffix.lower()

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError(f"{path}: top-level JSON must be an object.")
        return obj

    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        obj = {k: data[k] for k in data.files}

        # Common pattern: a pickled dict saved as arr_0 or frame.
        for key in ("frame", "arr_0", "data"):
            if key in obj:
                candidate = obj[key]
                try:
                    if isinstance(candidate, np.ndarray) and candidate.shape == ():
                        candidate = candidate.item()
                    if isinstance(candidate, dict):
                        return _to_builtin(candidate)
                except Exception:
                    pass

        return _to_builtin(obj)

    raise ValueError(f"Unsupported file type: {path}")


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_builtin(obj), f, indent=2, ensure_ascii=False)


def extract_frame_id_from_name(path: Path) -> Optional[int]:
    """
    Extract frame index from names such as:
        frame_000020.json
        000020.json
        detection_frame20.npz
    """
    nums = re.findall(r"\d+", path.stem)
    if not nums:
        return None
    return int(nums[-1])


def discover_frame_files(directory: Path) -> Dict[int, Path]:
    """
    Find per-frame .json/.npz files and index them by frame_id.
    frame_id inside the file takes priority over filename digits.
    """
    files = sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".json", ".npz"}]
    )
    result: Dict[int, Path] = {}

    for path in files:
        try:
            obj = load_file(path)
            fid = obj.get("frame_id")
            if fid is None:
                fid = extract_frame_id_from_name(path)
            if fid is None:
                print(f"[WARN] Could not determine frame_id for {path.name}; skipped.")
                continue
            fid = int(fid)
            if fid in result:
                raise ValueError(f"Duplicate frame_id {fid}: {result[fid]} and {path}")
            result[fid] = path
        except Exception as e:
            print(f"[WARN] Could not inspect {path.name}: {e}")

    return result


# ---------------------------------------------------------------------------
# Plane schema normalisation
# ---------------------------------------------------------------------------

def canonicalise_plane_dict(plane: Dict[str, Any], default_id: int) -> Dict[str, Any]:
    """
    Preserve all original fields while normalising the core geometry fields.
    Accepts common aliases:
        normal: normal / n / plane_normal
        offset: offset / d / plane_offset
        centroid: centroid / center / centre
    """
    p = dict(plane)

    normal = None
    for key in ("normal", "n", "plane_normal"):
        if key in p:
            normal = np.asarray(p[key], dtype=float).reshape(-1)
            break
    if normal is None or normal.size != 3:
        raise ValueError(f"Plane {p.get('plane_id', default_id)} has no valid 3D normal.")

    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        raise ValueError(f"Plane {p.get('plane_id', default_id)} has zero normal.")
    normal = normal / norm

    offset = None
    for key in ("offset", "d", "plane_offset"):
        if key in p:
            offset = float(p[key]) / norm
            break
    if offset is None:
        raise ValueError(f"Plane {p.get('plane_id', default_id)} has no plane offset.")

    centroid = None
    for key in ("centroid", "center", "centre"):
        if key in p and p[key] is not None:
            arr = np.asarray(p[key], dtype=float).reshape(-1)
            if arr.size == 3:
                centroid = arr.tolist()
                break

    p["plane_id"] = p.get("plane_id", default_id)
    p["normal"] = normal.tolist()
    p["offset"] = float(offset)
    if centroid is not None:
        p["centroid"] = centroid

    return _to_builtin(p)


def extract_planes(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Supports:
      frame["planes"] = [...]
    and a few array-style NPZ exports:
      normals, offsets, centroids
    """
    if "planes" in frame:
        planes = frame["planes"]
        if isinstance(planes, np.ndarray):
            planes = planes.tolist()
        if not isinstance(planes, list):
            raise ValueError("'planes' must be a list.")
        return [canonicalise_plane_dict(p, i) for i, p in enumerate(planes)]

    if "normals" in frame and ("offsets" in frame or "d" in frame):
        normals = np.asarray(frame["normals"], dtype=float)
        offsets = np.asarray(frame.get("offsets", frame.get("d")), dtype=float).reshape(-1)
        centroids = frame.get("centroids")
        centroids = None if centroids is None else np.asarray(centroids, dtype=float)

        planes = []
        for i in range(len(offsets)):
            p = {
                "plane_id": i,
                "normal": normals[i].tolist(),
                "offset": float(offsets[i]),
            }
            if centroids is not None and i < len(centroids):
                p["centroid"] = centroids[i].tolist()
            planes.append(canonicalise_plane_dict(p, i))
        return planes

    raise ValueError("Could not find plane list. Expected 'planes' or normals/offsets arrays.")


# ---------------------------------------------------------------------------
# Matching geometry
# ---------------------------------------------------------------------------

def orient_estimate_like_reference(
    n_est: np.ndarray,
    d_est: float,
    n_ref: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Resolve plane sign ambiguity so estimated normal points in the same
    hemisphere as reference normal.
    """
    if float(np.dot(n_est, n_ref)) < 0.0:
        return -n_est, -float(d_est)
    return n_est, float(d_est)


def support_iou(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[float]:
    """
    Compute point-index IoU if BOTH planes contain support_indices.
    Returns None if support indices are unavailable.
    """
    if "support_indices" not in a or "support_indices" not in b:
        return None
    if a["support_indices"] is None or b["support_indices"] is None:
        return None

    sa = set(int(x) for x in np.asarray(a["support_indices"]).reshape(-1))
    sb = set(int(x) for x in np.asarray(b["support_indices"]).reshape(-1))
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def match_metrics(
    det: Dict[str, Any],
    ref: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    n_det = np.asarray(det["normal"], dtype=float).reshape(3)
    d_det = float(det["offset"])
    n_ref = np.asarray(ref["normal"], dtype=float).reshape(3)
    d_ref = float(ref["offset"])

    n_det, d_det = orient_estimate_like_reference(n_det, d_det, n_ref)

    dot = float(np.clip(np.dot(n_det, n_ref), -1.0, 1.0))
    normal_deg = float(np.degrees(np.arccos(dot)))
    offset_m = abs(d_det - d_ref)

    centroid_m: Optional[float] = None
    if det.get("centroid") is not None and ref.get("centroid") is not None:
        cd = np.asarray(det["centroid"], dtype=float).reshape(3)
        cr = np.asarray(ref["centroid"], dtype=float).reshape(3)
        centroid_m = float(np.linalg.norm(cd - cr))

    iou = support_iou(det, ref)

    return {
        "normal_deg": normal_deg,
        "offset_m": offset_m,
        "centroid_m": centroid_m,
        "support_iou": iou,
    }


def build_cost(
    m: Dict[str, Optional[float]],
    max_normal_deg: float,
    max_offset_m: float,
    max_centroid_m: float,
    min_iou: float,
    w_normal: float,
    w_offset: float,
    w_centroid: float,
    w_iou: float,
) -> float:
    """
    Eligibility first, then a normalised Hungarian cost.

    IoU is only used if both planes contain support_indices.
    Centroid is only used if both planes contain a centroid.
    """
    if m["normal_deg"] is None or m["normal_deg"] > max_normal_deg:
        return 1e9
    if m["offset_m"] is None or m["offset_m"] > max_offset_m:
        return 1e9

    if m["centroid_m"] is not None and max_centroid_m > 0:
        if m["centroid_m"] > max_centroid_m:
            return 1e9

    if m["support_iou"] is not None and min_iou > 0:
        if m["support_iou"] < min_iou:
            return 1e9

    c = 0.0
    c += w_normal * (float(m["normal_deg"]) / max(max_normal_deg, 1e-12))
    c += w_offset * (float(m["offset_m"]) / max(max_offset_m, 1e-12))

    if m["centroid_m"] is not None and max_centroid_m > 0:
        c += w_centroid * (float(m["centroid_m"]) / max(max_centroid_m, 1e-12))

    if m["support_iou"] is not None:
        c += w_iou * (1.0 - float(m["support_iou"]))

    return float(c)


def match_frame_planes(
    det_planes: List[Dict[str, Any]],
    ref_planes: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    One-to-one Hungarian assignment.

    Returns:
      enriched detected planes
      mapping rows for CSV
    """
    if not det_planes:
        return [], []

    if not ref_planes:
        enriched = []
        mapping = []
        for p in det_planes:
            q = dict(p)
            q["global_plane_id"] = None
            q["gt_assignment_status"] = "no_reference_planes"
            enriched.append(q)
            mapping.append({
                "detected_plane_id": p.get("plane_id"),
                "global_plane_id": None,
                "status": "no_reference_planes",
            })
        return enriched, mapping

    C = np.full((len(det_planes), len(ref_planes)), 1e9, dtype=float)
    metrics: Dict[Tuple[int, int], Dict[str, Optional[float]]] = {}

    for i, dp in enumerate(det_planes):
        for j, rp in enumerate(ref_planes):
            m = match_metrics(dp, rp)
            metrics[(i, j)] = m
            C[i, j] = build_cost(
                m,
                args.max_normal_deg,
                args.max_offset_m,
                args.max_centroid_m,
                args.min_iou,
                args.w_normal,
                args.w_offset,
                args.w_centroid,
                args.w_iou,
            )

    rows, cols = linear_sum_assignment(C)
    assignment = {
        int(i): int(j)
        for i, j in zip(rows, cols)
        if C[int(i), int(j)] < args.invalid_cost
    }

    enriched: List[Dict[str, Any]] = []
    mapping: List[Dict[str, Any]] = []

    for i, dp in enumerate(det_planes):
        q = dict(dp)

        if i not in assignment:
            q["global_plane_id"] = None
            q["gt_assignment_status"] = "unmatched"
            q["gt_assignment_cost"] = None
            q["gt_assignment_errors"] = None

            mapping.append({
                "detected_plane_id": dp.get("plane_id"),
                "reference_plane_id": None,
                "global_plane_id": None,
                "status": "unmatched",
                "cost": None,
                "normal_error_deg": None,
                "offset_error_m": None,
                "centroid_error_m": None,
                "support_iou": None,
            })
        else:
            j = assignment[i]
            rp = ref_planes[j]
            gid = rp.get("global_plane_id", rp.get("plane_id"))
            m = metrics[(i, j)]

            q["global_plane_id"] = str(gid)
            q["gt_reference_plane_id"] = rp.get("plane_id")
            q["gt_assignment_status"] = "matched"
            q["gt_assignment_cost"] = float(C[i, j])
            q["gt_assignment_errors"] = _to_builtin(m)

            mapping.append({
                "detected_plane_id": dp.get("plane_id"),
                "reference_plane_id": rp.get("plane_id"),
                "global_plane_id": str(gid),
                "status": "matched",
                "cost": float(C[i, j]),
                "normal_error_deg": m["normal_deg"],
                "offset_error_m": m["offset_m"],
                "centroid_error_m": m["centroid_m"],
                "support_iou": m["support_iou"],
            })

        enriched.append(q)

    return enriched, mapping


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Append ETH3D-derived persistent plane IDs to TUM Plane Detection output."
    )

    # Default project paths: with these defaults you can simply run:
    #     python export_tum_plane_features_with_gt_ids.py
    ap.add_argument(
        "--detections",
        type=Path,
        default=Path(r"D:\PlaneBasedSLAM\results\plane_detection\tum\final\detections\iterative_ransac"),
        help="Plane Detection per-frame output directory."
    )
    ap.add_argument(
        "--reference",
        type=Path,
        default=Path(r"D:\PlaneBasedSLAM\data\derived_gt\fr1_xyz\frame_plane_gt\frames"),
        help="ETH3D-derived per-frame plane reference directory."
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\PlaneBasedSLAM\results\candidate_gating\tum_gt_export"),
        help="Output directory."
    )

    # Conservative defaults; use the same thresholds as the existing
    # evaluate_plane_detection_eth3d_gt.py if they differ.
    ap.add_argument("--max-normal-deg", type=float, default=10.0)
    ap.add_argument("--max-offset-m", type=float, default=0.10)
    ap.add_argument("--max-centroid-m", type=float, default=0.75)
    ap.add_argument("--min-iou", type=float, default=0.0,
                    help="Only applied if both plane records contain support_indices.")

    ap.add_argument("--w-normal", type=float, default=1.0)
    ap.add_argument("--w-offset", type=float, default=1.0)
    ap.add_argument("--w-centroid", type=float, default=0.5)
    ap.add_argument("--w-iou", type=float, default=0.5)

    ap.add_argument("--invalid-cost", type=float, default=1e8)

    args = ap.parse_args()

    if not args.detections.is_dir():
        raise FileNotFoundError(f"Detection directory not found: {args.detections}")
    if not args.reference.is_dir():
        raise FileNotFoundError(f"Reference directory not found: {args.reference}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    det_files = discover_frame_files(args.detections)
    ref_files = discover_frame_files(args.reference)

    common_ids = sorted(set(det_files) & set(ref_files))
    only_det = sorted(set(det_files) - set(ref_files))
    only_ref = sorted(set(ref_files) - set(det_files))

    if not common_ids:
        raise RuntimeError(
            "No common frame IDs were found between detections and ETH3D reference.\n"
            "Check the frame numbering / filenames."
        )

    print(f"[INFO] Detection frames: {len(det_files)}")
    print(f"[INFO] Reference frames: {len(ref_files)}")
    print(f"[INFO] Common frames:    {len(common_ids)}")

    if only_det:
        print(f"[WARN] {len(only_det)} detection frames have no reference frame.")
    if only_ref:
        print(f"[INFO] {len(only_ref)} reference frames have no detection frame.")

    output_frames: List[Dict[str, Any]] = []
    mapping_rows: List[Dict[str, Any]] = []

    n_det_total = 0
    n_matched_total = 0

    for frame_id in common_ids:
        det_frame = load_file(det_files[frame_id])
        ref_frame = load_file(ref_files[frame_id])

        det_planes = extract_planes(det_frame)
        ref_planes = extract_planes(ref_frame)

        enriched, mapping = match_frame_planes(det_planes, ref_planes, args)

        timestamp = det_frame.get("timestamp", ref_frame.get("timestamp"))

        output_frames.append({
            "frame_id": int(frame_id),
            "timestamp": timestamp,
            "source_detection_file": str(det_files[frame_id]),
            "source_reference_file": str(ref_files[frame_id]),
            "planes": enriched,
        })

        for row in mapping:
            row = dict(row)
            row["frame_id"] = int(frame_id)
            row["timestamp"] = timestamp
            mapping_rows.append(row)

        n_det_total += len(det_planes)
        n_matched_total += sum(1 for p in enriched if p.get("global_plane_id") is not None)

        print(
            f"[FRAME {frame_id:06d}] "
            f"detected={len(det_planes):2d}, "
            f"reference={len(ref_planes):2d}, "
            f"matched={sum(1 for p in enriched if p.get('global_plane_id') is not None):2d}"
        )

    combined = {
        "description": (
            "Plane Detection output enriched with ETH3D-derived persistent physical-plane IDs. "
            "global_plane_id is evaluation-only and must not be used by Candidate Gating algorithms."
        ),
        "matching_config": {
            "max_normal_deg": args.max_normal_deg,
            "max_offset_m": args.max_offset_m,
            "max_centroid_m": args.max_centroid_m,
            "min_iou": args.min_iou,
            "w_normal": args.w_normal,
            "w_offset": args.w_offset,
            "w_centroid": args.w_centroid,
            "w_iou": args.w_iou,
        },
        "frames": output_frames,
    }

    json_out = args.output_dir / "tum_plane_features_with_gt_ids.json"
    csv_out = args.output_dir / "tum_plane_features_with_gt_ids.csv"
    summary_out = args.output_dir / "tum_plane_gt_export_summary.txt"

    save_json(combined, json_out)

    csv_fields = [
        "frame_id",
        "timestamp",
        "detected_plane_id",
        "reference_plane_id",
        "global_plane_id",
        "status",
        "cost",
        "normal_error_deg",
        "offset_error_m",
        "centroid_error_m",
        "support_iou",
    ]

    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in mapping_rows:
            writer.writerow({k: row.get(k) for k in csv_fields})

    match_rate = n_matched_total / n_det_total if n_det_total else 0.0

    summary_text = f"""TUM Plane Feature + ETH3D GT ID Export
====================================

Detection directory:
{args.detections}

Reference directory:
{args.reference}

Common frames:
{len(common_ids)}

Detected planes:
{n_det_total}

Matched detected planes:
{n_matched_total}

Unmatched detected planes:
{n_det_total - n_matched_total}

Matched fraction:
{match_rate:.4f}

Output JSON:
{json_out}

Output CSV:
{csv_out}

IMPORTANT:
global_plane_id is ground-truth/evaluation metadata only.
Do not pass it into Candidate Gating algorithms.
"""

    summary_out.write_text(summary_text, encoding="utf-8")

    print("\n[DONE]")
    print(f"JSON:    {json_out}")
    print(f"CSV:     {csv_out}")
    print(f"Summary: {summary_out}")
    print(f"Matched detected planes: {n_matched_total}/{n_det_total} ({match_rate:.1%})")


if __name__ == "__main__":
    main()
