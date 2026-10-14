"""
export_candidate_gating_input_eth3d_final.py

Export Candidate Gating input for exactly the same ETH3D-evaluated frames
used by evaluate_plane_detection_eth3d_gt.py: frame 0003 ... 0078.

It re-runs only Iterative RANSAC with the same random seed convention and
attaches the already-computed ETH3D gt_plane_id from matched_planes.csv.

Frames with no successful ETH3D match are still exported; their
global_plane_ids are -1.

The output folder is cleared first so stale frame_0000/0001/0002 files from
older runs cannot remain.
"""

from pathlib import Path
import csv
import shutil
import sys

import numpy as np
from scipy.spatial import ConvexHull, QhullError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = PROJECT_ROOT / "experiments" / "feature_finding"

if str(FEATURE_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_DIR))

import iterative_ransac


SEQUENCE = "rgbd_dataset_freiburg1_xyz"
BASE_SEED = 20260919
CONTOUR_SAMPLES = 32

# Must match evaluate_plane_detection_eth3d_gt.py exactly.
FRAME_IDS = list(range(3, 79))

FRAMES_DIR = (
    PROJECT_ROOT / "data" / "tum" / "plane_detection"
    / SEQUENCE / "frames"
)

GT_FRAMES_DIR = (
    PROJECT_ROOT / "data" / "derived_gt"
    / "fr1_xyz" / "frame_plane_gt" / "frames"
)

MATCHED_CSV = (
    PROJECT_ROOT / "results" / "plane_detection"
    / "tum_eth3d_gt" / "matched_planes.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "candidate_gating"
    / "tum_plane_features_with_gt_ids"
)


def load_gt_ids():
    labels = {}

    with MATCHED_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] != "Iterative RANSAC":
                continue

            fid = int(row["frame_id"])
            model_index = int(row["original_model_index"])
            gt_id = int(row["gt_plane_id"])

            labels.setdefault(fid, {})[model_index] = gt_id

    return labels


def plane_contour(points, mask, normal):
    support = points[mask]

    if len(support) < 3:
        centroid = np.mean(support, axis=0) if len(support) else np.zeros(3)
        return centroid, np.repeat(
            centroid[None, :], CONTOUR_SAMPLES, axis=0
        )

    centroid = np.mean(support, axis=0)

    n = np.asarray(normal, dtype=float)
    n /= np.linalg.norm(n)

    ref = (
        np.array([0.0, 0.0, 1.0])
        if abs(n[2]) < 0.9
        else np.array([1.0, 0.0, 0.0])
    )

    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    centred = support - centroid
    xy = np.column_stack((centred @ u, centred @ v))

    try:
        hull = ConvexHull(xy)
        polygon_xy = xy[hull.vertices]
    except QhullError:
        polygon_xy = np.array([
            [xy[:, 0].min(), xy[:, 1].min()],
            [xy[:, 0].max(), xy[:, 1].min()],
            [xy[:, 0].max(), xy[:, 1].max()],
            [xy[:, 0].min(), xy[:, 1].max()],
        ])

    polygon = (
        centroid
        + polygon_xy[:, 0:1] * u
        + polygon_xy[:, 1:2] * v
    )

    closed = np.vstack([polygon, polygon[0]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(np.sum(lengths))

    if perimeter < 1e-12:
        return centroid, np.repeat(
            centroid[None, :], CONTOUR_SAMPLES, axis=0
        )

    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    targets = np.linspace(
        0.0, perimeter, CONTOUR_SAMPLES, endpoint=False
    )

    samples = []

    for target in targets:
        k = np.searchsorted(cumulative, target, side="right") - 1
        k = min(k, len(lengths) - 1)

        alpha = (
            (target - cumulative[k]) / lengths[k]
            if lengths[k] > 1e-12
            else 0.0
        )

        samples.append(
            (1.0 - alpha) * closed[k]
            + alpha * closed[k + 1]
        )

    return centroid, np.asarray(samples)


def main():
    if not MATCHED_CSV.exists():
        raise FileNotFoundError(MATCHED_CSV)

    # Important: remove stale files from previous versions/runs.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = load_gt_ids()

    exported = 0

    for fid in FRAME_IDS:
        frame_file = FRAMES_DIR / f"frame_{fid:04d}.npz"
        gt_file = GT_FRAMES_DIR / f"frame_{fid:04d}.npz"

        if not frame_file.exists():
            raise FileNotFoundError(frame_file)
        if not gt_file.exists():
            raise FileNotFoundError(gt_file)

        with np.load(frame_file, allow_pickle=False) as data:
            points = np.asarray(
                data["detector_points"],
                dtype=np.float64,
            )

        # Same Iterative-RANSAC seed used by the ETH3D evaluator:
        # BASE_SEED + frame_id * 100 + method_index, method_index = 0.
        rng = np.random.default_rng(BASE_SEED + fid * 100)

        models = iterative_ransac.detect_planes(points, rng)
        models = [] if models is None else models

        n = len(models)

        normals = np.empty((n, 3), dtype=np.float64)
        offsets = np.empty(n, dtype=np.float64)
        centroids = np.empty((n, 3), dtype=np.float64)
        contours = np.empty(
            (n, CONTOUR_SAMPLES, 3), dtype=np.float64
        )
        support_counts = np.empty(n, dtype=np.int32)
        support_masks = np.empty(
            (n, len(points)), dtype=bool
        )
        global_ids = np.full(n, -1, dtype=np.int32)

        frame_labels = labels.get(fid, {})

        for model_index, model in enumerate(models):
            mask = np.asarray(model["mask"], dtype=bool)

            centroid, contour = plane_contour(
                points, mask, model["normal"]
            )

            normals[model_index] = model["normal"]
            offsets[model_index] = model["d"]
            centroids[model_index] = centroid
            contours[model_index] = contour
            support_counts[model_index] = int(np.sum(mask))
            support_masks[model_index] = mask

            if model_index in frame_labels:
                global_ids[model_index] = frame_labels[model_index]

        np.savez_compressed(
            OUTPUT_DIR / f"frame_{fid:04d}.npz",
            normals=normals,
            offsets=offsets,
            centroids=centroids,
            contours=contours,
            support_counts=support_counts,
            support_masks=support_masks,
            global_plane_ids=global_ids,
            gt_known=(global_ids >= 0),
        )

        exported += 1

        print(
            f"frame_{fid:04d}: "
            f"{n} planes, "
            f"{int(np.sum(global_ids >= 0))} with ETH3D ID"
        )

    print("\nDone.")
    print(f"Frames exported: {exported}")
    print(f"Expected       : {len(FRAME_IDS)}")
    print(f"Output         : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
