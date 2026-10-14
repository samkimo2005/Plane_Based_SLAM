from __future__ import annotations

"""
build_tum_plane_gt_from_eth3d.py

Build per-frame plane reference data for TUM Freiburg1 XYZ from:
  1. the ETH3D-derived global plane reference,
  2. the simplified labelled mesh topology,
  3. TUM RGB-D prepared frames,
  4. the refined ETH3D <- TUM rigid transform.

The generated labels are mesh-derived reference labels, NOT official TUM
plane annotations.

Outputs:
    data/derived_gt/fr1_xyz/frame_plane_gt/
        frames/frame_0000.npz ...
        overlays/frame_0000.png ...
        summary.csv
        metadata.json
"""

from pathlib import Path
import csv
import json

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


SEQUENCE_NAME = "rgbd_dataset_freiburg1_xyz"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TUM_FRAMES_DIR = (
    PROJECT_ROOT / "data" / "tum" / "plane_detection"
    / SEQUENCE_NAME / "frames"
)

GLOBAL_REF_DIR = (
    PROJECT_ROOT / "data" / "derived_gt" / "fr1_xyz"
    / "global_plane_reference"
)

SIMPLIFIED_MESH_FILE = GLOBAL_REF_DIR / "simplified_mesh.ply"
COMPONENT_LABELS_FILE = GLOBAL_REF_DIR / "plane_component_labels.npy"
GLOBAL_PLANES_FILE = GLOBAL_REF_DIR / "global_planes.npz"

REFINED_TRANSFORM_FILE = (
    PROJECT_ROOT / "data" / "derived_gt" / "fr1_xyz"
    / "alignment_refined" / "T_eth3d_from_tum_refined.npy"
)

FALLBACK_TRANSFORM_FILE = (
    PROJECT_ROOT / "data" / "derived_gt" / "fr1_xyz"
    / "alignment_check" / "T_eth3d_from_tum.npy"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "derived_gt" / "fr1_xyz"
    / "frame_plane_gt"
)
OUTPUT_FRAMES_DIR = OUTPUT_DIR / "frames"
OVERLAY_DIR = OUTPUT_DIR / "overlays"

# Mesh hit must agree with the observed RGB-D depth.
DEPTH_CONSISTENCY_M = 0.080

# Observed 3D point must also be close to the global plane equation.
PLANE_RESIDUAL_M = 0.050

# Detector points inherit labels from their nearest raw RGB-D point.
DETECTOR_TO_RAW_MAX_M = 0.035

# A plane is considered evaluable in a frame only when enough detector points
# support it. This matches the common project minimum plane support scale.
MIN_EVALUABLE_DETECTOR_POINTS = 120

RAY_BATCH_SIZE = 100_000
SAVE_OVERLAYS = True
OVERLAY_EVERY_N_FRAMES = 5


def transform_points(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


def infer_uv_order(raw_pixels, raw_points, depth_m):
    """
    Decide whether raw_pixels stores (u,v) or (v,u) by checking which
    interpretation reproduces raw_points[:,2] from depth_m more accurately.
    """
    h, w = depth_m.shape
    p = np.asarray(raw_pixels, dtype=np.int64)
    z = np.asarray(raw_points[:, 2], dtype=np.float64)

    candidates = []

    for name, u, v in [
        ("uv", p[:, 0], p[:, 1]),
        ("vu", p[:, 1], p[:, 0]),
    ]:
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(inside):
            candidates.append((np.inf, name, u, v))
            continue

        sample_idx = np.flatnonzero(inside)
        if len(sample_idx) > 20000:
            sample_idx = sample_idx[:: max(1, len(sample_idx) // 20000)]

        depth_values = depth_m[v[sample_idx], u[sample_idx]].astype(np.float64)
        errors = np.abs(depth_values - z[sample_idx])
        errors = errors[np.isfinite(errors)]

        score = float(np.median(errors)) if len(errors) else np.inf
        candidates.append((score, name, u, v))

    candidates.sort(key=lambda x: x[0])
    score, name, u, v = candidates[0]

    if not np.isfinite(score) or score > 0.02:
        raise RuntimeError(
            "Could not reliably infer raw_pixels coordinate order. "
            f"Best median depth mismatch = {score:.6f} m."
        )

    return name, u.astype(np.int32), v.astype(np.int32), score


def build_face_to_plane(component_labels, accepted_component_ids):
    face_to_plane = np.full(len(component_labels), -1, dtype=np.int32)

    for plane_id, component_id in enumerate(accepted_component_ids):
        face_to_plane[component_labels == component_id] = plane_id

    return face_to_plane


def raycast_valid_pixels(
    scene,
    raw_points_cam,
    u,
    v,
    K,
    T_eth_cam,
    face_to_plane,
    normals_world,
    offsets_world,
):
    """
    Raycast only the valid-depth pixels. Returns one global plane ID per
    raw point, with -1 for unknown / non-planar / inconsistent.
    """
    n = len(raw_points_cam)
    labels = np.full(n, -1, dtype=np.int32)

    K_inv = np.linalg.inv(K)

    homog = np.stack(
        [
            u.astype(np.float64),
            v.astype(np.float64),
            np.ones(n, dtype=np.float64),
        ],
        axis=1,
    )

    dirs_cam = homog @ K_inv.T
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)

    R_eth_cam = T_eth_cam[:3, :3]
    t_eth_cam = T_eth_cam[:3, 3]

    dirs_eth = dirs_cam @ R_eth_cam.T
    origins = np.repeat(t_eth_cam[None, :], n, axis=0)

    raw_points_eth = transform_points(raw_points_cam, T_eth_cam)

    mesh_face_count = len(face_to_plane)

    for start in range(0, n, RAY_BATCH_SIZE):
        end = min(start + RAY_BATCH_SIZE, n)

        rays_np = np.concatenate(
            [
                origins[start:end],
                dirs_eth[start:end],
            ],
            axis=1,
        ).astype(np.float32)

        result = scene.cast_rays(o3d.core.Tensor(rays_np))
        t_hit = result["t_hit"].numpy().astype(np.float64)
        primitive_ids = result["primitive_ids"].numpy().astype(np.int64)

        hit_ok = np.isfinite(t_hit)
        hit_ok &= primitive_ids >= 0
        hit_ok &= primitive_ids < mesh_face_count

        if not np.any(hit_ok):
            continue

        local_indices = np.flatnonzero(hit_ok)
        global_indices = start + local_indices

        prim = primitive_ids[local_indices]
        candidate_plane_ids = face_to_plane[prim]

        plane_ok = candidate_plane_ids >= 0
        if not np.any(plane_ok):
            continue

        local_indices = local_indices[plane_ok]
        global_indices = global_indices[plane_ok]
        candidate_plane_ids = candidate_plane_ids[plane_ok]

        # Compute rendered mesh hit position and convert it back to camera frame.
        hit_points_eth = (
            origins[global_indices]
            + t_hit[local_indices, None] * dirs_eth[global_indices]
        )

        hit_points_cam = (
            hit_points_eth - t_eth_cam[None, :]
        ) @ R_eth_cam

        z_mesh = hit_points_cam[:, 2]
        z_observed = raw_points_cam[global_indices, 2]

        depth_ok = (
            np.isfinite(z_mesh)
            & (z_mesh > 0)
            & (np.abs(z_mesh - z_observed) <= DEPTH_CONSISTENCY_M)
        )

        p_eth = raw_points_eth[global_indices]
        n_world = normals_world[candidate_plane_ids]
        d_world = offsets_world[candidate_plane_ids]

        plane_residual = np.abs(
            np.einsum("ij,ij->i", p_eth, n_world) + d_world
        )

        residual_ok = plane_residual <= PLANE_RESIDUAL_M
        final_ok = depth_ok & residual_ok

        labels[global_indices[final_ok]] = candidate_plane_ids[final_ok]

    return labels


def plane_to_camera(normal_world, offset_world, T_world_cam):
    """
    Plane world equation:
        n_w^T x_w + d_w = 0
    with x_w = R_wc x_c + t_wc

    Therefore:
        n_c = R_wc^T n_w
        d_c = n_w^T t_wc + d_w
    """
    R_wc = T_world_cam[:3, :3]
    t_wc = T_world_cam[:3, 3]

    n_c = R_wc.T @ normal_world
    d_c = float(np.dot(normal_world, t_wc) + offset_world)

    return n_c, d_c


def save_overlay(rgb, label_image, evaluable_ids, output_path):
    try:
        from PIL import Image
    except ImportError:
        return False

    image = np.asarray(rgb, dtype=np.uint8).copy()
    overlay = image.astype(np.float32)

    palette_rng = np.random.default_rng(20260919)

    # Stable colour table indexed by global plane ID.
    max_id = int(max(evaluable_ids)) if len(evaluable_ids) else -1
    colours = {}
    for pid in range(max_id + 1):
        colours[pid] = palette_rng.integers(30, 240, size=3, dtype=np.uint8)

    for pid in evaluable_ids:
        mask = label_image == int(pid)
        if not np.any(mask):
            continue

        colour = colours[int(pid)].astype(np.float32)
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * colour

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(output_path)
    return True


def main():
    OUTPUT_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    required = [
        SIMPLIFIED_MESH_FILE,
        COMPONENT_LABELS_FILE,
        GLOBAL_PLANES_FILE,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing:\n{path}")

    if REFINED_TRANSFORM_FILE.exists():
        T_eth_from_tum = np.load(REFINED_TRANSFORM_FILE)
        alignment_source = str(REFINED_TRANSFORM_FILE)
    elif FALLBACK_TRANSFORM_FILE.exists():
        T_eth_from_tum = np.load(FALLBACK_TRANSFORM_FILE)
        alignment_source = str(FALLBACK_TRANSFORM_FILE)
    else:
        raise FileNotFoundError(
            "No ETH3D <- TUM alignment transform found. "
            "Run the alignment scripts first."
        )

    with np.load(GLOBAL_PLANES_FILE, allow_pickle=False) as gp:
        normals_world = np.asarray(gp["normals_world"], dtype=np.float64)
        offsets_world = np.asarray(gp["offsets_world"], dtype=np.float64)
        centroids_world = np.asarray(gp["centroids_world"], dtype=np.float64)
        areas_m2 = np.asarray(gp["areas_m2"], dtype=np.float64)
        component_ids = np.asarray(gp["component_ids"], dtype=np.int32)

    component_labels = np.load(COMPONENT_LABELS_FILE).astype(np.int32)

    mesh_legacy = o3d.io.read_triangle_mesh(str(SIMPLIFIED_MESH_FILE))
    if len(mesh_legacy.triangles) != len(component_labels):
        raise RuntimeError(
            "simplified_mesh.ply triangle count does not match "
            "plane_component_labels.npy."
        )

    face_to_plane = build_face_to_plane(component_labels, component_ids)

    mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_tensor)

    frame_files = sorted(TUM_FRAMES_DIR.glob("frame_*.npz"))
    if not frame_files:
        raise FileNotFoundError(f"No TUM prepared frames in:\n{TUM_FRAMES_DIR}")

    print()
    print("=" * 94)
    print("BUILD TUM PER-FRAME PLANE REFERENCE FROM ETH3D")
    print("=" * 94)
    print(f"Frames              : {len(frame_files)}")
    print(f"Global planes       : {len(normals_world)}")
    print(f"Alignment transform : {alignment_source}")
    print(f"Depth consistency   : {DEPTH_CONSISTENCY_M*1000:.0f} mm")
    print(f"Plane residual      : {PLANE_RESIDUAL_M*1000:.0f} mm")
    print()

    summary_rows = []
    inferred_pixel_order = None

    for frame_index, frame_path in enumerate(frame_files):
        with np.load(frame_path, allow_pickle=False) as f:
            raw_points = np.asarray(f["raw_points"], dtype=np.float64)
            raw_pixels = np.asarray(f["raw_pixels"], dtype=np.int32)
            detector_points = np.asarray(f["detector_points"], dtype=np.float64)
            depth_m = np.asarray(f["depth_m"], dtype=np.float64)
            rgb = np.asarray(f["rgb"], dtype=np.uint8)
            K = np.asarray(f["camera_intrinsics"], dtype=np.float64)
            T_tum_cam = np.asarray(f["T_world_cam"], dtype=np.float64)
            image_size = np.asarray(f["image_size"], dtype=np.int32)

        h, w = depth_m.shape

        pixel_order, u, v, pixel_depth_error = infer_uv_order(
            raw_pixels,
            raw_points,
            depth_m,
        )

        if inferred_pixel_order is None:
            inferred_pixel_order = pixel_order
            print(
                f"Detected raw_pixels order: {pixel_order} "
                f"(median depth mismatch {pixel_depth_error*1000:.3f} mm)"
            )
        elif pixel_order != inferred_pixel_order:
            raise RuntimeError(
                f"Pixel coordinate order changed at {frame_path.name}: "
                f"{pixel_order} vs {inferred_pixel_order}"
            )

        T_eth_cam = T_eth_from_tum @ T_tum_cam

        raw_labels = raycast_valid_pixels(
            scene,
            raw_points,
            u,
            v,
            K,
            T_eth_cam,
            face_to_plane,
            normals_world,
            offsets_world,
        )

        # Build label image.
        label_image = np.full((h, w), -1, dtype=np.int32)
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        label_image[v[inside], u[inside]] = raw_labels[inside]

        # Transfer raw labels to detector points.
        raw_tree = cKDTree(raw_points)
        nearest_dist, nearest_idx = raw_tree.query(
            detector_points,
            k=1,
            workers=-1,
        )

        detector_labels = np.full(len(detector_points), -1, dtype=np.int32)
        transferable = (
            np.isfinite(nearest_dist)
            & (nearest_dist <= DETECTOR_TO_RAW_MAX_M)
        )
        detector_labels[transferable] = raw_labels[nearest_idx[transferable]]

        # Find planes that are sufficiently supported at detector resolution.
        counts = np.bincount(
            detector_labels[detector_labels >= 0],
            minlength=len(normals_world),
        )

        evaluable_ids = np.flatnonzero(
            counts >= MIN_EVALUABLE_DETECTOR_POINTS
        ).astype(np.int32)

        # Rows correspond exactly to evaluable_ids.
        gt_support_masks = np.stack(
            [detector_labels == int(pid) for pid in evaluable_ids],
            axis=0,
        ) if len(evaluable_ids) else np.zeros(
            (0, len(detector_points)),
            dtype=bool,
        )

        gt_pixel_masks = np.stack(
            [label_image == int(pid) for pid in evaluable_ids],
            axis=0,
        ) if len(evaluable_ids) else np.zeros(
            (0, h, w),
            dtype=bool,
        )

        gt_normals = []
        gt_offsets = []
        gt_centroids = []
        gt_raw_support = []
        gt_detector_support = []

        for pid in evaluable_ids:
            n_c, d_c = plane_to_camera(
                normals_world[pid],
                offsets_world[pid],
                T_eth_cam,
            )

            support_raw = raw_labels == int(pid)

            if np.any(support_raw):
                centroid_cam = np.mean(raw_points[support_raw], axis=0)
            else:
                # Fallback: transform global centroid into camera coordinates.
                T_cam_eth = np.linalg.inv(T_eth_cam)
                centroid_cam = transform_points(
                    centroids_world[pid][None, :],
                    T_cam_eth,
                )[0]

            gt_normals.append(n_c)
            gt_offsets.append(d_c)
            gt_centroids.append(centroid_cam)
            gt_raw_support.append(int(np.sum(support_raw)))
            gt_detector_support.append(int(counts[pid]))

        gt_normals = (
            np.asarray(gt_normals, dtype=np.float64).reshape(-1, 3)
        )
        gt_offsets = np.asarray(gt_offsets, dtype=np.float64)
        gt_centroids = (
            np.asarray(gt_centroids, dtype=np.float64).reshape(-1, 3)
        )
        gt_raw_support = np.asarray(gt_raw_support, dtype=np.int32)
        gt_detector_support = np.asarray(
            gt_detector_support,
            dtype=np.int32,
        )

        out_path = OUTPUT_FRAMES_DIR / frame_path.name

        np.savez_compressed(
            out_path,
            gt_plane_ids=evaluable_ids,
            gt_normals=gt_normals,
            gt_offsets=gt_offsets,
            gt_centroids=gt_centroids,
            gt_support_masks=gt_support_masks,
            gt_pixel_masks=gt_pixel_masks,
            gt_raw_point_labels=raw_labels,
            gt_detector_point_labels=detector_labels,
            gt_label_image=label_image,
            gt_raw_support=gt_raw_support,
            gt_detector_support=gt_detector_support,
            gt_global_plane_areas_m2=areas_m2[evaluable_ids],
            T_eth3d_world_cam=T_eth_cam,
            T_eth3d_from_tum=T_eth_from_tum,
            source_frame_name=np.asarray(frame_path.name),
        )

        labelled_raw_fraction = float(np.mean(raw_labels >= 0))
        labelled_detector_fraction = float(np.mean(detector_labels >= 0))

        summary_rows.append(
            {
                "frame": frame_path.name,
                "evaluable_plane_count": int(len(evaluable_ids)),
                "evaluable_plane_ids": " ".join(map(str, evaluable_ids.tolist())),
                "labelled_raw_fraction": labelled_raw_fraction,
                "labelled_detector_fraction": labelled_detector_fraction,
                "raw_support_total": int(np.sum(raw_labels >= 0)),
                "detector_support_total": int(np.sum(detector_labels >= 0)),
            }
        )

        if (
            SAVE_OVERLAYS
            and frame_index % OVERLAY_EVERY_N_FRAMES == 0
        ):
            save_overlay(
                rgb,
                label_image,
                evaluable_ids,
                OVERLAY_DIR / f"{frame_path.stem}.png",
            )

        print(
            f"{frame_path.name}: "
            f"planes={len(evaluable_ids):2d}, "
            f"raw-labelled={labelled_raw_fraction*100:5.1f}%, "
            f"detector-labelled={labelled_detector_fraction*100:5.1f}%"
        )

    summary_path = OUTPUT_DIR / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata = {
        "sequence": SEQUENCE_NAME,
        "reference_type": "ETH3D mesh-derived global plane reference",
        "not_official_tum_plane_annotation": True,
        "global_plane_count": int(len(normals_world)),
        "frame_count": int(len(frame_files)),
        "alignment_transform_source": alignment_source,
        "depth_consistency_m": DEPTH_CONSISTENCY_M,
        "plane_residual_m": PLANE_RESIDUAL_M,
        "detector_to_raw_max_m": DETECTOR_TO_RAW_MAX_M,
        "min_evaluable_detector_points": MIN_EVALUABLE_DETECTOR_POINTS,
        "raw_pixels_order": inferred_pixel_order,
    }

    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    plane_counts = np.asarray(
        [r["evaluable_plane_count"] for r in summary_rows],
        dtype=np.float64,
    )
    raw_fractions = np.asarray(
        [r["labelled_raw_fraction"] for r in summary_rows],
        dtype=np.float64,
    )

    print()
    print("=" * 94)
    print("PER-FRAME PLANE REFERENCE COMPLETE")
    print("=" * 94)
    print(f"Frames written              : {len(summary_rows)}")
    print(f"Mean evaluable planes/frame : {np.mean(plane_counts):.2f}")
    print(f"Median planes/frame         : {np.median(plane_counts):.2f}")
    print(f"Mean labelled raw fraction  : {np.mean(raw_fractions)*100:.1f}%")
    print(f"Summary                     : {summary_path}")
    print(f"Frame GT directory          : {OUTPUT_FRAMES_DIR}")
    print(f"Overlay directory           : {OVERLAY_DIR}")
    print()
    print(
        "Inspect several overlays before using these references for the "
        "formal Plane Detection decision matrix."
    )


if __name__ == "__main__":
    main()
