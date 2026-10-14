from __future__ import annotations

"""
refine_eth3d_tum_alignment.py

Place in:
    D:/PlaneBasedSLAM/experiments/feature_finding/refine_eth3d_tum_alignment.py

Refines the previously estimated rigid transform between the TUM ground-truth
world frame and the ETH3D reconstruction frame.

Input:
    data/tum/plane_detection/rgbd_dataset_freiburg1_xyz/frames/frame_*.npz
    data/derived_gt/fr1_xyz/eth3d_mesh/fr1_xyz_eth3d_sampled.ply
    data/derived_gt/fr1_xyz/alignment_check/T_eth3d_from_tum.npy

Output:
    data/derived_gt/fr1_xyz/alignment_refined/
        T_eth3d_from_tum_refined.npy
        alignment_refined_report.json
        tum_world_refined_to_eth3d.ply
"""

from pathlib import Path
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

ETH3D_CLOUD_FILE = (
    PROJECT_ROOT / "data" / "derived_gt" / "fr1_xyz"
    / "eth3d_mesh" / "fr1_xyz_eth3d_sampled.ply"
)

INITIAL_TRANSFORM_FILE = (
    PROJECT_ROOT / "data" / "derived_gt" / "fr1_xyz"
    / "alignment_check" / "T_eth3d_from_tum.npy"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "derived_gt" / "fr1_xyz"
    / "alignment_refined"
)

NUMBER_OF_FRAMES = 30
MAX_POINTS_PER_FRAME = 12000
SOURCE_VOXEL_M = 0.015
INITIAL_OVERLAP_GATE_M = 0.15

# Multi-scale point-to-plane ICP: (voxel size, correspondence distance, iterations)
ICP_LEVELS = [
    (0.050, 0.100, 80),
    (0.030, 0.060, 80),
    (0.020, 0.040, 100),
]

RNG = np.random.default_rng(20260919)
SHOW_VIEWER = True


def transform_points(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


def make_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def nn_stats(source_points, target_points):
    tree = cKDTree(target_points)
    d, _ = tree.query(source_points, k=1, workers=-1)
    d = d[np.isfinite(d)]
    return {
        "count": int(len(d)),
        "mean_m": float(np.mean(d)),
        "median_m": float(np.median(d)),
        "p90_m": float(np.percentile(d, 90)),
        "p95_m": float(np.percentile(d, 95)),
        "within_0.02m": float(np.mean(d <= 0.02)),
        "within_0.05m": float(np.mean(d <= 0.05)),
        "within_0.10m": float(np.mean(d <= 0.10)),
    }


def rotation_angle_deg(T):
    R = T[:3, :3]
    x = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(x)))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(TUM_FRAMES_DIR.glob("frame_*.npz"))
    if not frame_files:
        raise FileNotFoundError(f"No prepared TUM frames in:\n{TUM_FRAMES_DIR}")
    if not ETH3D_CLOUD_FILE.exists():
        raise FileNotFoundError(f"ETH3D cloud missing:\n{ETH3D_CLOUD_FILE}")
    if not INITIAL_TRANSFORM_FILE.exists():
        raise FileNotFoundError(
            f"Initial alignment transform missing:\n{INITIAL_TRANSFORM_FILE}"
        )

    n = min(NUMBER_OF_FRAMES, len(frame_files))
    idx = np.unique(np.round(np.linspace(0, len(frame_files) - 1, n)).astype(int))
    selected = [frame_files[i] for i in idx]

    chunks = []
    print()
    print("=" * 88)
    print("REFINE ETH3D <-> TUM ALIGNMENT")
    print("=" * 88)
    print(f"Frames used: {len(selected)}")

    for path in selected:
        with np.load(path, allow_pickle=False) as f:
            pts = np.asarray(f["detector_points"], dtype=np.float64)
            T_wc = np.asarray(f["T_world_cam"], dtype=np.float64)

        if len(pts) > MAX_POINTS_PER_FRAME:
            keep = RNG.choice(len(pts), MAX_POINTS_PER_FRAME, replace=False)
            pts = pts[keep]

        pts_w = transform_points(pts, T_wc)
        pts_w = pts_w[np.all(np.isfinite(pts_w), axis=1)]
        chunks.append(pts_w)

    source_tum = make_pcd(np.vstack(chunks)).voxel_down_sample(SOURCE_VOXEL_M)
    target_eth = o3d.io.read_point_cloud(str(ETH3D_CLOUD_FILE))

    if len(target_eth.points) == 0:
        raise RuntimeError("ETH3D target cloud is empty.")

    T_initial = np.load(INITIAL_TRANSFORM_FILE)
    if T_initial.shape != (4, 4):
        raise ValueError("Initial transform is not 4x4.")

    source_initial = o3d.geometry.PointCloud(source_tum)
    source_initial.transform(T_initial)

    src_init_np = np.asarray(source_initial.points)
    tgt_np = np.asarray(target_eth.points)

    # Remove obvious non-overlap points before refinement.
    tree = cKDTree(tgt_np)
    d, _ = tree.query(src_init_np, k=1, workers=-1)
    overlap_mask = np.isfinite(d) & (d <= INITIAL_OVERLAP_GATE_M)

    source_overlap_eth = make_pcd(src_init_np[overlap_mask])
    if len(source_overlap_eth.points) < 1000:
        raise RuntimeError(
            "Too few overlapping points after the initial alignment gate."
        )

    print(f"TUM fused source       : {len(source_tum.points):,} points")
    print(f"Initial overlap points : {len(source_overlap_eth.points):,}")
    print(f"Initial transform      :\n{T_initial}")

    # We are now operating in the ETH3D frame, so ICP starts at identity.
    delta_total = np.eye(4)

    for level, (voxel, max_corr, iterations) in enumerate(ICP_LEVELS, start=1):
        src = source_overlap_eth.voxel_down_sample(voxel)
        tgt = target_eth.voxel_down_sample(voxel)

        tgt.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(0.08, voxel * 4.0),
                max_nn=50,
            )
        )

        result = o3d.pipelines.registration.registration_icp(
            src,
            tgt,
            max_corr,
            delta_total,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=iterations
            ),
        )

        delta_total = np.asarray(result.transformation)

        print()
        print(
            f"Level {level}: voxel={voxel:.3f} m, "
            f"max_corr={max_corr:.3f} m"
        )
        print(f"  fitness     : {result.fitness:.6f}")
        print(f"  inlier RMSE : {result.inlier_rmse * 1000:.2f} mm")

    T_refined = delta_total @ T_initial

    source_refined = o3d.geometry.PointCloud(source_tum)
    source_refined.transform(T_refined)

    refined_stats = nn_stats(
        np.asarray(source_refined.points),
        np.asarray(target_eth.points),
    )

    correction = T_refined @ np.linalg.inv(T_initial)
    correction_translation = float(np.linalg.norm(correction[:3, 3]))
    correction_rotation = rotation_angle_deg(correction)

    print()
    print("=" * 88)
    print("REFINED ALIGNMENT RESULT")
    print("=" * 88)
    print("T_ETH3D_from_TUM_refined =")
    print(T_refined)
    print()
    print(f"Additional translation correction : {correction_translation*1000:.2f} mm")
    print(f"Additional rotation correction    : {correction_rotation:.4f} deg")
    print(f"NN median                         : {refined_stats['median_m']*1000:.2f} mm")
    print(f"NN p90                            : {refined_stats['p90_m']*1000:.2f} mm")
    print(f"<= 50 mm                          : {refined_stats['within_0.05m']*100:.1f}%")
    print(f"<=100 mm                          : {refined_stats['within_0.10m']*100:.1f}%")

    transform_path = OUTPUT_DIR / "T_eth3d_from_tum_refined.npy"
    cloud_path = OUTPUT_DIR / "tum_world_refined_to_eth3d.ply"
    report_path = OUTPUT_DIR / "alignment_refined_report.json"

    np.save(transform_path, T_refined)
    o3d.io.write_point_cloud(str(cloud_path), source_refined)

    report = {
        "sequence": SEQUENCE_NAME,
        "frames_used": [p.name for p in selected],
        "initial_transform": T_initial.tolist(),
        "refined_transform": T_refined.tolist(),
        "additional_correction_translation_m": correction_translation,
        "additional_correction_rotation_deg": correction_rotation,
        "nearest_neighbour_stats_after_refinement": refined_stats,
        "icp_levels": [
            {
                "voxel_m": v,
                "max_correspondence_m": c,
                "iterations": it,
            }
            for v, c, it in ICP_LEVELS
        ],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print()
    print(f"Saved transform: {transform_path}")
    print(f"Saved cloud    : {cloud_path}")
    print(f"Saved report   : {report_path}")

    if SHOW_VIEWER:
        o3d.visualization.draw_geometries(
            [target_eth, source_refined],
            window_name="Refined ETH3D + TUM alignment",
            width=1280,
            height=800,
        )


if __name__ == "__main__":
    main()
