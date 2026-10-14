from __future__ import annotations

"""
verify_eth3d_tum_alignment.py

Place in:
    D:\PlaneBasedSLAM\experiments\feature_finding\verify_eth3d_tum_alignment.py

Purpose
-------
Check whether the ETH3D Freiburg1-XYZ ground-truth reconstruction is already
expressed in (or very close to) the TUM ground-truth world coordinate frame.

Method
------
1. Load several prepared TUM frames.
2. Transform each frame's detector_points to world coordinates using T_world_cam.
3. Fuse / voxel-downsample those world points.
4. Compare them directly against the ETH3D sampled reconstruction.
5. Also run ICP from the TUM fused cloud to ETH3D to estimate any residual rigid
   transform.

This script does not modify the original datasets.
"""

from pathlib import Path
import json
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


# ============================================================
# USER SETTINGS
# ============================================================

SEQUENCE_NAME = "rgbd_dataset_freiburg1_xyz"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TUM_FRAMES_DIR = (
    PROJECT_ROOT
    / "data"
    / "tum"
    / "plane_detection"
    / SEQUENCE_NAME
    / "frames"
)

ETH3D_SAMPLED_FILE = (
    PROJECT_ROOT
    / "data"
    / "derived_gt"
    / "fr1_xyz"
    / "eth3d_mesh"
    / "fr1_xyz_eth3d_sampled.ply"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "derived_gt"
    / "fr1_xyz"
    / "alignment_check"
)

# Use evenly spaced prepared frames.
NUMBER_OF_FRAMES = 10

# Keep the alignment check manageable.
MAX_POINTS_PER_FRAME = 20_000

# Common voxel size for comparison.
VOXEL_SIZE_M = 0.02

# ICP settings.
ICP_MAX_CORRESPONDENCE_M = 0.10
ICP_MAX_ITERATIONS = 100

# Purely diagnostic thresholds. They are not evaluation GT thresholds.
DIRECT_ALIGNMENT_GOOD_MEDIAN_M = 0.05
DIRECT_ALIGNMENT_GOOD_P90_M = 0.10

SHOW_VIEWER = True

RNG = np.random.default_rng(20260919)


# ============================================================
# HELPERS
# ============================================================

def load_frame(path: Path):
    with np.load(path, allow_pickle=False) as data:
        if "detector_points" not in data:
            raise KeyError(
                f"{path.name}: detector_points not found. "
                f"Available keys: {list(data.keys())}"
            )

        if "T_world_cam" not in data:
            raise KeyError(
                f"{path.name}: T_world_cam not found. "
                f"Available keys: {list(data.keys())}"
            )

        points = np.asarray(data["detector_points"], dtype=np.float64)
        T_world_cam = np.asarray(data["T_world_cam"], dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{path.name}: detector_points must be (N,3).")

    if T_world_cam.shape != (4, 4):
        raise ValueError(f"{path.name}: T_world_cam must be 4x4.")

    return points, T_world_cam


def transform_points(points, T):
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


def make_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def nearest_distance_stats(source_points, target_points):
    tree = cKDTree(target_points)
    distances, _ = tree.query(source_points, k=1, workers=-1)

    distances = distances[np.isfinite(distances)]

    if len(distances) == 0:
        raise RuntimeError("No valid nearest-neighbour distances.")

    return {
        "count": int(len(distances)),
        "mean_m": float(np.mean(distances)),
        "median_m": float(np.median(distances)),
        "p90_m": float(np.percentile(distances, 90)),
        "p95_m": float(np.percentile(distances, 95)),
        "max_m": float(np.max(distances)),
        "within_0.02m": float(np.mean(distances <= 0.02)),
        "within_0.05m": float(np.mean(distances <= 0.05)),
        "within_0.10m": float(np.mean(distances <= 0.10)),
    }


def print_stats(title, stats):
    print(title)
    print(f"  points         : {stats['count']:,}")
    print(f"  mean NN error  : {stats['mean_m']*1000:.2f} mm")
    print(f"  median         : {stats['median_m']*1000:.2f} mm")
    print(f"  p90            : {stats['p90_m']*1000:.2f} mm")
    print(f"  p95            : {stats['p95_m']*1000:.2f} mm")
    print(f"  <= 20 mm       : {stats['within_0.02m']*100:.1f}%")
    print(f"  <= 50 mm       : {stats['within_0.05m']*100:.1f}%")
    print(f"  <=100 mm       : {stats['within_0.10m']*100:.1f}%")


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(TUM_FRAMES_DIR.glob("frame_*.npz"))

    if not frame_files:
        raise FileNotFoundError(
            f"No prepared TUM frames found:\n{TUM_FRAMES_DIR}"
        )

    if not ETH3D_SAMPLED_FILE.exists():
        raise FileNotFoundError(
            f"ETH3D sampled cloud not found:\n{ETH3D_SAMPLED_FILE}\n"
            "Run inspect_eth3d_gt_mesh.py first."
        )

    # Evenly-spaced frame selection.
    n = min(NUMBER_OF_FRAMES, len(frame_files))
    positions = np.linspace(0, len(frame_files) - 1, n)
    selected_indices = np.unique(np.round(positions).astype(int))
    selected_frames = [frame_files[i] for i in selected_indices]

    print()
    print("=" * 90)
    print("ETH3D <-> TUM WORLD-FRAME ALIGNMENT CHECK")
    print("=" * 90)
    print(f"TUM prepared frames : {len(frame_files)}")
    print(f"Frames used         : {[p.name for p in selected_frames]}")
    print(f"ETH3D cloud         : {ETH3D_SAMPLED_FILE}")
    print()

    # --------------------------------------------------------
    # Fuse TUM observations in GT world coordinates
    # --------------------------------------------------------

    world_chunks = []

    for path in selected_frames:
        points_cam, T_world_cam = load_frame(path)

        if len(points_cam) > MAX_POINTS_PER_FRAME:
            chosen = RNG.choice(
                len(points_cam),
                size=MAX_POINTS_PER_FRAME,
                replace=False,
            )
            points_cam = points_cam[chosen]

        points_world = transform_points(points_cam, T_world_cam)

        finite = np.all(np.isfinite(points_world), axis=1)
        points_world = points_world[finite]

        world_chunks.append(points_world)

        print(
            f"{path.name}: "
            f"{len(points_world):,} points transformed to TUM world frame"
        )

    fused_points = np.vstack(world_chunks)

    tum_world = make_pcd(fused_points)
    tum_world = tum_world.voxel_down_sample(VOXEL_SIZE_M)

    eth = o3d.io.read_point_cloud(str(ETH3D_SAMPLED_FILE))

    if len(eth.points) == 0:
        raise RuntimeError("ETH3D sampled cloud is empty.")

    eth = eth.voxel_down_sample(VOXEL_SIZE_M)

    tum_np = np.asarray(tum_world.points)
    eth_np = np.asarray(eth.points)

    print()
    print(f"Fused TUM cloud     : {len(tum_np):,} points after voxelisation")
    print(f"ETH3D cloud         : {len(eth_np):,} points after voxelisation")
    print()

    # --------------------------------------------------------
    # Direct alignment check
    # --------------------------------------------------------

    direct_stats = nearest_distance_stats(tum_np, eth_np)
    print_stats("DIRECT TUM-world -> ETH3D nearest-neighbour comparison", direct_stats)

    directly_aligned = (
        direct_stats["median_m"] <= DIRECT_ALIGNMENT_GOOD_MEDIAN_M
        and direct_stats["p90_m"] <= DIRECT_ALIGNMENT_GOOD_P90_M
    )

    print()
    if directly_aligned:
        print(
            "DIRECT ALIGNMENT STATUS: GOOD\n"
            "The ETH3D geometry is already close to the TUM GT world frame."
        )
    else:
        print(
            "DIRECT ALIGNMENT STATUS: NOT CONFIRMED\n"
            "A residual coordinate-frame transform may be present. "
            "ICP will now estimate it."
        )

    # --------------------------------------------------------
    # ICP diagnostic alignment
    # --------------------------------------------------------

    # Estimate normals on target for visualization / future extension.
    eth.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.08,
            max_nn=30,
        )
    )

    result = o3d.pipelines.registration.registration_icp(
        tum_world,
        eth,
        ICP_MAX_CORRESPONDENCE_M,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=ICP_MAX_ITERATIONS
        ),
    )

    T_eth_from_tum = np.asarray(result.transformation)

    tum_icp = o3d.geometry.PointCloud(tum_world)
    tum_icp.transform(T_eth_from_tum)

    icp_np = np.asarray(tum_icp.points)
    icp_stats = nearest_distance_stats(icp_np, eth_np)

    print()
    print("ICP RESULT")
    print(f"  fitness        : {result.fitness:.6f}")
    print(f"  inlier RMSE    : {result.inlier_rmse*1000:.2f} mm")
    print("  T_ETH3D_from_TUM =")
    print(T_eth_from_tum)
    print()

    print_stats("AFTER ICP", icp_stats)

    # How far is the correction from identity?
    R = T_eth_from_tum[:3, :3]
    t = T_eth_from_tum[:3, 3]

    trace_value = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    rotation_deg = float(np.degrees(np.arccos(trace_value)))
    translation_m = float(np.linalg.norm(t))

    print()
    print("Residual transform magnitude:")
    print(f"  translation : {translation_m*1000:.2f} mm")
    print(f"  rotation    : {rotation_deg:.4f} deg")

    # --------------------------------------------------------
    # Save diagnostics
    # --------------------------------------------------------

    tum_path = OUTPUT_DIR / "tum_gt_fused_world.ply"
    icp_path = OUTPUT_DIR / "tum_gt_fused_aligned_to_eth3d.ply"
    transform_path = OUTPUT_DIR / "T_eth3d_from_tum.npy"
    report_path = OUTPUT_DIR / "alignment_report.json"

    o3d.io.write_point_cloud(str(tum_path), tum_world)
    o3d.io.write_point_cloud(str(icp_path), tum_icp)
    np.save(transform_path, T_eth_from_tum)

    report = {
        "sequence": SEQUENCE_NAME,
        "frames_used": [p.name for p in selected_frames],
        "direct_alignment_good_by_diagnostic_rule": bool(directly_aligned),
        "direct_nn_stats": direct_stats,
        "icp_fitness": float(result.fitness),
        "icp_inlier_rmse_m": float(result.inlier_rmse),
        "icp_nn_stats": icp_stats,
        "T_eth3d_from_tum": T_eth_from_tum.tolist(),
        "residual_translation_m": translation_m,
        "residual_rotation_deg": rotation_deg,
        "voxel_size_m": VOXEL_SIZE_M,
        "icp_max_correspondence_m": ICP_MAX_CORRESPONDENCE_M,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print()
    print("Saved:")
    print(f"  {tum_path}")
    print(f"  {icp_path}")
    print(f"  {transform_path}")
    print(f"  {report_path}")

    if SHOW_VIEWER:
        # Blue/orange default-style distinction is intentionally not hard-coded.
        # Open3D's own default display is used.
        print()
        print("Opening comparison viewer...")
        print("Close the window when finished.")
        o3d.visualization.draw_geometries(
            [eth, tum_icp],
            window_name="ETH3D reconstruction + TUM GT-fused cloud",
            width=1280,
            height=800,
        )

    print()
    print("=" * 90)
    print("ALIGNMENT CHECK COMPLETE")
    print("=" * 90)
    print(
        "Send the DIRECT comparison, ICP fitness/RMSE, residual translation, "
        "residual rotation, and T_ETH3D_from_TUM output for interpretation."
    )

if __name__ == "__main__":
    main()
