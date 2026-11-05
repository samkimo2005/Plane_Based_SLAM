"""Visualise saved Run I Iterative RANSAC plane segmentation with Open3D.

Designed for the 5 mm real-scene downsampled point clouds already generated
in D:\\PlaneBasedSLAM. This script DOES NOT rerun plane detection.

Run from your activated virtual environment:
    python show_plane_detection_3d.py
    python show_plane_detection_3d.py --frame 000002

Controls:
    Drag left mouse: rotate    | Mouse wheel: zoom
    Drag right mouse: pan      | S: save PNG screenshot
    R: reset to camera-facing viewpoint
    Q / Esc: close window

The plane ID -> colour map is saved alongside every screenshot.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

# Change this ONE line if your project is not on D:.
PROJECT_ROOT = Path(r"D:\PlaneBasedSLAM")

# Fixed distinct RGB colours for plane IDs 0-19; repeat only if >20 planes.
HEX_COLORS = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00",
    "#56B4E9", "#F0E442", "#882255", "#44AA99", "#AA4499",
    "#117733", "#332288", "#DDCC77", "#88CCEE", "#CC6677",
    "#999933", "#661100", "#6699CC", "#AA4466", "#4477AA",
]
UNASSIGNED_COLOR = "#D3D7DB"


def hex_to_rgb(code: str) -> np.ndarray:
    code = code.lstrip("#")
    return np.array([int(code[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def make_paths(frame: str) -> tuple[Path, Path]:
    ply = (PROJECT_ROOT / "experiments" / "pointcloud" / "results"
           / "real_scene_downsampling_20261023_151330_374491"
           / "voxel_0005p00mm" / f"{frame}_depth.ply")
    labels = (PROJECT_ROOT / "experiments" / "plane_detection_real" / "results"
              / "runH_maxplanes20_test" / "voxel_0005p00mm"
              / "iterative_ransac" / f"{frame}_depth_iterative_ransac_labels.npz")
    return ply, labels


def reset_camera(vis) -> bool:
    """Optical frame: X right, Y down, Z forward; camera looks along +Z."""
    ctr = vis.get_view_control()
    ctr.set_lookat(VIEW_CENTRE)
    ctr.set_front([0.0, 0.0, -1.0])
    ctr.set_up([0.0, -1.0, 0.0])
    ctr.set_zoom(0.76)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", default="000000", choices=[f"{i:06d}" for i in range(5)])
    parser.add_argument("--ply", type=Path, default=None,
                        help="Override 5 mm downsampled PLY location")
    parser.add_argument("--labels", type=Path, default=None,
                        help="Override the matching Iterative RANSAC .npz location")
    parser.add_argument("--point-size", type=float, default=2.2)
    args = parser.parse_args()

    default_ply, default_labels = make_paths(args.frame)
    ply_path = args.ply if args.ply is not None else default_ply
    labels_path = args.labels if args.labels is not None else default_labels

    if not ply_path.is_file() or not labels_path.is_file():
        missing = [str(p) for p in (ply_path, labels_path) if not p.is_file()]
        raise FileNotFoundError(
            "Cannot find saved experiment output(s):\n  " + "\n  ".join(missing)
            + "\n\nEdit PROJECT_ROOT at the top of this script, or specify "
            "--ply and --labels. Use the *downsampled* 5 mm PLY, "
            "not the original depth PLY."
        )

    # Import late to make missing dependencies easy to diagnose.
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit("Open3D missing. Run: python -m pip install open3d") from exc

    cloud = o3d.io.read_point_cloud(str(ply_path))
    xyz = np.asarray(cloud.points)
    with np.load(labels_path, allow_pickle=False) as data:
        if "labels" not in data.files:
            raise ValueError(f"NPZ has no 'labels' array. Found: {data.files}")
        labels = np.asarray(data["labels"], dtype=np.int32).reshape(-1)

    if len(xyz) == 0:
        raise ValueError(f"Point cloud is empty: {ply_path}")
    if len(xyz) != len(labels):
        raise ValueError(
            f"POINT COUNT MISMATCH: PLY has {len(xyz):,} points; "
            f"labels has {len(labels):,}.\n"
            "Use the downsampled PLY for the *same frame and voxel size*. "
            "Do NOT use the original un-downsampled depth PLY."
        )
    if not np.isfinite(xyz).all():
        raise ValueError("Point cloud contains non-finite coordinates.")

    rgb = np.tile(hex_to_rgb(UNASSIGNED_COLOR), (len(xyz), 1))
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)

    output_dir = Path(__file__).resolve().parent / "plane_detection_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    legend_path = output_dir / f"{args.frame}_plane_colour_key.csv"
    with legend_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["plane_id", "hex_colour", "num_points"])
        for plane_id, count in zip(ids.tolist(), counts.tolist()):
            colour = HEX_COLORS[plane_id % len(HEX_COLORS)]
            rgb[labels == plane_id] = hex_to_rgb(colour)
            writer.writerow([plane_id, colour, count])
            print(f"Plane {plane_id:2d}: {colour} | {count:6,d} points")
        writer.writerow(["unassigned", UNASSIGNED_COLOR, int((labels < 0).sum())])

    print(f"\nFrame {args.frame}: {len(xyz):,} total points, {len(ids)} detected planes")
    print(f"PLY:       {ply_path}")
    print(f"Labels:    {labels_path}")
    print(f"Colour key:{legend_path}")
    print("\nControls: drag = rotate, wheel = zoom, right drag = pan, ")
    print("          R = camera-facing view, S = save PNG, Q = close")

    cloud.colors = o3d.utility.Vector3dVector(rgb)

    global VIEW_CENTRE
    VIEW_CENTRE = np.median(xyz, axis=0).tolist()
    viewer = o3d.visualization.VisualizerWithKeyCallback()
    if not viewer.create_window(
        window_name=f"Run I: Iterative RANSAC | Frame {args.frame} | S=Save, R=Reset",
        width=1280, height=900, visible=True
    ):
        raise RuntimeError("Could not open Open3D viewer window.")
    viewer.add_geometry(cloud)

    opts = viewer.get_render_option()
    opts.background_color = np.array([1.0, 1.0, 1.0])
    opts.point_size = max(1.0, args.point_size)
    opts.show_coordinate_frame = False

    reset_camera(viewer)

    def save_screenshot(vis):
        path = output_dir / f"{args.frame}_iterative_ransac_screenshot.png"
        vis.capture_screen_image(str(path), do_render=True)
        print(f"\nSaved screenshot: {path}")
        print(f"Saved colour key: {legend_path}")
        return False

    viewer.register_key_callback(ord("S"), save_screenshot)
    viewer.register_key_callback(ord("R"), reset_camera)
    try:
        viewer.run()
    finally:
        viewer.destroy_window()


if __name__ == "__main__":
    main()
