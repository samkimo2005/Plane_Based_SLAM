"""Colour native RealSense depth points with calibrated and factory RGB models.

Both output clouds have identical XYZ coordinates in the ORIGINAL DEPTH optical
frame. Invalid RGB projections receive neutral grey, and an explicit valid-colour
mask is exported. The optional coloured-only PLY contains only visible pixels.
No calibration files, source images or prior experiment results are modified.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np
import open3d as o3d

SESSION_NAME = "alignment_session_20261005_171922_220185"
INTRINSICS_RUN = "intrinsics_20261005_150235_328385"
EXTRINSICS_RUN = "extrinsics_20261005_153349_643800"


def project_root():
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for p in (start, *start.parents):
            if (p / "data" / "camera_calibration").is_dir():
                return p
    raise FileNotFoundError("Cannot find project root with data/camera_calibration")


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig") as f:
        return json.load(f)


def read_png(path, flags):
    if not path.is_file():
        raise FileNotFoundError(path)
    image = cv.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def check_source_frame(camera_parameters, depth, rgb):
    streams = camera_parameters["streams"]
    for name, array in (("depth", depth), ("rgb", rgb)):
        p = streams[name]
        if array.shape[:2] != (int(p["height"]), int(p["width"])):
            raise ValueError(f"{name} image size does not match session metadata")
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("Expected native, unaligned, single-channel uint16 depth PNG")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected BGR RGB image")

    depth_to_left = camera_parameters["factory_extrinsics"]["depth_to_left"]
    rot = np.asarray(depth_to_left["rotation_matrix"], dtype=np.float64)
    translation = np.asarray(depth_to_left["translation_m"], dtype=np.float64)
    if not np.allclose(rot, np.eye(3), atol=1e-8) or not np.allclose(translation, 0, atol=1e-8):
        raise ValueError("Depth and left IR frames differ: transformation composition required")

    depth_intr = streams["depth"]
    if not ("brown_conrady" in str(depth_intr["distortion_model"]).lower()
            or "none" in str(depth_intr["distortion_model"]).lower()):
        raise ValueError("Unsupported depth distortion model")
    if np.any(np.abs(np.asarray(depth_intr["coeffs"], dtype=float)) > 1e-12):
        raise ValueError("Non-zero native-depth distortion not supported by pinhole back-projection")
    return depth_intr


def native_points(depth, intr, scale, near, far):
    fx, fy, cx, cy = (float(intr[k]) for k in ("fx", "fy", "cx", "cy"))
    if fx <= 0 or fy <= 0 or scale <= 0:
        raise ValueError("Invalid depth intrinsics or scale")
    z_image = depth.astype(np.float64) * scale
    ok = (depth > 0) & np.isfinite(z_image) & (z_image >= near) & (z_image <= far)
    v, u = np.nonzero(ok)
    z = z_image[v, u]
    xyz = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    if len(xyz) == 0:
        raise ValueError("No valid depth points in selected range")
    return xyz, u, v


def factory_rgb_model(camera_parameters):
    p = camera_parameters["streams"]["rgb"]
    coeff = np.asarray(p["coeffs"], dtype=np.float64)
    model = str(p["distortion_model"]).lower()
    if "inverse_brown_conrady" in model and np.any(np.abs(coeff) > 1e-12):
        raise ValueError("Non-zero inverse Brown-Conrady requires a special RGB projector")
    if not ("brown_conrady" in model or "none" in model):
        raise ValueError(f"Unsupported RGB factory distortion: {model}")
    K = np.array([[p["fx"], 0, p["cx"]], [0, p["fy"], p["cy"]], [0, 0, 1]], dtype=float)
    pair = camera_parameters["factory_extrinsics"]["depth_to_rgb"]
    return K, coeff, pair


def calibrated_rgb_model(intrinsics, extrinsics):
    p = intrinsics["cameras"]["rgb"]
    K = np.asarray(p["camera_matrix"], dtype=float)
    coeff = np.asarray(p["distortion_k1_k2_p1_p2_k3"], dtype=float)
    pair = extrinsics["pairs"]["left_to_rgb"]
    if pair.get("source", "ir_left") != "ir_left" or pair.get("target", "rgb") != "rgb":
        raise ValueError("Unexpected fitted extrinsics camera pair")
    return K, coeff, pair


def rgb_colours(native_xyz, rgb_bgr, model):
    K, D, pair = model
    if K.shape != (3, 3):
        raise ValueError("RGB camera matrix must be 3x3")
    if pair.get("convention", "p_target = R @ p_source + t") != "p_target = R @ p_source + t":
        raise ValueError("Unexpected extrinsic transform convention")
    R = np.asarray(pair["rotation_matrix"], dtype=float)
    t = np.asarray(pair["translation_m"], dtype=float)
    if R.shape != (3, 3) or t.shape != (3,):
        raise ValueError("Invalid extrinsic shape")
    rgb_xyz = native_xyz @ R.T + t
    valid_3d = np.isfinite(rgb_xyz).all(axis=1) & (rgb_xyz[:, 2] > 0)
    samples = np.full((len(native_xyz), 3), 0.5, dtype=np.float64)  # grey marks invalid colour
    valid = np.zeros(len(native_xyz), dtype=bool)
    if not np.any(valid_3d):
        return samples, valid
    indexes = np.flatnonzero(valid_3d)
    uv, _ = cv.projectPoints(rgb_xyz[indexes], np.zeros(3), np.zeros(3), K, D)
    uv = uv.reshape(-1, 2)
    finite = np.isfinite(uv).all(axis=1)
    selected = indexes[finite]
    pixels = uv[finite]
    # Nearest-neighbour colour samples; require continuous pixel positions inside image.
    h, w = rgb_bgr.shape[:2]
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] <= w - 1)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] <= h - 1))
    selected = selected[inside]
    pixels = pixels[inside]
    u = np.rint(pixels[:, 0]).astype(np.int32)
    v = np.rint(pixels[:, 1]).astype(np.int32)
    samples[selected] = rgb_bgr[v, u, ::-1].astype(np.float64) / 255.0
    valid[selected] = True
    return samples, valid


def write_cloud(path, xyz, colours):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colours)
    if not o3d.io.write_point_cloud(str(path), pcd, write_ascii=False):
        raise OSError(f"Failed saving point cloud: {path}")
    return pcd


def make_colour_preview(rgb, pixels_u, pixels_v, masks, output_path):
    # A small visual diagnostic in the original DEPTH image coordinates.
    h, w = rgb.shape[:2]
    panels = []
    for title, mask in masks.items():
        view = np.zeros((h, w, 3), dtype=np.uint8)
        view[pixels_v[mask], pixels_u[mask]] = (0, 200, 0)
        view[pixels_v[~mask], pixels_u[~mask]] = (0, 0, 180)
        cv.putText(view, title, (12, 28), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        panels.append(view)
    if not cv.imwrite(str(output_path), np.hstack(panels)):
        raise OSError(f"Cannot write preview: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=None)
    parser.add_argument("--intrinsics", type=Path, default=None)
    parser.add_argument("--extrinsics", type=Path, default=None)
    parser.add_argument("--frame", default="0000")
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=5.0)
    parser.add_argument("--visualize", action="store_true", help="Open Open3D windows sequentially")
    args = parser.parse_args()
    if not (0 < args.near < args.far):
        parser.error("Expected 0 < --near < --far")

    start = time.perf_counter()
    root = project_root()
    session = args.session or root / "data" / "camera_calibration" / SESSION_NAME
    intr_path = args.intrinsics or root / "experiments" / "calibration" / "results" / INTRINSICS_RUN / "intrinsics.json"
    extr_path = args.extrinsics or root / "experiments" / "calibration" / "results" / EXTRINSICS_RUN / "extrinsics.json"
    session, intr_path, extr_path = (p.resolve() for p in (session, intr_path, extr_path))
    params_path = session / "camera_parameters.json"
    rgb_path = session / "rgb" / f"{args.frame}.png"
    depth_path = session / "depth" / f"{args.frame}.png"
    params, fitted_intr, fitted_extr = (read_json(p) for p in (params_path, intr_path, extr_path))
    rgb = read_png(rgb_path, cv.IMREAD_COLOR)
    depth = read_png(depth_path, cv.IMREAD_UNCHANGED)
    depth_intr = check_source_frame(params, depth, rgb)
    if tuple(fitted_intr["cameras"]["rgb"]["image_size_wh"]) != (rgb.shape[1], rgb.shape[0]):
        raise ValueError("Fitted RGB intrinsics have a different image size")
    if params.get("serial") and params.get("serial") != "261322074081":
        print("Warning: session serial differs from the original calibration device; confirm parameters")

    xyz, pu, pv = native_points(depth, depth_intr, float(params["depth_scale_m_per_unit"]), args.near, args.far)
    models = {
        "calibrated": calibrated_rgb_model(fitted_intr, fitted_extr),
        "factory": factory_rgb_model(params),
    }
    output_dir = root / "experiments" / "pointcloud" / "results" / (
        "coloured_pointcloud_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    results = {}
    masks = {}
    for name, model in models.items():
        colours, mask = rgb_colours(xyz, rgb, model)
        masks[name] = mask
        full_path = output_dir / f"{args.frame}_{name}.ply"
        valid_path = output_dir / f"{args.frame}_{name}_coloured_only.ply"
        cloud = write_cloud(full_path, xyz, colours)
        if np.any(mask):
            write_cloud(valid_path, xyz[mask], colours[mask])
        results[name] = {
            "file": str(full_path),
            "coloured_only_file": str(valid_path) if np.any(mask) else None,
            "total_xyz_points": len(xyz),
            "valid_rgb_colours": int(mask.sum()),
            "valid_rgb_fraction_percent": round(100 * float(mask.mean()), 4),
            "invalid_rgb_colours": int((~mask).sum()),
            "invalid_colour_value": "neutral grey (0.5, 0.5, 0.5)",
            "coordinate_frame": "native_depth_optical_frame",
        }
        print(f"{name}: {len(xyz):,} identical XYZ; {mask.sum():,} RGB colours ")
        if args.visualize:
            print(f"Opening {name} cloud; close window to continue...")
            o3d.visualization.draw_geometries([cloud], window_name=f"{name.title()} RGB colours")

    preview_path = output_dir / f"{args.frame}_colour_validity.png"
    make_colour_preview(rgb, pu, pv, masks, preview_path)
    report = {
        "frame_id": args.frame,
        "session": str(session),
        "depth_source": str(depth_path),
        "rgb_source": str(rgb_path),
        "fitted_intrinsics": str(intr_path),
        "fitted_extrinsics": str(extr_path),
        "original_depth_intrinsics": {k: depth_intr[k] for k in ("fx", "fy", "cx", "cy")},
        "depth_scale_m_per_unit": float(params["depth_scale_m_per_unit"]),
        "depth_filter_m": [args.near, args.far],
        "input_pixels": int(depth.size),
        "native_xyz_points": int(len(xyz)),
        "native_depth_coverage_percent": round(100 * len(xyz) / depth.size, 4),
        "same_xyz_for_both_models": True,
        "colour_sampling": "nearest-neighbour in RGB; invalid RGB projections coloured grey",
        "note": "Depth XYZ uses SDK-native depth intrinsics; calibrated/factory models affect only RGB colour mapping.",
        "processing_seconds": round(time.perf_counter() - start, 4),
        "outputs": results,
        "colour_validity_preview": str(preview_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved results: {output_dir}")
    print(f"Elapsed: {report['processing_seconds']:.3f} s")


if __name__ == "__main__":
    main()
