"""Align native depth to RGB using consistent method-specific RGB models."""

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def load_data(session_dir, intrinsics_path, extrinsics_path, frame_id):
    """Read one RGB-depth pair and the parameters needed for alignment."""
    session_dir = Path(session_dir)
    rgb_path = session_dir / "rgb" / f"{frame_id}.png"
    depth_path = session_dir / "depth" / f"{frame_id}.png"
    camera_parameters_path = session_dir / "camera_parameters.json"

    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if rgb is None:
        raise FileNotFoundError(rgb_path)
    if depth is None:
        raise FileNotFoundError(depth_path)
    if depth.dtype != np.uint16:
        raise ValueError(f"Depth image must be uint16: {depth_path}")

    with camera_parameters_path.open("r", encoding="utf-8") as f:
        camera_parameters = json.load(f)
    with Path(intrinsics_path).open("r", encoding="utf-8") as f:
        intrinsics = json.load(f)
    with Path(extrinsics_path).open("r", encoding="utf-8") as f:
        extrinsics = json.load(f)

    return rgb, depth, camera_parameters, intrinsics, extrinsics


def verify_depth_left_geometry(camera_parameters):
    """The fitted left-IR transform can only act directly on depth-frame points
    when their reference frames are identical. Retain native depth intrinsics.
    """
    streams = camera_parameters["streams"]
    a, b = streams["depth"], streams["ir_left"]
    for field in ("width", "height", "fx", "fy", "cx", "cy"):
        if not np.isclose(a[field], b[field], rtol=0, atol=1e-7):
            raise ValueError(f"Depth and left IR stream mismatch: {field}")
    for camera in (a, b):
        coefficients = np.asarray(camera["coeffs"], dtype=float)
        if np.any(np.abs(coefficients) > 1e-12):
            raise ValueError("Depth/IR nonzero distortion needs inverse deprojection")
    depth_to_left = camera_parameters["factory_extrinsics"]["depth_to_left"]
    if not np.allclose(depth_to_left["rotation_matrix"], np.eye(3), atol=1e-8):
        raise ValueError("Depth to left IR rotation is not identity")
    if not np.allclose(depth_to_left["translation_m"], 0.0, atol=1e-8):
        raise ValueError("Depth to left IR translation is not zero")


def rgb_model(camera_parameters, fitted_intrinsics, method):
    """Return (K, D) for a consistent target RGB projection model."""
    if method == "calibrated":
        values = fitted_intrinsics["cameras"]["rgb"]
        K = np.asarray(values["camera_matrix"], dtype=np.float64)
        D = np.asarray(values["distortion_k1_k2_p1_p2_k3"], dtype=np.float64)
    elif method == "factory":
        values = camera_parameters["streams"]["rgb"]
        D = np.asarray(values["coeffs"], dtype=np.float64)
        model = str(values["distortion_model"]).lower()
        if "inverse_brown_conrady" in model and np.any(np.abs(D) > 1e-12):
            raise ValueError("Nonzero factory inverse Brown-Conrady needs special projection")
        if not any(x in model for x in ("brown_conrady", "none")):
            raise ValueError(f"Unsupported factory RGB model: {model}")
        K = np.array([[values["fx"], 0, values["cx"]],
                      [0, values["fy"], values["cy"]], [0, 0, 1]], dtype=np.float64)
    else:
        raise ValueError(f"Unknown method: {method}")
    if K.shape != (3, 3) or D.size != 5:
        raise ValueError(f"Invalid RGB camera model: {method}")
    return K, D


def make_aligned_depths(depth, camera_parameters, intrinsics, extrinsics, image_shape):
    """Use native depth intrinsics, with method-specific RGB intrinsics/extrinsics."""
    verify_depth_left_geometry(camera_parameters)
    points_depth = deproject_depth(depth, camera_parameters)
    transformations = {
        "calibrated": extrinsics["pairs"]["left_to_rgb"],
        "factory": camera_parameters["factory_extrinsics"]["depth_to_rgb"],
    }
    aligned = {}
    for method, transform in transformations.items():
        points_rgb = transform_points(points_depth, transform)
        aligned[method] = project_to_rgb(
            points_rgb, rgb_model(camera_parameters, intrinsics, method), image_shape
        )
    return aligned


def deproject_depth(depth, camera_parameters):
    """Convert valid depth pixels to 3D points in depth-camera coordinates, in metres."""
    depth_intrinsics = camera_parameters["streams"]["depth"]
    depth_scale = camera_parameters["depth_scale_m_per_unit"]

    v, u = np.nonzero(depth)
    z = depth[v, u].astype(np.float64) * depth_scale
    x = (u - depth_intrinsics["cx"]) * z / depth_intrinsics["fx"]
    y = (v - depth_intrinsics["cy"]) * z / depth_intrinsics["fy"]

    return np.column_stack((x, y, z))


def transform_points(points, transform):
    """Apply a source-to-target extrinsic transform to row-wise 3D points."""
    rotation = np.array(transform["rotation_matrix"], dtype=np.float64)
    translation = np.array(transform["translation_m"], dtype=np.float64)
    return points @ rotation.T + translation


def project_to_rgb(points_rgb, rgb_parameters, image_shape):
    """Project RGB-camera points to an RGB-sized depth map measured in metres."""
    height, width = image_shape
    aligned_depth = np.full((height, width), np.inf, dtype=np.float32)

    camera_matrix, distortion = rgb_parameters

    valid_points = points_rgb[
        np.isfinite(points_rgb).all(axis=1) & (points_rgb[:, 2] > 0)
    ]
    if len(valid_points) == 0:
        return np.zeros((height, width), dtype=np.float32)

    projected, _ = cv2.projectPoints(
        valid_points,
        np.zeros(3),
        np.zeros(3),
        camera_matrix,
        distortion,
    )

    pixels = projected.reshape(-1, 2)
    valid_pixels = np.isfinite(pixels).all(axis=1)
    pixels = pixels[valid_pixels]
    z = valid_points[valid_pixels, 2]

    u = np.rint(pixels[:, 0]).astype(np.int32)
    v = np.rint(pixels[:, 1]).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    np.minimum.at(aligned_depth, (v[inside], u[inside]), z[inside])
    aligned_depth[np.isinf(aligned_depth)] = 0
    return aligned_depth


def make_comparison(rgb, calibrated_depth, factory_depth, near_m, far_m):
    """Show calibrated and factory foreground depth overlays side by side."""
    panels = []

    for title, aligned_depth, color in (
        ("Calibrated", calibrated_depth, (0, 255, 0)),
        ("Factory", factory_depth, (0, 0, 255)),
    ):
        mask = (
            (aligned_depth >= near_m) & (aligned_depth <= far_m)
        ).astype(np.uint8)

        panel = rgb.copy()
        selected = mask > 0
        panel[selected] = (
            0.6 * panel[selected] + 0.4 * np.array(color)
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = [contour for contour in contours if cv2.contourArea(contour) > 200]
        cv2.drawContours(panel, contours, -1, color, 2)
        cv2.putText(
            panel, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )
        panels.append(panel)

    return np.hstack(panels)


def main():
    """Align selected frames and save comparisons and both depth maps."""
    project_dir = Path(__file__).resolve().parents[2]
    session_dir = (
        project_dir
        / "data"
        / "camera_calibration"
        / "alignment_session_20261005_171922_220185"
    )
    results_dir = project_dir / "experiments" / "calibration" / "results"

    intrinsics_path = results_dir / "intrinsics_20261005_150235_328385" / "intrinsics.json"
    extrinsics_path = results_dir / "extrinsics_20261005_153349_643800" / "extrinsics.json"
    for path in (session_dir, intrinsics_path, extrinsics_path):
        if not path.exists():
            raise FileNotFoundError(path)
    print(f"Intrinsic parameters: {intrinsics_path}")
    print(f"Extrinsic parameters: {extrinsics_path}")

    output_dir = results_dir / (
        "alignment_complete_model_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    frame_ranges = {
        "0000": (0.45, 0.75),
        "0001": (0.45, 0.75),
        "0002": (0.45, 0.75),
        "0003": (0.90, 1.10),
        "0004": (0.90, 1.10),
        "0005": (0.90, 1.10),
        "0006": (0.68, 0.85),
        "0007": (0.68, 0.85),
        "0008": (0.68, 0.85),
        "0009": (0.40, 0.60),
        "0010": (0.40, 0.60),
        "0011": (0.40, 0.60),
        "0012": (0.62, 0.78),
        "0013": (0.62, 0.78),
        "0014": (0.62, 0.78),
    }

    comparisons = []
    for frame_id, (near_m, far_m) in frame_ranges.items():
        rgb, depth, camera_parameters, intrinsics, extrinsics = load_data(
            session_dir, intrinsics_path, extrinsics_path, frame_id
        )

        aligned = make_aligned_depths(
            depth, camera_parameters, intrinsics, extrinsics, rgb.shape[:2]
        )
        calibrated_depth = aligned["calibrated"]
        factory_depth = aligned["factory"]

        comparison = make_comparison(
            rgb, calibrated_depth, factory_depth, near_m, far_m
        )
        cv2.putText(
            comparison, frame_id, (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        comparisons.append(comparison)
        comparison_path = output_dir / f"{frame_id}_comparison.png"
        if not cv2.imwrite(str(comparison_path), comparison):
            raise OSError(f"Could not save {comparison_path}")

        depth_scale = camera_parameters["depth_scale_m_per_unit"]
        for name, aligned_depth in (
            ("calibrated", calibrated_depth),
            ("factory", factory_depth),
        ):
            depth_uint16 = np.clip(
                np.rint(aligned_depth / depth_scale), 0, 65535
            ).astype(np.uint16)
            depth_path = output_dir / f"{frame_id}_{name}_depth.png"
            if not cv2.imwrite(str(depth_path), depth_uint16):
                raise OSError(f"Could not save {depth_path}")

        print(f"Saved frame {frame_id}: {comparison_path}")

    overview_rows = [
        np.hstack(comparisons[i:i + 3])
        for i in range(0, len(comparisons), 3)
    ]
    overview_path = output_dir / "all_frames_comparison.png"
    if not cv2.imwrite(str(overview_path), np.vstack(overview_rows)):
        raise OSError(f"Could not save {overview_path}")

    print(f"Overview: {overview_path}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
