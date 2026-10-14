"""Measure aligned depth consistency with the checkerboard plane in RGB images."""

import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from align_rgb_depth import load_data, make_aligned_depths, rgb_model


def estimate_board_plane(rgb, board, camera_parameters, intrinsics, method):
    """Estimate the checkerboard plane from corners detected in an RGB image.

    Args:
        rgb: BGR image captured by the RGB camera.
        board: Checkerboard geometry.
        camera_parameters: Acquisition-time SDK stream parameters.
        intrinsics: Project-fitted intrinsic models.
        method: Either calibrated or factory (each uses its own RGB model).

    Returns:
        A tuple containing the unit plane normal and a point on the plane in
        RGB camera coordinates (metres), an eroded uint8 checkerboard mask,
        the RGB camera matrix, distortion coefficients and the checkerboard
        pose reprojection RMS in pixels.

    Raises:
        ValueError: The checkerboard or its pose cannot be found, or the
            selected interior region is empty.
    """
    columns, rows = board["inner_corners"]
    square_size = board["square_size_m"]
    camera_matrix, distortion = rgb_model(camera_parameters, intrinsics, method)

    grey = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(grey, (columns, rows))
    if not found:
        found, corners = cv2.findChessboardCorners(grey, (columns, rows))
        if found:
            corners = cv2.cornerSubPix(
                grey,
                corners,
                (5, 5),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
    if not found:
        raise ValueError("Could not detect all checkerboard corners in the RGB image")

    object_points = np.zeros((columns * rows, 3), dtype=np.float64)
    object_points[:, :2] = (
        np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size
    )
    success, rvec, tvec = cv2.solvePnP(
        object_points, corners, camera_matrix, distortion
    )
    if not success:
        raise ValueError("Could not estimate the RGB checkerboard pose")

    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, distortion
    )
    projected_xy = projected.reshape(-1, 2)
    detected_xy = corners.reshape(-1, 2)
    pose_rms_px = float(
        np.sqrt(np.mean(np.sum((projected_xy - detected_xy) ** 2, axis=1)))
    )
    if not np.isfinite(pose_rms_px) or pose_rms_px > 5.0:
        raise ValueError(
            f"Checkerboard pose reprojection RMS is {pose_rms_px:.2f} px; "
            "check the detected corners and camera parameters"
        )
    rotation, _ = cv2.Rodrigues(rvec)
    normal = rotation[:, 2]
    point_on_plane = tvec.reshape(3)

    hull = cv2.convexHull(corners.reshape(-1, 2).astype(np.float32))
    polygon = np.rint(hull).astype(np.int32)
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon.reshape(-1, 2), 255)
    mask = cv2.erode(mask, np.ones((21, 21), dtype=np.uint8))
    if not np.any(mask):
        raise ValueError("The checkerboard interior mask is empty")

    return normal, point_on_plane, mask, camera_matrix, distortion, pose_rms_px


def measure_plane_distances(aligned_depth, mask, normal, point_on_plane,
                            camera_matrix, distortion):
    """Measure distances from aligned depth points to the RGB checkerboard plane.

    Args:
        aligned_depth: RGB-sized map of RGB-camera Z depth in metres.
        mask: uint8 mask selecting the checkerboard interior in the RGB image.
        normal: Unit normal of the checkerboard plane in RGB coordinates.
        point_on_plane: One RGB-camera point on that plane, in metres.
        camera_matrix: RGB camera matrix for this method.
        distortion: RGB distortion coefficients for this method.

    Returns:
        The valid pixel count, coverage as a percentage of mask pixels,
        median absolute point-to-plane distance in millimetres and the 90th
        percentile distance in millimetres. If no valid depth is present,
        the count and coverage are zero and both distances are NaN.
    """
    board_pixels = int(np.count_nonzero(mask))
    valid = (mask > 0) & np.isfinite(aligned_depth) & (aligned_depth > 0)
    v, u = np.nonzero(valid)
    count = len(u)
    if count == 0:
        return 0, 0.0, float("nan"), float("nan")

    image_points = np.column_stack((u, v)).astype(np.float64).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(image_points, camera_matrix, distortion).reshape(-1, 2)
    z = aligned_depth[v, u].astype(np.float64)
    points_rgb = np.column_stack((rays[:, 0] * z, rays[:, 1] * z, z))
    distances_mm = np.abs((points_rgb - point_on_plane) @ normal) * 1000.0

    return (
        count,
        100.0 * count / board_pixels,
        float(np.median(distances_mm)),
        float(np.percentile(distances_mm, 90)),
    )


def save_mask_preview(rgb, mask, path):
    """Save an RGB preview showing the interior region used for measurement.

    Args:
        rgb: Original BGR image.
        mask: uint8 checkerboard interior mask.
        path: Destination path for the preview PNG image.

    Raises:
        OSError: OpenCV cannot write the preview image.
    """
    preview = rgb.copy()
    inside = mask > 0
    preview[inside] = (
        0.65 * preview[inside] + 0.35 * np.array([0, 255, 255])
    ).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, contours, -1, (0, 255, 255), 2)
    if not cv2.imwrite(str(path), preview):
        raise OSError(f"Could not save {path}")


def save_tables(records, output_dir):
    """Write per-frame measurements and summaries for five board positions.

    Args:
        records: Per-frame dictionaries for the calibrated and factory maps.
            Each group contains three consecutive captured frames.
        output_dir: Directory in which both CSV files are written.

    Returns:
        Paths to ``frame_metrics.csv`` and ``group_summary.csv``.
    """
    frame_path = output_dir / "frame_metrics.csv"
    with frame_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    group_path = output_dir / "group_summary.csv"
    with group_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "group", "method", "frame_count", "median_of_frame_medians_mm",
            "median_of_frame_p90_mm", "mean_coverage_pct",
        ])
        for group in range(1, 6):
            for method in ("calibrated", "factory"):
                selected = [
                    row for row in records
                    if row["group"] == group and row["method"] == method
                ]
                writer.writerow([
                    group,
                    method,
                    len(selected),
                    float(np.median([r["median_mm"] for r in selected])),
                    float(np.median([r["p90_mm"] for r in selected])),
                    float(np.mean([r["coverage_pct"] for r in selected])),
                ])
    return frame_path, group_path


def main():
    """Measure all 15 frames and save plane-distance results and mask previews.

    The input session and the exact calibration result folders are selected
    below. Outputs are placed in a new timestamped results directory.

    Raises:
        FileNotFoundError: A required session or calibration file is missing.
        ValueError: A frame has no detectable checkerboard or valid depth
            within its measurement mask.
    """
    project_dir = Path(__file__).resolve().parents[2]
    session_dir = (
        project_dir / "data" / "camera_calibration"
        / "alignment_session_20261005_171922_220185"
    )
    results_dir = project_dir / "experiments" / "calibration" / "results"
    intrinsics_path = (
        results_dir / "intrinsics_20261005_150235_328385" / "intrinsics.json"
    )
    extrinsics_path = (
        results_dir / "extrinsics_20261005_153349_643800" / "extrinsics.json"
    )
    for path in (session_dir, intrinsics_path, extrinsics_path):
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = results_dir / (
        "depth_plane_complete_model_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []

    for index in range(15):
        frame_id = f"{index:04d}"
        rgb, depth, camera_parameters, intrinsics, extrinsics = load_data(
            session_dir, intrinsics_path, extrinsics_path, frame_id
        )
        aligned_maps = make_aligned_depths(
            depth, camera_parameters, intrinsics, extrinsics, rgb.shape[:2]
        )
        for method, aligned_depth in aligned_maps.items():
            normal, point_on_plane, mask, camera_matrix, distortion, pose_rms = (
                estimate_board_plane(rgb, intrinsics["board"], camera_parameters,
                                     intrinsics, method)
            )
            save_mask_preview(rgb, mask, output_dir / f"{frame_id}_{method}_mask.png")
            count, coverage, median, p90 = measure_plane_distances(
                aligned_depth, mask, normal, point_on_plane,
                camera_matrix, distortion,
            )
            if count == 0:
                raise ValueError(f"No valid {method} depth inside frame {frame_id} mask")
            records.append({
                "frame_id": frame_id,
                "group": index // 3 + 1,
                "method": method,
                "board_pose_rms_px": round(pose_rms, 5),
                "valid_pixels": count,
                "coverage_pct": round(coverage, 3),
                "median_mm": round(median, 3),
                "p90_mm": round(p90, 3),
            })
            print(
                f"{frame_id} {method:10s} "
                f"median={median:.2f} mm  p90={p90:.2f} mm  "
                f"coverage={coverage:.1f}%"
            )

    frame_path, group_path = save_tables(records, output_dir)
    print(f"Frame results: {frame_path}")
    print(f"Group results: {group_path}")
    print(f"Mask previews: {output_dir}")


if __name__ == "__main__":
    main()
