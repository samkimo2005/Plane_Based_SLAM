"""Inspect left-IR-to-RGB reprojection residuals on held-out checkerboard images."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path, PureWindowsPath

import cv2 as cv
import numpy as np


def load_inputs(project, session_arg, intrinsics_arg, extrinsics_arg):
    """Locate the paired calibration files and check their session and board metadata."""
    results = project / "experiments" / "calibration" / "results"
    if extrinsics_arg is None:
        candidates = list(results.glob("extrinsics_*/extrinsics.json"))
        if not candidates:
            raise FileNotFoundError(f"No extrinsics.json under {results}")
        extrinsics_path = max(candidates, key=lambda p: p.stat().st_mtime)
    else:
        extrinsics_path = extrinsics_arg
    extrinsics = json.loads(extrinsics_path.read_text(encoding="utf-8-sig"))

    recorded_intrinsics = PureWindowsPath(extrinsics["intrinsics_file"])
    intrinsics_path = intrinsics_arg or (
        results / recorded_intrinsics.parent.name / recorded_intrinsics.name
    )
    if (intrinsics_path.parent.name, intrinsics_path.name) != (
        recorded_intrinsics.parent.name, recorded_intrinsics.name
    ):
        raise ValueError("The extrinsics refer to a different intrinsics result folder.")
    intrinsics = json.loads(intrinsics_path.read_text(encoding="utf-8-sig"))

    recorded_session = PureWindowsPath(extrinsics["session"]).name
    session = session_arg or (
        project / "data" / "camera_calibration" / recorded_session
    )
    if session.name != recorded_session:
        raise ValueError("The extrinsics refer to a different capture session.")
    factory = json.loads(
        (session / "factory_parameters.json").read_text(encoding="utf-8-sig")
    )
    if intrinsics["board"] != factory["board"] or extrinsics["board"] != factory["board"]:
        raise ValueError("Checkerboard settings differ between the files.")
    if intrinsics.get("session") and PureWindowsPath(intrinsics["session"]).name != session.name:
        raise ValueError("The intrinsics refer to a different capture session.")
    if not extrinsics.get("intrinsics_fixed"):
        raise ValueError("Extrinsics were not computed with fixed intrinsics.")
    pair = extrinsics["pairs"]["left_to_rgb"]
    if pair["source"] != "ir_left" or pair["target"] != "rgb":
        raise ValueError("The selected extrinsics are not left IR to RGB.")
    return session, intrinsics_path, extrinsics_path, factory, intrinsics, pair


def read_image_and_corners(path, pattern, expected_size):
    """Read one image and find its checkerboard corners in the original order."""
    image_bytes = np.fromfile(str(path), np.uint8)
    gray = cv.imdecode(image_bytes, cv.IMREAD_GRAYSCALE)
    image = cv.imdecode(image_bytes, cv.IMREAD_COLOR)
    if gray is None or image is None or gray.shape[1::-1] != expected_size:
        raise ValueError(f"Unreadable image or unexpected resolution: {path}")
    found, corners = cv.findChessboardCornersSB(
        gray, pattern, flags=cv.CALIB_CB_ACCURACY
    )
    if not found or len(corners) != pattern[0] * pattern[1]:
        raise ValueError(f"Could not find all checkerboard corners: {path}")
    return image, corners.reshape(-1, 2).astype(np.float64)


def check_corner_order(left, rgb, pattern, name):
    """Reject a pair if its horizontal or vertical grid directions are reversed."""
    left_grid = left.reshape(pattern[1], pattern[0], 2)
    rgb_grid = rgb.reshape(pattern[1], pattern[0], 2)
    for axis in (0, 1):
        left_direction = np.diff(left_grid, axis=axis).reshape(-1, 2).mean(axis=0)
        rgb_direction = np.diff(rgb_grid, axis=axis).reshape(-1, 2).mean(axis=0)
        if np.dot(left_direction, rgb_direction) <= 0:
            raise ValueError(f"Corner ordering differs between views: {name}")


def make_object_points(pattern, square_size):
    """Create the measured checkerboard corner grid in metres."""
    points = np.zeros((pattern[0] * pattern[1], 3), dtype=np.float64)
    points[:, :2] = (
        np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_size
    )
    return points


def project_from_left(object_points, left_corners, intrinsics, pair):
    """Estimate the board pose in left IR and predict corners in RGB."""
    left = intrinsics["cameras"]["ir_left"]
    rgb = intrinsics["cameras"]["rgb"]
    K_left = np.asarray(left["camera_matrix"], dtype=np.float64)
    D_left = np.asarray(left["distortion_k1_k2_p1_p2_k3"], dtype=np.float64)
    K_rgb = np.asarray(rgb["camera_matrix"], dtype=np.float64)
    D_rgb = np.asarray(rgb["distortion_k1_k2_p1_p2_k3"], dtype=np.float64)
    found, rvec, tvec = cv.solvePnP(object_points, left_corners, K_left, D_left)
    if not found:
        raise ValueError("Could not estimate the checkerboard pose in left IR.")
    R_board, _ = cv.Rodrigues(rvec)
    points_left = (R_board @ object_points.T + tvec.reshape(3, 1)).T
    R_pair = np.asarray(pair["rotation_matrix"], dtype=np.float64)
    t_pair = np.asarray(pair["translation_m"], dtype=np.float64).reshape(3, 1)
    points_rgb = (R_pair @ points_left.T + t_pair).T
    predicted, _ = cv.projectPoints(
        points_rgb, np.zeros(3), np.zeros(3), K_rgb, D_rgb
    )
    return predicted.reshape(-1, 2)


def frame_statistics(residuals):
    """Summarise signed RGB pixel residuals and their radial magnitudes."""
    lengths = np.linalg.norm(residuals, axis=1)
    bias = residuals.mean(axis=0)
    scatter = residuals - bias
    return {
        "rms_px": float(np.sqrt(np.mean(lengths**2))),
        "mean_dx_px": float(bias[0]),
        "mean_dy_px": float(bias[1]),
        "bias_magnitude_px": float(np.linalg.norm(bias)),
        "scatter_rms_px": float(np.sqrt(np.mean(np.sum(scatter**2, axis=1)))),
        "p90_error_px": float(np.percentile(lengths, 90)),
        "maximum_error_px": float(lengths.max()),
    }


def timestamp_difference(record_path):
    """Return the absolute RGB versus left IR timestamp gap, if recorded."""
    if not record_path.is_file():
        return None
    record = json.loads(record_path.read_text(encoding="utf-8-sig"))
    rgb = record["streams"]["rgb"]
    left = record["streams"]["ir_left"]
    if rgb["timestamp_domain"] != left["timestamp_domain"]:
        return None
    return abs(float(rgb["timestamp_ms"]) - float(left["timestamp_ms"]))


def draw_residuals(image, observed, predicted, stats, frame_id, arrow_scale):
    """Show detected and projected corners with visibly enlarged error arrows."""
    canvas = image.copy()
    for actual, estimate in zip(observed, predicted):
        start = tuple(np.rint(estimate).astype(int))
        actual_pixel = tuple(np.rint(actual).astype(int))
        end = tuple(np.rint(estimate + arrow_scale * (actual - estimate)).astype(int))
        cv.arrowedLine(canvas, start, end, (0, 165, 255), 1, cv.LINE_AA, tipLength=0.22)
        cv.circle(canvas, start, 2, (255, 0, 255), -1, cv.LINE_AA)
        cv.circle(canvas, actual_pixel, 2, (0, 255, 0), -1, cv.LINE_AA)
    cv.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 52), (20, 20, 20), -1)
    cv.putText(canvas, f"{frame_id}  RMS {stats['rms_px']:.3f} px", (10, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1, cv.LINE_AA)
    cv.putText(canvas, f"green observed | magenta projected | arrows x{arrow_scale:g}",
               (10, 42), cv.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv.LINE_AA)
    return canvas


def write_csv(path, fields, rows):
    """Save diagnostic rows without modifying any input file."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Process all held-out frames using the existing fixed calibration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--session", type=Path)
    parser.add_argument("--intrinsics", type=Path)
    parser.add_argument("--extrinsics", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--arrow-scale", type=float, default=20.0)
    args = parser.parse_args()
    if not np.isfinite(args.arrow_scale) or args.arrow_scale <= 0:
        parser.error("--arrow-scale must be a positive number")
    project = args.project.resolve()
    session, intrinsics_path, extrinsics_path, factory, intrinsics, pair = load_inputs(
        project, args.session, args.intrinsics, args.extrinsics
    )
    pattern = tuple(factory["board"]["inner_corners"])
    square_size = float(factory["board"]["square_size_m"])
    if len(pattern) != 2 or min(pattern) < 2 or not np.isfinite(square_size) or square_size <= 0:
        raise ValueError("Invalid checkerboard dimensions.")
    object_points = make_object_points(pattern, square_size)
    left_folder = session / "validation" / "ir_left"
    rgb_folder = session / "validation" / "rgb"
    names = sorted(path.name for path in left_folder.glob("*.png"))
    if len(names) != factory["planned_sets"]["validation"]:
        raise ValueError("The number of validation frames does not match the capture record.")
    if names != sorted(path.name for path in rgb_folder.glob("*.png")):
        raise ValueError("Left IR and RGB validation frame names do not match.")
    if args.out is None:
        args.out = (project / "experiments" / "calibration" / "results" /
                    ("residual_check_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Choose an empty output directory.")
    args.out.mkdir(parents=True, exist_ok=True)

    summaries = []
    corner_rows = []
    for name in names:
        frame_id = Path(name).stem
        _, left_corners = read_image_and_corners(
            left_folder / name, pattern, tuple(intrinsics["cameras"]["ir_left"]["image_size_wh"])
        )
        rgb_image, rgb_corners = read_image_and_corners(
            rgb_folder / name, pattern, tuple(intrinsics["cameras"]["rgb"]["image_size_wh"])
        )
        check_corner_order(left_corners, rgb_corners, pattern, name)
        predicted = project_from_left(object_points, left_corners, intrinsics, pair)
        residuals = rgb_corners - predicted
        stats = frame_statistics(residuals)
        stats.update({
            "frame_id": frame_id,
            "corner_count": len(rgb_corners),
            "timestamp_gap_ms": timestamp_difference(
                session / "validation" / "records" / f"{frame_id}.json"
            ),
            "above_0_5_px": stats["rms_px"] >= 0.5,
        })
        summaries.append(stats)
        for index, (observed, estimate, residual) in enumerate(
            zip(rgb_corners, predicted, residuals)
        ):
            corner_rows.append({
                "frame_id": frame_id, "corner_index": index,
                "observed_u_px": observed[0], "observed_v_px": observed[1],
                "predicted_u_px": estimate[0], "predicted_v_px": estimate[1],
                "dx_px": residual[0], "dy_px": residual[1],
                "distance_px": float(np.linalg.norm(residual)),
            })
        overlay = draw_residuals(
            rgb_image, rgb_corners, predicted, stats, frame_id, args.arrow_scale
        )
        encoded_ok, encoded_image = cv.imencode(".png", overlay)
        if not encoded_ok:
            raise OSError(f"Could not write residual image for {frame_id}")
        encoded_image.tofile(str(args.out / f"{frame_id}_residuals.png"))
        print(f"{frame_id}: {stats['rms_px']:.3f} px, "
              f"bias=({stats['mean_dx_px']:+.3f}, {stats['mean_dy_px']:+.3f}) px")

    write_csv(args.out / "frame_summary.csv", tuple(summaries[0]), summaries)
    write_csv(args.out / "corner_residuals.csv", tuple(corner_rows[0]), corner_rows)
    all_distances = np.array([row["distance_px"] for row in corner_rows])
    overall = float(np.sqrt(np.mean(all_distances**2)))
    (args.out / "run_info.json").write_text(json.dumps({
        "session": str(session), "intrinsics_file": str(intrinsics_path),
        "extrinsics_file": str(extrinsics_path), "overall_rms_px": overall,
        "frame_count": len(names), "corner_count": len(corner_rows),
        "arrow_scale_display_only": args.arrow_scale,
        "residual_definition": "observed RGB corner minus predicted RGB corner",
    }, indent=2), encoding="utf-8")
    print(f"Overall RMS: {overall:.6f} px")
    print(f"Results: {args.out.resolve()}")


if __name__ == "__main__":
    main()
