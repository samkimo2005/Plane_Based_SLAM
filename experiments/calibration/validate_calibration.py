import argparse
import csv
import json
from datetime import datetime
from pathlib import Path, PureWindowsPath

import cv2 as cv
import numpy as np


def select_path(paths, prompt):
    paths = sorted(set(path.resolve() for path in paths))
    if len(paths) == 1:
        return paths[0]
    if paths:
        for index, path in enumerate(paths, 1):
            print(f"{index}: {path}")
        index = int(input(prompt + " number: "))
        if not 1 <= index <= len(paths):
            raise ValueError("Invalid selection.")
        return paths[index - 1]
    value = input(prompt + " path: ").strip().strip('"').strip("'")
    if not value:
        raise ValueError("No path entered.")
    return Path(value).expanduser().resolve()


def find_result(project, roots, filename):
    preferred = project / "experiments" / "calibration" / "results"
    candidates = list(preferred.glob(f"*/{filename}"))
    candidates += [preferred / filename] if (preferred / filename).is_file() else []
    if not candidates:
        for root in (project, *roots):
            candidates.extend((root / "results").glob(f"*/{filename}"))
            if (root / filename).is_file():
                candidates.append(root / filename)
    return select_path(candidates, filename)


def detect_validation(session, pattern, sizes, expected_count):
    cameras = ("ir_left", "ir_right", "rgb")
    paths = {camera: sorted((session / "validation" / camera).glob("*.png")) for camera in cameras}
    filenames = [path.name for path in paths["ir_left"]]
    if len(filenames) != expected_count or expected_count < 1:
        raise ValueError("Incorrect validation image count for ir_left.")
    if any([path.name for path in paths[camera]] != filenames for camera in cameras):
        raise ValueError("Validation filenames do not match across cameras.")
    corners = {camera: {} for camera in cameras}
    for camera in cameras:
        for path in paths[camera]:
            gray = cv.imdecode(np.fromfile(str(path), np.uint8), cv.IMREAD_GRAYSCALE)
            if gray is None or gray.shape[1::-1] != sizes[camera]:
                raise ValueError(f"Unreadable image or incorrect dimensions: {path}")
            found, points = cv.findChessboardCornersSB(gray, pattern, flags=cv.CALIB_CB_ACCURACY)
            if not found or len(points) != pattern[0] * pattern[1]:
                raise ValueError(f"Checkerboard detection failed: {path}")
            corners[camera][path.name] = points.reshape(-1, 2).astype(np.float64)
    for filename in filenames:
        grids = [corners[camera][filename].reshape(pattern[1], pattern[0], 2) for camera in cameras]
        for target in (1, 2):
            for axis in (0, 1):
                directions = [np.diff(grid, axis=axis).reshape(-1, 2).mean(axis=0)
                              for grid in (grids[0], grids[target])]
                if np.dot(directions[0], directions[1]) <= 0:
                    raise ValueError(f"Corner ordering differs between camera views: {filename}")
    return filenames, corners


def rms_error(predicted, observed):
    errors = np.sum((predicted.reshape(-1, 2) - observed.reshape(-1, 2)) ** 2, axis=1)
    return float(np.sqrt(np.mean(errors)))


def summarise(per_image):
    return {
        "overall_rms_px": float(np.sqrt(np.mean(np.square(list(per_image.values()))))),
        "maximum_image_rms_px": float(max(per_image.values())),
        "per_image_rms_px": per_image,
    }


def save_left_to_rgb_per_frame(path, filenames, corner_count, intrinsics_results, extrinsics_results):
    """Write the existing per-image validation RMS values to a CSV file."""
    left_intrinsics = intrinsics_results["ir_left"]["per_image_rms_px"]
    rgb_intrinsics = intrinsics_results["rgb"]["per_image_rms_px"]
    comparison = extrinsics_results["left_to_rgb"]
    calibrated = comparison["calibrated"]["per_image_rms_px"]
    factory = comparison["factory"]["per_image_rms_px"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow((
            "frame_id", "corner_count", "left_intrinsics_rms_px",
            "rgb_intrinsics_rms_px", "calibrated_rms_px", "factory_rms_px",
        ))
        for filename in filenames:
            writer.writerow((
                Path(filename).stem, corner_count,
                left_intrinsics[filename], rgb_intrinsics[filename],
                calibrated[filename], factory[filename],
            ))


def validate_intrinsics(object_points, filenames, corners, intrinsics):
    results = {}
    left_poses = {}
    for camera in ("ir_left", "ir_right", "rgb"):
        info = intrinsics["cameras"][camera]
        K = np.array(info["camera_matrix"], dtype=np.float64)
        D = np.array(info["distortion_k1_k2_p1_p2_k3"], dtype=np.float64)
        per_image = {}
        for filename in filenames:
            observed = corners[camera][filename]
            found, rvec, tvec = cv.solvePnP(object_points, observed, K, D)
            if not found:
                raise ValueError(f"Board pose estimation failed: {camera}/{filename}")
            projected = cv.projectPoints(object_points, rvec, tvec, K, D)[0]
            per_image[filename] = rms_error(projected, observed)
            if camera == "ir_left":
                R, _ = cv.Rodrigues(rvec)
                left_poses[filename] = (R, tvec.reshape(3, 1))
        results[camera] = summarise(per_image)
    return results, left_poses


def validate_extrinsics(object_points, filenames, corners, left_poses, intrinsics, extrinsics, factory):
    results = {}
    for target, name in (("ir_right", "left_to_right"), ("rgb", "left_to_rgb")):
        K = np.array(intrinsics["cameras"][target]["camera_matrix"], dtype=np.float64)
        D = np.array(intrinsics["cameras"][target]["distortion_k1_k2_p1_p2_k3"], dtype=np.float64)
        candidates = ("calibrated", "factory")
        parameters = (extrinsics["pairs"][name], factory["extrinsics"][name])
        per_image = {candidate: {} for candidate in candidates}
        for filename in filenames:
            R_left, t_left = left_poses[filename]
            points_left = (R_left @ object_points.T + t_left).T
            observed = corners[target][filename]
            for candidate, parameters_pair in zip(candidates, parameters):
                R = np.array(parameters_pair["rotation_matrix"], dtype=np.float64)
                t = np.array(parameters_pair["translation_m"], dtype=np.float64).reshape(3, 1)
                points_target = (R @ points_left.T + t).T
                projected = cv.projectPoints(points_target, np.zeros(3), np.zeros(3), K, D)[0]
                per_image[candidate][filename] = rms_error(projected, observed)
        results[name] = {candidate: summarise(per_image[candidate]) for candidate in candidates}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path)
    parser.add_argument("--intrinsics", type=Path)
    parser.add_argument("--extrinsics", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    roots = []
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for root in (start, *start.parents):
            if root not in roots:
                roots.append(root)
    if args.session is None:
        candidates = [p.parent for root in roots
                      for p in (root / "data" / "camera_calibration").glob("session_*/factory_parameters.json")]
        args.session = select_path(candidates, "Session folder")
    args.session = args.session.resolve()
    project = args.session.parents[2] if args.session.parent.parent.name == "data" else Path(__file__).resolve().parent
    if args.extrinsics is None:
        args.extrinsics = find_result(project, roots, "extrinsics.json")
    args.extrinsics = args.extrinsics.resolve()
    extrinsics = json.loads(args.extrinsics.read_text(encoding="utf-8-sig"))
    recorded_intrinsics = PureWindowsPath(extrinsics["intrinsics_file"])
    if args.intrinsics is None:
        matching_intrinsics = (
            project / "experiments" / "calibration" / "results"
            / recorded_intrinsics.parent.name / recorded_intrinsics.name
        )
        args.intrinsics = (
            matching_intrinsics if matching_intrinsics.is_file()
            else find_result(project, roots, "intrinsics.json")
        )
    args.intrinsics = args.intrinsics.resolve()
    factory = json.loads((args.session / "factory_parameters.json").read_text(encoding="utf-8-sig"))
    intrinsics = json.loads(args.intrinsics.read_text(encoding="utf-8-sig"))
    if intrinsics["board"] != factory["board"] or extrinsics["board"] != factory["board"]:
        raise ValueError("Checkerboard settings differ between files.")
    if "session" in intrinsics and Path(intrinsics["session"].replace("\\", "/")).name != args.session.name:
        raise ValueError("Intrinsics belong to a different session.")
    if Path(extrinsics["session"].replace("\\", "/")).name != args.session.name:
        raise ValueError("Extrinsics belong to a different session.")
    if (recorded_intrinsics.parent.name, recorded_intrinsics.name) != (
        args.intrinsics.parent.name, args.intrinsics.name
    ):
        raise ValueError(
            f"Extrinsics expect {recorded_intrinsics.parent.name}/{recorded_intrinsics.name}, "
            f"but the selected intrinsics file is {args.intrinsics}"
        )
    if not extrinsics.get("intrinsics_fixed"):
        raise ValueError("Extrinsics were not computed with fixed intrinsics.")
    for name, target in (("left_to_right", "ir_right"), ("left_to_rgb", "rgb")):
        pair = extrinsics["pairs"][name]
        if pair["source"] != "ir_left" or pair["target"] != target:
            raise ValueError(f"Unexpected camera pair: {name}")
    pattern = tuple(factory["board"]["inner_corners"])
    square = float(factory["board"]["square_size_m"])
    if len(pattern) != 2 or min(pattern) < 2 or not np.isfinite(square) or square <= 0:
        raise ValueError("Invalid checkerboard dimensions.")
    object_points = np.zeros((pattern[0] * pattern[1], 3), dtype=np.float64)
    object_points[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square
    sizes = {camera: tuple(intrinsics["cameras"][camera]["image_size_wh"])
             for camera in ("ir_left", "ir_right", "rgb")}
    for camera, size in sizes.items():
        parameters = factory["stream_intrinsics"][camera]
        if size != (parameters["width"], parameters["height"]):
            raise ValueError(f"Calibration image dimensions differ for {camera}.")
    filenames, corners = detect_validation(
        args.session, pattern, sizes, factory["planned_sets"]["validation"],
    )
    print(f"Session: {args.session}")
    print(f"Validation image sets: {len(filenames)}", flush=True)
    intrinsics_results, left_poses = validate_intrinsics(object_points, filenames, corners, intrinsics)
    extrinsics_results = validate_extrinsics(
        object_points, filenames, corners, left_poses, intrinsics, extrinsics, factory,
    )
    results = {
        "session": str(args.session), "intrinsics_file": str(args.intrinsics),
        "extrinsics_file": str(args.extrinsics), "board": factory["board"],
        "validation_image_count": len(filenames),
        "intrinsics_validation": intrinsics_results,
        "extrinsics_validation": extrinsics_results,
        "extrinsics_comparison_uses_calibrated_intrinsics": True,
    }
    for camera, result in intrinsics_results.items():
        print(f"{camera}: intrinsics validation RMS = {result['overall_rms_px']:.6f} px")
    for name, comparison in extrinsics_results.items():
        for candidate, result in comparison.items():
            print(f"{name} ({candidate}): transfer RMS = {result['overall_rms_px']:.6f} px")
    if args.out is None:
        args.out = project / "experiments" / "calibration" / "results" / (
            "validation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Choose an empty output directory.")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "validation.json").write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8",
    )
    save_left_to_rgb_per_frame(
        args.out / "left_to_rgb_per_frame.csv", filenames, len(object_points),
        intrinsics_results, extrinsics_results,
    )
    print(f"Saved to {args.out.resolve()}")


if __name__ == "__main__":
    main()
