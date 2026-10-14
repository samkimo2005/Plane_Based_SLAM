import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np


def find_session(location=None):
    if location is None:
        roots = []
        for start in (Path.cwd(), Path(__file__).resolve().parent):
            for root in (start, *start.parents):
                if root not in roots:
                    roots.append(root)
        folders = [root / "data" / "camera_calibration" for root in roots]
    else:
        path = Path(location).expanduser().resolve()
        if path.is_file() and path.name == "factory_parameters.json":
            path = path.parent
        if path.name == "train":
            path = path.parent
        folders = [path, path / "data" / "camera_calibration"]

    candidates = set()
    for folder in folders:
        if (folder / "factory_parameters.json").is_file():
            candidates.add(folder.resolve())
        if folder.is_dir():
            for child in folder.iterdir():
                if child.is_dir() and (child / "factory_parameters.json").is_file():
                    candidates.add(child.resolve())

    complete = []
    for candidate in sorted(candidates):
        try:
            metadata = json.loads((candidate / "factory_parameters.json").read_text(encoding="utf-8-sig"))
            expected = int(metadata["planned_sets"]["train"])
            counts = [len(list((candidate / "train" / camera).glob("*.png")))
                      for camera in ("rgb", "ir_left", "ir_right")]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if expected >= 3 and counts == [expected] * 3:
            complete.append(candidate)
        else:
            print(f"Incomplete training data: {candidate} ({counts}; expected {expected} per camera)")

    if not complete:
        if location is not None:
            raise ValueError(f"No complete training session found in: {location}")
        print(f"Working directory: {Path.cwd()}")
        print(f"Script directory: {Path(__file__).resolve().parent}")
        value = input("Session folder or camera_calibration folder: ").strip().strip('\"').strip("'")
        if not value:
            raise ValueError("No folder entered.")
        return find_session(value)
    if len(complete) == 1:
        selected = complete[0]
    else:
        for index, candidate in enumerate(complete, 1):
            print(f"{index}: {candidate}")
        index = int(input("Select session number: "))
        if index < 1 or index > len(complete):
            raise ValueError("Invalid session number.")
        selected = complete[index - 1]
    print(f"Session: {selected}", flush=True)
    return selected


def detect_corners(folder, pattern, expected_count, size):
    paths = sorted(folder.glob("*.png"))
    if len(paths) != expected_count or len(paths) < 3:
        raise ValueError(f"Check the training image count in {folder}.")
    image_points = []

    for path in paths:
        gray = cv.imdecode(np.fromfile(str(path), np.uint8), cv.IMREAD_GRAYSCALE)
        if gray is None or gray.shape[1::-1] != size:
            raise ValueError(f"Unreadable image or incorrect dimensions: {path}")
        found, corners = cv.findChessboardCornersSB(gray, pattern, flags=cv.CALIB_CB_ACCURACY)
        if not found or len(corners) != pattern[0] * pattern[1]:
            raise ValueError(f"Checkerboard detection failed: {path}")
        image_points.append(corners.astype(np.float32))
        if path == paths[0]:
            preview = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
            cv.drawChessboardCorners(preview, pattern, corners, found)

    return paths, image_points, preview


def calibrate_camera(camera, object_points, paths, image_points, size, factory):
    initial_K = np.array([
        [factory["fx"], 0, factory["cx"]],
        [0, factory["fy"], factory["cy"]],
        [0, 0, 1],
    ], dtype=np.float64)
    flags = cv.CALIB_USE_INTRINSIC_GUESS | cv.CALIB_FIX_K3
    if camera != "rgb":
        flags |= cv.CALIB_FIX_K1 | cv.CALIB_FIX_K2 | cv.CALIB_ZERO_TANGENT_DIST
    rms, K, D, rvecs, tvecs = cv.calibrateCamera(
        [object_points] * len(paths), image_points, size,
        initial_K, np.zeros(5), flags=flags,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 150, 1e-10),
    )
    per_image = {}
    squared_errors = []
    for path, points, rvec, tvec in zip(paths, image_points, rvecs, tvecs):
        projected = cv.projectPoints(object_points.astype(np.float64), rvec, tvec, K, D)[0]
        residuals = projected.reshape(-1, 2) - points.reshape(-1, 2)
        errors = np.sum(residuals ** 2, axis=1)
        per_image[path.name] = float(np.sqrt(np.mean(errors)))
        squared_errors.extend(errors)
    manual_rms = float(np.sqrt(np.mean(squared_errors)))
    if not np.isclose(manual_rms, rms, atol=1e-4):
        raise ValueError(f"RMS calculation mismatch for {camera}.")
    return {
        "image_size_wh": list(size),
        "training_image_count": len(paths),
        "camera_matrix": K.tolist(),
        "distortion_k1_k2_p1_p2_k3": D.ravel().tolist(),
        "training_rms_px": manual_rms,
        "per_image_training_rms_px": per_image,
        "flags": int(flags),
    }


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    args.session = find_session(args.session)
    if args.out is None:
        if args.session.parent.name == "camera_calibration" and args.session.parent.parent.name == "data":
            base = args.session.parents[2]
        args.out = base / "results" / ("intrinsics_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Choose an empty output directory.")

    config = json.loads((args.session / "factory_parameters.json").read_text(encoding="utf-8-sig"))
    pattern = tuple(config["board"]["inner_corners"])
    square = float(config["board"]["square_size_m"])
    if len(pattern) != 2 or min(pattern) < 2 or not np.isfinite(square) or square <= 0:
        raise ValueError("Invalid checkerboard dimensions.")
    object_points = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square
    results = {"session": str(args.session), "board": config["board"], "cameras": {}}
    previews = {}

    for camera in ("rgb", "ir_left", "ir_right"):
        print(f"Processing {camera}...", flush=True)
        factory = config["stream_intrinsics"][camera]
        size = (factory["width"], factory["height"])
        paths, image_points, preview = detect_corners(
            args.session / "train" / camera, pattern,
            config["planned_sets"]["train"], size,
        )
        result = calibrate_camera(camera, object_points, paths, image_points, size, factory)
        results["cameras"][camera] = result
        previews[camera] = preview
        print(f"{camera}: training RMS = {result['training_rms_px']:.6f} px", flush=True)
        print(np.array(result["camera_matrix"]))
        print(np.array(result["distortion_k1_k2_p1_p2_k3"]))

    args.out.mkdir(parents=True, exist_ok=True)
    for camera, preview in previews.items():
        ok, encoded = cv.imencode(".png", preview)
        if not ok:
            raise ValueError(f"Could not save the preview for {camera}.")
        encoded.tofile(str(args.out / f"{camera}_corners.png"))
    (args.out / "intrinsics.json").write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8",
    )
    print(f"Saved to {args.out.resolve()}")


if __name__ == "__main__":
    main()