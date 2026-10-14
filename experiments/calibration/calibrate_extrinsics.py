import argparse
import json
from datetime import datetime
from pathlib import Path

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
    value = input(prompt + " path: ").strip().strip('\"').strip("'")
    if not value:
        raise ValueError("No path entered.")
    return Path(value).expanduser().resolve()


def load_paired_corners(session, source, target, pattern, expected_count, sizes):
    paths_source = sorted((session / "train" / source).glob("*.png"))
    paths_target = sorted((session / "train" / target).glob("*.png"))
    if len(paths_source) != expected_count or expected_count < 3:
        raise ValueError(f"Incorrect training image count for {source}.")
    if [p.name for p in paths_source] != [p.name for p in paths_target]:
        raise ValueError(f"Training filenames do not match: {source}, {target}.")
    points_source, points_target = [], []
    for path_source, path_target in zip(paths_source, paths_target):
        pair = []
        for path, size in zip((path_source, path_target), sizes):
            gray = cv.imdecode(np.fromfile(str(path), np.uint8), cv.IMREAD_GRAYSCALE)
            if gray is None or gray.shape[1::-1] != size:
                raise ValueError(f"Unreadable image or incorrect dimensions: {path}")
            found, corners = cv.findChessboardCornersSB(gray, pattern, flags=cv.CALIB_CB_ACCURACY)
            if not found or len(corners) != pattern[0] * pattern[1]:
                raise ValueError(f"Checkerboard detection failed: {path}")
            pair.append(corners.astype(np.float32).reshape(-1, 1, 2))
        grids = [points.reshape(pattern[1], pattern[0], 2) for points in pair]
        for axis in (0, 1):
            directions = [np.diff(grid, axis=axis).reshape(-1, 2).mean(axis=0) for grid in grids]
            if np.dot(directions[0], directions[1]) <= 0:
                raise ValueError(f"Corner ordering differs between the D435i views: {path_source.name}")
        points_source.append(pair[0])
        points_target.append(pair[1])
    return [p.name for p in paths_source], points_source, points_target


def calibrate_pair(object_points, points_source, points_target, source_info, target_info):
    K_source = np.array(source_info["camera_matrix"], dtype=np.float64)
    D_source = np.array(source_info["distortion_k1_k2_p1_p2_k3"], dtype=np.float64)
    K_target = np.array(target_info["camera_matrix"], dtype=np.float64)
    D_target = np.array(target_info["distortion_k1_k2_p1_p2_k3"], dtype=np.float64)
    rms, K1, D1, K2, D2, R, t, _, _ = cv.stereoCalibrate(
        [object_points] * len(points_source), points_source, points_target,
        K_source.copy(), D_source.copy(), K_target.copy(), D_target.copy(),
        tuple(source_info["image_size_wh"]), flags=cv.CALIB_FIX_INTRINSIC,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 150, 1e-10),
    )
    for fitted, original in ((K1, K_source), (D1, D_source), (K2, K_target), (D2, D_target)):
        if not np.allclose(fitted.ravel(), original.ravel(), rtol=0, atol=1e-12):
            raise ValueError("Intrinsics changed during stereo calibration.")
    return {
        "rotation_matrix": R.tolist(),
        "translation_m": t.ravel().tolist(),
        "camera_separation_m": float(np.linalg.norm(t)),
        "joint_training_rms_px": float(rms),
        "convention": "p_target = R @ p_source + t",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path)
    parser.add_argument("--intrinsics", type=Path)
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
    config = json.loads((args.session / "factory_parameters.json").read_text(encoding="utf-8-sig"))
    project = args.session.parents[2] if args.session.parent.parent.name == "data" else Path(__file__).resolve().parent
    if args.intrinsics is None:
        candidates = [p for root in (project, *roots)
                      for p in (root / "results").glob("*/intrinsics.json")]
        candidates += [root / "intrinsics.json" for root in (project, *roots)
                       if (root / "intrinsics.json").is_file()]
        args.intrinsics = select_path(candidates, "Intrinsics JSON")
    args.intrinsics = args.intrinsics.resolve()
    intrinsics = json.loads(args.intrinsics.read_text(encoding="utf-8-sig"))
    print(f"Session: {args.session}", flush=True)
    print(f"Intrinsics: {args.intrinsics}", flush=True)
    if intrinsics["board"] != config["board"]:
        raise ValueError("Checkerboard settings differ between the session and intrinsics JSON.")
    pattern = tuple(config["board"]["inner_corners"])
    square = float(config["board"]["square_size_m"])
    if len(pattern) != 2 or min(pattern) < 2 or not np.isfinite(square) or square <= 0:
        raise ValueError("Invalid checkerboard dimensions.")
    object_points = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square
    results = {"session": str(args.session), "intrinsics_file": str(args.intrinsics),
               "board": config["board"], "intrinsics_fixed": True, "pairs": {}}

    for target, name in (("ir_right", "left_to_right"), ("rgb", "left_to_rgb")):
        print(f"Processing ir_left -> {target}...", flush=True)
        source_info, target_info = (intrinsics["cameras"][camera] for camera in ("ir_left", target))
        sizes = [tuple(info["image_size_wh"]) for info in (source_info, target_info)]
        for camera, size in zip(("ir_left", target), sizes):
            factory = config["stream_intrinsics"][camera]
            if size != (factory["width"], factory["height"]):
                raise ValueError(f"Calibration image dimensions differ for {camera}.")
        files, points_source, points_target = load_paired_corners(
            args.session, "ir_left", target, pattern, config["planned_sets"]["train"], sizes,
        )
        result = calibrate_pair(object_points, points_source, points_target, source_info, target_info)
        result.update(source="ir_left", target=target, training_pair_count=len(files), files=files)
        results["pairs"][name] = result
        print(f"Joint training RMS: {result['joint_training_rms_px']:.6f} px")
        print(np.array(result["rotation_matrix"]))
        print(np.array(result["translation_m"]))

    if args.out is None:
        args.out = project / "experiments" / "calibration" / "results" / ("extrinsics_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Choose an empty output directory.")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "extrinsics.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved to {args.out.resolve()}")


if __name__ == "__main__":
    main()
