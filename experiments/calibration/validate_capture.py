"""Capture 15 RGB/left-IR/right-IR checkerboard sets with measured board distance.

Run from the project root, for example:
    python experiments/calibration/capture_multidistance_diagnostic.py --targets-m 0.50 0.65 0.80

Choose the middle and far targets for your own setup. The program asks which
first-round intrinsics.json to use if more than one is found. Press L to arm
capture after the automatic exposure has settled, S to save a complete set,
and Q or Esc to stop. L leaves automatic exposure enabled.
"""

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2 as cv
import numpy as np


CAMERAS = ("rgb", "ir_left", "ir_right")
GROUPS = ("near", "middle", "far")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-m", nargs=3, type=float, required=True,
                        metavar=("NEAR", "MIDDLE", "FAR"))
    parser.add_argument("--per-distance", type=int, default=5)
    parser.add_argument("--tolerance-m", type=float, default=0.06)
    parser.add_argument("--square-mm", type=float, default=32.0)
    parser.add_argument("--cols", type=int, default=11)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--intrinsics", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/camera_calibration"))
    parser.add_argument("--serial")
    args = parser.parse_args()
    if (not all(np.isfinite(args.targets_m)) or
            not 0 < args.targets_m[0] < args.targets_m[1] < args.targets_m[2]):
        parser.error("--targets-m must be three increasing positive distances.")
    if args.per_distance < 1 or args.tolerance_m <= 0 or args.square_mm <= 0:
        parser.error("Counts, distance tolerance and square size must be positive.")
    if min(args.cols, args.rows) < 2:
        parser.error("The checkerboard must have at least 2x2 inner corners.")
    return args


def project_root():
    script = Path(__file__).resolve().parent
    for start in (Path.cwd(), script):
        for root in (start, *start.parents):
            if (root / "experiments" / "calibration" / "results").is_dir():
                return root
    return Path.cwd()


def load_rgb_intrinsics(args):
    if args.intrinsics is None:
        results = project_root() / "experiments" / "calibration" / "results"
        choices = sorted(results.glob("intrinsics_*/intrinsics.json"))
        if not choices:
            raise FileNotFoundError("Specify the first-round intrinsics.json with --intrinsics.")
        if len(choices) > 1:
            for i, path in enumerate(choices, 1):
                print(f"{i}: {path}")
            selected = int(input("Select FIRST-ROUND intrinsics number: "))
            if not 1 <= selected <= len(choices):
                raise ValueError("Invalid intrinsics selection.")
            path = choices[selected - 1]
        else:
            path = choices[0]
    else:
        path = args.intrinsics.expanduser().resolve()
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    pattern = [args.cols, args.rows]
    if data["board"]["inner_corners"] != pattern:
        raise ValueError("The first-round intrinsics use a different checkerboard pattern.")
    if not np.isclose(float(data["board"]["square_size_m"]), args.square_mm / 1000):
        raise ValueError("The first-round intrinsics use a different square size.")
    rgb = data["cameras"]["rgb"]
    if rgb["image_size_wh"] != [640, 480]:
        raise ValueError("The first-round RGB intrinsics must be for 640x480 images.")
    K = np.asarray(rgb["camera_matrix"], dtype=np.float64)
    D = np.asarray(rgb["distortion_k1_k2_p1_p2_k3"], dtype=np.float64)
    if K.shape != (3, 3) or D.size not in (4, 5, 8) or not np.isfinite(K).all() or not np.isfinite(D).all():
        raise ValueError("Invalid RGB camera matrix or distortion coefficients.")
    print(f"Distance estimate uses: {path}")
    return path, raw, hashlib.sha256(raw).hexdigest(), K, D


def board_points(args):
    points = np.zeros((args.cols * args.rows, 3), np.float64)
    points[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * (args.square_mm / 1000)
    center = np.array([(args.cols - 1) * args.square_mm / 2000,
                       (args.rows - 1) * args.square_mm / 2000, 0], dtype=np.float64)
    return points, center


def find_corners(image, pattern):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
    found, corners = cv.findChessboardCornersSB(gray, pattern, flags=cv.CALIB_CB_ACCURACY)
    return corners if found and len(corners) == pattern[0] * pattern[1] else None


def board_distance(corners, points, center, K, D):
    found, rvec, tvec = cv.solvePnP(points, corners, K, D, flags=cv.SOLVEPNP_ITERATIVE)
    if not found:
        raise ValueError("Could not estimate the checkerboard pose in RGB.")
    R, _ = cv.Rodrigues(rvec)
    center_rgb = R @ center + tvec.reshape(3)
    if center_rgb[2] <= 0:
        raise ValueError("The estimated checkerboard lies behind the RGB camera.")
    projected, _ = cv.projectPoints(points, rvec, tvec, K, D)
    residual = projected.reshape(-1, 2) - corners.reshape(-1, 2)
    pose_rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return float(np.linalg.norm(center_rgb)), float(center_rgb[2]), pose_rms


def matching_corner_order(corners, pattern):
    grids = [corners[name].reshape(pattern[1], pattern[0], 2) for name in CAMERAS]
    for other in (1, 2):
        for axis in (0, 1):
            a = np.diff(grids[0], axis=axis).reshape(-1, 2).mean(axis=0)
            b = np.diff(grids[other], axis=axis).reshape(-1, 2).mean(axis=0)
            if np.dot(a, b) <= 0:
                return False
    return True


def profile_info(profile):
    p = profile.get_intrinsics()
    return dict(width=p.width, height=p.height, fps=profile.fps(), format=str(profile.format()),
                fx=p.fx, fy=p.fy, cx=p.ppx, cy=p.ppy, distortion_model=str(p.model),
                distortion_coefficients=list(p.coeffs))


def extrinsics_info(source, target):
    e = source.get_extrinsics_to(target)
    return dict(rotation_matrix=np.asarray(e.rotation).reshape(3, 3, order="F").tolist(),
                translation_m=list(e.translation), convention="p_target = R @ p_source + t")


def snapshot_sensor(sensor, rs):
    values = {}
    for name in ("enable_auto_exposure", "exposure", "gain", "enable_auto_white_balance",
                 "white_balance", "emitter_enabled", "laser_power", "visual_preset"):
        option = getattr(rs.option, name)
        if sensor.supports(option):
            try:
                values[name] = sensor.get_option(option)
            except RuntimeError:
                pass
    return values


def restore_sensor(sensor, saved, rs):
    for name in ("enable_auto_exposure", "enable_auto_white_balance"):
        if name in saved:
            try:
                sensor.set_option(getattr(rs.option, name), 0)
            except RuntimeError:
                pass
    for name in ("exposure", "gain", "white_balance", "emitter_enabled",
                 "enable_auto_exposure", "enable_auto_white_balance"):
        if name in saved:
            try:
                sensor.set_option(getattr(rs.option, name), saved[name])
            except RuntimeError:
                print(f"Could not restore {name}.")


def write_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def draw_preview(images, pattern, rgb_corners, group, target, distance, z, rms,
                 count, total, tolerance, status):
    panels = []
    for name in CAMERAS:
        image = images[name]
        panel = cv.cvtColor(image, cv.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        if name == "rgb" and rgb_corners is not None:
            cv.drawChessboardCorners(panel, pattern, rgb_corners, True)
        panel = cv.resize(panel, (400, 300))
        cv.putText(panel, name, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        panels.append(panel)
    canvas = np.vstack((np.hstack(panels), np.zeros((120, 1200, 3), np.uint8)))
    distance_text = (f"center distance {distance:.3f} m | Z {z:.3f} m | RGB pose RMS {rms:.3f} px"
                     if distance is not None else "Checkerboard not detected in RGB")
    lines = (f"{group} target {target:.2f} +/- {tolerance:.2f} m | saved {count}/{total}",
             distance_text, status[:130], "L ready (auto exposure stays on) | S save | Q/Esc quit")
    for index, line in enumerate(lines):
        cv.putText(canvas, line, (10, 321 + index * 27), cv.FONT_HERSHEY_SIMPLEX,
                   0.58, (255, 255, 255), 1, cv.LINE_AA)
    cv.imshow("Multi-distance calibration diagnostic", canvas)


def save_set(root, index, group, target, distance, z, rms, images, trio, corners):
    name = f"{index:04d}"
    brightness = {camera: float(cv.cvtColor(images[camera], cv.COLOR_BGR2GRAY).mean()
                 if images[camera].ndim == 3 else images[camera].mean()) for camera in CAMERAS}
    record = dict(split="validation", index=index, distance_group=group,
                  target_distance_m=target, board_center_distance_m=distance,
                  board_center_z_m=z, rgb_pose_rms_px=rms,
                  mean_brightness_0_255=brightness,
                  saved_utc=datetime.now(timezone.utc).isoformat(), streams={})
    written = []
    try:
        for camera in CAMERAS:
            folder = root / "validation" / camera
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{name}.png"
            ok, encoded = cv.imencode(".png", images[camera])
            if not ok:
                raise OSError(f"Could not encode {camera} image.")
            encoded.tofile(str(path))
            written.append(path)
            frame = trio[camera]
            record["streams"][camera] = dict(file=path.relative_to(root).as_posix(),
                timestamp_ms=frame.get_timestamp(),
                timestamp_domain=str(frame.get_frame_timestamp_domain()),
                frame_number=frame.get_frame_number(),
                corners=corners[camera].reshape(-1, 2).tolist())
        write_json(root / "validation" / "records" / f"{name}.json", record)
        with (root / "distance_summary.csv").open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow((name, group, target, distance, z, rms,
                                    brightness["rgb"], brightness["ir_left"],
                                    brightness["ir_right"]))
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        (root / "validation" / "records" / f"{name}.json").unlink(missing_ok=True)
        raise


def capture(args, rs, intrinsic_path, intrinsic_bytes, sha256, K, D):
    pattern = (args.cols, args.rows)
    points, center = board_points(args)
    context = rs.context()
    devices = context.query_devices()
    if not devices:
        raise RuntimeError("No RealSense camera found. Check USB and close RealSense Viewer.")
    if len(devices) > 1 and not args.serial:
        raise RuntimeError("Multiple RealSense cameras found. Specify --serial.")
    pipeline = rs.pipeline(context)
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
    started = False
    originals = []
    root = None
    count = 0
    try:
        selected = pipeline.start(config)
        started = True
        device = selected.get_device()
        sensors = [s for s in device.query_sensors() if s.supports(rs.option.enable_auto_exposure)]
        if len(sensors) < 2:
            raise RuntimeError("Expected RGB and stereo exposure controls on D435i.")
        for sensor in sensors:
            originals.append((sensor, snapshot_sensor(sensor, rs)))
            if sensor.supports(rs.option.emitter_enabled):
                sensor.set_option(rs.option.emitter_enabled, 0)
            sensor.set_option(rs.option.enable_auto_exposure, 1)
            if sensor.supports(rs.option.enable_auto_white_balance):
                sensor.set_option(rs.option.enable_auto_white_balance, 1)

        profiles = dict(rgb=selected.get_stream(rs.stream.color).as_video_stream_profile(),
                        ir_left=selected.get_stream(rs.stream.infrared, 1).as_video_stream_profile(),
                        ir_right=selected.get_stream(rs.stream.infrared, 2).as_video_stream_profile())
        device_name = device.get_info(rs.camera_info.name)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        root = args.out.expanduser().resolve() / ("distance_diagnostic_" + stamp)
        root.mkdir(parents=True, exist_ok=False)
        (root / "validation" / "records").mkdir(parents=True)
        (root / "distance_intrinsics.json").write_bytes(intrinsic_bytes)
        metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                        serial=device.get_info(rs.camera_info.serial_number),
                        device=device_name, firmware=device.get_info(rs.camera_info.firmware_version),
                        python=sys.version,
                        board=dict(inner_corners=list(pattern), square_size_m=args.square_mm / 1000),
                        planned_sets=dict(train=0, validation=3 * args.per_distance),
                        stream_intrinsics={k: profile_info(v) for k, v in profiles.items()},
                        extrinsics=dict(left_to_right=extrinsics_info(profiles["ir_left"], profiles["ir_right"]),
                                        left_to_rgb=extrinsics_info(profiles["ir_left"], profiles["rgb"])),
                        distance_estimate=dict(method="RGB checkerboard solvePnP; Euclidean camera-to-board-center distance",
                            intrinsics_source=str(intrinsic_path), intrinsics_sha256=sha256,
                            targets_m=args.targets_m, tolerance_m=args.tolerance_m,
                            samples_per_target=args.per_distance),
                        notes=["Distance is an RGB-PnP estimate, not a direct range measurement.",
                               "Images are diagnostic validation images; none is calibration training data.",
                               "Auto exposure stays enabled; image brightness is logged per frame.",
                               "IR images are rectified SDK output; frameset is not guaranteed hardware sync.",
                               "No depth images or software alignment were used."])
        write_json(root / "factory_parameters.json", metadata)
        with (root / "distance_summary.csv").open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(("frame_id", "group", "target_m", "board_center_distance_m",
                                    "board_center_z_m", "rgb_pose_rms_px", "rgb_mean_brightness",
                                    "ir_left_mean_brightness", "ir_right_mean_brightness"))
        print(f"New diagnostic session: {root}")
        print("Wait 5 seconds with the board visible, press L to begin, then S to save each set.")
        print("Automatic exposure remains enabled; per-frame image brightness is recorded.")
        ready = False
        auto_since = time.monotonic()
        status = "AUTO: wait at least 5 seconds, then press L"
        last_saved_frames = None
        total = 3 * args.per_distance
        while count < total:
            frames = pipeline.wait_for_frames(10000)
            trio = dict(rgb=frames.get_color_frame(), ir_left=frames.get_infrared_frame(1),
                        ir_right=frames.get_infrared_frame(2))
            if not all(trio.values()):
                continue
            images = {name: np.asanyarray(frame.get_data()).copy() for name, frame in trio.items()}
            group_index = count // args.per_distance
            group, target = GROUPS[group_index], args.targets_m[group_index]
            rgb_corners = find_corners(images["rgb"], pattern)
            distance = z = rms = None
            if rgb_corners is not None:
                try:
                    distance, z, rms = board_distance(rgb_corners, points, center, K, D)
                except ValueError:
                    pass
            draw_preview(images, pattern, rgb_corners, group, target, distance, z, rms,
                         count, total, args.tolerance_m, status)
            key = cv.waitKey(1) & 0xff
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("l"), ord("L")) and not ready:
                if time.monotonic() - auto_since < 5:
                    status = "Please wait 5 seconds for auto exposure to settle."
                else:
                    ready = True
                    write_json(root / "capture_settings.json",
                               dict(exposure_mode="automatic",
                                    sensors={s.get_info(rs.camera_info.name): snapshot_sensor(s, rs)
                                             for s in sensors}))
                    status = "READY: auto exposure stays on; hold board still for 2 seconds before S"
                continue
            if key not in (ord("s"), ord("S")):
                continue
            if not ready:
                status = "Press L to prepare capture before saving."
                continue
            if distance is None:
                status = "Not saved: RGB checkerboard pose unavailable."
                continue
            if abs(distance - target) > args.tolerance_m:
                status = f"Not saved: distance {distance:.3f} m is outside {target:.2f} +/- {args.tolerance_m:.2f} m"
                continue
            numbers = tuple(trio[name].get_frame_number() for name in CAMERAS)
            if numbers == last_saved_frames:
                status = "Not saved: duplicate frameset."
                continue
            corners = {"rgb": rgb_corners}
            for camera in ("ir_left", "ir_right"):
                corners[camera] = find_corners(images[camera], pattern)
            if any(corners[name] is None for name in CAMERAS):
                status = "Not saved: checkerboard must be found in all three images."
                continue
            if not matching_corner_order(corners, pattern):
                status = "Not saved: checkerboard corner order differs between cameras."
                continue
            save_set(root, count, group, target, distance, z, rms, images, trio, corners)
            count += 1
            last_saved_frames = numbers
            status = f"Saved {count}/{total}: {group}, measured {distance:.3f} m"
            print(status, flush=True)
            draw_preview(images, pattern, rgb_corners, group, target, distance, z, rms,
                         count, total, args.tolerance_m, status)
            cv.waitKey(800)
        print(f"Saved {count}/{total} sets in {root}")
    finally:
        if started:
            for sensor, saved in originals:
                restore_sensor(sensor, saved, rs)
            pipeline.stop()
        cv.destroyAllWindows()


def main():
    args = parse_args()
    intrinsic_path, intrinsic_bytes, sha256, K, D = load_rgb_intrinsics(args)
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(f"pyrealsense2 is required in the active Python environment: {exc}")
    try:
        capture(args, rs, intrinsic_path, intrinsic_bytes, sha256, K, D)
    except (OSError, RuntimeError) as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
