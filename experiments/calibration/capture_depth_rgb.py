import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2 as cv
import numpy as np
import pyrealsense2 as rs


def intrinsics_data(profile):
    intrinsics = profile.get_intrinsics()
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.ppx,
        "cy": intrinsics.ppy,
        "distortion_model": str(intrinsics.model),
        "coeffs": list(intrinsics.coeffs),
    }


def extrinsics_data(source, target):
    extrinsics = source.get_extrinsics_to(target)
    rotation = np.array(extrinsics.rotation).reshape(3, 3).T
    return {
        "rotation_matrix": rotation.tolist(),
        "translation_m": list(extrinsics.translation),
        "convention": "p_target = R @ p_source + t",
    }


def frame_data(frame, filename):
    return {
        "file": filename,
        "timestamp_ms": frame.get_timestamp(),
        "timestamp_domain": str(frame.get_frame_timestamp_domain()),
        "frame_number": frame.get_frame_number(),
    }


def write_png(path, image):
    success, encoded = cv.imencode(".png", image)
    if not success:
        raise ValueError(f"Could not encode {path}")
    encoded.tofile(str(path))


def save_capture(folder, index, color_frame, depth_frame, ir_frame):
    filename = f"{index:04d}.png"
    frames = {"rgb": color_frame, "depth": depth_frame, "ir_left": ir_frame}
    images = {camera: np.asanyarray(frame.get_data()).copy() for camera, frame in frames.items()}
    if images["rgb"].dtype != np.uint8 or images["rgb"].ndim != 3:
        raise ValueError("RGB image must be BGR8.")
    if images["depth"].dtype != np.uint16 or images["depth"].ndim != 2:
        raise ValueError("Depth image must be raw Z16.")
    if images["ir_left"].dtype != np.uint8 or images["ir_left"].ndim != 2:
        raise ValueError("Left IR image must be Y8.")
    for camera, image in images.items():
        write_png(folder / camera / filename, image)
    record = {
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "streams": {
            camera: frame_data(frame, f"{camera}/{filename}")
            for camera, frame in frames.items()
        },
    }
    (folder / "records" / f"{index:04d}.json").write_text(
        json.dumps(record, indent=2, allow_nan=False), encoding="utf-8",
    )
    return images


def make_preview(color, depth, index, depth_scale):
    depth_preview = cv.applyColorMap(cv.convertScaleAbs(depth, alpha=0.05), cv.COLORMAP_JET)
    preview = cv.copyMakeBorder(
        np.hstack((color, depth_preview)), 0, 100, 0, 0,
        cv.BORDER_CONSTANT, value=(25, 25, 25),
    )
    middle = depth[depth.shape[0] // 2 - 10:depth.shape[0] // 2 + 10,
                   depth.shape[1] // 2 - 10:depth.shape[1] // 2 + 10]
    valid = middle[middle > 0]
    distance = f"{np.median(valid) * depth_scale:.2f} m" if valid.size else "no depth"
    lines = (
        "SPACE: save a set    Q / ESC: finish",
        f"Saved: {index}    Image-center depth: {distance}",
        "Suggested distances: 0.6 m, 1.0 m, 1.5 m; center and off-center",
    )
    for line, y in zip(lines, (511, 543, 574)):
        cv.putText(preview, line, (12, y), cv.FONT_HERSHEY_SIMPLEX,
                   0.65, (255, 255, 255), 1, cv.LINE_AA)
    return preview


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    script_folder = Path(__file__).resolve().parent
    project = script_folder.parents[1] if script_folder.name == "calibration" and script_folder.parent.name == "experiments" else Path.cwd()
    if args.out is None:
        args.out = project / "data" / "camera_calibration" / (
            "alignment_session_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
    args.out = args.out.expanduser().resolve()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Choose an empty output directory.")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    try:
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        ir_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        device = profile.get_device()
        settings = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "serial": device.get_info(rs.camera_info.serial_number),
            "firmware": device.get_info(rs.camera_info.firmware_version),
            "depth_scale_m_per_unit": device.first_depth_sensor().get_depth_scale(),
            "streams": {
                "depth": intrinsics_data(depth_profile),
                "ir_left": intrinsics_data(ir_profile),
                "rgb": intrinsics_data(color_profile),
            },
            "factory_extrinsics": {
                "depth_to_left": extrinsics_data(depth_profile, ir_profile),
                "left_to_rgb": extrinsics_data(ir_profile, color_profile),
                "depth_to_rgb": extrinsics_data(depth_profile, color_profile),
            },
            "image_format": {"depth": "raw_uint16_png", "ir_left": "y8_png", "rgb": "bgr8_png"},
        }
        for camera in ("rgb", "depth", "ir_left", "records"):
            (args.out / camera).mkdir(parents=True, exist_ok=True)
        (args.out / "camera_parameters.json").write_text(
            json.dumps(settings, indent=2, allow_nan=False), encoding="utf-8",
        )
        for _ in range(30):
            pipeline.wait_for_frames()
        print(f"Session: {args.out}")
        print("Press SPACE to save one RGB + depth + left IR set; press Q or ESC to finish.")
        index = 0
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            ir_frame = frames.get_infrared_frame(1)
            if not color_frame or not depth_frame or not ir_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            preview = make_preview(color, depth, index, settings["depth_scale_m_per_unit"])
            cv.imshow("RGB | depth preview", preview)
            key = cv.waitKey(1) & 255
            if key in (ord("q"), 27):
                break
            if key == 32:
                save_capture(args.out, index, color_frame, depth_frame, ir_frame)
                print(f"Saved {index:04d}.png", flush=True)
                index += 1
        print(f"Saved {index} image sets to {args.out}")
    finally:
        pipeline.stop()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
