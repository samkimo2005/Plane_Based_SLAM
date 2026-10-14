#!/usr/bin/env python3
"""Lab Book 9: native, unaligned RealSense D435i RGB-D acquisition.

Windows: pip install pyrealsense2 opencv-python numpy
Run from project root:
    python experiments/data_acquisition/capture_rgbd_scenes.py --scene wall_corner --lighting normal

Controls (click the preview window first): S = single frame; R = start/stop
continuous capture; Q or Esc = exit. Each run creates a new session. Raw depth
is NOT aligned to RGB: deproject depth with the saved DEPTH intrinsics.
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2 as cv
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

FIELDS = [
    "image_id", "mode", "rgb_file", "depth_file", "host_utc_iso",
    "host_monotonic_ns", "rgb_timestamp_ms", "depth_timestamp_ms",
    "rgb_frame_number", "depth_frame_number", "rgb_timestamp_domain",
    "depth_timestamp_domain", "rgb_depth_timestamp_delta_ms",
    "depth_valid_percent", "depth_median_valid_m", "rgb_png_bytes",
    "depth_png_bytes",
]


def safe_name(value: str) -> str:
    import re
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    if not text:
        raise ValueError("Scene and lighting names must contain letters or numbers.")
    return text[:70]


def auto_project_root() -> Path:
    here = Path(__file__).resolve()
    # Installed at D:/PlaneBasedSLAM/experiments/data_acquisition/...
    if here.parent.name == "data_acquisition" and here.parent.parent.name == "experiments":
        return here.parent.parent.parent
    return Path.cwd()


def intrinsics_dict(profile):
    i = profile.as_video_stream_profile().get_intrinsics()
    return {
        "width": int(i.width), "height": int(i.height),
        "fx": float(i.fx), "fy": float(i.fy), "cx": float(i.ppx), "cy": float(i.ppy),
        "distortion_model": str(i.model),
        "coeffs": [float(x) for x in i.coeffs],
    }


def extrinsics_dict(source, target):
    ex = source.get_extrinsics_to(target)
    # rs extrinsics rotation is a column-major flattened 3x3 matrix.
    rotation = np.asarray(ex.rotation, dtype=float).reshape((3, 3), order="F")
    return {
        "rotation_matrix": rotation.tolist(),
        "translation_m": [float(t) for t in ex.translation],
        "convention": "p_target = R @ p_source + t",
    }


def atomic_json(path: Path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    temp.replace(path)


def camera_metadata(profile, device, args):
    rgb_prof = profile.get_stream(rs.stream.color)
    depth_prof = profile.get_stream(rs.stream.depth)
    sensor = profile.get_device().first_depth_sensor()
    result = {
        "experiment": "Lab Book 9 - real-scene RGB-D acquisition",
        "scene": args.scene,
        "lighting": args.lighting,
        "scene_notes": args.notes,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": {},
        "depth_scale_m_per_unit": float(sensor.get_depth_scale()),
        "streams": {
            "depth": intrinsics_dict(depth_prof),
            "rgb": intrinsics_dict(rgb_prof),
        },
        "factory_extrinsics": {
            "depth_to_rgb": extrinsics_dict(depth_prof, rgb_prof),
        },
        "image_format": {
            "depth": "raw_uint16_png; NOT aligned to RGB; native depth optical frame",
            "rgb": "bgr8_png",
        },
        "settings": {
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "warmup_frames": args.warmup_frames,
            "record_every_n_frames": args.every,
            "automatic_exposure": "sensor SDK defaults; not manually changed",
        },
        "timestamp_notes": (
            "Per-stream timestamps and their SDK domains are stored in frames.csv. "
            "Host monotonic time is captured when the frameset is received; it is "
            "not a hardware-exposure timestamp. RGB/depth difference is meaningful "
            "only when both SDK timestamp domains agree."
        ),
    }
    for key in (rs.camera_info.name, rs.camera_info.serial_number,
                rs.camera_info.firmware_version, rs.camera_info.usb_type_descriptor):
        try:
            result["device"][str(key).split(".")[-1]] = device.get_info(key)
        except Exception:
            pass
    return result


class Writer:
    def __init__(self, session: Path, max_queue: int):
        self.session = session
        self.rgb_dir = session / "rgb"
        self.depth_dir = session / "depth"
        self.rgb_dir.mkdir(parents=True, exist_ok=False)
        self.depth_dir.mkdir(parents=True, exist_ok=False)
        self.q = queue.Queue(maxsize=max_queue)
        self.error = None
        self.count = 0
        self.dropped = 0
        self.stats = {"snapshot": 0, "sequence": 0}
        self.thread = threading.Thread(target=self._work, daemon=False)
        self.thread.start()

    def enqueue(self, record, rgb, depth):
        if self.error:
            raise RuntimeError("Image writer failed") from self.error
        try:
            self.q.put_nowait((record, rgb.copy(), depth.copy()))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _work(self):
        try:
            with (self.session / "frames.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                while True:
                    item = self.q.get()
                    try:
                        if item is None:
                            break
                        rec, rgb, depth = item
                        name = f"{self.count:06d}.png"
                        rgb_path = self.rgb_dir / name
                        depth_path = self.depth_dir / name
                        if not cv.imwrite(str(rgb_path), rgb):
                            raise IOError(f"Cannot save {rgb_path}")
                        if not cv.imwrite(str(depth_path), depth):
                            raise IOError(f"Cannot save {depth_path}")
                        rec["image_id"] = f"{self.count:06d}"
                        rec["rgb_file"] = f"rgb/{name}"
                        rec["depth_file"] = f"depth/{name}"
                        rec["rgb_png_bytes"] = rgb_path.stat().st_size
                        rec["depth_png_bytes"] = depth_path.stat().st_size
                        writer.writerow(rec)
                        f.flush()
                        self.count += 1
                        self.stats[rec["mode"]] += 1
                    finally:
                        self.q.task_done()
        except Exception as exc:
            self.error = exc
            # Drain the queue so closing cannot deadlock after a disk/write failure.
            while True:
                try:
                    self.q.get_nowait()
                    self.q.task_done()
                except queue.Empty:
                    break

    def close(self):
        if self.thread.is_alive():
            while True:
                if self.error or not self.thread.is_alive():
                    break
                try:
                    self.q.put(None, timeout=0.2)
                    break
                except queue.Full:
                    continue
            self.thread.join()
        if self.error:
            raise RuntimeError(f"Failed to save frames: {self.error}") from self.error


def frame_record(rgb_frame, depth_frame, depth_raw, scale, mode):
    valid = depth_raw > 0
    depths = depth_raw[valid]
    domain_rgb = str(rgb_frame.get_frame_timestamp_domain())
    domain_depth = str(depth_frame.get_frame_timestamp_domain())
    return {
        "image_id": "", "mode": mode, "rgb_file": "", "depth_file": "",
        "host_utc_iso": datetime.now(timezone.utc).isoformat(),
        "host_monotonic_ns": time.monotonic_ns(),
        "rgb_timestamp_ms": float(rgb_frame.get_timestamp()),
        "depth_timestamp_ms": float(depth_frame.get_timestamp()),
        "rgb_frame_number": int(rgb_frame.get_frame_number()),
        "depth_frame_number": int(depth_frame.get_frame_number()),
        "rgb_timestamp_domain": domain_rgb,
        "depth_timestamp_domain": domain_depth,
        "rgb_depth_timestamp_delta_ms": (
            round(float(rgb_frame.get_timestamp()) - float(depth_frame.get_timestamp()), 4)
            if domain_rgb == domain_depth else ""
        ),
        "depth_valid_percent": round(float(np.mean(valid) * 100), 4),
        "depth_median_valid_m": (
            round(float(np.median(depths) * scale), 5) if len(depths) else ""
        ),
        "rgb_png_bytes": "", "depth_png_bytes": "",
    }


def depth_visual(depth):
    # Only for display; original uint16 is always saved without colourising.
    vis = cv.convertScaleAbs(depth, alpha=255.0 / 2500.0)
    colour = cv.applyColorMap(vis, cv.COLORMAP_TURBO)
    colour[depth == 0] = (0, 0, 0)
    return colour


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True, help="wall_corner, desk, complex_indoor, etc.")
    p.add_argument("--lighting", default="normal", help="normal, dim, side_light, etc.")
    p.add_argument("--notes", default="", help="Optional scene notes")
    p.add_argument("--output-root", type=Path, default=None,
                   help="Defaults to <project>/data/real_scenes")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--warmup-frames", type=int, default=60)
    p.add_argument("--every", type=int, default=1,
                   help="Store every Nth frameset during continuous recording")
    p.add_argument("--queue-size", type=int, default=100)
    p.add_argument("--min-free-gb", type=float, default=1.0)
    args = p.parse_args()
    args.scene = safe_name(args.scene)
    args.lighting = safe_name(args.lighting)
    if args.every < 1 or args.queue_size < 1 or args.fps < 1:
        p.error("--every, --queue-size and --fps must be positive")
    if rs is None:
        sys.exit("pyrealsense2 is missing. Activate your project venv and run: pip install pyrealsense2")

    root = (args.output_root or (auto_project_root() / "data" / "real_scenes")).resolve()
    scene_dir = root / f"scene_{args.scene}" / f"lighting_{args.lighting}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(scene_dir).free / 1e9
    if free_gb < args.min_free_gb:
        sys.exit(f"Only {free_gb:.2f} GB free at {scene_dir}; free space before capturing.")
    session = scene_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = None
    writer = None
    recording = False
    frame_index = 0
    attempted = 0
    status = "Press S to save a frame; R to record; Q to quit"
    try:
        profile = pipeline.start(cfg)
        device = profile.get_device()
        meta = camera_metadata(profile, device, args)
        print("Camera:", meta["device"])
        print("Depth scale:", meta["depth_scale_m_per_unit"], "m/unit")
        print(f"Warming up for {args.warmup_frames} frames...")
        for _ in range(max(0, args.warmup_frames)):
            pipeline.wait_for_frames(timeout_ms=10000)

        session.mkdir(parents=True, exist_ok=False)
        atomic_json(session / "camera_parameters.json", meta)
        writer = Writer(session, args.queue_size)
        print("Saving to:", session)
        cv.namedWindow("Lab Book 9 - RGB | Depth", cv.WINDOW_NORMAL)
        while True:
            if writer.error:
                raise RuntimeError("Disk/image writing failed") from writer.error
            frames = pipeline.wait_for_frames(timeout_ms=10000)
            rgb_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not rgb_frame or not depth_frame:
                continue
            rgb = np.asanyarray(rgb_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            if depth.dtype != np.uint16:
                raise RuntimeError(f"Expected raw uint16 depth; got {depth.dtype}")
            if rgb.shape != (args.height, args.width, 3) or depth.shape != (args.height, args.width):
                raise RuntimeError(f"Unexpected RGB/depth dimensions: {rgb.shape}, {depth.shape}")
            valid = round(float(np.count_nonzero(depth)) / depth.size * 100, 1)
            depth_col = depth_visual(depth)
            preview = np.hstack((rgb, depth_col))
            cv.putText(preview, f"Scene: {args.scene}  Light: {args.lighting}",
                       (15, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv.putText(preview, f"Depth valid: {valid:.1f}%  Saved: {writer.count}  Queue: {writer.q.qsize()}",
                       (15, 51), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
            cv.putText(preview, ("REC  " if recording else "IDLE ") + status,
                       (15, 78), cv.FONT_HERSHEY_SIMPLEX, 0.50,
                       (40,40,255) if recording else (255,255,255), 2)
            cv.imshow("Lab Book 9 - RGB | Depth", preview)

            if recording and frame_index % args.every == 0:
                attempted += 1
                rec = frame_record(rgb_frame, depth_frame, depth, meta["depth_scale_m_per_unit"], "sequence")
                if not writer.enqueue(rec, rgb, depth):
                    status = f"WARNING: writer queue full, dropped {writer.dropped}"
            frame_index += 1
            key = cv.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                recording = not recording
                status = "Recording started" if recording else "Recording stopped"
                print(status)
            if key == ord("s"):
                rec = frame_record(rgb_frame, depth_frame, depth, meta["depth_scale_m_per_unit"], "snapshot")
                attempted += 1
                status = "Snapshot queued" if writer.enqueue(rec, rgb, depth) else "WARNING: queue full; snapshot NOT saved"
                print(status)
    except KeyboardInterrupt:
        print("Interrupted, closing safely...")
    finally:
        if profile is not None:
            pipeline.stop()
        cv.destroyAllWindows()
        if writer is not None:
            try:
                writer.close()
            finally:
                atomic_json(session / "capture_summary.json", {
                    "scene": args.scene, "lighting": args.lighting,
                    "saved_frames": writer.count,
                    "saved_snapshots": writer.stats["snapshot"],
                    "saved_sequence_frames": writer.stats["sequence"],
                    "queue_full_dropped_frames": writer.dropped,
                    "total_save_attempts": attempted,
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "data_integrity_warning": ("Some frames could not be queued" if writer.dropped else None),
                })
                print(f"Saved {writer.count} frames; dropped {writer.dropped}. Folder: {session}")


if __name__ == "__main__":
    main()
