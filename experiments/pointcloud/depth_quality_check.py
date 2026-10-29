"""Depth Quality Check for Intel RealSense raw 16-bit depth PNGs.

Examples:
  python depth_quality_check.py
  python depth_quality_check.py --input "E:\\...\\depth"
  python depth_quality_check.py --input "E:\\...\\depth\\000000.png" --scale 0.001

The script reads the saved images and writes reports; it never modifies inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2 as cv
import numpy as np


RANGES = [(0.0, 1.0, "0-1m"), (1.0, 3.0, "1-3m"),
          (3.0, 5.0, "3-5m"), (5.0, 8.0, "5-8m"),
          (8.0, float("inf"), "8m+")]


def load_depth(path: Path) -> np.ndarray:
    # np.fromfile / cv.imdecode also support non-ASCII Windows paths.
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv.imdecode(raw, cv.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Cannot decode PNG")
    if image.ndim != 2 or image.dtype != np.uint16:
        raise ValueError(f"Expected single-channel uint16 raw Depth PNG; got {image.shape} {image.dtype}")
    return image


def find_scale(image_path: Path, override: float | None) -> tuple[float, str]:
    if override is not None:
        return override, "--scale"
    candidates = []
    for parent in (image_path.parent, *list(image_path.parents)[1:5]):
        for name in ("camera_parameters.json", "metadata.json"):
            p = parent / name
            if p.is_file() and p not in candidates:
                candidates.append(p)
    for path in candidates:
        try:
            meta = json.loads(path.read_text(encoding="utf-8-sig"))
            scale = meta.get("depth_scale_m_per_unit")
            if scale is not None and float(scale) > 0:
                return float(scale), str(path)
        except (ValueError, TypeError, OSError, AttributeError):
            pass
    print("  WARNING: no depth scale metadata found; assuming 0.001 m/unit. Verify camera_parameters.json!")
    return 0.001, "ASSUMED 0.001"


def draw_report(depth: np.ndarray, scale: float, row: dict, path: Path, out: Path) -> None:
    """Depth image, missing-pixel map and 0–10 m histogram as one PNG."""
    z = depth.astype(np.float32) * scale
    shown_max = 8.0  # visualization only; does NOT truncate data used for metrics
    grey = np.uint8(np.clip(z / shown_max, 0, 1) * 255)
    colour = cv.applyColorMap(grey, cv.COLORMAP_TURBO)
    colour[depth == 0] = (0, 0, 0)
    colour = cv.resize(colour, (640, 480))

    invalid = np.zeros((*depth.shape, 3), dtype=np.uint8)
    invalid[depth == 0] = (255, 255, 255)
    invalid = cv.resize(invalid, (640, 480), interpolation=cv.INTER_NEAREST)

    hist_img = np.full((330, 1280, 3), 245, dtype=np.uint8)
    values = z[depth > 0]
    bins = np.linspace(0, 10, 81)
    hist, _ = np.histogram(values, bins=bins)
    x0, y0, w, h = 55, 278, 1190, 205
    cv.line(hist_img, (x0, y0), (x0 + w, y0), (30, 30, 30), 2)
    maximum = int(hist.max()) if hist.size else 0
    if maximum:
        points = []
        for i, n in enumerate(hist):
            xx = x0 + round((i + 0.5) / len(hist) * w)
            yy = y0 - round(int(n) / maximum * h)
            points.append([xx, yy])
        cv.polylines(hist_img, [np.asarray(points, dtype=np.int32)], False, (125, 75, 30), 2)
    for m in (0, 1, 3, 5, 8, 10):
        x = x0 + round(m / 10 * w)
        cv.line(hist_img, (x, y0), (x, y0 + 5), (30, 30, 30), 1)
        cv.putText(hist_img, f"{m}m", (max(0, x - 13), 302), cv.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 20), 1)
    cv.putText(hist_img, f"Depth distribution (0-10 m; >10 m not plotted) | valid: {row['valid_percent']}% | median: {row['median_m']} m",
               (35, 33), cv.FONT_HERSHEY_SIMPLEX, 0.67, (30, 30, 30), 2)
    cv.putText(hist_img, f"Distance bins (% of VALID pixels): " + "   ".join(f"{label} {row[label + '_pct_valid']}%" for _, _, label in RANGES),
               (35, 69), cv.FONT_HERSHEY_SIMPLEX, 0.53, (30, 30, 30), 1)

    top = np.hstack([colour, invalid])
    cv.putText(top, "DEPTH: 0-8 m colour scale", (12, 30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv.putText(top, "INVALID (white=missing, black=valid)", (653, 30), cv.FONT_HERSHEY_SIMPLEX, 0.65, (90, 90, 255), 2)
    report = np.vstack([top, hist_img])
    ok, enc = cv.imencode(".png", report)
    if not ok:
        raise RuntimeError("Unable to encode diagnostic PNG")
    enc.tofile(str(out))


def analyse(path: Path, out: Path, scale_arg: float | None) -> dict:
    depth = load_depth(path)
    scale, scale_source = find_scale(path, scale_arg)
    nonzero = depth > 0
    values = depth[nonzero].astype(np.float64) * scale
    count = int(nonzero.sum())
    total = int(depth.size)
    row = {
        "file": str(path.resolve()),
        "width": depth.shape[1], "height": depth.shape[0],
        "depth_scale_m_per_unit": round(scale, 12), "scale_source": scale_source,
        "total_pixels": total, "valid_pixels": count,
        "invalid_pixels": total-count,
        "valid_percent": round(100*count/total, 2),
        "min_m": round(float(values.min()), 3) if count else "",
        "median_m": round(float(np.median(values)), 3) if count else "",
        "max_m": round(float(values.max()), 3) if count else "",
    }
    for lower, upper, label in RANGES:
        in_bin = (values >= lower) & (values < upper)
        n = int(np.count_nonzero(in_bin))
        row[label + "_count"] = n
        row[label + "_pct_valid"] = round(100*n/count, 2) if count else 0.0
    # Percent with any missing sensor return, not a ground-truth error metric.
    filename = path.stem + "_depth_quality.png"
    draw_report(depth, scale, row, path, out/filename)
    print(f"  {path.name}: valid {row['valid_percent']}%, median {row['median_m']} m, >5m {row['5-8m_pct_valid'] + row['8m+_pct_valid']:.2f}% of valid")
    return row


def choose_files(input_path: str | None) -> list[Path]:
    if input_path:
        p = Path(input_path.strip('"').strip("'"))
        if p.is_file():
            return [p]
        if not p.is_dir():
            raise FileNotFoundError(f"Not found: {p}")
        pngs = sorted(p.glob("*.png"))
        if p.name.lower() != "depth":
            pngs = [f for f in pngs if "depth" in f.stem.lower()]
        if not pngs:
            raise ValueError("No depth PNGs found; pass the actual 'depth' directory or select individual depth PNG files.")
        return pngs
    try:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        filenames = filedialog.askopenfilenames(title="Select RAW 16-bit Depth PNG(s)", filetypes=[("PNG files", "*.png")])
        root.destroy()
        return [Path(f) for f in filenames]
    except Exception as exc:
        raise RuntimeError("File picker unavailable. Use --input PATH.") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect recorded RealSense 16-bit depth images; never edits input data")
    parser.add_argument("--input", help="One depth PNG, or folder containing 16-bit depth PNGs. Omit to pick files in a dialog.")
    parser.add_argument("--scale", type=float, help="Depth scale in metres per raw unit (overrides metadata; D435i often ~0.001).")
    parser.add_argument("--output", type=Path, help="Output folder (default: depth_quality_results next to selected image).")
    args = parser.parse_args()
    if args.scale is not None and not (0 < args.scale < 1):
        parser.error("--scale must be positive and less than 1 metre per raw unit")
    files = choose_files(args.input)
    if not files:
        print("No files selected.")
        return 1
    out = args.output or (files[0].parent / "depth_quality_results")
    out.mkdir(parents=True, exist_ok=True)
    rows=[]
    for file in files:
        try:
            rows.append(analyse(file, out, args.scale))
        except (ValueError, OSError, RuntimeError, cv.error) as exc:
            print(f"  SKIPPED {file.name}: {exc}")
    if not rows:
        print("No usable 16-bit depth PNGs were found.")
        return 2
    csv_path = out / "depth_quality_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nAnalysed {len(rows)}/{len(files)} frames. Reports: {out.resolve()}")
    print(f"Metrics: {csv_path}")
    print("Note: Valid pixels measure completeness, NOT depth accuracy or plane quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
