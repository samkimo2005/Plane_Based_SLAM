#!/usr/bin/env python3
"""Lab Book 9 — batch native-depth point-cloud generation for RealSense D435i.

Works on a single uint16 depth PNG, a depth/ folder, a scene folder, or a
session folder containing multiple scenes. No RealSense camera is required.

Typical run (from D:\\PlaneBasedSLAM):
    python experiments\\plane_detection\\generate_real_scene_pointclouds.py
    python experiments\\plane_detection\\generate_real_scene_pointclouds.py --input "data\\real_scenes\\lab9_capture_...\\wall_corner"

Dependencies: numpy, opencv-python. PLY is written directly in binary format.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def project_root() -> Path:
    # If installed at PROJECT/experiments/plane_detection/..., use that project.
    script = Path(__file__).resolve()
    for folder in [Path.cwd(), *Path.cwd().parents, script.parent, *script.parents]:
        if (folder / 'data' / 'real_scenes').is_dir():
            return folder
    return Path.cwd()


def normalise_path(raw: str) -> Path:
    value = raw.strip().strip('"').strip("'")
    if not value:
        raise ValueError('No input path supplied.')
    p = Path(value).expanduser()
    return (p if p.is_absolute() else Path.cwd() / p).resolve()


def list_depth_images(root: Path) -> list[Path]:
    if root.is_file():
        if root.suffix.lower() != '.png':
            raise ValueError('Single input file must be a .png depth image.')
        # Reject RGB images even if supplied as a PNG.
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f'Input does not exist: {root}')
    if root.name.lower() == 'depth':
        images = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == '.png')
    else:
        # Only named depth folders, avoiding accidental processing of RGB PNGs.
        images = sorted(p for d in root.rglob('*') if d.is_dir() and d.name.lower() == 'depth'
                        for p in d.iterdir() if p.is_file() and p.suffix.lower() == '.png')
    if not images:
        raise ValueError(f'No depth PNG files found beneath {root}. Expected a folder named depth/.')
    return images


def find_parameters(image: Path, override: Path | None) -> Path:
    if override is not None:
        if not override.is_file():
            raise FileNotFoundError(f'Parameter file not found: {override}')
        return override
    for parent in image.parents:
        candidate = parent / 'camera_parameters.json'
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        'camera_parameters.json not found in parents of ' + str(image) +
        '. Provide --params PATH to choose it explicitly.')


def read_calibration(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        meta = json.load(f)
    depth = meta['streams']['depth']
    scale = float(meta['depth_scale_m_per_unit'])
    for name in ('fx', 'fy', 'cx', 'cy'):
        if not np.isfinite(float(depth[name])):
            raise ValueError(f'Nonfinite depth calibration: {name}')
    if float(depth['fx']) <= 0 or float(depth['fy']) <= 0 or not (0 < scale < 1):
        raise ValueError('Invalid focal length or depth scale.')
    coeffs = np.asarray(depth.get('coeffs', [0]*5), dtype=float)
    if np.any(np.abs(coeffs) > 1e-12):
        raise ValueError('Nonzero depth lens distortion is not supported by simple pinhole back-projection. Rectify depth first.')
    return {'width': int(depth['width']), 'height': int(depth['height']),
            'fx': float(depth['fx']), 'fy': float(depth['fy']),
            'cx': float(depth['cx']), 'cy': float(depth['cy']), 'scale': scale}


def reconstruct(depth: np.ndarray, k: dict, min_depth: float, max_depth: float):
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(f'Expected native single-channel uint16 depth PNG, received {depth.dtype} with shape {depth.shape}')
    h, w = depth.shape
    if (w, h) != (k['width'], k['height']):
        raise ValueError(f'Depth image {w}x{h} does not match intrinsics {k["width"]}x{k["height"]}')
    z = depth.astype(np.float32) * np.float32(k['scale'])
    valid_raw = depth > 0
    good = valid_raw & np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
    vv, uu = np.nonzero(good)
    zz = z[vv, uu]
    xx = (uu.astype(np.float32) - np.float32(k['cx'])) * zz / np.float32(k['fx'])
    yy = (vv.astype(np.float32) - np.float32(k['cy'])) * zz / np.float32(k['fy'])
    xyz = np.column_stack((xx, yy, zz)).astype('<f4', copy=False)
    return xyz, int(valid_raw.sum()), int(good.sum())


def write_binary_ply(path: Path, xyz: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment Lab Book 9: native depth optical frame; units metres\n'
              f'element vertex {len(xyz)}\n'
              'property float x\nproperty float y\nproperty float z\nend_header\n')
    with path.open('wb') as f:
        f.write(header.encode('ascii'))
        xyz.astype('<f4', copy=False).tofile(f)


def group_for(image: Path) -> str:
    if image.parent.name.lower() == 'depth':
        return image.parent.parent.name
    return 'single_image'


def unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        alternative = path.with_name(f'{path.stem}_{counter:02d}{path.suffix}')
        if not alternative.exists():
            return alternative
        counter += 1


def run(args: argparse.Namespace) -> int:
    if args.input is None:
        print('Enter the path to a depth PNG, depth folder, scene folder, or full capture session.')
        print('Example: D:\\PlaneBasedSLAM\\data\\real_scenes\\lab9_capture_...\\wall_corner')
        raw = input('Input path: ')
    else:
        raw = args.input
    selected = normalise_path(raw)
    images = list_depth_images(selected)
    params_override = normalise_path(args.params) if args.params else None
    if args.min_depth <= 0 or args.max_depth <= args.min_depth:
        raise ValueError('Require 0 < --min-depth < --max-depth.')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out_root = normalise_path(args.output) if args.output else (
        project_root() / 'experiments' / 'plane_detection' / 'results' / f'pointcloud_generation_{stamp}')
    out_root.mkdir(parents=True, exist_ok=True)

    calibration_cache = {}
    records = []
    print(f'Found {len(images)} depth image(s).')
    print(f'Output: {out_root}')
    failures = []
    for index, image in enumerate(images, start=1):
        try:
            params_file = find_parameters(image, params_override)
            if params_file not in calibration_cache:
                calibration_cache[params_file] = read_calibration(params_file)
            k = calibration_cache[params_file]
            depth = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise ValueError('OpenCV could not read depth PNG.')
            xyz, valid_raw, valid_filtered = reconstruct(depth, k, args.min_depth, args.max_depth)
            if len(xyz) == 0:
                raise ValueError('No points survived depth range filtering.')
            scene = group_for(image)
            output = unique_output_path(out_root / scene / 'native_pointclouds' / f'{image.stem}_depth.ply')
            write_binary_ply(output, xyz)
            record = {'scene': scene, 'frame': image.stem, 'input_depth_png': str(image),
                      'parameters_file': str(params_file), 'output_ply': str(output),
                      'pixels_total': int(depth.size), 'valid_raw_depth_points': valid_raw,
                      'valid_raw_percent': round(100 * valid_raw / depth.size, 4),
                      'points_after_depth_filter': valid_filtered,
                      'retained_percent': round(100 * valid_filtered / depth.size, 4),
                      'min_depth_m': round(float(xyz[:,2].min()), 5),
                      'median_depth_m': round(float(np.median(xyz[:,2])), 5),
                      'max_depth_m': round(float(xyz[:,2].max()), 5)}
            records.append(record)
            print(f'[{index}/{len(images)}] {scene}/{image.name}: '
                  f'{valid_filtered:,} points, {record["retained_percent"]:.2f}% retained')
        except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
            failures.append({'image': str(image), 'error': str(exc)})
            print(f'[{index}/{len(images)}] ERROR {image}: {exc}', file=sys.stderr)
            if args.strict:
                break

    if records:
        csv_path = out_root / 'pointcloud_metrics.csv'
        with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        summary = {
            'experiment': 'Lab Book 9 — real scene native point cloud generation',
            'input': str(selected), 'output': str(out_root),
            'coordinate_frame': 'native depth optical frame', 'xyz_units': 'metres',
            'depth_filter_m': [args.min_depth, args.max_depth],
            'number_successful': len(records), 'number_failed': len(failures),
            'frames': records, 'failures': failures,
        }
        (out_root / 'pointcloud_summary.json').write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'\nDone: {len(records)} successful, {len(failures)} failed. Metrics: {csv_path}')
    else:
        print('No point cloud generated. Check errors above.', file=sys.stderr)
    return 1 if failures or not records else 0


def main():
    parser = argparse.ArgumentParser(description='Generate native depth PLY from a single PNG or entire RealSense scene/session.')
    parser.add_argument('--input', '-i', help='Depth PNG, depth folder, scene folder or capture session; prompts if omitted.')
    parser.add_argument('--params', '-p', help='Override camera_parameters.json path (optional).')
    parser.add_argument('--output', '-o', help='Output results directory (default: project experiments/plane_detection/results/time-stamped).')
    parser.add_argument('--min-depth', type=float, default=0.1, help='Minimum valid depth in metres (default: 0.1).')
    parser.add_argument('--max-depth', type=float, default=5.0, help='Maximum valid depth in metres (default: 5.0).')
    parser.add_argument('--strict', action='store_true', help='Stop at first failed frame rather than skipping it.')
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, FileNotFoundError, EOFError, OSError) as error:
        parser.exit(status=2, message=f'Error: {error}\n')


if __name__ == '__main__':
    sys.exit(main())
