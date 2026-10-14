"""Lab Book 9: batch voxel downsampling of native RealSense depth point clouds.

Run from project root:
  python experiments/pointcloud/downsample_real_scene_pointclouds.py
  python experiments/pointcloud/downsample_real_scene_pointclouds.py --input "path/to/native_pointclouds"

Enter one PLY or a folder of PLYs. Makes 5 mm and 10 mm outputs by default.
Original PLYs are never modified. Requires open3d and numpy.
"""

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open3d as o3d


def get_project_root():
    here = Path(__file__).resolve()
    for folder in [Path.cwd().resolve(), *here.parents]:
        if (folder / 'experiments' / 'pointcloud').is_dir():
            return folder
    return Path.cwd().resolve()


def get_input_files(path):
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != '.ply':
            raise ValueError('Input file must be a .ply point cloud.')
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f'Input path does not exist: {path}')
    files = sorted(p for p in path.glob('*.ply') if p.is_file())
    if not files and (path / 'native_pointclouds').is_dir():
        files = sorted((path / 'native_pointclouds').glob('*.ply'))
    if not files:
        raise FileNotFoundError(f'No PLY files found in: {path} (or native_pointclouds subfolder)')
    return files


def clean_cloud(cloud):
    xyz = np.asarray(cloud.points)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
        raise ValueError('Empty or invalid point cloud')
    mask = np.isfinite(xyz).all(axis=1)
    removed = int(len(xyz) - mask.sum())
    if removed:
        cloud = cloud.select_by_index(np.flatnonzero(mask).tolist())
    if cloud.is_empty():
        raise ValueError('No valid XYZ coordinates')
    return cloud, removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Single PLY or directory containing PLYs')
    parser.add_argument('--out-parent', type=Path, help='Directory in which to create a new results folder')
    parser.add_argument('--voxels-mm', type=float, nargs='+', default=[5.0, 10.0], help='Voxel sizes in mm (default 5 10)')
    args = parser.parse_args()

    if any(not math.isfinite(v) or v <= 0 for v in args.voxels_mm):
        parser.error('Voxel sizes must be finite and positive')
    if len(set(args.voxels_mm)) != len(args.voxels_mm):
        parser.error('Duplicate voxel sizes are not allowed')

    if args.input is None:
        print('Paste the FULL PATH of a native point-cloud folder, or a single .ply file.')
        raw = input('Input path: ').strip().strip('"').strip("'")
        if not raw:
            parser.error('Input path is required')
        source = Path(raw)
    else:
        source = args.input
    files = get_input_files(source)
    root = get_project_root()
    parent = args.out_parent.expanduser().resolve() if args.out_parent else (root / 'experiments' / 'pointcloud' / 'results')
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = parent / f'real_scene_downsampling_{stamp}'
    output.mkdir(exist_ok=False)

    print(f'Found {len(files)} PLY files; output: {output}')
    records = []
    for i, filename in enumerate(files, start=1):
        cloud = o3d.io.read_point_cloud(str(filename))
        cloud, discarded = clean_cloud(cloud)
        original_count = len(cloud.points)
        for voxel_mm in args.voxels_mm:
            voxel_m = voxel_mm / 1000.0
            # The rounded folder label is only for neat display; avoid collisions
            # by including two decimal places if needed.
            label = f'voxel_{voxel_mm:07.2f}mm'.replace('.', 'p')
            out_folder = output / label
            out_folder.mkdir(exist_ok=True)
            start = time.perf_counter()
            reduced = cloud.voxel_down_sample(voxel_size=voxel_m)
            runtime_ms = (time.perf_counter() - start) * 1000.0
            outfile = out_folder / filename.name
            if not o3d.io.write_point_cloud(str(outfile), reduced, write_ascii=False):
                raise OSError(f'Failed to write: {outfile}')
            count = len(reduced.points)
            reduction = 100.0 * (1.0 - count / original_count)
            records.append({
                'input_ply': str(filename), 'frame': filename.stem,
                'voxel_size_mm': voxel_mm, 'original_points': original_count,
                'discarded_invalid_xyz': discarded, 'downsampled_points': count,
                'point_reduction_percent': round(reduction, 5),
                'downsampling_runtime_ms': round(runtime_ms, 4),
                'output_ply': str(outfile),
            })
            print(f'[{i}/{len(files)}] {filename.name} | {voxel_mm:g} mm: '
                  f'{original_count:,} -> {count:,} pts '
                  f'({reduction:.2f}% reduction) | {runtime_ms:.2f} ms')

    csv_file = output / 'downsampling_metrics.csv'
    with csv_file.open('w', newline='', encoding='utf-8-sig') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        'experiment': 'Lab Book 9 - real-scene voxel downsampling',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'input': str(source.expanduser().resolve()),
        'coordinate_frame': 'unchanged from input PLY (native depth optical frame if input was native depth)',
        'units': 'metres', 'voxel_sizes_mm': args.voxels_mm,
        'input_files': len(files), 'output_clouds': len(records),
        'runtime_note': 'Single measurement per cloud/size; downsampling only, excludes PLY I/O',
        'results': records,
    }
    summary_file = output / 'downsampling_summary.json'
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nFinished: {len(files)} input clouds -> {len(records)} output clouds')
    print(f'Metrics: {csv_file}')
    print(f'Summary: {summary_file}')


if __name__ == '__main__':
    main()
