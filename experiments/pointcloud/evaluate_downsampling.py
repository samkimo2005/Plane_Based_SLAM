"""Lab Book 8: evaluate existing voxel-grid downsampling outputs.

Run from PlaneBasedSLAM root:
    python experiments/pointcloud/evaluate_downsampling.py
Or specify an existing downsampling result directory:
    python experiments/pointcloud/evaluate_downsampling.py --run-dir ".../downsampling_YYYY..."

All XYZ units are metres. Errors are relative to the ORIGINAL depth cloud,
NOT independent geometric ground truth. No new camera calibration is performed.
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d


def project_root():
    script = Path(__file__).resolve()
    for base in [Path.cwd().resolve(), *script.parents]:
        if (base / 'experiments' / 'pointcloud').is_dir():
            return base
    raise FileNotFoundError('Could not find project root (experiments/pointcloud).')


def load_cloud(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    cloud = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(cloud.points)
    if pts.ndim != 2 or pts.shape[1] != 3 or not len(pts):
        raise ValueError(f'No 3D points in {path}')
    finite = np.isfinite(pts).all(axis=1)
    if not finite.all():
        raise ValueError(f'Non-finite XYZ detected in {path}')
    return cloud


def fit_plane_pca(points):
    if len(points) < 3:
        return None
    centre = points.mean(axis=0)
    A = points - centre
    _, singular, vt = np.linalg.svd(A, full_matrices=False)
    if singular[1] < 1e-10:
        return None
    n = vt[-1]
    n /= np.linalg.norm(n)
    return n, centre


def plane_metrics(points, reference_normal=None, reference_point=None):
    fit = fit_plane_pca(points)
    if fit is None:
        return None
    n, centre = fit
    dist = (points - centre) @ n
    out = {
        'rms_mm': float(np.sqrt(np.mean(dist ** 2)) * 1000),
        'median_abs_mm': float(np.median(np.abs(dist)) * 1000),
        'p90_abs_mm': float(np.percentile(np.abs(dist), 90) * 1000),
    }
    if reference_normal is not None:
        cos = np.clip(np.abs(np.dot(n, reference_normal)), 0, 1)
        out['normal_difference_deg'] = float(np.degrees(np.arccos(cos)))
        # Evaluate downsampled points against the ORIGINAL reference plane as well.
        ref_dist = (points - reference_point) @ reference_normal
        out['reference_plane_rms_mm'] = float(np.sqrt(np.mean(ref_dist ** 2)) * 1000)
    return out


def original_dominant_plane(original, threshold_m, iterations, minimum):
    # Fix the random seed where supported, for reproducibility.
    if hasattr(o3d.utility, 'random') and hasattr(o3d.utility.random, 'seed'):
        o3d.utility.random.seed(17)
    coeff, indices = original.segment_plane(
        distance_threshold=threshold_m,
        ransac_n=3,
        num_iterations=iterations,
    )
    if len(indices) < minimum:
        raise ValueError(
            f'Dominant-plane RANSAC identified only {len(indices)} points '
            f'(minimum {minimum}). Try --plane-threshold-mm or inspect the scene.'
        )
    pts = np.asarray(original.points)[np.asarray(indices, dtype=int)]
    ref = fit_plane_pca(pts)
    if ref is None:
        raise ValueError('Dominant-plane points do not support a 2D plane fit.')
    return pts, ref, coeff, len(indices)


def stats_nn(original, sampled):
    # Distance from EVERY original point to nearest voxel representative.
    # This measures approximation of the original sampled surface, NOT accuracy.
    d = np.asarray(original.compute_point_cloud_distance(sampled), dtype=np.float64)
    if not len(d) or not np.isfinite(d).all():
        raise ValueError('Non-finite nearest-neighbour distances.')
    return {
        'nn_median_mm': float(np.median(d) * 1000),
        'nn_p90_mm': float(np.percentile(d, 90) * 1000),
        'nn_rms_mm': float(np.sqrt(np.mean(d * d)) * 1000),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, help='Existing downsampling_... directory; defaults to latest')
    p.add_argument('--out-parent', type=Path, help='Results parent; default experiments/pointcloud/results')
    p.add_argument('--plane-threshold-mm', type=float, default=8.0,
                   help='RANSAC and fixed-plane ROI half-thickness (default 8 mm)')
    p.add_argument('--ransac-iterations', type=int, default=1000)
    p.add_argument('--min-plane-points', type=int, default=500)
    args = p.parse_args()
    if not np.isfinite(args.plane_threshold_mm) or args.plane_threshold_mm <= 0:
        p.error('--plane-threshold-mm must be positive.')
    if args.ransac_iterations < 1 or args.min_plane_points < 3:
        p.error('RANSAC iterations must be >=1; minimum plane points >=3.')

    root = project_root()
    result_root = root / 'experiments' / 'pointcloud' / 'results'
    if args.run_dir:
        run_dir = args.run_dir.expanduser().resolve()
    else:
        candidates = sorted(x for x in result_root.glob('downsampling_*')
                            if (x / 'summary.json').is_file())
        if not candidates:
            raise FileNotFoundError(f'No downsampling results under {result_root}')
        run_dir = candidates[-1]
    meta_path = run_dir / 'summary.json'
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding='utf-8-sig'))
    original_path = Path(meta['input_ply'])
    if not original_path.is_file():
        # Recover if project was relocated: identify by basename in sibling depth folders.
        matches = sorted(result_root.glob(f'depth_pointcloud_*/{original_path.name}'))
        if len(matches) != 1:
            raise FileNotFoundError(f'Original PLY is missing or ambiguous: {original_path}')
        original_path = matches[0]
    original = load_cloud(original_path)
    original_count = len(original.points)
    if original_count != meta['original_points']:
        raise ValueError('Original PLY point count differs from saved summary.')

    threshold_m = args.plane_threshold_mm / 1000
    t0 = time.perf_counter()
    ref_points, (ref_normal, ref_centre), coeff, n_ref = original_dominant_plane(
        original, threshold_m, args.ransac_iterations, args.min_plane_points
    )
    ref_stats = plane_metrics(ref_points, ref_normal, ref_centre)
    print(f'Original cloud: {original_count:,} points')
    print(f'Dominant original plane: {n_ref:,} inliers; original plane RMS {ref_stats["rms_mm"]:.3f} mm')
    print('NOTE: Automatically selected DOMINANT plane; it may be wall/background, NOT checkerboard.')

    rows = []
    for item in meta['results']:
        # Prefer the filename recorded for that actual downsampling run.
        candidate = Path(item['output_ply'])
        if not candidate.is_file():
            candidate = run_dir / candidate.name
        sampled = load_cloud(candidate)
        if len(sampled.points) != item['downsampled_points']:
            raise ValueError(f'Count mismatch for {candidate}')
        start = time.perf_counter()
        nn = stats_nn(original, sampled)
        # Consistent spatial selection: same fixed plane and same threshold for ALL sizes.
        pts = np.asarray(sampled.points)
        plane_dist = np.abs((pts - ref_centre) @ ref_normal)
        selected = pts[plane_dist <= threshold_m]
        metrics = plane_metrics(selected, ref_normal, ref_centre)
        if metrics is None:
            print(f'WARNING: {item["voxel_size_mm"]} mm: too few plane-region points')
        row = {
            'voxel_size_mm': item['voxel_size_mm'],
            'original_points': original_count,
            'downsampled_points': len(sampled.points),
            'point_reduction_percent': item['point_reduction_percent'],
            'downsampling_runtime_ms_single_run': item['runtime_ms'],
            **{k: round(v, 5) for k, v in nn.items()},
            'fixed_plane_region_points': len(selected),
            'plane_region_fraction_percent': round(100 * len(selected) / len(pts), 4),
            'plane_fit_rms_mm': round(metrics['rms_mm'], 5) if metrics else None,
            'plane_fit_p90_abs_mm': round(metrics['p90_abs_mm'], 5) if metrics else None,
            'normal_difference_deg': round(metrics['normal_difference_deg'], 5) if metrics else None,
            'original_reference_plane_rms_mm': round(metrics['reference_plane_rms_mm'], 5) if metrics else None,
            'evaluation_runtime_ms': round((time.perf_counter() - start) * 1000, 3),
            'source_ply': str(candidate),
        }
        rows.append(row)
        print(f'{item["voxel_size_mm"]:>5g} mm: {len(pts):>7,} pts | '
              f'NN median={nn["nn_median_mm"]:.2f}, P90={nn["nn_p90_mm"]:.2f}, '
              f'RMS={nn["nn_rms_mm"]:.2f} mm | '
              f'normal={row["normal_difference_deg"]} deg')

    parent = args.out_parent.expanduser().resolve() if args.out_parent else result_root
    parent.mkdir(parents=True, exist_ok=True)
    out = parent / ('downsampling_evaluation_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    csv_path = out / 'evaluation_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        'experiment': 'Lab Book 8 - point-cloud downsampling geometric retention evaluation',
        'original_ply': str(original_path),
        'downsampling_run': str(run_dir),
        'geometry_units': 'metres',
        'metrics_units': 'millimetres and degrees unless stated',
        'plane_region_method': 'Original dominant RANSAC plane; one reference PCA plane; select each downsampled cloud by distance <= threshold to that same reference',
        'plane_warning': 'Dominant plane is not guaranteed to be the checkerboard. ROI is reference-plane band, not independently annotated ground truth. The reference-plane band can contain unrelated coplanar structures.',
        'plane_threshold_mm': args.plane_threshold_mm,
        'ransac_iterations': args.ransac_iterations,
        'reference_plane_inliers': n_ref,
        'reference_plane_coefficients_ransac': list(map(float, coeff)),
        'reference_plane_normal_pca': ref_normal.tolist(),
        'reference_plane_centre_pca_m': ref_centre.tolist(),
        'reference_original_plane_rms_mm': ref_stats['rms_mm'],
        'nearest_neighbour_direction': 'original -> downsampled',
        'accuracy_warning': 'All errors compare to the original depth cloud/reference derived from it; NOT independent absolute geometry accuracy or RGB extrinsic accuracy.',
        'results': rows,
    }
    (out / 'evaluation_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f'\nResults saved to: {out}')
    print('Files: evaluation_metrics.csv, evaluation_summary.json')
    print('No camera images or calibration parameters were changed.')


if __name__ == '__main__':
    main()
