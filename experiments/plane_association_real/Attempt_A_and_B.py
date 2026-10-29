#!/usr/bin/env python3
"""Lab Note 11 - Attempt A: two-frame plane association, Run I (Iterative RANSAC).

One command integrates:
  (1) Load plane_detection_real outputs (plane_parameters.csv, labels.npz, PLY)
  (2) Prepare per-plane descriptors (normal, offset, centroid, extent)
  (3) Estimate source->target relative pose by FPFH RANSAC + point-to-plane ICP
  (4) Candidate gating -> geometric similarity -> Hungarian assignment
  (5) Write plane matches, candidates, registration, descriptors, visualisation.

All geometric quantities are in metres. Planes are in each frame's native
DEPTH optical coordinate system, not the RGB coordinate system. This is an
initial integration BASELINE; without manual correspondence ground truth it
must not be presented as matching accuracy or SLAM pose accuracy.

Example from E:\\PlaneBasedSLAM (PowerShell, single line recommended):
  python experiments\\plane_association\\attemptA_two_frame_plane_association.py --run "E:\\PlaneBasedSLAM\\experiments\\plane_detection_real\\results\\runH_maxplanes20_test" --cloud-root "E:\\PlaneBasedSLAM\\experiments\\pointcloud\\results\\real_scene_downsampling_20261023_151330_374491" --expected-batch real_scene_downsampling_20261023_151330_374491 --source 000000 --target 000001 --voxel-mm 5

Requirements: numpy, scipy, open3d.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from xml.sax.saxutils import escape

import numpy as np
from scipy.optimize import linear_sum_assignment

METHOD = 'iterative_ransac'


def rows_from_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing required file: {path}')
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def basename(path: str) -> str:
    return PureWindowsPath(path.replace('/', '\\')).name


def winpath(path: str) -> PureWindowsPath:
    return PureWindowsPath(path.replace('/', '\\'))


def frame_of(path: str) -> str:
    match = re.fullmatch(r'(\d{6})(?:_depth)?\.ply', basename(path))
    if match is None:
        raise ValueError(f'Unrecognised input PLY filename: {path}')
    return match.group(1)


def resolve_inputs(run: Path, cloud_root: Path, source: str, target: str,
                   voxel_mm: int, expected_batch: str | None):
    metrics = rows_from_csv(run / 'detection_metrics.csv')
    params = rows_from_csv(run / 'plane_parameters.csv')
    group = f'voxel_{voxel_mm:04d}p00mm'
    selected = {}
    for row in metrics:
        if row.get('algorithm') != METHOD or row.get('scene_group') != group:
            continue
        frame = frame_of(row['input_ply'])
        if frame not in (source, target):
            continue
        if frame in selected:
            raise ValueError(f'Duplicate {METHOD} metric rows for frame {frame}')
        batch = winpath(row['input_ply']).parent.parent.name
        if expected_batch and batch != expected_batch:
            raise ValueError(f'Wrong dataset for frame {frame}: expected {expected_batch}, found {batch}.')
        path = cloud_root / group / basename(row['input_ply'])
        if not path.is_file():
            # Support working directly in Windows, where the CSV's original path exists.
            original = Path(row['input_ply'])
            if original.is_file():
                path = original
            else:
                raise FileNotFoundError(f'Missing point cloud: {path}\n'
                                        'Check --cloud-root points to the Run I downsampling batch.')
        label_path = run / group / METHOD / f'{path.stem}_{METHOD}_labels.npz'
        if not label_path.is_file():
            original_label = Path(row.get('labels_npz', ''))
            if original_label.is_file():
                label_path = original_label
            else:
                raise FileNotFoundError(f'Missing segmentation labels: {label_path}\n'
                                        'You need the original Run I detection results, not only the CSVs.')
        selected[frame] = (row, path, label_path)
    missing = [frame for frame in (source, target) if frame not in selected]
    if missing:
        raise ValueError(f'Missing {METHOD}, {group} rows for frames: {missing}')
    params_by_frame = {source: [], target: []}
    src_ply_keys = {frame: selected[frame][0]['input_ply'] for frame in (source, target)}
    for p in params:
        if p.get('algorithm') != METHOD:
            continue
        for frame, src_ply in src_ply_keys.items():
            if p['input_ply'] == src_ply:
                params_by_frame[frame].append(p)
                break
    return selected, params_by_frame


def normalised_plane(row):
    n = np.array([float(row['nx']), float(row['ny']), float(row['nz'])], dtype=float)
    d = float(row['d_m'])
    k = float(np.linalg.norm(n))
    if not np.isfinite(k) or k < 1e-10 or not np.isfinite(d):
        raise ValueError(f'Bad plane normal/offset: {row}')
    n /= k
    d /= k
    # Sign canonicalised only for stable output: matching remains sign invariant.
    if n[np.argmax(np.abs(n))] < 0:
        n, d = -n, -d
    return n, d


def extract_frame(o3d, frame: str, entry, param_rows):
    metric, ply, label_file = entry
    pcd = o3d.io.read_point_cloud(str(ply))
    raw = np.asarray(pcd.points, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f'Bad PLY points for {frame}: shape={raw.shape}')
    points = raw[np.isfinite(raw).all(axis=1)]
    if len(points) != int(metric['valid_points']):
        raise ValueError(f'Point count mismatch for {frame}; wrong PLY?')
    with np.load(label_file, allow_pickle=False) as z:
        if 'labels' not in z.files:
            raise KeyError(f'Missing labels in {label_file}')
        labels = np.asarray(z['labels'])
    if labels.ndim != 1 or len(labels) != len(points) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f'Labels are not a valid integer vector aligned with {frame} PLY')
    count = int(metric['detected_planes'])
    if np.any(labels < -1) or np.any(labels >= count):
        raise ValueError(f'Out-of-range labels in {frame}')
    if int(np.count_nonzero(labels >= 0)) != int(metric['assigned_points']):
        raise ValueError(f'Assigned-point count mismatch for {frame}')
    by_id = {int(p['plane_id']): p for p in param_rows}
    if len(by_id) != len(param_rows) or set(by_id) != set(range(count)):
        raise ValueError(f'Plane parameters for frame {frame} disagree with metrics (expected {count}).')
    planes = []
    for pid in range(count):
        p = by_id[pid]
        support = points[labels == pid]
        if len(support) != int(p['inlier_count']) or len(support) < 3:
            raise ValueError(f'Plane {frame}:{pid} support count disagrees with CSV.')
        n, d = normalised_plane(p)
        centre = support.mean(axis=0)
        plane_centre = centre - (n @ centre + d) * n
        # PCA projected extent; bounding rectangle is not true physical plane area.
        helper = np.eye(3)[np.argmin(np.abs(n))]
        u = np.cross(helper, n)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        uv = np.column_stack(((support - centre) @ u, (support - centre) @ v))
        _, eigvec = np.linalg.eigh((uv.T @ uv) / len(uv))
        width, height = sorted(np.ptp(uv @ eigvec, axis=0), reverse=True)
        error_mm = np.abs(support @ n + d) * 1000.0
        rms_mm = float(np.sqrt(np.mean(error_mm ** 2)))
        p90_mm = float(np.percentile(error_mm, 90))
        if abs(rms_mm - float(p['fit_rms_mm'])) > 0.2 or abs(p90_mm - float(p['fit_p90_mm'])) > 0.2:
            raise ValueError(f'Plane {frame}:{pid} RMS/P90 mismatch; labels and PLY may be from different runs.')
        planes.append({'id': pid, 'n': n, 'd': float(d), 'c': plane_centre,
                       'support': len(support), 'extent': [float(width), float(height)],
                       'area_bbox': float(width * height), 'rms_mm': rms_mm,
                       'p90_mm': p90_mm})
    return points, planes


def estimate_relative_pose(o3d, source_points, target_points, args):
    """Return T mapping source camera XYZ to target camera XYZ."""
    reg = o3d.pipelines.registration
    voxel = args.registration_voxel
    def make_features(points):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud = cloud.voxel_down_sample(voxel)
        if len(cloud.points) < 50:
            raise ValueError('Not enough downsampled points for FPFH registration.')
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2.5, max_nn=40))
        f = reg.compute_fpfh_feature(
            cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*5.0, max_nn=100))
        return cloud, f
    t0 = time.perf_counter()
    src, src_fpfh = make_features(source_points)
    tgt, tgt_fpfh = make_features(target_points)
    result = reg.registration_ransac_based_on_feature_matching(
        src, tgt, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=args.ransac_distance,
        estimation_method=reg.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[reg.CorrespondenceCheckerBasedOnEdgeLength(0.90),
                  reg.CorrespondenceCheckerBasedOnDistance(args.ransac_distance)],
        criteria=reg.RANSACConvergenceCriteria(100000, 0.999))
    if result.fitness < args.min_registration_fitness:
        raise ValueError('Initial FPFH registration is unreliable: '
                         f'fitness={result.fitness:.3f} < {args.min_registration_fitness:.3f}. '
                         'Review overlap and try a manual initial pose if available.')
    # Align the already downsampled clouds using a tighter geometric objective.
    refined = reg.registration_icp(
        src, tgt, args.icp_distance, np.asarray(result.transformation),
        reg.TransformationEstimationPointToPlane(),
        reg.ICPConvergenceCriteria(max_iteration=60))
    if refined.fitness < args.min_registration_fitness:
        raise ValueError('ICP registration is unreliable: '
                         f'fitness={refined.fitness:.3f} < {args.min_registration_fitness:.3f}.')
    T = np.asarray(refined.transformation, dtype=float)
    return T, {'method': 'FPFH_RANSAC_then_point_to_plane_ICP',
               'registration_voxel_m': voxel,
               'ransac_fitness': float(result.fitness), 'ransac_rmse_m': float(result.inlier_rmse),
               'icp_fitness': float(refined.fitness), 'icp_rmse_m': float(refined.inlier_rmse),
               'registration_runtime_s': time.perf_counter()-t0}


def validate_pose(T):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError('Relative pose must be a finite 4x4 matrix')
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError('Last row of 4x4 pose must be [0,0,0,1]')
    if not np.allclose(T[:3,:3].T @ T[:3,:3], np.eye(3), atol=0.02) or abs(np.linalg.det(T[:3,:3])-1)>0.02:
        raise ValueError('Pose rotation must be approximately orthonormal with determinant +1')
    return T


def from_pose_file(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    T = data['T_source_to_target'] if isinstance(data, dict) else data
    return validate_pose(T)


def transformed_plane(plane, T):
    R, t = T[:3,:3], T[:3,3]
    n = R @ plane['n']
    d = plane['d'] - float(n @ t)
    c = R @ plane['c'] + t
    return n, d, c


def compare_planes(source_plane, target_plane, T, args):
    ns, ds, cs = transformed_plane(source_plane, T)
    nt, dt, ct = target_plane['n'], target_plane['d'], target_plane['c']
    sign = 1 if float(ns @ nt) >= 0 else -1
    angle = math.degrees(math.acos(float(np.clip(abs(ns @ nt), 0, 1))))
    offset = abs(ds - sign * dt)
    # Lateral centroid distance in target plane; more useful than 3-D distance
    # when a surface is partially visible from different viewpoints.
    displacement = cs - ct
    lateral = float(np.linalg.norm(displacement - (displacement @ nt) * nt))
    areas = [max(source_plane['area_bbox'], 1e-6), max(target_plane['area_bbox'], 1e-6)]
    ratio = max(areas) / min(areas)
    tests = [angle <= args.max_angle_deg,
             offset <= args.max_offset_m,
             lateral <= args.max_centroid_m,
             ratio <= args.max_area_ratio]
    reasons = [name for name, ok in zip(('angle', 'offset', 'centroid', 'area_ratio'), tests) if not ok]
    # Scale-normalised score; bounded, monotone, and interpretable.
    score = (0.35 * math.exp(-(angle / args.max_angle_deg)**2)
             + 0.30 * math.exp(-(offset / args.max_offset_m)**2)
             + 0.25 * math.exp(-(lateral / args.max_centroid_m)**2)
             + 0.10 * math.exp(-(math.log(ratio) / math.log(args.max_area_ratio))**2))
    return {'source_plane_id': source_plane['id'], 'target_plane_id': target_plane['id'],
            'normal_angle_deg': angle, 'offset_difference_m': offset,
            'lateral_centroid_distance_m': lateral, 'area_ratio': ratio,
            'passed_gating': all(tests), 'rejection_reason': ','.join(reasons),
            'similarity': score}


def assign_candidates(candidates, n_source, n_target, min_similarity):
    """Rectangular Hungarian assignment with dummy unmatched columns."""
    large = 1e6
    # n_source individual dummy target columns make unassignment always possible.
    cost = np.full((n_source, n_target + n_source), large, dtype=float)
    cost[:, n_target:] = 1.0 - min_similarity
    index = {}
    for row in candidates:
        i, j = row['source_plane_id'], row['target_plane_id']
        index[(i, j)] = row
        if row['passed_gating'] and row['similarity'] >= min_similarity:
            cost[i, j] = 1.0 - row['similarity']
    pairs = []
    if n_source:
        rr, cc = linear_sum_assignment(cost)
        for i, j in zip(rr, cc):
            if j < n_target and cost[i, j] < large:
                pairs.append(index[(int(i), int(j))])
    pairs.sort(key=lambda x: x['source_plane_id'])
    return pairs


def serialize_plane(frame, plane):
    return {'frame_id': frame, 'plane_id': plane['id'],
            'nx': float(plane['n'][0]), 'ny': float(plane['n'][1]), 'nz': float(plane['n'][2]),
            'd_m': plane['d'],
            'cx_m': float(plane['c'][0]), 'cy_m': float(plane['c'][1]), 'cz_m': float(plane['c'][2]),
            'support_points': plane['support'], 'bbox_area_m2': plane['area_bbox'],
            'extent_long_m': plane['extent'][0], 'extent_short_m': plane['extent'][1],
            'rms_mm': plane['rms_mm'], 'p90_mm': plane['p90_mm']}


def write_csv(path, data, columns):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(data)


def write_match_svg(path, source, target, pairs, src_name, tgt_name):
    """Visual index diagram, not a 3-D geometric verification."""
    h = 90 + 38 * max(len(source), len(target), 4)
    w = 720
    left_x, right_x = 175, 545
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             f'<text x="{left_x}" y="30" text-anchor="middle" font-family="sans-serif" font-size="16">Frame {escape(src_name)}</text>',
             f'<text x="{right_x}" y="30" text-anchor="middle" font-family="sans-serif" font-size="16">Frame {escape(tgt_name)}</text>']
    ys = lambda i: 64 + i*38
    matched_src = {m['source_plane_id'] for m in pairs}
    matched_tgt = {m['target_plane_id'] for m in pairs}
    for m in pairs:
        i, j = m['source_plane_id'], m['target_plane_id']
        lines.append(f'<line x1="{left_x+50}" y1="{ys(i)}" x2="{right_x-50}" y2="{ys(j)}" stroke="#357fbc" stroke-width="2"/>')
        lines.append(f'<text x="360" y="{(ys(i)+ys(j))/2-4:.1f}" fill="#166ba4" font-size="10" font-family="sans-serif" text-anchor="middle">{m["similarity"]:.2f}</text>')
    for side, planes, matches in ((left_x,source,matched_src),(right_x,target,matched_tgt)):
        for p in planes:
            y=ys(p['id'])
            fill = '#d9eefa' if p['id'] in matches else '#e5e7eb'
            lines.append(f'<rect x="{side-45}" y="{y-13}" width="90" height="26" rx="5" fill="{fill}" stroke="#94a3b8"/>')
            lines.append(f'<text x="{side}" y="{y+4}" text-anchor="middle" font-size="12" font-family="sans-serif">Plane {p["id"]}</text>')
    lines.append('</svg>')
    path.write_text('\n'.join(lines), encoding='utf-8')


def save_overlay(o3d, source_points, target_points, T, path):
    # Visually inspect FPFH/ICP alignment in any PLY viewer; grey vs cyan.
    pts_src = source_points @ T[:3,:3].T + T[:3,3]
    pts = np.vstack([pts_src, target_points])
    colors = np.empty((len(pts), 3), dtype=float)
    colors[:len(pts_src)] = [0.18, 0.55, 0.93]
    colors[len(pts_src):] = [0.95, 0.64, 0.23]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise IOError(f'Could not save registered cloud overlay: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True, help='Run I detection results containing CSV + labels')
    parser.add_argument('--cloud-root', type=Path, required=True, help='Run I point-cloud downsampling batch')
    parser.add_argument('--expected-batch', default='real_scene_downsampling_20261023_151330_374491')
    parser.add_argument('--source', default='000000', help='First six-digit frame id')
    parser.add_argument('--target', default='000001', help='Second six-digit frame id')
    parser.add_argument('--voxel-mm', type=int, choices=(5,10), default=5)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--pose-file', type=Path, default=None,
                        help='Optional JSON 4x4 T_source_to_target if a trusted pose is available')
    parser.add_argument('--registration-voxel', type=float, default=0.025)
    parser.add_argument('--ransac-distance', type=float, default=0.075)
    parser.add_argument('--icp-distance', type=float, default=0.04)
    parser.add_argument('--min-registration-fitness', type=float, default=0.20)
    parser.add_argument('--max-angle-deg', type=float, default=20.0)
    parser.add_argument('--max-offset-m', type=float, default=0.12)
    parser.add_argument('--max-centroid-m', type=float, default=0.75)
    parser.add_argument('--max-area-ratio', type=float, default=8.0)
    parser.add_argument('--min-similarity', type=float, default=0.55)
    args = parser.parse_args()
    if args.source == args.target or any(not re.fullmatch(r'\d{6}', f) for f in (args.source, args.target)):
        parser.error('Choose two different six-digit frame IDs, e.g. --source 000000 --target 000001')
    if (args.registration_voxel <= 0 or args.ransac_distance <= 0 or args.icp_distance <= 0
            or not 0 < args.min_registration_fitness <= 1 or args.max_angle_deg <= 0
            or args.max_offset_m <= 0 or args.max_centroid_m <= 0 or args.max_area_ratio <= 1
            or not 0 < args.min_similarity < 1):
        parser.error('Invalid positive distance/angle/ratio/fitness/similarity threshold')
    try:
        import open3d as o3d
    except ImportError:
        parser.error('Open3D is missing; activate .venv and install open3d')
    run = args.run.expanduser().resolve()
    cloud_root = args.cloud_root.expanduser().resolve()
    out = (args.output or (Path(__file__).resolve().parent / 'results' /
           f'attemptA_{args.source}_{args.target}_{datetime.now():%Y%m%d_%H%M%S}')).expanduser().resolve()
    try:
        selected, params = resolve_inputs(run,cloud_root,args.source,args.target,
                                          args.voxel_mm,args.expected_batch)
        frame_data = {}
        for f in (args.source,args.target):
            frame_data[f] = extract_frame(o3d, f, selected[f], params[f])
            print(f'Frame {f}: {len(frame_data[f][0]):,} points, {len(frame_data[f][1])} planes',flush=True)
        src_xyz, src_planes = frame_data[args.source]
        tgt_xyz, tgt_planes = frame_data[args.target]
        if args.pose_file:
            T = from_pose_file(args.pose_file)
            reginfo = {'method': 'supplied_T_source_to_target',
                       'registration_runtime_s': 0.0}
        else:
            print('Estimating source->target pose: FPFH RANSAC + ICP ...',flush=True)
            T, reginfo = estimate_relative_pose(o3d,src_xyz,tgt_xyz,args)
            validate_pose(T)
        print('Pose source -> target:')
        print(np.array2string(T,precision=5))
        print('Registration:', reginfo)
        t0 = time.perf_counter()
        candidates = [compare_planes(a,b,T,args) for a in src_planes for b in tgt_planes]
        matches = assign_candidates(candidates,len(src_planes),len(tgt_planes),args.min_similarity)
        assoc_ms = (time.perf_counter()-t0)*1000
        matched_src = {v['source_plane_id'] for v in matches}
        matched_tgt = {v['target_plane_id'] for v in matches}
        unmatched_src = [p['id'] for p in src_planes if p['id'] not in matched_src]
        unmatched_tgt = [p['id'] for p in tgt_planes if p['id'] not in matched_tgt]
        out.mkdir(parents=True,exist_ok=True)
        candidate_cols = ['source_plane_id','target_plane_id','normal_angle_deg','offset_difference_m',
                          'lateral_centroid_distance_m','area_ratio','passed_gating','rejection_reason','similarity']
        write_csv(out/'candidate_pairs.csv',candidates,candidate_cols)
        write_csv(out/'matched_planes.csv',matches,candidate_cols)
        descriptors = [serialize_plane(args.source,p) for p in src_planes] + [serialize_plane(args.target,p) for p in tgt_planes]
        write_csv(out/'plane_features.csv', descriptors, list(descriptors[0]) if descriptors else
                  ['frame_id','plane_id'])
        (out/'relative_pose.json').write_text(json.dumps({
            'source_frame':args.source, 'target_frame':args.target,
            'convention':'p_target = R @ p_source + t; native DEPTH optical coordinates',
            'T_source_to_target':T.tolist(), 'registration':reginfo}, indent=2),encoding='utf-8')
        write_match_svg(out/'matches.svg',src_planes,tgt_planes,matches,args.source,args.target)
        save_overlay(o3d,src_xyz,tgt_xyz,T,out/'registered_overlay.ply')
        summary = {
            'experiment':'Lab Note 11 - Attempt A: integrated two-frame plane association',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'source_frame':args.source,'target_frame':args.target,'method':METHOD,
            'voxel_mm':args.voxel_mm,'input_batch':args.expected_batch,
            'source_planes':len(src_planes),'target_planes':len(tgt_planes),
            'candidate_pairs':len(candidates),
            'gated_pairs':sum(bool(c['passed_gating']) for c in candidates),
            'matched_pairs':len(matches),
            'unmatched_source_ids':unmatched_src,'unmatched_target_ids':unmatched_tgt,
            'association_runtime_ms':assoc_ms,
            'registration':reginfo,
            'gating_thresholds':{'max_angle_deg':args.max_angle_deg,'max_offset_m':args.max_offset_m,
                                 'max_centroid_m':args.max_centroid_m,'max_area_ratio':args.max_area_ratio,
                                 'min_similarity':args.min_similarity},
            'notes':[
                'Gating: normal angle, aligned plane offset, lateral centroid distance, rectangle area ratio.',
                'Similarity: normalised weighted geometric similarity (baseline).',
                'Assignment: Hungarian linear sum assignment with unmatched dummy options.',
                'The initial camera pose is estimated independently (FPFH/ICP), NOT from plane matches.',
                'Matches are predictions, NOT validated accuracy; manually inspect registered_overlay.ply and matches.svg.',
                'Use manually annotated true correspondences for quantitative Precision/Recall/F1.',
                'Projected bounding-box area is approximate; disconnected coplanar supports may inflate its value.'
            ]}
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(f'Gated candidates: {summary["gated_pairs"]}/{len(candidates)}; matched pairs: {len(matches)}')
        for m in matches:
            print(f'  {args.source}:P{m["source_plane_id"]} -> {args.target}:P{m["target_plane_id"]}  '
                  f'sim={m["similarity"]:.3f}; angle={m["normal_angle_deg"]:.1f} deg; '
                  f'offset={m["offset_difference_m"]:.3f} m')
        print('Unmatched source:',unmatched_src,'Unmatched target:',unmatched_tgt)
        print('Saved to:',out)
        return 0
    except (OSError,ValueError,KeyError,RuntimeError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':
    sys.exit(main())
