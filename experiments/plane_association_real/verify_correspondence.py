#!/usr/bin/env python3
"""Lab Note 11 / Attempt A: visually verify existing two-frame plane matches.

Reads outputs of Attempt_A.py AND its original Run I Iterative RANSAC labels/PLYs.
No detection, pose estimation, or association is repeated.

Examples (PowerShell, from D:\\PlaneBasedSLAM):
  python experiments\\plane_association_real\\verify_attemptA_plane_matches.py --results "D:\\PlaneBasedSLAM\\experiments\\plane_association_real\\results\\attemptA_000000_000001_20261029_124609"
  python experiments\\plane_association_real\\verify_attemptA_plane_matches.py --results "D:\\PlaneBasedSLAM\\experiments\\plane_association_real\\results\\attemptA_000000_000001_20261029_124609" --pair 4:10 --show

Defaults infer the project directory from the Attempt A results location. Supply
--run and --cloud-root if you moved/renamed the Run I data. The output files are
review aids, NOT ground-truth validation or automatic matching accuracy scores.

Requires: numpy, open3d.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path, PureWindowsPath

import numpy as np

METHOD = 'iterative_ransac'
BLUE = (0.13, 0.41, 0.95)       # Source plane, registered into target coordinates
ORANGE = (0.97, 0.47, 0.10)     # Target plane
GREY_SOURCE = (0.52, 0.56, 0.62)
GREY_TARGET = (0.76, 0.78, 0.80)


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f'Required file not found: {path}')
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def find_project_root(results: Path) -> Path:
    for p in (results, *results.parents):
        if p.name.lower() == 'experiments':
            return p.parent
    raise ValueError('Could not locate project root from --results. Specify --run and --cloud-root.')


def fname(csv_path: str) -> str:
    return PureWindowsPath(str(csv_path).replace('/', '\\')).name


def expected_frame(filename: str) -> str:
    m = re.fullmatch(r'(\d{6})(?:_depth)?\.ply', filename)
    if m is None:
        raise ValueError(f'Unrecognised point cloud name: {filename}')
    return m.group(1)


def load_inputs(o3d, *, run: Path, cloud_root: Path, frame: str,
                group: str, batch: str) -> tuple[np.ndarray, np.ndarray]:
    metrics = read_csv(run / 'detection_metrics.csv')
    selected = []
    for row in metrics:
        if row.get('algorithm') != METHOD or row.get('scene_group') != group:
            continue
        if expected_frame(fname(row['input_ply'])) != frame:
            continue
        csv_batch = PureWindowsPath(row['input_ply'].replace('/', '\\')).parent.parent.name
        if csv_batch != batch:
            raise ValueError(f'Wrong dataset in detection_metrics.csv for {frame}: '
                             f'{csv_batch!r}, expected {batch!r}. Do not mix Run H and Run I.')
        selected.append(row)
    if len(selected) != 1:
        raise ValueError(f'Expected exactly one {METHOD} result for frame {frame} / {group}; got {len(selected)}.')
    row = selected[0]
    ply = cloud_root / group / fname(row['input_ply'])
    labels_path = run / group / METHOD / f'{ply.stem}_{METHOD}_labels.npz'
    if not ply.is_file():
        raise FileNotFoundError(f'Missing Run I PLY: {ply}')
    if not labels_path.is_file():
        raise FileNotFoundError(f'Missing matching labels file: {labels_path}')
    pcd = o3d.io.read_point_cloud(str(ply))
    coords = np.asarray(pcd.points, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.isfinite(coords).all():
        raise ValueError(f'PLY is empty or contains non-finite points: {ply}')
    with np.load(labels_path, allow_pickle=False) as archive:
        if 'labels' not in archive.files:
            raise ValueError(f"No 'labels' array in {labels_path}")
        labels = np.asarray(archive['labels'])
    if labels.ndim != 1 or len(labels) != len(coords) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f'PLY and label array do not match for frame {frame}')
    if len(coords) != int(row['valid_points']):
        raise ValueError(f'Point count differs from Run I metrics for frame {frame}')
    plane_count = int(row['detected_planes'])
    if np.any(labels < -1) or np.any(labels >= plane_count):
        raise ValueError(f'Invalid plane IDs in labels for frame {frame}')
    if int(np.count_nonzero(labels >= 0)) != int(row['assigned_points']):
        raise ValueError(f'Assigned support count differs from Run I metrics for frame {frame}')
    print(f'Frame {frame}: {len(coords):,} points, {plane_count} planes, labels verified')
    return coords, labels


def make_cloud(o3d, blocks: list[tuple[np.ndarray, tuple[float, float, float]]]):
    point_arrays, colour_arrays = [], []
    for coords, colour in blocks:
        if len(coords) == 0:
            continue
        point_arrays.append(coords)
        colour_arrays.append(np.repeat(np.array([colour], dtype=np.float64), len(coords), axis=0))
    if not point_arrays:
        raise ValueError('No points available to draw')
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.concatenate(point_arrays, axis=0))
    cloud.colors = o3d.utility.Vector3dVector(np.concatenate(colour_arrays, axis=0))
    return cloud


def save_cloud(o3d, path: Path, blocks):
    cloud = make_cloud(o3d, blocks)
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False):
        raise OSError(f'Could not save visualisation: {path}')
    return cloud


def build_review_outputs(o3d, output: Path, pairs: list[dict], src, src_labels,
                         tgt, tgt_labels, T, context_step: int, show: bool):
    src_reg = src @ T[:3, :3].T + T[:3, 3]
    review_rows = []
    for pair in pairs:
        i, j = int(pair['source_plane_id']), int(pair['target_plane_id'])
        src_mask, tgt_mask = src_labels == i, tgt_labels == j
        source_support, target_support = int(src_mask.sum()), int(tgt_mask.sum())
        if source_support == 0 or target_support == 0:
            raise ValueError(f'Plane match P{i} -> P{j} has zero support in its labels')
        stem = f'pair_P{i:02d}_P{j:02d}'
        # Context only is thinned for visibility. Matched supports are kept in full.
        overlay = save_cloud(o3d, output / f'{stem}_overlay.ply', [
            (src_reg[~src_mask][::context_step], GREY_SOURCE),
            (tgt[~tgt_mask][::context_step], GREY_TARGET),
            (src_reg[src_mask], BLUE),
            (tgt[tgt_mask], ORANGE),
        ])
        save_cloud(o3d, output / f'{stem}_source_context.ply', [
            (src[~src_mask][::context_step], GREY_SOURCE), (src[src_mask], BLUE),
        ])
        save_cloud(o3d, output / f'{stem}_target_context.ply', [
            (tgt[~tgt_mask][::context_step], GREY_TARGET), (tgt[tgt_mask], ORANGE),
        ])
        review_rows.append({
            'source_plane_id': i, 'target_plane_id': j,
            'assignment_status': pair.get('assignment_status','ASSIGNED'),
            'similarity': pair.get('similarity', ''),
            'normal_angle_deg': pair.get('normal_angle_deg', ''),
            'offset_difference_m': pair.get('offset_difference_m', ''),
            'source_support_points': source_support, 'target_support_points': target_support,
            'physical_match': '', 'notes': '',
            'overlay_file': f'{stem}_overlay.ply',
            'source_file': f'{stem}_source_context.ply',
            'target_file': f'{stem}_target_context.ply',
        })
        print(f'P{i} -> P{j}: {source_support:,} / {target_support:,} support points; saved {stem}_*.ply')
        if show:
            o3d.visualization.draw_geometries([overlay],
                window_name=f'Frame-to-frame physical match: P{i} -> P{j} (blue=source, orange=target)',
                width=1200, height=850)
    review_path = output / 'manual_match_review.csv'
    with review_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(review_rows[0].keys()))
        writer.writeheader()
        writer.writerows(review_rows)
    (output / 'README.txt').write_text(
        'Attempt A manual physical plane correspondence review\n'
        '====================================================\n'
        'Open pair_Pxx_Pyy_overlay.ply in Open3D.\n'
        'BLUE: source frame plane transformed into target camera coordinates.\n'
        'ORANGE: target frame plane.\n'
        'GREY: sampled context points (not matched planes).\n\n'
        'Use source_context.ply and target_context.ply to identify each physical\n'
        'surface separately. Inspect from several viewpoints: whether the blue\n'
        'and orange highlighted regions actually represent the SAME wall, cabinet,\n'
        'or worktop, not just nearby parallel or coplanar surfaces.\n\n'
        'The assignment_status column distinguishes a selected match from an\n'
        'alternative candidate you asked to inspect. Alternative candidates are\n'
        'NOT extra final assignments.\n\n'
        'In manual_match_review.csv, set physical_match to YES / NO / UNCERTAIN,\n'
        'and add the physical surface identity and any problems in notes.\n'
        'Do not interpret similarity scores as confidence probabilities.\n'
        'This review can flag false positives, but does NOT measure recall; recall\n'
        'requires a separate complete ground-truth list of true plane pairs.\n',
        encoding='utf-8'
    )
    return review_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', required=True, type=Path,
                        help='Folder generated by Attempt_A.py with matched_planes.csv and relative_pose.json')
    parser.add_argument('--run', type=Path, default=None,
                        help='Run I plane detection results folder, if not in the default location')
    parser.add_argument('--cloud-root', type=Path, default=None,
                        help='Run I downsampling batch, if not in the default location')
    parser.add_argument('--pair', default=None, help='Inspect any candidate pair, including unassigned alternatives (e.g. 1:5); omit for all final matches')
    parser.add_argument('--context-step', type=int, default=5,
                        help='Show every Nth context point (matched plane points are never thinned); default 5')
    parser.add_argument('--show', action='store_true', help='Open each generated match overlay interactively in Open3D')
    parser.add_argument('--output', type=Path, default=None, help='Output directory (default: RESULTS/match_verification)')
    args = parser.parse_args(argv)
    if args.context_step < 1:
        parser.error('--context-step must be positive')
    try:
        import open3d as o3d
    except ImportError:
        parser.error('Open3D is not installed in the active Python environment')
    try:
        results = args.results.expanduser().resolve()
        summary = json.loads((results / 'summary.json').read_text(encoding='utf-8'))
        pose = json.loads((results / 'relative_pose.json').read_text(encoding='utf-8'))
        source, target = str(summary['source_frame']), str(summary['target_frame'])
        if (str(pose['source_frame']), str(pose['target_frame'])) != (source, target):
            raise ValueError('Attempt A summary.json and relative_pose.json refer to different frame pairs')
        if summary.get('method') != METHOD:
            raise ValueError('This visualiser expects Iterative RANSAC Attempt A outputs')
        T = np.asarray(pose['T_source_to_target'], dtype=np.float64)
        if T.shape != (4,4) or not np.isfinite(T).all() or not np.allclose(T[3], [0,0,0,1]):
            raise ValueError('Invalid 4x4 relative pose')
        R = T[:3,:3]
        if not np.allclose(R.T @ R, np.eye(3), atol=0.02) or abs(np.linalg.det(R)-1) > 0.02:
            raise ValueError('Invalid relative-pose rotation')
        matches = read_csv(results / 'matched_planes.csv')
        if not matches:
            raise ValueError('No matched planes: cannot perform physical match review')
        if len(matches) != int(summary['matched_pairs']):
            raise ValueError('Matched-plane count does not agree with Attempt A summary')
        ids = [(int(m['source_plane_id']), int(m['target_plane_id'])) for m in matches]
        if len(set(i for i,j in ids)) != len(ids) or len(set(j for i,j in ids)) != len(ids):
            raise ValueError('Matched pairs are not one-to-one')
        if args.pair is not None:
            m = re.fullmatch(r'\s*(\d+)\s*[:,-]\s*(\d+)\s*', args.pair)
            if not m:
                parser.error('--pair must have the form 1:5')
            pair = (int(m.group(1)), int(m.group(2)))
            candidates = read_csv(results / 'candidate_pairs.csv')
            selected = [row for row in candidates if
                        (int(row['source_plane_id']), int(row['target_plane_id'])) == pair]
            if len(selected) != 1:
                raise ValueError(f'Candidate pair {pair} not uniquely found in candidate_pairs.csv')
            matches = [dict(selected[0], assignment_status=(
                'ASSIGNED' if pair in ids else 'NOT_ASSIGNED'))]
            if not bool(str(selected[0].get('passed_gating', '')).lower() == 'true'):
                print(f'NOTE: Pair {pair} FAILED the original gating thresholds; visualised for diagnosis only.')
        else:
            matches = [dict(row, assignment_status='ASSIGNED') for row in matches]
        batch = str(summary['input_batch'])
        group = f'voxel_{int(summary["voxel_mm"]):04d}p00mm'
        proj = find_project_root(results) if args.run is None or args.cloud_root is None else None
        run = (args.run or (proj / 'experiments' / 'plane_detection_real' / 'results' / 'runH_maxplanes20_test')).expanduser().resolve()
        cloud_root = (args.cloud_root or (proj / 'experiments' / 'pointcloud' / 'results' / batch)).expanduser().resolve()
        print('Plane Detection results:',run)
        print('Downsampled point clouds:',cloud_root)
        print('Plane group:',group)
        src, src_labels = load_inputs(o3d,run=run,cloud_root=cloud_root,frame=source,group=group,batch=batch)
        tgt, tgt_labels = load_inputs(o3d,run=run,cloud_root=cloud_root,frame=target,group=group,batch=batch)
        default_out = results / 'match_verification'
        if args.pair is not None:
            default_out = default_out / f'pair_P{pair[0]:02d}_P{pair[1]:02d}'
        out = (args.output or default_out).expanduser().resolve()
        out.mkdir(parents=True,exist_ok=True)
        review_path = build_review_outputs(o3d,out,matches,src,src_labels,tgt,tgt_labels,T,args.context_step,args.show)
        print(f'\nSaved {len(matches)} match visualisation(s) to: {out}')
        print('Manual review table:',review_path)
        print('Each pair has an overlay and separate highlighted source/target contexts.')
        return 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
