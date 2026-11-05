#!/usr/bin/env python3
"""Lab Note 11 - Attempt D: compare four existing plane-similarity methods.

INPUTS (all produced previously; no plane detection or gating reruns):
 - Attempt C: similarity_input_all.csv, plane_contours_XXXXXX.npz,
   and XXXX_to_YYYY/relative_pose.json.
 - Run I: detection_metrics.csv, saved plane-label NPZs and input PLYs.

Place this script in the SAME directory as these five unchanged modules:
 similarity_common.py, descriptor_6d_similarity.py, hia_tcd_similarity.py,
 planematch_similarity.py, popup_weighted_similarity.py

Run from the PlaneBasedSLAM project root:
 python experiments/plane_association_real/Attempt_D.py \
   --attempt-c-dir "D:/PlaneBasedSLAM/experiments/plane_association_real/results/attemptC_candidate_gating_YYYYMMDD_HHMMSS"

The --run and --cloud-root options are optional if Attempt C's summary.json
records accessible paths. This experiment uses the original four compute_score()
functions and writes scores/ranks for downstream Assignment. It does NOT perform
Assignment or report accuracy without ground-truth correspondence labels.

Important: The uploaded planematch_similarity.py is a SIMPLIFIED GEOMETRIC
PlaneMatch-inspired residual, NOT the learned network in the original paper.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PureWindowsPath

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from similarity_common import PlaneObservation, SimilaritySample
import descriptor_6d_similarity
import hia_tcd_similarity
import planematch_similarity
import popup_weighted_similarity

METHODS = {
    'descriptor_6d_similarity': descriptor_6d_similarity,
    'hia_tcd_similarity': hia_tcd_similarity,
    'planematch_similarity': planematch_similarity,
    'popup_weighted_similarity': popup_weighted_similarity,
}

FEATURE_FIELDS = [
    'source_frame', 'target_frame', 'gating_algorithm',
    'source_plane_id', 'target_plane_id',
]
SCORE_FIELDS = FEATURE_FIELDS + [
    'similarity_algorithm', 'score', 'higher_is_better',
    'rank_within_source', 'candidates_for_source', 'runtime_ms',
    'gt_match', 'source_normal_x', 'source_normal_y', 'source_normal_z',
    'source_offset_m', 'target_normal_x', 'target_normal_y', 'target_normal_z',
    'target_offset_m', 'normal_angle_deg', 'offset_difference_m',
    'centroid_distance_m', 'source_bbox_area_m2', 'target_bbox_area_m2',
]
SUMMARY_FIELDS = [
    'gating_algorithm', 'similarity_algorithm', 'frame_pairs',
    'candidate_records', 'mean_score', 'median_score', 'min_score',
    'max_score', 'mean_runtime_ms_per_candidate',
    'total_runtime_ms', 'top1_source_choices',
    'gt_evaluated', 'top1_correct', 'top1_precision_selected',
]


def csv_read(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing file: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f'No data records in {path}')
    return rows


def csv_write(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def get_value(row: dict, key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Invalid or missing {key!r} in candidate row {row}') from exc
    if not np.isfinite(value):
        raise ValueError(f'Non-finite {key!r} in candidate row')
    return value


def vec3(row: dict, prefix: str, kind: str) -> np.ndarray:
    suffix = '_m' if kind == 'centroid' else ''
    return np.array([get_value(row, f'{prefix}_{kind}_{a}{suffix}')
                     for a in 'xyz'], dtype=float)


def valid_id(value: str) -> str:
    value = str(value).strip()
    if not re.fullmatch(r'\d{6}', value):
        raise ValueError(f'Expected six-digit frame ID, got {value!r}')
    return value


def read_ply_xyz(path: Path) -> np.ndarray:
    # MUST preserve original point order because Run I labels are row-aligned.
    try:
        import open3d as o3d
        points = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=float)
    except ImportError:
        try:
            import trimesh
        except ImportError as exc:
            raise ImportError('Install open3d or trimesh to load Run I PLYs') from exc
        points = np.asarray(trimesh.load(str(path), process=False).vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f'Invalid PLY coordinates: {path}')
    return points


def load_contours(c_dir: Path, frames: set[str]) -> dict[tuple[str, int], np.ndarray]:
    result = {}
    for frame in sorted(frames):
        path = c_dir / f'plane_contours_{frame}.npz'
        if not path.is_file():
            raise FileNotFoundError(f'Attempt C contour data required for HIA-TCD: {path}')
        with np.load(path, allow_pickle=False) as z:
            for key in z.files:
                match = re.fullmatch(r'plane_(\d+)', key)
                if match is None:
                    continue
                contour = np.asarray(z[key], dtype=float)
                if (contour.ndim != 2 or contour.shape[1] != 3 or
                        len(contour) < 3 or not np.isfinite(contour).all()):
                    raise ValueError(f'Bad contour {frame}/{key}: {contour.shape}')
                result[(frame, int(match.group(1)))] = contour
    return result


def load_pose(c_dir: Path, source: str, target: str) -> tuple[np.ndarray, str]:
    pose_path = c_dir / f'{source}_to_{target}' / 'relative_pose.json'
    if not pose_path.is_file():
        raise FileNotFoundError(
            f'Missing Attempt C pose {pose_path}. Pose must NOT be re-estimated.')
    info = json.loads(pose_path.read_text(encoding='utf-8-sig'))
    if isinstance(info, dict):
        if info.get('source_frame', source) != source or info.get('target_frame', target) != target:
            raise ValueError(f'Pose frame IDs mismatch: {pose_path}')
        data = info.get('T_source_to_target')
    else:
        data = info
    T = np.asarray(data, dtype=float)
    if (T.shape != (4, 4) or not np.isfinite(T).all() or
            not np.allclose(T[3], [0, 0, 0, 1], atol=1e-5) or
            not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=0.02) or
            abs(np.linalg.det(T[:3, :3]) - 1.0) > 0.02):
        raise ValueError(f'Invalid source-to-target pose in {pose_path}')
    return T, str(pose_path)


def load_run_patches(run: Path, cloud_root: Path, frames: set[str],
                     voxel_mm: int, detector: str) -> dict[tuple[str, int], np.ndarray]:
    """Load original PLY/label pairs, without detecting or refitting planes."""
    metrics = csv_read(run / 'detection_metrics.csv')
    group = f'voxel_{voxel_mm:04d}p00mm'
    found = {}
    for row in metrics:
        if row.get('algorithm') != detector or row.get('scene_group') != group:
            continue
        basename = PureWindowsPath(str(row.get('input_ply', '')).replace('/', '\\')).name
        match = re.fullmatch(r'(\d{6})(?:_depth)?\.ply', basename, re.I)
        if match is None or match.group(1) not in frames:
            continue
        frame = match.group(1)
        if frame in found:
            raise ValueError(f'Duplicate Run I detection metric for {frame}')
        cloud_path = cloud_root / group / basename
        labels_path = run / group / detector / f'{Path(basename).stem}_{detector}_labels.npz'
        if not cloud_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError(f'Missing Run I PLY/label files: {cloud_path}, {labels_path}')
        found[frame] = (cloud_path, labels_path, row)
    if set(found) != frames:
        raise ValueError(f'Run I frames absent: {sorted(frames - set(found))}')

    patches = {}
    for frame, (ply, labels_file, metric) in found.items():
        points = read_ply_xyz(ply)
        with np.load(labels_file, allow_pickle=False) as arr:
            if 'labels' not in arr:
                raise ValueError(f'No labels array in {labels_file}')
            labels = np.asarray(arr['labels'])
        plane_count = int(metric['detected_planes'])
        if (labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer)
                or len(labels) != len(points)
                or int(metric['valid_points']) != len(points)
                or np.any(labels < -1) or np.any(labels >= plane_count)
                or int(np.count_nonzero(labels >= 0)) != int(metric['assigned_points'])):
            raise ValueError(f'Labels/PLY/metrics inconsistent for frame {frame}')
        for i in range(plane_count):
            pts = points[labels == i]
            if len(pts) < 3:
                raise ValueError(f'Insufficient patch points: {frame}, plane {i}')
            patches[(frame, i)] = pts
    return patches


def load_ground_truth(path: Path | None, pairs: set[tuple[str, str]]) -> dict | None:
    if path is None:
        return None
    rows = csv_read(path)
    required = {'source_frame', 'target_frame', 'source_plane_id', 'target_plane_id'}
    if not required.issubset(rows[0]):
        raise ValueError(f'Ground-truth CSV requires columns: {sorted(required)}')
    truth = defaultdict(set)
    covered = set()
    for row in rows:
        key = (valid_id(row['source_frame']), valid_id(row['target_frame']))
        if key not in pairs:
            raise ValueError(f'Unexpected frame pair in GT CSV: {key}')
        covered.add(key)
        if not row['source_plane_id'].strip() and not row['target_plane_id'].strip():
            continue  # An explicitly annotated frame pair with zero positive matches.
        a, b = int(row['source_plane_id']), int(row['target_plane_id'])
        if a < 0 or b < 0 or (a, b) in truth[key]:
            raise ValueError(f'Duplicate/invalid GT pair {key}: {(a, b)}')
        truth[key].add((a, b))
    if covered != pairs:
        raise ValueError(f'Ground truth must explicitly cover every frame pair; missing {pairs - covered}')
    return truth


# ============================================================================
# STAGE 1. LOAD THE INPUTS AND THE FOUR EXISTING SIMILARITY ALGORITHMS
# ============================================================================

def load_inputs():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--attempt-c-dir', type=Path, required=True,
                        help='Directory of completed Attempt C results')
    parser.add_argument('--run', type=Path, default=None,
                        help='Run I detector directory (defaults to Attempt C summary.json)')
    parser.add_argument('--cloud-root', type=Path, default=None,
                        help='Run I downsampled PLY directory (defaults to Attempt C summary.json)')
    parser.add_argument('--ground-truth', type=Path, default=None,
                        help='Optional exhaustive manual positive plane pairs CSV')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output folder (must not already exist)')
    args = parser.parse_args()

    c_dir = args.attempt_c_dir.expanduser().resolve()
    cfg_path = c_dir / 'summary.json'
    if not cfg_path.is_file():
        parser.error(f'Missing Attempt C summary.json: {cfg_path}')
    cfg = json.loads(cfg_path.read_text(encoding='utf-8-sig'))
    run = Path(args.run or cfg['run']).expanduser().resolve()
    cloud_root = Path(args.cloud_root or cfg['cloud_root']).expanduser().resolve()
    for required in (run / 'detection_metrics.csv', cloud_root):
        if not required.exists():
            parser.error(f'Required Run I input not found: {required}\n'
                         'Use --run and --cloud-root to override stale saved paths.')
    voxel = int(cfg['voxel_mm'])
    detector = 'iterative_ransac'
    if 'iterative ransac' not in str(cfg.get('input', '')).lower().replace('_', ' '):
        parser.error('Attempt C must use Iterative RANSAC Run I data for this adapter')

    c_rows = csv_read(c_dir / 'similarity_input_all.csv')
    if not all(set(FEATURE_FIELDS).issubset(row) for row in c_rows):
        parser.error(f'Attempt C CSV lacks mandatory fields {FEATURE_FIELDS}')
    pairs = set()
    frames = set()
    unique_rows = set()
    for row in c_rows:
        a, b = valid_id(row['source_frame']), valid_id(row['target_frame'])
        if a == b:
            raise ValueError(f'Identical frame IDs for candidate: {a}')
        pairs.add((a, b)); frames.update((a, b))
        i, j = int(row['source_plane_id']), int(row['target_plane_id'])
        if i < 0 or j < 0:
            raise ValueError('Negative plane IDs are invalid')
        key = (a, b, row['gating_algorithm'], i, j)
        if key in unique_rows:
            raise ValueError(f'Duplicate candidate record {key}')
        unique_rows.add(key)

    c_expected = set(cfg.get('gating_methods', []))
    if c_expected and set(r['gating_algorithm'] for r in c_rows) - c_expected:
        raise ValueError('Unexpected gating algorithms in Attempt C CSV')
    # If a gating method yields zero pairs, its per-pair score table remains empty.
    # All method evaluations still use the exact candidates reported by Attempt C.
    print('STAGE 1 | Reading saved Attempt C candidate pairs and Run I geometry...', flush=True)
    contours = load_contours(c_dir, frames)
    poses = {pair: load_pose(c_dir, *pair) for pair in pairs}
    patches = load_run_patches(run, cloud_root, frames, voxel, detector)
    truth = load_ground_truth(args.ground_truth, pairs)

    # Validate all referenced plane IDs before evaluation begins.
    for row in c_rows:
        a, b = row['source_frame'], row['target_frame']
        i, j = int(row['source_plane_id']), int(row['target_plane_id'])
        for key in ((a, i), (b, j)):
            if key not in contours or key not in patches:
                raise ValueError(f'Plane {key} missing a contour or support-point patch')
        if (len(patches[(a, i)]) != int(row['source_support_points']) or
                len(patches[(b, j)]) != int(row['target_support_points'])):
            raise ValueError('Run I support sizes do not agree with Attempt C CSV')

    output = (args.output or SCRIPT_DIR / 'results' /
              f'attemptD_similarity_{datetime.now():%Y%m%d_%H%M%S}').expanduser().resolve()
    if output.exists():
        parser.error(f'Refusing to overwrite existing output: {output}')
    print(f'  {len(c_rows)} gating-specific candidates; {len(pairs)} frame pairs; '
          f'{len(METHODS)} similarity algorithms', flush=True)
    return args, output, cfg, c_rows, contours, patches, poses, truth


# ============================================================================
# STAGE 2. APPLY THE SAME CANDIDATES TO THE FOUR ALGORITHMS
# ============================================================================

def make_sample(row, contours, patches, T, truth):
    """Build the ACTUAL SimilaritySample expected by similarity_common.py.

    Attempt C CSV already contains BOTH plane equations in TARGET coordinates.
    Source contours and patch points are transformed to TARGET coordinates as
    well. Therefore R_ref_cur/t_ref_cur are deliberately None: no double pose.
    """
    source, target = row['source_frame'], row['target_frame']
    i, j = int(row['source_plane_id']), int(row['target_plane_id'])
    R, t = T[:3, :3], T[:3, 3]

    source_contour = contours[(source, i)] @ R.T + t
    target_contour = contours[(target, j)]
    source_patch = patches[(source, i)] @ R.T + t
    target_patch = patches[(target, j)]

    # Confirm source-plane transformed geometry agrees with the original
    # source-to-target pose stored by Attempt C; no silent coordinate mixups.
    # Source normal is already in target frame in Attempt C's CSV. Explicit
    # geometry quality checks also confirm patches are close to fitted planes.

    def plane(prefix, plane_id, contour, patch):
        normal = vec3(row, prefix, 'normal')
        offset = get_value(row, f'{prefix}_offset_m')
        if abs(np.linalg.norm(normal) - 1.0) > 1e-3:
            raise ValueError(f'Nonunit {prefix} normal in Attempt C CSV')
        centroid = vec3(row, prefix, 'centroid')
        area = get_value(row, f'{prefix}_bbox_area_m2')
        support = int(row[f'{prefix}_support_points'])
        # Do not silently accept geometry in different coordinate frames.
        median_residual = float(np.median(np.abs(patch @ normal + offset)))
        if median_residual > 0.035:
            raise ValueError(
                f'Plane geometry/patch frame mismatch {source}->{target} '
                f'{prefix} plane {plane_id}: median residual {median_residual:.3f} m')
        return PlaneObservation(
            plane_id=str(plane_id), normal=normal, offset=offset,
            centroid=centroid, area=area, support_size=support,
            metadata={'contour': contour, 'patch': patch},
        )

    # The source/current plane and target/reference plane are both expressed
    # in TARGET coordinates, including their contour and patch metadata.
    gt_known = truth is not None
    label = ((i, j) in truth[(source, target)]) if gt_known else False
    sample = SimilaritySample(
        sample_id=f'{source}_{target}_{i}_{j}',
        current_plane=plane('source', i, source_contour, source_patch),
        reference_plane=plane('target', j, target_contour, target_patch),
        gt_match=label, data_source='real', split='final',
        sequence_id='run_I', current_frame_id=source,
        reference_frame_id=target,
        R_ref_cur=None, t_ref_cur=None,
        metadata={'gt_known': gt_known, 'source_in_target_frame': True},
    )
    return sample


def run_similarity_algorithms(c_rows, contours, patches, poses, truth):
    print('\nSTAGE 2 | Running four original Similarity algorithms...', flush=True)
    results = []
    # Precalculate unique physical pair scores only once, but do not alter
    # candidate membership of each gating method.
    unique_pairs = {}
    for row in c_rows:
        key = (row['source_frame'], row['target_frame'],
               int(row['source_plane_id']), int(row['target_plane_id']))
        if key not in unique_pairs:
            unique_pairs[key] = row
    computed = {}
    for k, row in unique_pairs.items():
        pair = (k[0], k[1])
        sample = make_sample(row, contours, patches, poses[pair][0], truth)
        for name, module in METHODS.items():
            t0 = time.perf_counter()
            score = float(module.compute_score(sample))
            duration = (time.perf_counter() - t0) * 1000.0
            if not math.isfinite(score) or score < -1e-8 or score > 1 + 1e-8:
                raise ValueError(f'{name} produced invalid score {score} for {k}')
            computed[k, name] = (score, duration)
    for row in c_rows:
        source = row['source_frame']; target = row['target_frame']
        i, j = int(row['source_plane_id']), int(row['target_plane_id'])
        for name in METHODS:
            score, duration = computed[(source, target, i, j), name]
            out = {key: row.get(key, '') for key in FEATURE_FIELDS}
            out.update({
                'similarity_algorithm': name, 'score': score,
                'higher_is_better': True, 'runtime_ms': duration,
                'gt_match': (int((i, j) in truth[(source, target)]) if truth is not None else ''),
            })
            for k in SCORE_FIELDS:
                if k not in out:
                    out[k] = row.get(k, '')
            results.append(out)
    print(f'  Computed {len(unique_pairs)} unique geometric plane pairs '
          f'x 4 methods; generated {len(results)} gating-specific scores.', flush=True)
    return results


# ============================================================================
# STAGE 3. EVALUATE SCORES / RANKS AND CREATE COMPARISON SUMMARY
# ============================================================================

def evaluate_outputs(results, ground_truth_available: bool):
    print('\nSTAGE 3 | Ranking candidate matches and summarising...', flush=True)
    sources = defaultdict(list)
    for row in results:
        key = (row['source_frame'], row['target_frame'],
               row['gating_algorithm'], row['similarity_algorithm'],
               int(row['source_plane_id']))
        sources[key].append(row)
    for group in sources.values():
        group.sort(key=lambda r: (-float(r['score']), int(r['target_plane_id'])))
        for position, row in enumerate(group, 1):
            row['rank_within_source'] = position
            row['candidates_for_source'] = len(group)

    groups = defaultdict(list)
    for r in results:
        groups[r['gating_algorithm'], r['similarity_algorithm']].append(r)
    summary = []
    for (gating, method), rows in sorted(groups.items()):
        scores = np.array([float(r['score']) for r in rows], dtype=float)
        # Runtime is measured on each unique geometric pair ONCE, not once
        # per gating reuse. Restrict runtime summation to unique pairs here.
        seen = set()
        runtime = []
        for row in rows:
            key = (row['source_frame'], row['target_frame'],
                   row['source_plane_id'], row['target_plane_id'])
            if key not in seen:
                seen.add(key)
                runtime.append(float(row['runtime_ms']))
        first = [r for r in rows if r['rank_within_source'] == 1]
        correct = sum(int(r['gt_match']) for r in first) if ground_truth_available else None
        summary.append({
            'gating_algorithm': gating, 'similarity_algorithm': method,
            'frame_pairs': len(set((r['source_frame'], r['target_frame']) for r in rows)),
            'candidate_records': len(rows),
            'mean_score': float(np.mean(scores)),
            'median_score': float(np.median(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'mean_runtime_ms_per_candidate': float(np.mean(runtime)),
            'total_runtime_ms': float(sum(runtime)),
            'top1_source_choices': len(first),
            'gt_evaluated': ground_truth_available,
            'top1_correct': correct if ground_truth_available else '',
            # Accuracy among selected top-1 source choices, not recall across
            # every plane; cannot interpret as full correspondence accuracy.
            'top1_precision_selected': (correct / len(first) if first and ground_truth_available else ''),
        })
    return results, summary


# ============================================================================
# STAGE 4. SAVE THE SUMMARY AND SIMILARITY SCORES FOR ASSIGNMENT
# ============================================================================

def save_results(out: Path, cfg: dict, c_rows: list[dict], results: list[dict],
                 summary: list[dict], poses: dict, truth: dict | None) -> None:
    print('\nSTAGE 4 | Saving scores, rankings and comparison summary...', flush=True)
    out.mkdir(parents=True)
    csv_write(out / 'similarity_results_all.csv', results, SCORE_FIELDS)
    csv_write(out / 'similarity_summary.csv', summary, SUMMARY_FIELDS)
    # One file per gating method, directly suitable for Assignment selection.
    for gating in sorted(set(r['gating_algorithm'] for r in results)):
        rows = [r for r in results if r['gating_algorithm'] == gating]
        csv_write(out / f'assignment_input_{gating}.csv', rows, SCORE_FIELDS)
    (out / 'summary.json').write_text(json.dumps({
        'experiment': 'Lab Note 11 - Attempt D: four Similarity methods',
        'method_names': list(METHODS),
        'candidate_records': len(c_rows), 'scored_records': len(results),
        'ground_truth_supplied': truth is not None,
        'pose_files': {f'{a}_to_{b}': origin for (a, b), (_, origin) in poses.items()},
        'methodology': [
            'Uses existing Attempt C candidates; no detection, gating or pose re-estimation.',
            'All source planes, contours and patch points are in target coordinates.',
            'All four original compute_score() implementations called unchanged.',
            'Similarity scores from different methods are not directly calibrated against one another.',
            'PlaneMatch here is a simplified geometric, non-learned adaptation.',
            'No automatic assignment is performed.',
            'No GT-based accuracy is claimed without exhaustive correspondence annotations.',
            'Runtime values measure direct method calls only, not data loading or preparation.',
        ],
    }, indent=2), encoding='utf-8')
    print(f'  Results directory: {out}', flush=True)
    print(f'  Scores: {out / "similarity_results_all.csv"}', flush=True)
    print(f'  Summary: {out / "similarity_summary.csv"}', flush=True)


def main():
    args, output, cfg, rows, contours, patches, poses, truth = load_inputs()
    scores = run_similarity_algorithms(rows, contours, patches, poses, truth)
    results, summary = evaluate_outputs(scores, truth is not None)
    save_results(output, cfg, rows, results, summary, poses, truth)


if __name__ == '__main__':
    main()
