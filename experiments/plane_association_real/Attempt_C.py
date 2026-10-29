#!/usr/bin/env python3
"""Lab Note 11 - Attempt C: REAL-DATA COMPARISON OF FOUR CANDIDATE GATING METHODS.

Uses Run I's EXISTING Iterative RANSAC segmentation. NO plane detection is run.
Executes the four original gating modules, UNMODIFIED:
  contour_point_gating, motion_model_distance_gating,
  normal_based_gating, pcpm_gating.

All algorithms receive exactly the same plane features and camera pose per
frame pair. Exports per-method candidate lists plus a unified, feature-rich
CSV to be passed to the next Similarity Evaluation stage. Does not perform
similarity scoring, final assignment, or bad-match rejection.

From D:\\PlaneBasedSLAM (PowerShell):
 python experiments\\plane_association_real\\Attempt_C.py `
   --run "D:\\PlaneBasedSLAM\\experiments\\plane_detection_real\\results\\runH_maxplanes20_test" `
   --cloud-root "D:\\PlaneBasedSLAM\\experiments\\pointcloud\\results\\real_scene_downsampling_20261023_151330_374491" `
   --source 000000 --targets 000001 000002 000003 000004 --voxel-mm 5

Python dependencies: numpy, scipy, and open3d (or trimesh for point-cloud PLY
reading). Open3D is needed only if a relative pose must be newly estimated.
Existing Attempt A relative_pose.json files are loaded automatically when
present in plane_association_real/results. Place the four algorithm .py files
and candidate_gating_common.py BESIDE THIS FILE; no code copying is needed.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

import numpy as np
from scipy.spatial import ConvexHull, QhullError

# The existing four algorithms and their shared module are in THIS folder.
# Python resolves imports relative to the running script; no extra paths needed.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from candidate_gating_common import PlaneSet, transform_plane_set, normal_angle_deg, align_plane_sign
import contour_point_gating
import motion_model_distance_gating
import normal_based_gating
import pcpm_gating

# EXPERIMENT FLOW (the detailed functions appear below):
#   Stage 1 -> load_inputs()
#   Stage 2 -> run_gating_algorithms()
#   Stage 3 -> evaluate_outputs()
#   Stage 4 -> save_results()
# The main() function at the bottom calls them in that order.

DETECTOR = 'iterative_ransac'  # Already completed in Lab Note 10, Run I
METHODS = {
    'contour_point_gating': contour_point_gating,
    'motion_model_distance_gating': motion_model_distance_gating,
    'normal_based_gating': normal_based_gating,
    'pcpm_gating': pcpm_gating,
}
BATCH = 'real_scene_downsampling_20261023_151330_374491'


# ============================================================================
# SHARED HELPERS: reading Run I data, extracting plane geometry, and poses
# ============================================================================

def csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f'Required Run I file does not exist: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, records: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def portable_name(value: str) -> str:
    return PureWindowsPath(str(value).replace('/', '\\')).name


def get_frame(value: str) -> str:
    m = re.fullmatch(r'(\d{6})(?:_depth)?\.ply', portable_name(value), re.I)
    if not m:
        raise ValueError(f'Unrecognised Run I PLY name: {value}')
    return m.group(1)


def read_ply(path: Path) -> np.ndarray:
    """PLY point order MUST be identical to the saved labels.npz order."""
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(path))
        xyz = np.asarray(pcd.points, dtype=float)
    except ImportError:
        try:
            import trimesh
        except ImportError as e:
            raise ImportError('Please install open3d in your project environment.') from e
        obj = trimesh.load(str(path), process=False)
        xyz = np.asarray(obj.vertices, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f'Invalid point-cloud geometry: {path}')
    # Do NOT remove points: each labels.npz entry corresponds to a PLY row.
    if not np.isfinite(xyz).all():
        raise ValueError(f'Non-finite XYZ in {path}; filtering would invalidate Run I labels')
    return xyz


def locate_run_frames(run, cloud_root, frames, voxel_mm, expected_batch):
    if expected_batch and cloud_root.name != expected_batch:
        raise ValueError(f'Wrong cloud batch: {cloud_root.name}; expected {expected_batch}')
    group = f'voxel_{voxel_mm:04d}p00mm'
    metrics = csv_rows(run / 'detection_metrics.csv')
    parameters = csv_rows(run / 'plane_parameters.csv')
    entries = {}
    for row in metrics:
        if row.get('algorithm') != DETECTOR or row.get('scene_group') != group:
            continue
        frame = get_frame(row['input_ply'])
        if frame not in frames:
            continue
        if frame in entries:
            raise ValueError(f'Duplicate {DETECTOR} detection entry for {frame}, {group}')
        if expected_batch:
            original_batch = PureWindowsPath(str(row['input_ply']).replace('/', '\\')).parent.parent.name
            if original_batch != expected_batch:
                raise ValueError(f'Metric row {frame} belongs to the wrong batch {original_batch}')
        basename = portable_name(row['input_ply'])
        ply = cloud_root / group / basename
        if not ply.is_file():
            old_path = Path(row['input_ply'])
            if old_path.is_file():
                ply = old_path
            else:
                raise FileNotFoundError(f'Cannot find input point cloud: {ply}')
        labels = run / group / DETECTOR / f'{ply.stem}_{DETECTOR}_labels.npz'
        if not labels.is_file():
            old_label = Path(row.get('labels_npz', ''))
            if old_label.is_file():
                labels = old_label
            else:
                raise FileNotFoundError(f'Cannot find Run I plane labels: {labels}')
        entries[frame] = {'metric': row, 'ply': ply, 'labels': labels,
                          'original_ply': row['input_ply'], 'parameter_rows': []}
    missing = set(frames) - set(entries)
    if missing:
        raise ValueError(f'Run I {DETECTOR} result is missing frames: {sorted(missing)}')
    keyed = defaultdict(list)
    for row in parameters:
        if row.get('algorithm') == DETECTOR:
            keyed[row['input_ply']].append(row)
    for frame, entry in entries.items():
        entry['parameter_rows'] = keyed[entry['original_ply']]
        expected = int(entry['metric']['detected_planes'])
        ids = sorted(int(r['plane_id']) for r in entry['parameter_rows'])
        if ids != list(range(expected)):
            raise ValueError(f'Plane parameters missing/duplicated for frame {frame}: {ids}')
    return entries


def contour_points(support: np.ndarray, n: np.ndarray, max_hull=18000) -> np.ndarray:
    """Subset of REAL support points on a 2D convex hull (not fitted corners).

    Convex hull may bridge disconnected coplanar regions: the script exports
    the actual hull vertices only. No interpolation or artificial outline.
    """
    if len(support) <= 3:
        return support.copy()
    # Deterministic subsampling if necessary, while retaining physical points.
    sample = support if len(support) <= max_hull else support[np.linspace(
        0, len(support) - 1, max_hull, dtype=int)]
    base = np.eye(3)[np.argmin(np.abs(n))]
    u = np.cross(n, base)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    uv = np.column_stack([sample @ u, sample @ v])
    try:
        hull = ConvexHull(uv)
        boundary = sample[hull.vertices]
    except QhullError:
        extrema = set()
        for j in range(2):
            extrema.add(int(np.argmin(uv[:, j])))
            extrema.add(int(np.argmax(uv[:, j])))
        boundary = sample[sorted(extrema)]
    return boundary


def pack_contours(contours: list[np.ndarray]) -> np.ndarray:
    """Convert variable-length boundaries to a dense (planes, vertices, 3) array.

    The user's real candidate_gating_common.transform_plane_set accesses
    `planes.contours.shape`, so a Python list is NOT a valid PlaneSet input.
    Repeat the last REAL boundary point for padding; this does not alter any
    min point-to-plane distance. The original unpadded boundaries are saved
    separately for downstream interpretation.
    """
    arrays = [np.asarray(c, dtype=np.float64).reshape(-1, 3) for c in contours]
    if any(len(c) == 0 or not np.isfinite(c).all() for c in arrays):
        raise ValueError('Every detected plane must have finite contour vertices')
    if not arrays:
        return np.empty((0, 0, 3), dtype=np.float64)
    max_vertices = max(len(c) for c in arrays)
    dense = np.empty((len(arrays), max_vertices, 3), dtype=np.float64)
    for idx, points in enumerate(arrays):
        dense[idx, :len(points), :] = points
        dense[idx, len(points):, :] = points[-1]  # repeat, NOT zeros or NaNs
    assert dense.ndim == 3 and dense.shape[0] == len(arrays) and dense.shape[2] == 3
    return dense


def create_plane_set(normals, offsets, centres, contours, widths, heights):
    """Construct the user's PlaneSet using a dense numeric contour array.

    No modifications to candidate_gating_common or the four algorithms.
    Constructor field names are resolved explicitly to avoid positional errors.
    """
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    offsets = np.asarray(offsets, dtype=np.float64).reshape(-1)
    centres = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    widths = np.asarray(widths, dtype=np.float64).reshape(-1)
    heights = np.asarray(heights, dtype=np.float64).reshape(-1)
    contours_dense = pack_contours(contours)
    n = len(normals)
    if any(len(x) != n for x in (offsets, centres, widths, heights)):
        raise ValueError('Plane geometry fields have incompatible lengths')
    if not (np.isfinite(normals).all() and np.isfinite(offsets).all()
            and np.isfinite(centres).all()):
        raise ValueError('Plane geometry contains non-finite values')

    values = {
        'normals': normals, 'offsets': offsets, 'centroids': centres,
        'centers': centres, 'centres': centres,
        'contours': contours_dense,
        'widths': widths, 'heights': heights,
        'areas': widths * heights, 'area': widths * heights,
        'sizes': np.column_stack((widths, heights)),
        'extents': np.column_stack((widths, heights)),
        'plane_ids': np.arange(n, dtype=int), 'ids': np.arange(n, dtype=int),
    }
    signature = inspect.signature(PlaneSet)
    kwargs = {}
    for name, spec in signature.parameters.items():
        if spec.kind in (inspect.Parameter.VAR_POSITIONAL,
                         inspect.Parameter.VAR_KEYWORD):
            continue
        if name in values:
            kwargs[name] = values[name]
        elif spec.default is inspect.Parameter.empty:
            raise RuntimeError('Unsupported PlaneSet required parameter '
                               f'{name!r}; constructor is {signature}')
    ps = PlaneSet(**kwargs)
    if len(ps) != n:
        raise ValueError('PlaneSet unexpectedly changed the number of planes')
    # Fail here (STAGE 1), rather than during the algorithm loop (STAGE 2).
    if not isinstance(ps.contours, np.ndarray):
        # Some PlaneSet versions normalise input contours back to a list.
        # The production transform explicitly requires .shape on this field.
        ps.contours = pack_contours(ps.contours)
    if ps.contours.ndim != 3 or ps.contours.shape[0] != n or ps.contours.shape[2] != 3:
        raise ValueError(f'PlaneSet contour shape is invalid: {ps.contours.shape}')
    return ps

def build_frame(frame: str, entry: dict):
    xyz = read_ply(entry['ply'])
    metric = entry['metric']
    expected_points = int(metric['valid_points'])
    if len(xyz) != expected_points:
        raise ValueError(f'{frame}: PLY point count {len(xyz)} != Run I {expected_points}; '
                         'check that cloud-root and Run I results belong together')
    with np.load(entry['labels'], allow_pickle=False) as data:
        labels = np.asarray(data['labels'])
    num_planes = int(metric['detected_planes'])
    if (labels.ndim != 1 or len(labels) != len(xyz) or
            not np.issubdtype(labels.dtype, np.integer) or
            (len(labels) and (labels.min() < -1 or labels.max() >= num_planes)) or
            int(np.count_nonzero(labels >= 0)) != int(metric['assigned_points'])):
        raise ValueError(f'{frame}: Run I labels do not match its PLY and metrics')
    by_id = {int(row['plane_id']): row for row in entry['parameter_rows']}
    normals, offsets, centres, contours, widths, heights, features = [], [], [], [], [], [], []
    for pid in range(num_planes):
        row = by_id[pid]
        support = xyz[labels == pid]
        if len(support) < 3 or len(support) != int(row['inlier_count']):
            raise ValueError(f'{frame}: plane {pid} has inconsistent support count')
        n = np.array([float(row[k]) for k in ('nx','ny','nz')], dtype=float)
        d = float(row['d_m'])
        norm = np.linalg.norm(n)
        if not np.isfinite(norm) or norm < 1e-10 or not np.isfinite(d):
            raise ValueError(f'{frame}: invalid plane {pid} parameters')
        n, d = n/norm, d/norm
        centre = support.mean(axis=0)
        centre -= (centre @ n + d)*n   # Project centroid onto its fitted plane
        residual_mm = np.abs(support @ n + d)*1000.0
        rms = float(np.sqrt(np.mean(residual_mm**2)))
        p90 = float(np.percentile(residual_mm,90))
        if (abs(rms-float(row['fit_rms_mm'])) > 0.35 or
                abs(p90-float(row['fit_p90_mm'])) > 0.35):
            raise ValueError(f'{frame}: plane {pid} residuals disagree with Run I; '
                             'this usually means the wrong labels or point cloud were loaded')
        boundary = contour_points(support,n)
        # Conservative bounding rectangle descriptor, not true occupied area.
        helper = np.eye(3)[np.argmin(np.abs(n))]
        u = np.cross(n,helper)
        u /= np.linalg.norm(u)
        v = np.cross(n,u)
        uv = np.column_stack((support @ u,support @ v))
        width,height = sorted(np.ptp(uv,axis=0),reverse=True)
        normals.append(n); offsets.append(d); centres.append(centre); contours.append(boundary)
        widths.append(float(width)); heights.append(float(height))
        features.append({'frame_id': frame,'plane_id': pid,
                         'normal_x':float(n[0]),'normal_y':float(n[1]),'normal_z':float(n[2]),
                         'offset_m':float(d),'centroid_x_m':float(centre[0]),
                         'centroid_y_m':float(centre[1]),'centroid_z_m':float(centre[2]),
                         'support_points':len(support),'bbox_area_m2':float(width*height),
                         'extent_long_m':float(width),'extent_short_m':float(height),
                         'fit_rms_mm':rms,'fit_p90_mm':p90,'contour_vertices':len(boundary),
                         'voxel_mm':int(entry['metric']['scene_group'].split('_')[1][0:4]),
                         'detector':DETECTOR})
    return {'points':xyz,'set':create_plane_set(
                normals, offsets, centres, contours, widths, heights),
            'raw_contours': contours, 'features':features,
            'metric':metric,'ply':entry['ply']}


def pose_from_json(path: Path, source: str, target: str):
    content = json.loads(path.read_text(encoding='utf-8-sig'))
    if isinstance(content,dict):
        if content.get('source_frame') not in (None,source) or content.get('target_frame') not in (None,target):
            raise ValueError(f'Pose JSON frame IDs disagree with requested {source}->{target}: {path}')
        data = content['T_source_to_target']
    else:
        data = content
    T = np.asarray(data,dtype=float)
    if T.shape != (4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1],atol=1e-5):
        raise ValueError(f'Invalid 4x4 camera pose matrix: {path}')
    R = T[:3,:3]
    if not np.allclose(R.T@R,np.eye(3),atol=0.02) or abs(np.linalg.det(R)-1)>0.02:
        raise ValueError(f'Invalid rotation matrix in {path}')
    return T,content.get('registration',{}) if isinstance(content,dict) else {}


def estimate_pose(src_points, tgt_points, args):
    """FPFH+ICP registration; this does NOT perform plane detection."""
    try:
        import open3d as o3d
    except ImportError as e:
        raise RuntimeError('No prior pose found; install open3d to compute FPFH+ICP, '
                           'or supply --pose-dir containing relative_pose.json files.') from e
    reg = o3d.pipelines.registration
    v = args.registration_voxel
    def fpfh(xyz):
        cloud = o3d.geometry.PointCloud()
        cloud.points=o3d.utility.Vector3dVector(xyz)
        cloud=cloud.voxel_down_sample(v)
        if len(cloud.points) < 50:
            raise ValueError('Too few registration points')
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v*2.5,max_nn=40))
        descriptor=reg.compute_fpfh_feature(cloud,o3d.geometry.KDTreeSearchParamHybrid(radius=v*5,max_nn=100))
        return cloud,descriptor
    t0=time.perf_counter()
    source,src_feature=fpfh(src_points)
    target,tgt_feature=fpfh(tgt_points)
    ransac=reg.registration_ransac_based_on_feature_matching(
        source,target,src_feature,tgt_feature,mutual_filter=True,
        max_correspondence_distance=args.ransac_distance,
        estimation_method=reg.TransformationEstimationPointToPoint(False),ransac_n=4,
        checkers=[reg.CorrespondenceCheckerBasedOnEdgeLength(0.90),
                  reg.CorrespondenceCheckerBasedOnDistance(args.ransac_distance)],
        criteria=reg.RANSACConvergenceCriteria(100000,0.999))
    if ransac.fitness < args.min_registration_fitness:
        raise RuntimeError(f'FPFH registration fitness too low: {ransac.fitness:.3f}. '
                           'Use a verified relative pose instead.')
    icp=reg.registration_icp(source,target,args.icp_distance,ransac.transformation,
        reg.TransformationEstimationPointToPlane(),reg.ICPConvergenceCriteria(max_iteration=60))
    if icp.fitness < args.min_registration_fitness:
        raise RuntimeError(f'ICP registration fitness too low: {icp.fitness:.3f}')
    T=np.asarray(icp.transformation,dtype=float)
    return T, {'method':'FPFH_RANSAC_then_point_to_plane_ICP',
               'ransac_fitness':float(ransac.fitness),'ransac_rmse_m':float(ransac.inlier_rmse),
               'icp_fitness':float(icp.fitness),'icp_rmse_m':float(icp.inlier_rmse),
               'registration_runtime_s':time.perf_counter()-t0}


def find_pose(args, source: str, target: str, src_points, tgt_points):
    if args.pose_file is not None:
        return *pose_from_json(args.pose_file,source,target),str(args.pose_file)
    search = []
    if args.pose_dir is not None:
        search.extend([args.pose_dir/f'pose_{source}_{target}.json',
                       args.pose_dir/f'{source}_{target}'/'relative_pose.json'])
        search.extend(sorted(args.pose_dir.glob(f'attemptA_{source}_{target}_*/relative_pose.json'),reverse=True))
        search.extend(sorted(args.pose_dir.glob(f'{source}_to_{target}/relative_pose.json'),reverse=True))
    # Auto reuse of previous Attempt A results if script extracted beside results/.
    # Typical project location: experiments/plane_association_real/results
    # Also handle earlier nested placements of Attempt C.
    for old_results in (SCRIPT_DIR/'results', SCRIPT_DIR.parent/'results'):
        if old_results.is_dir():
            search.extend(sorted(old_results.glob(f'attemptA_{source}_{target}_*/relative_pose.json'),
                                 key=lambda p: p.stat().st_mtime, reverse=True))
            search.extend(sorted(old_results.glob(f'**/pose_{source}_{target}.json'),
                                 key=lambda p: p.stat().st_mtime, reverse=True))
            search.extend(sorted(old_results.glob(f'**/{source}_to_{target}/relative_pose.json'),
                                 key=lambda p: p.stat().st_mtime, reverse=True))
    for path in dict.fromkeys(search):
        if path.is_file():
            T, info=pose_from_json(path,source,target)
            return T,info,str(path)
    print(f'  No saved pose for {source}->{target}; estimating once via FPFH+ICP...',flush=True)
    T,info=estimate_pose(src_points,tgt_points,args)
    return T,info,'newly estimated'


def common_pair_geometry(src, tgt, source_id, target_id):
    ns,ds,cs=src.normals[source_id],src.offsets[source_id],src.centroids[source_id]
    nt,dt,ct=tgt.normals[target_id],tgt.offsets[target_id],tgt.centroids[target_id]
    _,dt_aligned=align_plane_sign(ns,nt,dt)
    distance=cs-ct
    lateral=distance-(distance@nt)*nt
    return {'normal_angle_deg':normal_angle_deg(ns,nt),
            'offset_difference_m':float(abs(ds-dt_aligned)),
            'centroid_distance_m':float(np.linalg.norm(distance)),
            'lateral_centroid_distance_m':float(np.linalg.norm(lateral))}


def enriched_row(raw, algorithm, source_frame, target_frame, source_set, target_set,
                 transformed_set, source_features, target_features):
    i,j=int(raw['source_plane_id']),int(raw['target_plane_id'])
    if not (0<=i<len(source_set) and 0<=j<len(target_set)):
        raise ValueError(f'{algorithm} emitted invalid IDs: {i}->{j}')
    out={'source_frame':source_frame,'target_frame':target_frame,
         'gating_algorithm':algorithm,'source_plane_id':i,'target_plane_id':j,
         'passed_gate':True}
    out.update(common_pair_geometry(transformed_set,target_set,i,j))
    # Source plane has been transformed into TARGET camera coordinates.
    for prefix,ps,pid,props in [('source',transformed_set,i,source_features[i]),
                                ('target',target_set,j,target_features[j])]:
        for idx,axis in enumerate('xyz'):
            out[f'{prefix}_normal_{axis}']=float(ps.normals[pid,idx])
            out[f'{prefix}_centroid_{axis}_m']=float(ps.centroids[pid,idx])
        out[f'{prefix}_offset_m']=float(ps.offsets[pid])
        out[f'{prefix}_support_points']=props['support_points']
        out[f'{prefix}_bbox_area_m2']=props['bbox_area_m2']
        out[f'{prefix}_fit_rms_mm']=props['fit_rms_mm']
        out[f'{prefix}_contour_vertices']=props['contour_vertices']
    out['area_bbox_ratio']=max(source_features[i]['bbox_area_m2'],target_features[j]['bbox_area_m2'],1e-9)/max(
        min(source_features[i]['bbox_area_m2'],target_features[j]['bbox_area_m2']),1e-9)
    # Preserve ALL metrics supplied by each original gate, with no modification.
    for k,v in raw.items():
        if k in ('source_plane_id','target_plane_id','passed_gate','algorithm'):
            continue
        out[f'gate_{k}']=v
    return out


def load_truth(path):
    if path is None:
        return None
    rows=csv_rows(path)
    required={'source_frame','target_frame','source_plane_id','target_plane_id'}
    if rows and not required <= set(rows[0]):
        raise ValueError(f'GT CSV needs columns {sorted(required)}')
    gt=defaultdict(set)
    for row in rows:
        key=(row['source_frame'],row['target_frame'])
        pair=(int(row['source_plane_id']),int(row['target_plane_id']))
        if pair in gt[key]:
            raise ValueError(f'Duplicate GT pair {key}, {pair}')
        gt[key].add(pair)
    return gt


def write_frame_files(frame, frame_data, root):
    write_csv(root/f'plane_features_{frame}.csv',frame_data['features'],list(frame_data['features'][0])
              if frame_data['features'] else ['frame_id','plane_id'])
    np.savez_compressed(root/f'plane_contours_{frame}.npz',**{
        f'plane_{i:03d}': contour for i,contour in enumerate(frame_data['raw_contours'])})


def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')



# ============================================================================
# STAGE 1. LOAD THE INPUTS AND THE FOUR CANDIDATE GATING ALGORITHMS
# ============================================================================
# The imports at the top load your original algorithm modules from this folder.
# This stage only loads EXISTING Run I detector outputs and the associated PLYs.
# No plane detection, similarity scoring, or assignment occurs here.


def load_inputs():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--run', type=Path, required=True,
                        help='Completed Run I detector results directory')
    parser.add_argument('--cloud-root', type=Path, required=True,
                        help='Completed Run I point cloud batch directory')
    parser.add_argument('--expected-batch', default=BATCH)
    parser.add_argument('--source', default='000000')
    parser.add_argument('--targets', nargs='+', default=['000001'])
    parser.add_argument('--voxel-mm', type=int, choices=[5, 10], default=5)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--pose-file', type=Path, default=None,
                        help='Existing relative_pose.json, only for one target')
    parser.add_argument('--pose-dir', type=Path, default=None,
                        help='Folder containing saved relative poses for frame pairs')
    parser.add_argument('--registration-voxel', type=float, default=0.025)
    parser.add_argument('--ransac-distance', type=float, default=0.075)
    parser.add_argument('--icp-distance', type=float, default=0.04)
    parser.add_argument('--min-registration-fitness', type=float, default=0.20)
    parser.add_argument('--normal-sigma-deg', type=float, default=2.0,
                        help='Assumed normal uncertainty, not a measured covariance')
    parser.add_argument('--pose-rotation-sigma-deg', type=float, default=2.0,
                        help='Assumed pose rotation uncertainty, not measured')
    parser.add_argument('--ground-truth', type=Path, default=None,
                        help='Optional exhaustive ground-truth pairs CSV')
    args = parser.parse_args()

    if (not re.fullmatch(r'\d{6}', args.source) or not args.targets or
            any(not re.fullmatch(r'\d{6}', t) or t == args.source
                for t in args.targets) or
            len(set(args.targets)) != len(args.targets)):
        parser.error('Source/target IDs must be distinct six-digit frame IDs')
    if args.pose_file is not None and len(args.targets) != 1:
        parser.error('--pose-file requires a single target; use --pose-dir for several')
    if args.normal_sigma_deg <= 0 or args.pose_rotation_sigma_deg <= 0:
        parser.error('Uncertainty parameters must be positive')

    args.run = args.run.expanduser().resolve()
    args.cloud_root = args.cloud_root.expanduser().resolve()
    if not args.run.is_dir() or not args.cloud_root.is_dir():
        parser.error('The Run I results folder or point-cloud directory does not exist')
    if args.pose_dir is not None:
        args.pose_dir = args.pose_dir.expanduser().resolve()
    if args.pose_file is not None:
        args.pose_file = args.pose_file.expanduser().resolve()

    default_output = (SCRIPT_DIR / 'results' /
                      f'attemptC_candidate_gating_{datetime.now():%Y%m%d_%H%M%S}')
    output_dir = (args.output or default_output).expanduser().resolve()
    if output_dir.exists():
        parser.error(f'Refusing to overwrite an existing output directory: {output_dir}')

    frames = [args.source] + args.targets
    entries = locate_run_frames(args.run, args.cloud_root, set(frames),
                                args.voxel_mm, args.expected_batch)
    truth = load_truth(args.ground_truth)
    if truth is not None and not all((args.source, t) in truth for t in args.targets):
        parser.error('Ground-truth CSV must include every requested frame pair')

    print('STAGE 1 | Loading existing Run I results (NO detection)...', flush=True)
    frame_data = {}
    for frame in frames:
        frame_data[frame] = build_frame(frame, entries[frame])
        print(f'  Frame {frame}: {len(frame_data[frame]["points"]):,} points, '
              f'{len(frame_data[frame]["set"])} planes', flush=True)
    return args, output_dir, frame_data, truth


# ============================================================================
# STAGE 2. APPLY THE INPUTS TO THE FOUR ALGORITHMS AND GENERATE OUTPUTS
# ============================================================================
# Every method receives the exact same pair of PlaneSets and source-to-target
# pose. Its own gate_candidates() function is called without modification.
# The resulting candidates are enriched with plane features for Similarity.


def run_gating_algorithms(args, frame_data):
    results = []
    pose_records = {}
    print('\nSTAGE 2 | Running four Candidate Gating algorithms...', flush=True)

    for target_frame in args.targets:
        source_frame = args.source
        pair_id = f'{source_frame}_to_{target_frame}'
        source = frame_data[source_frame]
        target = frame_data[target_frame]

        # One COMMON pose for all four gating algorithms for this frame pair.
        T, registration, pose_origin = find_pose(
            args, source_frame, target_frame, source['points'], target['points']
        )
        R, t = T[:3, :3], T[:3, 3]
        source_planes = source['set']
        target_planes = target['set']
        # Preflight the exact shared-library transform BEFORE starting a method.
        # It requires contours to be a NumPy array (not a list).
        if not isinstance(source_planes.contours, np.ndarray):
            raise TypeError('Source PlaneSet.contours must be a NumPy array')
        if not isinstance(target_planes.contours, np.ndarray):
            raise TypeError('Target PlaneSet.contours must be a NumPy array')
        transformed_source = transform_plane_set(source_planes, R, t)
        possible_pairs = len(source_planes) * len(target_planes)
        pose_records[pair_id] = {
            'source_frame': source_frame,
            'target_frame': target_frame,
            'T_source_to_target': T.tolist(),
            'pose_origin': pose_origin,
            'registration': registration,
            'convention': 'p_target = R @ p_source + t; depth optical coordinates',
        }
        print(f'  {pair_id}: {possible_pairs} possible pairs; pose={pose_origin}',
              flush=True)

        for algorithm, module in METHODS.items():
            start = time.perf_counter()
            if algorithm == 'normal_based_gating':
                candidates = module.gate_candidates(
                    source_planes, target_planes, R, t,
                    normal_sigma_deg=args.normal_sigma_deg,
                    pose_rotation_sigma_deg=args.pose_rotation_sigma_deg,
                )
            else:
                candidates = module.gate_candidates(
                    source_planes, target_planes, R, t
                )
            runtime_ms = (time.perf_counter() - start) * 1000.0

            pair_ids = [(int(row['source_plane_id']), int(row['target_plane_id']))
                        for row in candidates]
            if len(set(pair_ids)) != len(pair_ids):
                raise RuntimeError(f'{algorithm} returned duplicate candidate pairs')

            # Uniform output for the next Similarity stage. No scoring occurs.
            similarity_inputs = [
                enriched_row(row, algorithm, source_frame, target_frame,
                             source_planes, target_planes, transformed_source,
                             source['features'], target['features'])
                for row in candidates
            ]
            results.append({
                'pair_id': pair_id,
                'source_frame': source_frame,
                'target_frame': target_frame,
                'algorithm': algorithm,
                'source_planes': len(source_planes),
                'target_planes': len(target_planes),
                'possible_pairs': possible_pairs,
                'runtime_ms': runtime_ms,
                'pose_origin': pose_origin,
                'raw_candidates': candidates,
                'candidate_ids': pair_ids,
                'similarity_inputs': similarity_inputs,
            })
            print(f'    {algorithm:<31} {len(candidates):>3} candidates '
                  f'({runtime_ms:.3f} ms)', flush=True)

    return results, pose_records


# ============================================================================
# STAGE 3. EVALUATE THE FOUR ALGORITHM OUTPUTS AND GENERATE A SUMMARY
# ============================================================================
# Without manual GT, reduction rates/counts are descriptive, NOT accuracy.
# Precision/recall/F1 are calculated only with exhaustive supplied GT.


def evaluate_outputs(args, results, truth):
    print('\nSTAGE 3 | Evaluating Candidate Gating outputs...', flush=True)
    comparison_by_frame = []

    for item in results:
        predicted = set(item['candidate_ids'])
        possible = item['possible_pairs']
        row = {
            'source_frame': item['source_frame'],
            'target_frame': item['target_frame'],
            'algorithm': item['algorithm'],
            'detector': DETECTOR,
            'voxel_mm': args.voxel_mm,
            'source_planes': item['source_planes'],
            'target_planes': item['target_planes'],
            'possible_pairs': possible,
            'candidate_pairs': len(predicted),
            'candidate_reduction_rate': (1 - len(predicted) / possible
                                         if possible else ''),
            'source_planes_with_candidates': len({i for i, _ in predicted}),
            'target_planes_with_candidates': len({j for _, j in predicted}),
            'candidate_gating_runtime_ms': item['runtime_ms'],
            'pose_origin': item['pose_origin'],
            'gt_pairs': '',
            'gt_retained': '',
            'candidate_precision': '',
            'candidate_recall': '',
            'candidate_f1': '',
        }
        if truth is not None:
            gt = truth[(item['source_frame'], item['target_frame'])]
            if any(not (0 <= i < item['source_planes'] and
                        0 <= j < item['target_planes']) for i, j in gt):
                raise ValueError('Ground truth contains an out-of-range plane ID')
            tp, fp, fn = len(predicted & gt), len(predicted - gt), len(gt - predicted)
            precision = tp / (tp + fp) if tp + fp else (1.0 if not gt else 0.0)
            recall = tp / (tp + fn) if tp + fn else 1.0
            row.update({
                'gt_pairs': len(gt), 'gt_retained': tp,
                'candidate_precision': precision,
                'candidate_recall': recall,
                'candidate_f1': (2 * precision * recall / (precision + recall)
                                 if precision + recall else 0.0),
            })
        comparison_by_frame.append(row)

    comparison_summary = []
    for algorithm in METHODS:
        rows = [r for r in comparison_by_frame if r['algorithm'] == algorithm]
        comparison_summary.append({
            'algorithm': algorithm,
            'evaluated_frame_pairs': len(rows),
            'mean_possible_pairs': float(np.mean([r['possible_pairs'] for r in rows])),
            'mean_candidates': float(np.mean([r['candidate_pairs'] for r in rows])),
            'mean_reduction_rate': float(np.mean([
                r['candidate_reduction_rate'] for r in rows
            ])) if all(r['candidate_reduction_rate'] != '' for r in rows) else '',
            'mean_runtime_ms': float(np.mean([
                r['candidate_gating_runtime_ms'] for r in rows
            ])),
            'mean_candidate_recall': (float(np.mean([
                r['candidate_recall'] for r in rows
            ])) if truth is not None else ''),
        })
        print(f'  {algorithm:<31} average candidates: '
              f'{comparison_summary[-1]["mean_candidates"]:.2f}', flush=True)

    return comparison_by_frame, comparison_summary


# ============================================================================
# STAGE 4. SAVE THE SUMMARY TO CSV AND EXPORT SIMILARITY INPUTS
# ============================================================================
# comparison_by_frame.csv and comparison_summary.csv are the evaluation tables.
# similarity_input_all.csv / per-algorithm similarity_input.csv are the direct
# handoff to Similarity Evaluation (not final matched plane correspondences).


def save_results(args, output_dir, frame_data, truth, results, pose_records,
                 comparison_by_frame, comparison_summary):
    print('\nSTAGE 4 | Saving CSV results and Similarity inputs...', flush=True)
    output_dir.mkdir(parents=True)

    for frame, data in frame_data.items():
        write_frame_files(frame, data, output_dir)

    for pair_id, pose_info in pose_records.items():
        pair_dir = output_dir / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        write_json(pair_dir / 'relative_pose.json', pose_info)

    all_similarity_rows = []
    union_fields = ['source_frame', 'target_frame', 'gating_algorithm',
                    'source_plane_id', 'target_plane_id']
    for item in results:
        folder = output_dir / item['pair_id'] / item['algorithm']
        folder.mkdir(parents=True, exist_ok=True)
        raw = item['raw_candidates']
        similarity = item['similarity_inputs']
        raw_fields = list(dict.fromkeys(k for row in raw for k in row)) or [
            'source_plane_id', 'target_plane_id'
        ]
        similarity_fields = list(dict.fromkeys(
            union_fields + [k for row in similarity for k in row]
        ))
        write_csv(folder / 'candidate_pairs_raw.csv', raw, raw_fields)
        write_csv(folder / 'similarity_input.csv', similarity, similarity_fields)
        all_similarity_rows.extend(similarity)
        union_fields = list(dict.fromkeys(union_fields + similarity_fields))

    for target_frame in args.targets:
        pair_id = f'{args.source}_to_{target_frame}'
        this_pair = [r for r in all_similarity_rows
                     if r['source_frame'] == args.source and
                     r['target_frame'] == target_frame]
        write_csv(output_dir / pair_id / 'similarity_input_all.csv',
                  this_pair, union_fields)

    write_csv(output_dir / 'comparison_by_frame.csv', comparison_by_frame,
              list(comparison_by_frame[0]))
    write_csv(output_dir / 'comparison_summary.csv', comparison_summary,
              list(comparison_summary[0]))
    write_csv(output_dir / 'similarity_input_all.csv',
              all_similarity_rows, union_fields)

    write_json(output_dir / 'summary.json', {
        'experiment': 'Lab Note 11 - Attempt C: Candidate Gating (four methods)',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'input': 'Completed Run I Iterative RANSAC plane outputs; NO detection rerun',
        'run': str(args.run),
        'cloud_root': str(args.cloud_root),
        'source_frame': args.source,
        'targets': args.targets,
        'voxel_mm': args.voxel_mm,
        'gating_methods': list(METHODS),
        'parameters': {name: module.DEFAULT_PARAMETERS for name, module in METHODS.items()},
        'assumed_uncertainty': {
            'normal_sigma_deg': args.normal_sigma_deg,
            'pose_rotation_sigma_deg': args.pose_rotation_sigma_deg,
            'measured': False,
        },
        'pose_provenance': {
            pair_id: {
                'pose_origin': info['pose_origin'],
                'registration': info['registration'],
            } for pair_id, info in pose_records.items()
        },
        'ground_truth_provided': truth is not None,
        'notes': [
            'Calls four original gate_candidates implementations without changing their rules.',
            'No Similarity Evaluation or Assignment is performed in Attempt C.',
            'Source plane features in Similarity CSV are transformed into target frame.',
            'Convex hull contours may span disconnected coplanar segments.',
            'Candidate reduction and counts are not matching accuracy.',
            'Mahalanobis normal covariance uses assumed rather than measured uncertainty.',
        ],
    })
    print(f'\nDone. Results: {output_dir}', flush=True)
    print(f'Comparison: {output_dir / "comparison_summary.csv"}', flush=True)
    print(f'Similarity inputs: {output_dir / "similarity_input_all.csv"}', flush=True)


# ============================================================================
# MAIN EXECUTION: FOUR STEPS, IN THE SAME ORDER AS THE EXPERIMENT PLAN
# ============================================================================

def main():
    # 1. Load the inputs and the four existing algorithm modules.
    args, output_dir, frames, ground_truth = load_inputs()

    # 2. Apply those inputs to each gating algorithm; collect raw candidates.
    results, poses = run_gating_algorithms(args, frames)

    # 3. Evaluate candidate counts, reduction, timing, and optional GT recall.
    by_frame, summary = evaluate_outputs(args, results, ground_truth)

    # 4. Save comparison CSVs and the candidate pairs for Similarity Evaluation.
    save_results(args, output_dir, frames, ground_truth, results,
                 poses, by_frame, summary)


if __name__ == '__main__':
    main()
