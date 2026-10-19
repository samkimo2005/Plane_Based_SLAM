#!/usr/bin/env python3
"""Lab Book 9: depth-only planar ROI diagnostic (native D435i depth coordinates).

Run without arguments to paste paths interactively, then drag a rectangular ROI
on the DEPTH preview. The RGB image is displayed only as a visual reference;
it is NOT pixel-aligned with depth and its coordinates must not be reused.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
import cv2 as cv
import numpy as np


def ask_path(prompt):
    return Path(input(prompt).strip().strip('"').strip("'")).expanduser().resolve()


def find_params(depth_path):
    for parent in (depth_path.parent, *depth_path.parents):
        candidate = parent / 'camera_parameters.json'
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('Cannot find camera_parameters.json above Depth PNG. Specify --params.')


def fit_pca(points):
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    n = vh[-1]
    n /= np.linalg.norm(n)
    return n, -float(n @ centroid)


def robust_plane(points, seed=42, threshold=0.016, iterations=400):
    rng = np.random.default_rng(seed)
    if len(points) > 12000:
        sample = points[rng.choice(len(points), 12000, replace=False)]
    else:
        sample = points
    best_n = None
    best_d = None
    best_count = -1
    for _ in range(iterations):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b-a, c-a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        d = -float(n @ a)
        count = int(np.count_nonzero(np.abs(sample @ n + d) <= threshold))
        if count > best_count:
            best_n, best_d, best_count = n, d, count
    if best_n is None:
        raise ValueError('RANSAC could not sample a valid plane.')
    inlier = np.abs(points @ best_n + best_d) <= threshold
    if np.count_nonzero(inlier) >= 3:
        n, d = fit_pca(points[inlier])
    else:
        n, d = best_n, best_d
    residual = np.abs(points @ n + d)
    return n, float(d), residual


def stats(a):
    return {'median_mm': float(np.median(a)*1000),
            'p90_mm': float(np.percentile(a,90)*1000),
            'rms_mm': float(np.sqrt(np.mean(a*a))*1000),
            'within_8mm_percent': float(np.mean(a <= 0.008)*100),
            'within_16mm_percent': float(np.mean(a <= 0.016)*100)}


def main():
    p = argparse.ArgumentParser(description='Analyse a wall ROI using unaligned native depth PNG.')
    p.add_argument('--depth', type=Path, help='Raw 16-bit depth PNG')
    p.add_argument('--params', type=Path, help='Camera parameter JSON')
    p.add_argument('--rgb', type=Path, help='RGB PNG (reference only; NOT registered)')
    p.add_argument('--roi', nargs=4, type=int, metavar=('X','Y','W','H'), help='Skip GUI and use pixel ROI in native depth image')
    p.add_argument('--min-depth', type=float, default=0.1)
    p.add_argument('--max-depth', type=float, default=5.0)
    p.add_argument('--output', type=Path, help='Output directory')
    a = p.parse_args()
    depth_path = a.depth.resolve() if a.depth else ask_path('Paste full path to RAW Depth PNG: ')
    params_path = a.params.resolve() if a.params else find_params(depth_path)
    with params_path.open(encoding='utf-8') as f:
        meta = json.load(f)
    K = meta['streams']['depth']
    scale = float(meta['depth_scale_m_per_unit'])
    raw = cv.imread(str(depth_path), cv.IMREAD_UNCHANGED)
    if raw is None or raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError('Input must be a single-channel uint16 RAW depth PNG.')
    h, w = raw.shape
    if (int(K['width']), int(K['height'])) != (w,h):
        raise ValueError('Depth dimensions differ from saved camera intrinsics.')
    z = raw.astype(np.float64) * scale
    valid = (raw > 0) & np.isfinite(z) & (z >= a.min_depth) & (z <= a.max_depth)
    if not np.any(valid):
        raise ValueError('No valid depth within selected range.')
    lo, hi = np.percentile(z[valid], [5,95])
    if hi <= lo: hi = lo + 0.1
    grey = (255 * np.clip((z - lo) / (hi-lo), 0,1)).astype(np.uint8)
    preview = cv.applyColorMap(255-grey, cv.COLORMAP_TURBO)
    preview[~valid] = (0,0,0)
    cv.putText(preview, 'Select ONLY one flat wall in DEPTH image (not RGB)', (8,24),
               cv.FONT_HERSHEY_SIMPLEX, 0.47,(255,255,255),1,cv.LINE_AA)
    if a.rgb:
        rgb = cv.imread(str(a.rgb.resolve()))
        if rgb is not None:
            cv.imshow('RGB reference ONLY - pixels NOT aligned to depth', rgb)
    if a.roi:
        x,y,rw,rh = a.roi
    else:
        x,y,rw,rh = map(int, cv.selectROI('Native depth ROI - ENTER accept / C cancel',preview,False,False))
        cv.destroyAllWindows()
    if rw < 15 or rh < 15 or x < 0 or y < 0 or x+rw>w or y+rh>h:
        raise ValueError('ROI must be at least 15x15 pixels and inside image.')
    mask = np.zeros((h,w), bool)
    mask[y:y+rh,x:x+rw] = True
    inside = mask & valid
    region_count = int(rw*rh)
    nvalid = int(inside.sum())
    if nvalid < 100:
        raise ValueError('ROI has fewer than 100 valid depth points. Choose another wall patch.')
    vv, uu = np.nonzero(inside)
    zz = z[vv,uu]
    pts = np.column_stack(((uu-float(K['cx']))*zz/float(K['fx']),
                           (vv-float(K['cy']))*zz/float(K['fy']),zz))
    n,d,res = robust_plane(pts)
    out_dir = a.output.resolve() if a.output else Path.cwd()/'experiments'/'plane_detection'/'results'/('depth_quality_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = preview.copy()
    cv.rectangle(overlay,(x,y),(x+rw-1,y+rh-1),(255,255,255),2)
    cv.imwrite(str(out_dir/'selected_depth_roi.png'),overlay)
    rmap = np.zeros((h,w),np.uint8)
    rmap[vv,uu] = np.clip(res*1000/32*255,0,255).astype(np.uint8)
    heat = cv.applyColorMap(rmap,cv.COLORMAP_TURBO)
    heat[~inside] = (240,240,240)
    cv.imwrite(str(out_dir/'plane_residual_map.png'),heat)
    result = {
        'depth_image': str(depth_path), 'camera_parameters':str(params_path),
        'roi_depth_pixels': {'x':x,'y':y,'width':rw,'height':rh},
        'note':'ROI selected in NATIVE DEPTH coordinates, no RGB alignment. Fit is internal reference, not GT.',
        'valid_depth_definition':f'raw>0 and {a.min_depth}<=z<={a.max_depth} metres',
        'roi_pixels':region_count,'valid_depth_points':nvalid,
        'valid_depth_percent':nvalid/region_count*100,
        'depth_m':{'min':float(zz.min()),'median':float(np.median(zz)),
                   'p90':float(np.percentile(zz,90)),'max':float(zz.max())},
        'plane_normal':n.tolist(),'plane_offset_m':d,
        'plane_fit':'RANSAC 16 mm then PCA refine; residuals evaluated on ALL valid ROI points',
        'residual_all_roi':stats(res),
        'residual_fit_inliers':stats(res[res<=0.016]) if np.any(res<=0.016) else None,
        'caveat':'ROI contamination, surface curvature, and distance affect errors. This is not independent plane ground truth.'
    }
    (out_dir/'depth_quality_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('\n--- Wall Depth Quality Diagnostic ---')
    print(f'ROI: x={x}, y={y}, w={rw}, h={rh}')
    print(f'Valid depth: {nvalid:,}/{region_count:,} = {result["valid_depth_percent"]:.2f}%')
    print(f'Median depth: {result["depth_m"]["median"]:.3f} m')
    print(f'Plane residual (ALL valid ROI points): RMS {result["residual_all_roi"]["rms_mm"]:.2f} mm; '
          f'P90 {result["residual_all_roi"]["p90_mm"]:.2f} mm')
    print(f'Within 8 mm: {result["residual_all_roi"]["within_8mm_percent"]:.2f}% | '
          f'within 16 mm: {result["residual_all_roi"]["within_16mm_percent"]:.2f}%')
    print(f'Results saved: {out_dir}')

if __name__ == '__main__':
    main()
