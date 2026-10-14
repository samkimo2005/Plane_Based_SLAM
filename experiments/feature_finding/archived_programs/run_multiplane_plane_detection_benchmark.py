import csv
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "synthetic_multiplane_benchmark"
RESULTS_DIR = PROJECT_ROOT / "results" / "feature_finding"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "multiplane_plane_detection_results.csv"


# ==========================================================
# Common benchmark rules
# ==========================================================

FINAL_DISTANCE_THRESHOLD = 0.02

MATCH_NORMAL_THRESHOLD_DEG = 10.0
MATCH_OFFSET_THRESHOLD_M = 0.10

MIN_PLANE_SUPPORT = 120
MAX_PLANES = 6

K_NEIGHBORS = 30

# Un-timed warm-up avoids the isolated first-call timing spike
# seen in the original single-plane benchmark.
DO_WARMUP = True


# ==========================================================
# Method-specific fixed parameters
# ==========================================================

# Iterative RANSAC
RANSAC_ITERATIONS = 500

# Multi-object / multi-model RANSAC-inspired baseline
MULTI_HYPOTHESES = 1000
MULTI_MIN_HYPOTHESIS_SUPPORT = 120
MULTI_NORMAL_CLUSTER_DEG = 5.0
MULTI_OFFSET_CLUSTER_M = 0.05
MULTI_MIN_CLUSTER_VOTES = 4

# Surface-normal / surfel grouping
SURFEL_MAX_CURVATURE = 0.08
SURFEL_NORMAL_GROUP_DEG = 12.0
SURFEL_OFFSET_BIN_M = 0.05
SURFEL_OFFSET_GROUP_M = 0.04
SURFEL_OFFSET_NMS_M = 0.08

# Region Growing
RG_SMOOTHNESS_DEG = 15.0
RG_LOCAL_DISTANCE_M = 0.03
RG_MIN_REGION_SIZE = 100

# Hough
HOUGH_MAX_CURVATURE = 0.08
HOUGH_THETA_BIN_DEG = 10.0
HOUGH_PHI_BIN_DEG = 10.0
HOUGH_D_BIN_M = 0.05
HOUGH_MIN_VOTES = 40


# ==========================================================
# Generic geometry
# ==========================================================

def fit_plane_three(p1, p2, p3):
    normal = np.cross(p2 - p1, p3 - p1)
    norm = np.linalg.norm(normal)

    if norm < 1e-10:
        return None

    normal = normal / norm
    d = -float(np.dot(normal, p1))

    return canonicalise_plane(normal, d)


def fit_plane_svd(points):
    if len(points) < 3:
        return None

    centroid = np.mean(points, axis=0)
    centred = points - centroid

    _, _, vh = np.linalg.svd(
        centred,
        full_matrices=False,
    )

    normal = vh[-1]
    norm = np.linalg.norm(normal)

    if norm < 1e-10:
        return None

    normal = normal / norm
    d = -float(np.dot(normal, centroid))

    return canonicalise_plane(normal, d)


def canonicalise_plane(normal, d):
    normal = np.asarray(normal, dtype=np.float64)

    index = int(np.argmax(np.abs(normal)))

    if normal[index] < 0:
        normal = -normal
        d = -d

    return normal, float(d)


def plane_angle_deg(normal_a, normal_b):
    dot_value = float(np.dot(normal_a, normal_b))
    dot_value = np.clip(abs(dot_value), 0.0, 1.0)
    return float(np.degrees(np.arccos(dot_value)))


def align_plane_to_reference(normal, d, ref_normal):
    if np.dot(normal, ref_normal) < 0:
        return -normal, -d

    return normal, d


def build_model(points, normal, d, source_support=None):
    """
    Common final adapter used by ALL methods.
    """
    normal, d = canonicalise_plane(normal, d)

    distances = np.abs(points @ normal + d)
    mask = distances < FINAL_DISTANCE_THRESHOLD
    support = int(np.sum(mask))

    if support < MIN_PLANE_SUPPORT:
        return None

    return {
        "normal": normal,
        "d": float(d),
        "mask": mask,
        "support": support,
        "source_support": source_support,
    }


def merge_duplicate_models(models, normal_threshold_deg=5.0, offset_threshold_m=0.05):
    """
    Remove duplicate detections of the same geometric plane.
    Keep the stronger model.
    """
    models = sorted(
        models,
        key=lambda model: model["support"],
        reverse=True,
    )

    accepted = []

    for model in models:
        duplicate = False

        for existing in accepted:
            n2, d2 = align_plane_to_reference(
                model["normal"].copy(),
                model["d"],
                existing["normal"],
            )

            if (
                plane_angle_deg(existing["normal"], n2)
                <= normal_threshold_deg
                and abs(existing["d"] - d2)
                <= offset_threshold_m
            ):
                duplicate = True
                break

        if not duplicate:
            accepted.append(model)

        if len(accepted) >= MAX_PLANES:
            break

    return accepted


# ==========================================================
# Local normals / curvature
# ==========================================================

def estimate_local_geometry(points):
    if len(points) <= K_NEIGHBORS:
        return None

    tree = cKDTree(points)

    _, neighbour_indices = tree.query(
        points,
        k=K_NEIGHBORS + 1,
    )

    neighbour_indices = neighbour_indices[:, 1:]

    neighbours = points[neighbour_indices]
    centroids = np.mean(neighbours, axis=1)

    centred = neighbours - centroids[:, None, :]

    covariance = np.einsum(
        "nki,nkj->nij",
        centred,
        centred,
    )

    covariance /= float(K_NEIGHBORS)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    normals = eigenvectors[:, :, 0]

    eigenvalue_sum = np.sum(eigenvalues, axis=1)

    curvature = np.ones(len(points), dtype=np.float64)

    valid = eigenvalue_sum > 1e-12

    curvature[valid] = (
        eigenvalues[valid, 0]
        / eigenvalue_sum[valid]
    )

    # Canonicalise local normal sign.
    for i in range(len(normals)):
        n, _ = canonicalise_plane(normals[i], 0.0)
        normals[i] = n

    return normals, curvature, neighbour_indices


# ==========================================================
# 1. Iterative RANSAC
# ==========================================================

def detect_iterative_ransac(points, rng):
    remaining_indices = np.arange(len(points))
    models = []

    for _ in range(MAX_PLANES):
        if len(remaining_indices) < MIN_PLANE_SUPPORT:
            break

        remaining_points = points[remaining_indices]

        best_support = 0
        best_plane = None
        best_mask_local = None

        for _ in range(RANSAC_ITERATIONS):
            sample_indices = rng.choice(
                len(remaining_points),
                size=3,
                replace=False,
            )

            fitted = fit_plane_three(
                remaining_points[sample_indices[0]],
                remaining_points[sample_indices[1]],
                remaining_points[sample_indices[2]],
            )

            if fitted is None:
                continue

            normal, d = fitted

            distances = np.abs(
                remaining_points @ normal + d
            )

            mask_local = (
                distances
                < FINAL_DISTANCE_THRESHOLD
            )

            support = int(np.sum(mask_local))

            if support > best_support:
                best_support = support
                best_plane = (normal, d)
                best_mask_local = mask_local

        if (
            best_plane is None
            or best_support < MIN_PLANE_SUPPORT
        ):
            break

        refined = fit_plane_svd(
            remaining_points[best_mask_local]
        )

        if refined is None:
            break

        normal, d = refined

        model = build_model(
            points,
            normal,
            d,
            source_support=best_support,
        )

        if model is None:
            break

        models.append(model)

        # Sequential extraction: remove only the points that
        # supported the current plane in the remaining cloud.
        remove_mask = np.abs(
            remaining_points @ normal + d
        ) < FINAL_DISTANCE_THRESHOLD

        remaining_indices = remaining_indices[
            ~remove_mask
        ]

    return merge_duplicate_models(models)


# ==========================================================
# 2. Multi-Object / Multi-Model RANSAC-inspired
# ==========================================================

def detect_multi_object_ransac(points, rng):
    hypothesis_normals = []
    hypothesis_ds = []
    hypothesis_supports = []

    for _ in range(MULTI_HYPOTHESES):
        sample = rng.choice(
            len(points),
            size=3,
            replace=False,
        )

        fitted = fit_plane_three(
            points[sample[0]],
            points[sample[1]],
            points[sample[2]],
        )

        if fitted is None:
            continue

        normal, d = fitted

        support = int(
            np.sum(
                np.abs(points @ normal + d)
                < FINAL_DISTANCE_THRESHOLD
            )
        )

        if support >= MULTI_MIN_HYPOTHESIS_SUPPORT:
            hypothesis_normals.append(normal)
            hypothesis_ds.append(d)
            hypothesis_supports.append(support)

    if len(hypothesis_normals) == 0:
        return []

    normals = np.asarray(hypothesis_normals)
    ds = np.asarray(hypothesis_ds)
    supports = np.asarray(hypothesis_supports)

    order = np.argsort(supports)[::-1]

    clusters = []

    for index in order:
        normal = normals[index]
        d = float(ds[index])
        support = int(supports[index])

        assigned = False

        for cluster in clusters:
            n2, d2 = align_plane_to_reference(
                normal.copy(),
                d,
                cluster["normal"],
            )

            if (
                plane_angle_deg(
                    cluster["normal"],
                    n2,
                )
                <= MULTI_NORMAL_CLUSTER_DEG
                and abs(
                    cluster["d"]
                    - d2
                )
                <= MULTI_OFFSET_CLUSTER_M
            ):
                cluster["votes"] += 1
                assigned = True
                break

        if not assigned:
            clusters.append(
                {
                    "normal": normal.copy(),
                    "d": d,
                    "representative_support": support,
                    "votes": 1,
                }
            )

    models = []

    for cluster in clusters:
        if cluster["votes"] < MULTI_MIN_CLUSTER_VOTES:
            continue

        initial_mask = (
            np.abs(
                points @ cluster["normal"]
                + cluster["d"]
            )
            < FINAL_DISTANCE_THRESHOLD
        )

        if np.sum(initial_mask) < MIN_PLANE_SUPPORT:
            continue

        refined = fit_plane_svd(
            points[initial_mask]
        )

        if refined is None:
            continue

        normal, d = refined

        model = build_model(
            points,
            normal,
            d,
            source_support=cluster["votes"],
        )

        if model is not None:
            models.append(model)

    return merge_duplicate_models(models)


# ==========================================================
# 3. Surface-normal / surfel grouping
# ==========================================================

def detect_surface_normal(points, rng=None):
    geometry = estimate_local_geometry(points)

    if geometry is None:
        return []

    normals, curvature, _ = geometry

    planar_indices = np.flatnonzero(
        curvature <= SURFEL_MAX_CURVATURE
    )

    if len(planar_indices) < MIN_PLANE_SUPPORT:
        return []

    # Greedy orientation grouping.
    groups = []

    for index in planar_indices[
        np.argsort(curvature[planar_indices])
    ]:
        normal = normals[index]

        assigned = False

        for group in groups:
            if (
                plane_angle_deg(
                    normal,
                    group["normal"],
                )
                <= SURFEL_NORMAL_GROUP_DEG
            ):
                group["indices"].append(int(index))
                assigned = True
                break

        if not assigned:
            groups.append(
                {
                    "normal": normal.copy(),
                    "indices": [int(index)],
                }
            )

    models = []

    for group in groups:
        indices = np.asarray(
            group["indices"],
            dtype=np.int64,
        )

        if len(indices) < MIN_PLANE_SUPPORT:
            continue

        representative_normal = group["normal"]

        offsets = -(
            points[indices]
            @ representative_normal
        )

        minimum = float(np.min(offsets))
        maximum = float(np.max(offsets))

        if maximum - minimum < 1e-12:
            candidate_centres = [
                float(np.mean(offsets))
            ]
        else:
            edges = np.arange(
                minimum,
                maximum + SURFEL_OFFSET_BIN_M,
                SURFEL_OFFSET_BIN_M,
            )

            if len(edges) < 2:
                edges = np.array(
                    [
                        minimum,
                        minimum + SURFEL_OFFSET_BIN_M,
                    ]
                )

            histogram, edges = np.histogram(
                offsets,
                bins=edges,
            )

            bin_order = np.argsort(histogram)[::-1]

            candidate_centres = []

            for bin_index in bin_order:
                if histogram[bin_index] < MIN_PLANE_SUPPORT:
                    break

                centre = float(
                    0.5
                    * (
                        edges[bin_index]
                        + edges[bin_index + 1]
                    )
                )

                if any(
                    abs(centre - old)
                    < SURFEL_OFFSET_NMS_M
                    for old in candidate_centres
                ):
                    continue

                candidate_centres.append(centre)

                if len(candidate_centres) >= MAX_PLANES:
                    break

        for centre_d in candidate_centres:
            selected = indices[
                np.abs(offsets - centre_d)
                <= SURFEL_OFFSET_GROUP_M
            ]

            if len(selected) < MIN_PLANE_SUPPORT:
                continue

            refined = fit_plane_svd(
                points[selected]
            )

            if refined is None:
                continue

            normal, d = refined

            model = build_model(
                points,
                normal,
                d,
                source_support=len(selected),
            )

            if model is not None:
                models.append(model)

    return merge_duplicate_models(models)


# ==========================================================
# 4. Region Growing
# ==========================================================

def detect_region_growing(points, rng=None):
    geometry = estimate_local_geometry(points)

    if geometry is None:
        return []

    normals, curvature, neighbour_indices = geometry

    assigned = np.zeros(
        len(points),
        dtype=bool,
    )

    cos_threshold = np.cos(
        np.deg2rad(
            RG_SMOOTHNESS_DEG
        )
    )

    valid_regions = []

    for seed_index in np.argsort(curvature):
        if assigned[seed_index]:
            continue

        assigned[seed_index] = True

        queue = deque([int(seed_index)])
        region = [int(seed_index)]

        while queue:
            current = queue.popleft()

            current_normal = normals[current]
            current_point = points[current]

            for neighbour in neighbour_indices[current]:
                neighbour = int(neighbour)

                if assigned[neighbour]:
                    continue

                normal_similarity = abs(
                    np.dot(
                        current_normal,
                        normals[neighbour],
                    )
                )

                if normal_similarity < cos_threshold:
                    continue

                local_distance = abs(
                    np.dot(
                        current_normal,
                        points[neighbour]
                        - current_point,
                    )
                )

                if local_distance > RG_LOCAL_DISTANCE_M:
                    continue

                assigned[neighbour] = True
                region.append(neighbour)
                queue.append(neighbour)

        if len(region) >= RG_MIN_REGION_SIZE:
            valid_regions.append(region)

    models = []

    for region in valid_regions:
        region_points = points[
            np.asarray(region, dtype=np.int64)
        ]

        refined = fit_plane_svd(
            region_points
        )

        if refined is None:
            continue

        normal, d = refined

        model = build_model(
            points,
            normal,
            d,
            source_support=len(region),
        )

        if model is not None:
            models.append(model)

    return merge_duplicate_models(models)


# ==========================================================
# 5. Multi-peak 3D Hough
# ==========================================================

def detect_hough(points, rng=None):
    geometry = estimate_local_geometry(points)

    if geometry is None:
        return []

    normals, curvature, _ = geometry

    planar_indices = np.flatnonzero(
        curvature <= HOUGH_MAX_CURVATURE
    )

    if len(planar_indices) < MIN_PLANE_SUPPORT:
        return []

    accumulator = defaultdict(list)

    for point_index in planar_indices:
        normal = normals[point_index]
        d = -float(
            np.dot(
                normal,
                points[point_index],
            )
        )

        normal, d = canonicalise_plane(
            normal,
            d,
        )

        theta = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        normal[2],
                        -1.0,
                        1.0,
                    )
                )
            )
        )

        phi = float(
            np.degrees(
                np.arctan2(
                    normal[1],
                    normal[0],
                )
            )
        )

        if phi < 0:
            phi += 360.0

        # Pole handling, matching the philosophy of Hough V2.
        if theta < HOUGH_THETA_BIN_DEG:
            phi = 0.0

        theta_bin = int(
            np.floor(
                theta
                / HOUGH_THETA_BIN_DEG
            )
        )

        phi_bin = int(
            np.floor(
                phi
                / HOUGH_PHI_BIN_DEG
            )
        )

        d_bin = int(
            np.floor(
                d
                / HOUGH_D_BIN_M
            )
        )

        accumulator[
            (
                theta_bin,
                phi_bin,
                d_bin,
            )
        ].append(int(point_index))

    sorted_bins = sorted(
        accumulator.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    models = []

    for _, indices in sorted_bins:
        if len(indices) < HOUGH_MIN_VOTES:
            break

        refined = fit_plane_svd(
            points[
                np.asarray(
                    indices,
                    dtype=np.int64,
                )
            ]
        )

        if refined is None:
            continue

        normal, d = refined

        model = build_model(
            points,
            normal,
            d,
            source_support=len(indices),
        )

        if model is None:
            continue

        models.append(model)

        models = merge_duplicate_models(models)

        if len(models) >= MAX_PLANES:
            break

    return merge_duplicate_models(models)


# ==========================================================
# GT / matching
# ==========================================================

def load_sample(npz_file):
    data = np.load(npz_file)

    points = np.asarray(
        data["points"],
        dtype=np.float64,
    )

    labels = np.asarray(
        data["labels"],
        dtype=np.int32,
    )

    gt_file = npz_file.with_name(
        npz_file.stem
        + "_gt.json"
    )

    with open(
        gt_file,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    gt_planes = []

    for plane in metadata["gt_planes"]:
        normal = np.asarray(
            plane["normal"],
            dtype=np.float64,
        )

        d = float(plane["d"])

        normal, d = canonicalise_plane(
            normal,
            d,
        )

        gt_planes.append(
            {
                "id": int(plane["id"]),
                "name": plane["name"],
                "normal": normal,
                "d": d,
            }
        )

    return points, labels, metadata, gt_planes


def match_planes(models, gt_planes, labels):
    num_detected = len(models)
    num_gt = len(gt_planes)

    if num_detected == 0 or num_gt == 0:
        return {
            "matches": [],
            "matched_count": 0,
            "plane_precision": 0.0 if num_detected == 0 else 0.0,
            "plane_recall": 0.0,
            "plane_f1": 0.0,
            "mean_normal_error_deg": np.nan,
            "mean_offset_error_m": np.nan,
            "mean_iou": 0.0,
        }

    cost = np.full(
        (num_detected, num_gt),
        1e6,
        dtype=np.float64,
    )

    pair_info = {}

    for i, model in enumerate(models):
        for j, gt in enumerate(gt_planes):
            n, d = align_plane_to_reference(
                model["normal"].copy(),
                model["d"],
                gt["normal"],
            )

            angle = plane_angle_deg(
                n,
                gt["normal"],
            )

            offset_error = abs(
                d
                - gt["d"]
            )

            if (
                angle <= MATCH_NORMAL_THRESHOLD_DEG
                and offset_error <= MATCH_OFFSET_THRESHOLD_M
            ):
                pair_cost = (
                    angle
                    / MATCH_NORMAL_THRESHOLD_DEG
                    + offset_error
                    / MATCH_OFFSET_THRESHOLD_M
                )

                cost[i, j] = pair_cost

                gt_mask = (
                    labels
                    == gt["id"]
                )

                det_mask = model["mask"]

                intersection = int(
                    np.sum(
                        gt_mask
                        & det_mask
                    )
                )

                union = int(
                    np.sum(
                        gt_mask
                        | det_mask
                    )
                )

                iou = (
                    intersection / union
                    if union > 0
                    else 0.0
                )

                pair_info[(i, j)] = {
                    "normal_error_deg": angle,
                    "offset_error_m": offset_error,
                    "iou": iou,
                }

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []

    for i, j in zip(row_ind, col_ind):
        if cost[i, j] >= 1e5:
            continue

        info = pair_info[(int(i), int(j))]

        matches.append(
            {
                "detected_index": int(i),
                "gt_index": int(j),
                **info,
            }
        )

    matched_count = len(matches)

    plane_precision = (
        matched_count / num_detected
        if num_detected > 0
        else 0.0
    )

    plane_recall = (
        matched_count / num_gt
        if num_gt > 0
        else 0.0
    )

    if plane_precision + plane_recall > 0:
        plane_f1 = (
            2.0
            * plane_precision
            * plane_recall
            / (
                plane_precision
                + plane_recall
            )
        )
    else:
        plane_f1 = 0.0

    if matched_count > 0:
        mean_normal_error = float(
            np.mean(
                [
                    match["normal_error_deg"]
                    for match in matches
                ]
            )
        )

        mean_offset_error = float(
            np.mean(
                [
                    match["offset_error_m"]
                    for match in matches
                ]
            )
        )

        mean_iou = float(
            np.mean(
                [
                    match["iou"]
                    for match in matches
                ]
            )
        )
    else:
        mean_normal_error = np.nan
        mean_offset_error = np.nan
        mean_iou = 0.0

    return {
        "matches": matches,
        "matched_count": matched_count,
        "plane_precision": plane_precision,
        "plane_recall": plane_recall,
        "plane_f1": plane_f1,
        "mean_normal_error_deg": mean_normal_error,
        "mean_offset_error_m": mean_offset_error,
        "mean_iou": mean_iou,
    }


# ==========================================================
# Main benchmark
# ==========================================================

METHODS = {
    "Iterative RANSAC": detect_iterative_ransac,
    "Multi-Object RANSAC": detect_multi_object_ransac,
    "Surface-Normal / Surfel": detect_surface_normal,
    "Region Growing": detect_region_growing,
    "3D Hough": detect_hough,
}


def main():
    sample_files = sorted(
        DATA_DIR.glob(
            "P*_N*_O*_R*.npz"
        )
    )

    if len(sample_files) != 60:
        raise RuntimeError(
            f"Expected 60 multi-plane samples, "
            f"found {len(sample_files)}"
        )

    print(
        "=== Multi-Plane Plane-Detection Benchmark ==="
    )

    print(
        "Samples:",
        len(sample_files)
    )

    # ------------------------------------------------------
    # One un-timed warm-up per method.
    # ------------------------------------------------------

    if DO_WARMUP:
        print("\nRunning un-timed warm-up...")

        warm_points, _, _, _ = load_sample(
            sample_files[0]
        )

        for method_index, (method_name, detector) in enumerate(
            METHODS.items()
        ):
            rng = np.random.default_rng(
                99000 + method_index
            )

            detector(
                warm_points,
                rng,
            )

            print(
                "  warm-up:",
                method_name
            )

    rows = []

    total_runs = (
        len(sample_files)
        * len(METHODS)
    )

    run_number = 0

    for sample_index, npz_file in enumerate(
        sample_files,
        start=1,
    ):
        points, labels, metadata, gt_planes = load_sample(
            npz_file
        )

        for method_index, (method_name, detector) in enumerate(
            METHODS.items()
        ):
            run_number += 1

            seed = (
                int(metadata["seed"])
                + method_index * 100000
            )

            rng = np.random.default_rng(seed)

            start = time.perf_counter()

            models = detector(
                points,
                rng,
            )

            end = time.perf_counter()

            runtime_ms = (
                end - start
            ) * 1000.0

            evaluation = match_planes(
                models,
                gt_planes,
                labels,
            )

            gt_count = len(gt_planes)
            detected_count = len(models)
            matched_count = evaluation["matched_count"]

            false_plane_count = (
                detected_count
                - matched_count
            )

            missed_plane_count = (
                gt_count
                - matched_count
            )

            correct_plane_count = int(
                detected_count
                == gt_count
            )

            row = {
                "method": method_name,
                "sample_id": metadata["sample_id"],
                "plane_count_gt": gt_count,
                "noise_sigma_m": metadata["noise_sigma_m"],
                "outlier_ratio": metadata["outlier_ratio"],
                "repeat_id": metadata["repeat_id"],
                "num_points": len(points),

                "detected_plane_count": detected_count,
                "matched_plane_count": matched_count,
                "false_plane_count": false_plane_count,
                "missed_plane_count": missed_plane_count,
                "correct_plane_count": correct_plane_count,

                "plane_precision": evaluation["plane_precision"],
                "plane_recall": evaluation["plane_recall"],
                "plane_f1": evaluation["plane_f1"],

                "matched_mean_iou": evaluation["mean_iou"],

                "mean_normal_error_deg": (
                    ""
                    if np.isnan(
                        evaluation["mean_normal_error_deg"]
                    )
                    else evaluation["mean_normal_error_deg"]
                ),

                "mean_offset_error_m": (
                    ""
                    if np.isnan(
                        evaluation["mean_offset_error_m"]
                    )
                    else evaluation["mean_offset_error_m"]
                ),

                "runtime_ms": runtime_ms,
                "success": int(matched_count > 0),
            }

            rows.append(row)

            print(
                f"[{run_number:03d}/{total_runs}] "
                f"{metadata['sample_id']} | "
                f"{method_name:<24} "
                f"GT={gt_count} "
                f"Det={detected_count} "
                f"Match={matched_count} "
                f"F1={evaluation['plane_f1']:.3f} "
                f"IoU={evaluation['mean_iou']:.3f} "
                f"{runtime_ms:.1f} ms"
            )

    fieldnames = [
        "method",
        "sample_id",

        "plane_count_gt",
        "noise_sigma_m",
        "outlier_ratio",
        "repeat_id",
        "num_points",

        "detected_plane_count",
        "matched_plane_count",
        "false_plane_count",
        "missed_plane_count",
        "correct_plane_count",

        "plane_precision",
        "plane_recall",
        "plane_f1",

        "matched_mean_iou",

        "mean_normal_error_deg",
        "mean_offset_error_m",

        "runtime_ms",
        "success",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nBenchmark complete.")
    print("Results:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
