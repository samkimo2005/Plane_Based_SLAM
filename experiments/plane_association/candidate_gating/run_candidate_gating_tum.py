"""
run_candidate_gating_tum.py

Run the four Candidate Gating methods on the TUM + ETH3D-derived
candidate-pair benchmark.

Inputs
------
results/candidate_gating/tum_plane_features_with_gt_ids/frame_XXXX.npz
results/candidate_gating/candidate_pairs.csv
results/candidate_gating/tuning/best_candidate_gating_parameters.json

Outputs
-------
results/candidate_gating/tum/
    candidate_gating_tum_results.csv
    candidate_gating_tum_runtime.csv
    run_settings.json

Important
---------
- All four methods receive the same full Cartesian candidate set.
- ETH3D GT labels are NOT passed into the gating algorithms.
- The TUM GT camera trajectory is NOT used.
- For this stage-level real-data test, the pose prior is identity
  (zero-motion prior). This avoids GT-pose leakage.
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import candidate_gating_common as common
import motion_model_distance_gating
import contour_point_gating
import normal_based_gating
import pcpm_gating


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = common.PROJECT_ROOT

PLANE_DIR = (
    PROJECT_ROOT
    / "results"
    / "candidate_gating"
    / "tum_plane_features_with_gt_ids"
)

PAIR_FILE = (
    PROJECT_ROOT
    / "results"
    / "candidate_gating"
    / "candidate_pairs.csv"
)

FROZEN_PARAMETER_FILE = (
    PROJECT_ROOT
    / "results"
    / "candidate_gating"
    / "tuning"
    / "best_candidate_gating_parameters.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "candidate_gating"
    / "tum"
)

RESULT_FILE = OUTPUT_DIR / "candidate_gating_tum_results.csv"
RUNTIME_FILE = OUTPUT_DIR / "candidate_gating_tum_runtime.csv"
SETTINGS_FILE = OUTPUT_DIR / "run_settings.json"


# ============================================================
# TUM PRIOR
# ============================================================

# Do NOT replace this with TUM ground-truth trajectory.
# Identity is used as a zero-motion prior for the current module-level test.
PRIOR_R = np.eye(3, dtype=float)
PRIOR_T = np.zeros(3, dtype=float)


# ============================================================
# LOADING
# ============================================================

def load_plane_set(frame_id: int) -> common.PlaneSet:
    path = PLANE_DIR / f"frame_{frame_id:04d}.npz"

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        normals = np.asarray(data["normals"], dtype=float)
        offsets = np.asarray(data["offsets"], dtype=float)
        centroids = np.asarray(data["centroids"], dtype=float)
        contours = np.asarray(data["contours"], dtype=float)

    n = len(normals)

    # The four current gating methods do not use area/width/height,
    # but PlaneSet requires these fields.
    zeros = np.zeros(n, dtype=float)

    return common.PlaneSet(
        normals=common.normalize_rows(normals),
        offsets=offsets,
        centroids=centroids,
        areas=zeros.copy(),
        contours=contours,
        widths=zeros.copy(),
        heights=zeros.copy(),
    )


def load_candidate_pairs():
    groups = defaultdict(list)

    with PAIR_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            parsed = {
                "pair_id": int(row["pair_id"]),
                "frame_a": int(row["frame_a"]),
                "plane_a": int(row["plane_a"]),
                "frame_b": int(row["frame_b"]),
                "plane_b": int(row["plane_b"]),
                "gt_label": int(row["gt_label"]),
                "gt_plane_id_a": int(row["gt_plane_id_a"]),
                "gt_plane_id_b": int(row["gt_plane_id_b"]),
            }

            key = (parsed["frame_a"], parsed["frame_b"])
            groups[key].append(parsed)

    return dict(sorted(groups.items()))


def load_frozen_parameters():
    if not FROZEN_PARAMETER_FILE.exists():
        raise FileNotFoundError(
            f"\nFrozen Candidate Gating parameters not found:\n"
            f"{FROZEN_PARAMETER_FILE}\n\n"
            "Run tune_candidate_gating.py first."
        )

    data = json.loads(
        FROZEN_PARAMETER_FILE.read_text(encoding="utf-8")
    )

    algorithms = data.get("algorithms", {})

    required = [
        "motion_model",
        "contour_point",
        "normal_based",
        "pcpm",
    ]

    for name in required:
        if name not in algorithms:
            raise KeyError(
                f"Frozen parameter file has no algorithm '{name}'."
            )

    return {
        name: algorithms[name]["parameters"]
        for name in required
    }


# ============================================================
# RUN ONE METHOD
# ============================================================

def run_method(
    method_name: str,
    source: common.PlaneSet,
    target: common.PlaneSet,
    parameters: dict,
):
    start = time.perf_counter()

    if method_name == "motion_model":
        candidates = motion_model_distance_gating.gate_candidates(
            source_planes=source,
            target_planes=target,
            prior_R=PRIOR_R,
            prior_t=PRIOR_T,
            parameters=parameters,
        )

    elif method_name == "contour_point":
        candidates = contour_point_gating.gate_candidates(
            source_planes=source,
            target_planes=target,
            prior_R=PRIOR_R,
            prior_t=PRIOR_T,
            parameters=parameters,
        )

    elif method_name == "normal_based":
        normal_sigma = float(
            parameters.get("default_normal_sigma_deg", 2.0)
        )
        pose_sigma = float(
            parameters.get("default_pose_rotation_sigma_deg", 2.0)
        )

        candidates = normal_based_gating.gate_candidates(
            source_planes=source,
            target_planes=target,
            prior_R=PRIOR_R,
            prior_t=PRIOR_T,
            normal_sigma_deg=normal_sigma,
            pose_rotation_sigma_deg=pose_sigma,
            parameters=parameters,
        )

    elif method_name == "pcpm":
        candidates = pcpm_gating.gate_candidates(
            source_planes=source,
            target_planes=target,
            prior_R=PRIOR_R,
            prior_t=PRIOR_T,
            parameters=parameters,
        )

    else:
        raise ValueError(method_name)

    runtime_ms = (time.perf_counter() - start) * 1000.0

    kept = {
        (
            int(c["source_plane_id"]),
            int(c["target_plane_id"]),
        )
        for c in candidates
    }

    return kept, runtime_ms


# ============================================================
# MAIN
# ============================================================

def main():
    if not PLANE_DIR.exists():
        raise FileNotFoundError(PLANE_DIR)

    if not PAIR_FILE.exists():
        raise FileNotFoundError(PAIR_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parameters = load_frozen_parameters()
    pair_groups = load_candidate_pairs()

    methods = [
        "motion_model",
        "contour_point",
        "normal_based",
        "pcpm",
    ]

    result_rows = []
    runtime_rows = []

    print()
    print("=" * 78)
    print("TUM CANDIDATE GATING")
    print("=" * 78)
    print(f"Frame transitions : {len(pair_groups)}")
    print(f"Pose prior        : identity / zero-motion")
    print(f"GT pose used      : NO")
    print()

    for transition_index, (
        (frame_a, frame_b),
        pair_rows,
    ) in enumerate(pair_groups.items(), start=1):

        source = load_plane_set(frame_a)
        target = load_plane_set(frame_b)

        expected_pairs = len(source) * len(target)

        if len(pair_rows) != expected_pairs:
            raise RuntimeError(
                f"frame_{frame_a:04d} -> frame_{frame_b:04d}: "
                f"candidate_pairs.csv has {len(pair_rows)} pairs, "
                f"but PlaneSets imply {expected_pairs}."
            )

        print(
            f"[{transition_index:02d}/{len(pair_groups):02d}] "
            f"frame_{frame_a:04d} -> frame_{frame_b:04d} | "
            f"{len(source)} x {len(target)} = {expected_pairs}"
        )

        for method in methods:
            kept, runtime_ms = run_method(
                method_name=method,
                source=source,
                target=target,
                parameters=parameters[method],
            )

            runtime_rows.append(
                {
                    "method": method,
                    "frame_a": frame_a,
                    "frame_b": frame_b,
                    "input_pairs": expected_pairs,
                    "kept_pairs": len(kept),
                    "runtime_ms": runtime_ms,
                }
            )

            for pair in pair_rows:
                key = (
                    pair["plane_a"],
                    pair["plane_b"],
                )

                result_rows.append(
                    {
                        "method": method,
                        **pair,
                        "keep": int(key in kept),
                    }
                )

            print(
                f"    {method:<14} "
                f"kept={len(kept):3d}/{expected_pairs:3d} | "
                f"{runtime_ms:8.3f} ms"
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    result_fields = [
        "method",
        "pair_id",
        "frame_a",
        "plane_a",
        "frame_b",
        "plane_b",
        "gt_label",
        "gt_plane_id_a",
        "gt_plane_id_b",
        "keep",
    ]

    with RESULT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=result_fields,
        )
        writer.writeheader()
        writer.writerows(result_rows)

    runtime_fields = [
        "method",
        "frame_a",
        "frame_b",
        "input_pairs",
        "kept_pairs",
        "runtime_ms",
    ]

    with RUNTIME_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=runtime_fields,
        )
        writer.writeheader()
        writer.writerows(runtime_rows)

    settings = {
        "dataset": "TUM fr1_xyz + ETH3D-derived plane correspondence GT",
        "candidate_pair_file": str(PAIR_FILE),
        "plane_feature_dir": str(PLANE_DIR),
        "frozen_parameter_file": str(FROZEN_PARAMETER_FILE),
        "pose_prior_mode": "identity_zero_motion",
        "tum_ground_truth_pose_used": False,
        "methods": methods,
        "frozen_parameters": parameters,
    }

    SETTINGS_FILE.write_text(
        json.dumps(settings, indent=4),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("TUM CANDIDATE GATING COMPLETE")
    print("=" * 78)
    print(f"Pair results : {RESULT_FILE}")
    print(f"Runtime      : {RUNTIME_FILE}")
    print(f"Settings     : {SETTINGS_FILE}")


if __name__ == "__main__":
    main()
