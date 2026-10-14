"""
prepare_bad_match_rejection_tum.py

Prepare TUM real-data cases for the Bad-Match Rejection stage.

Pipeline
--------
Prepared TUM frame data
    + per-frame detected-plane features
    + frozen Assignment case
    + frozen Assignment solver
    + ETH3D-derived correspondence GT
        ->
Bad-Match Rejection case

Important experimental policy
-----------------------------
1. The selected Assignment method is re-run with frozen parameters.
2. Plane geometry is loaded from the same per-frame feature files used
   upstream by Candidate Gating / Similarity / Assignment.
3. Assignment indices are kept in the same order as AssignmentCase
   current_ids/reference_ids, so no plane re-detection or re-ordering
   is performed here.
4. Ground truth is attached only after Assignment has produced its
   matches. GT is never used to construct assigned_pairs.
5. Unknown real-data correspondence labels (-1) are preserved as
   ambiguous and must be ignored by the later evaluator.

Default upstream choice
-----------------------
The final Assignment weighted decision matrix selected
Jonker-Volgenant, so this is the default Assignment method.

Outputs
-------
experiments/data/bad_match_rejection/tum/
    tum_0003_0004.npz
    tum_0004_0005.npz
    ...
    manifest.csv
    parameters_used.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ASSIGNMENT_CODE_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "assignment"
)

if ASSIGNMENT_CODE_DIR.exists():
    sys.path.insert(
        0,
        str(ASSIGNMENT_CODE_DIR),
    )

try:
    from assignment_common import (
        load_assignment_case,
    )

    from run_assignment_tum import (
        load_frozen_parameters,
        load_reference,
        run_algorithm,
    )

except ImportError as exc:
    raise ImportError(
        "Could not import Assignment modules.\n"
        f"Expected Assignment code directory:\n"
        f"  {ASSIGNMENT_CODE_DIR}\n\n"
        "If your Assignment code is stored elsewhere, add that folder "
        "to PYTHONPATH or change ASSIGNMENT_CODE_DIR."
    ) from exc


DEFAULT_ASSIGNMENT_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "assignment"
    / "tum"
)

DEFAULT_ASSIGNMENT_REFERENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "assignment"
    / "tum_reference"
)

DEFAULT_ASSIGNMENT_PARAMETER_FILE = (
    PROJECT_ROOT
    / "results"
    / "assignment"
    / "tuning"
    / "frozen_parameters.json"
)

DEFAULT_FEATURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "candidate_gating"
    / "tum_plane_features_with_gt_ids"
)

DEFAULT_FRAME_DIR = (
    PROJECT_ROOT
    / "data"
    / "tum"
    / "plane_detection"
    / "rgbd_dataset_freiburg1_xyz"
    / "frames"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "data"
    / "bad_match_rejection"
    / "tum"
)

DEFAULT_ASSIGNMENT_METHOD = (
    "jonker_volgenant"
)

VALID_ASSIGNMENT_METHODS = {
    "hungarian",
    "auction",
    "jonker_volgenant",
    "cost_scaling",
}


# ============================================================
# Small helpers
# ============================================================

def rotation_angle_deg(R: np.ndarray) -> float:
    """
    Return rotation magnitude in degrees.
    """

    value = (
        np.trace(R) - 1.0
    ) / 2.0

    value = float(
        np.clip(
            value,
            -1.0,
            1.0,
        )
    )

    return float(
        np.degrees(
            np.arccos(value)
        )
    )


def save_csv(
    path: Path,
    rows: list[dict],
) -> None:

    if not rows:
        raise ValueError(
            f"No rows to save: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = []

    seen = set()

    for row in rows:
        for key in row.keys():

            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Load per-frame plane features
# ============================================================

def load_plane_feature_frame(
    feature_dir: Path,
    frame_id: int,
) -> dict:
    """
    Load the detected-plane feature file used by the upstream
    plane-association pipeline.

    Required fields:
        normals      (N, 3)
        offsets      (N,)
        centroids    (N, 3)

    Plane IDs used by candidate_pairs.csv are indices into these
    arrays.
    """

    path = (
        feature_dir
        / f"frame_{frame_id:04d}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Plane-feature file not found:\n"
            f"{path}"
        )

    with np.load(
        path,
        allow_pickle=True,
    ) as data:

        required = {
            "normals",
            "offsets",
            "centroids",
        }

        missing = (
            required
            - set(data.files)
        )

        if missing:
            raise KeyError(
                f"{path.name} missing fields: "
                f"{sorted(missing)}"
            )

        normals = np.asarray(
            data["normals"],
            dtype=np.float64,
        )

        offsets = np.asarray(
            data["offsets"],
            dtype=np.float64,
        ).reshape(-1)

        centroids = np.asarray(
            data["centroids"],
            dtype=np.float64,
        )

        optional = {}

        for key in (
            "support_counts",
            "global_plane_ids",
            "gt_known",
        ):
            if key in data.files:
                optional[key] = np.asarray(
                    data[key]
                )

    if (
        normals.ndim != 2
        or normals.shape[1] != 3
    ):
        raise ValueError(
            f"{path.name}: normals must have shape (N,3), "
            f"found {normals.shape}."
        )

    n_planes = len(normals)

    if offsets.shape != (n_planes,):
        raise ValueError(
            f"{path.name}: offsets shape {offsets.shape} "
            f"does not match {n_planes} planes."
        )

    if centroids.shape != (n_planes, 3):
        raise ValueError(
            f"{path.name}: centroids must have shape "
            f"({n_planes},3), found {centroids.shape}."
        )

    normal_norms = np.linalg.norm(
        normals,
        axis=1,
    )

    if not np.all(
        np.isfinite(normal_norms)
    ):
        raise ValueError(
            f"{path.name}: non-finite plane normal."
        )

    if not np.allclose(
        normal_norms,
        1.0,
        atol=1e-3,
    ):
        raise ValueError(
            f"{path.name}: plane normals are not unit length. "
            f"Range = "
            f"[{normal_norms.min():.6f}, "
            f"{normal_norms.max():.6f}]"
        )

    if not (
        np.all(np.isfinite(offsets))
        and np.all(np.isfinite(centroids))
    ):
        raise ValueError(
            f"{path.name}: non-finite plane geometry."
        )

    return {
        "path":
            path,

        "normals":
            normals,

        "offsets":
            offsets,

        "centroids":
            centroids,

        **optional,
    }


def select_planes(
    feature_frame: dict,
    plane_ids,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select planes in exactly the AssignmentCase ID order.

    Returns
    -------
    planes:
        (N,4) rows [nx, ny, nz, d]

    centroids:
        (N,3)
    """

    ids = np.asarray(
        plane_ids,
        dtype=np.int64,
    )

    if ids.ndim != 1:
        raise ValueError(
            "Plane IDs must be one-dimensional."
        )

    n_available = len(
        feature_frame["normals"]
    )

    if len(ids) > 0:
        if (
            np.min(ids) < 0
            or np.max(ids) >= n_available
        ):
            raise IndexError(
                "Assignment plane ID outside feature-frame range.\n"
                f"IDs: {ids.tolist()}\n"
                f"Available planes: {n_available}"
            )

    normals = (
        feature_frame["normals"][ids]
    )

    offsets = (
        feature_frame["offsets"][ids]
    )

    centroids = (
        feature_frame["centroids"][ids]
    )

    planes = np.column_stack(
        (
            normals,
            offsets,
        )
    ).astype(
        np.float64,
        copy=False,
    )

    return (
        planes,
        centroids,
    )


# ============================================================
# Load per-frame TUM ground-truth pose
# ============================================================

def load_frame_pose(
    frame_dir: Path,
    frame_id: int,
) -> dict:

    path = (
        frame_dir
        / f"frame_{frame_id:04d}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Prepared TUM frame not found:\n"
            f"{path}"
        )

    with np.load(
        path,
        allow_pickle=True,
    ) as data:

        required = {
            "R_world_cam",
            "t_world_cam",
            "rgb_timestamp",
        }

        missing = (
            required
            - set(data.files)
        )

        if missing:
            raise KeyError(
                f"{path.name} missing fields: "
                f"{sorted(missing)}"
            )

        R_world_cam = np.asarray(
            data["R_world_cam"],
            dtype=np.float64,
        )

        t_world_cam = np.asarray(
            data["t_world_cam"],
            dtype=np.float64,
        ).reshape(3)

        rgb_timestamp = float(
            np.asarray(
                data["rgb_timestamp"]
            ).item()
        )

        depth_timestamp = (
            float(
                np.asarray(
                    data["depth_timestamp"]
                ).item()
            )
            if "depth_timestamp"
            in data.files
            else np.nan
        )

        gt_timestamp = (
            float(
                np.asarray(
                    data[
                        "groundtruth_timestamp"
                    ]
                ).item()
            )
            if "groundtruth_timestamp"
            in data.files
            else np.nan
        )

    if R_world_cam.shape != (3, 3):
        raise ValueError(
            f"{path.name}: R_world_cam must be (3,3), "
            f"found {R_world_cam.shape}."
        )

    if not (
        np.all(np.isfinite(R_world_cam))
        and np.all(np.isfinite(t_world_cam))
    ):
        raise ValueError(
            f"{path.name}: non-finite camera pose."
        )

    return {
        "path":
            path,

        "R_world_cam":
            R_world_cam,

        "t_world_cam":
            t_world_cam,

        "rgb_timestamp":
            rgb_timestamp,

        "depth_timestamp":
            depth_timestamp,

        "groundtruth_timestamp":
            gt_timestamp,
    }


def relative_pose_a_to_b(
    pose_a: dict,
    pose_b: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    TUM prepared-frame convention:

        x_world = R_world_cam x_cam + t_world_cam

    Therefore the transform taking a point from camera A coordinates
    to camera B coordinates is

        x_B = R_BA x_A + t_BA

    with

        R_BA = R_WB^T R_WA
        t_BA = R_WB^T (t_WA - t_WB)
    """

    R_wa = pose_a[
        "R_world_cam"
    ]

    t_wa = pose_a[
        "t_world_cam"
    ]

    R_wb = pose_b[
        "R_world_cam"
    ]

    t_wb = pose_b[
        "t_world_cam"
    ]

    R_ba = (
        R_wb.T
        @ R_wa
    )

    t_ba = (
        R_wb.T
        @ (
            t_wa
            - t_wb
        )
    )

    return (
        R_ba,
        t_ba,
    )


# ============================================================
# Prepare one case
# ============================================================

def prepare_case(
    case_path: Path,
    assignment_reference_dir: Path,
    feature_dir: Path,
    frame_dir: Path,
    params: dict,
    assignment_method: str,
) -> tuple[dict, dict]:

    case = load_assignment_case(
        case_path
    )

    reference = load_reference(
        assignment_reference_dir,
        case.case_id,
    )

    frame_a = int(
        case.metadata[
            "frame_current"
        ]
    )

    frame_b = int(
        case.metadata[
            "frame_reference"
        ]
    )

    # --------------------------------------------------------
    # 1. Load plane geometry in EXACT Assignment ID order.
    # --------------------------------------------------------

    features_a = (
        load_plane_feature_frame(
            feature_dir,
            frame_a,
        )
    )

    features_b = (
        load_plane_feature_frame(
            feature_dir,
            frame_b,
        )
    )

    (
        planes_a,
        centroids_a,
    ) = select_planes(
        features_a,
        case.current_ids,
    )

    (
        planes_b,
        centroids_b,
    ) = select_planes(
        features_b,
        case.reference_ids,
    )

    if len(planes_a) != len(
        case.current_ids
    ):
        raise RuntimeError(
            f"{case.case_id}: plane-A count mismatch."
        )

    if len(planes_b) != len(
        case.reference_ids
    ):
        raise RuntimeError(
            f"{case.case_id}: plane-B count mismatch."
        )

    # --------------------------------------------------------
    # 2. Re-run the frozen, selected Assignment solver.
    # --------------------------------------------------------

    assignment_result = run_algorithm(
        assignment_method,
        case,
        params,
    )

    assigned_pairs = np.asarray(
        sorted(
            (
                int(i),
                int(j),
            )
            for i, j
            in assignment_result.matches
        ),
        dtype=np.int32,
    ).reshape(-1, 2)

    # Safety: Assignment may only select valid candidate edges.
    for i, j in assigned_pairs:

        if not case.candidate_mask[i, j]:
            raise RuntimeError(
                f"{case.case_id}: Assignment selected "
                f"non-candidate pair ({i},{j})."
            )

    # --------------------------------------------------------
    # 3. Attach GT AFTER Assignment.
    # --------------------------------------------------------

    gt_matrix = np.asarray(
        reference[
            "gt_label_matrix"
        ],
        dtype=np.int8,
    )

    if (
        gt_matrix.shape
        != case.similarity.shape
    ):
        raise RuntimeError(
            f"{case.case_id}: GT matrix shape "
            f"{gt_matrix.shape} != Assignment shape "
            f"{case.similarity.shape}."
        )

    pair_gt_labels = np.asarray(
        [
            int(
                gt_matrix[i, j]
            )
            for i, j
            in assigned_pairs
        ],
        dtype=np.int8,
    )

    if not np.all(
        np.isin(
            pair_gt_labels,
            [-1, 0, 1],
        )
    ):
        raise RuntimeError(
            f"{case.case_id}: invalid GT label."
        )

    gt_match_labels = (
        pair_gt_labels == 1
    )

    gt_match_ambiguous = (
        pair_gt_labels == -1
    )

    assigned_similarity = np.asarray(
        [
            float(
                case.similarity[i, j]
            )
            for i, j
            in assigned_pairs
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # 4. Ground-truth relative camera pose A -> B.
    # --------------------------------------------------------

    pose_a = load_frame_pose(
        frame_dir,
        frame_a,
    )

    pose_b = load_frame_pose(
        frame_dir,
        frame_b,
    )

    (
        R_gt,
        t_gt,
    ) = relative_pose_a_to_b(
        pose_a,
        pose_b,
    )

    # --------------------------------------------------------
    # 5. Statistics.
    # --------------------------------------------------------

    known_mask = (
        pair_gt_labels != -1
    )

    known_good = int(
        np.sum(
            pair_gt_labels == 1
        )
    )

    known_bad = int(
        np.sum(
            pair_gt_labels == 0
        )
    )

    ambiguous = int(
        np.sum(
            pair_gt_labels == -1
        )
    )

    known_total = (
        known_good
        + known_bad
    )

    bad_ratio_known = (
        float(
            known_bad
            / known_total
        )
        if known_total > 0
        else np.nan
    )

    output = {
        "case_id":
            np.asarray(
                case.case_id
            ),

        "frame_a":
            np.asarray(
                frame_a,
                dtype=np.int32,
            ),

        "frame_b":
            np.asarray(
                frame_b,
                dtype=np.int32,
            ),

        # Plane geometry.
        "planes_a":
            planes_a,

        "planes_b":
            planes_b,

        "centroids_a":
            centroids_a,

        "centroids_b":
            centroids_b,

        # Original upstream plane IDs for auditability.
        "plane_ids_a":
            np.asarray(
                case.current_ids,
                dtype=np.int32,
            ),

        "plane_ids_b":
            np.asarray(
                case.reference_ids,
                dtype=np.int32,
            ),

        # Assignment output.
        "assigned_pairs":
            assigned_pairs,

        "assigned_similarity":
            assigned_similarity,

        # Real-data GT.
        "gt_match_labels":
            gt_match_labels,

        "gt_match_ambiguous":
            gt_match_ambiguous,

        "gt_match_labels_int8":
            pair_gt_labels,

        # Relative pose A -> B.
        "R_gt":
            R_gt,

        "t_gt":
            t_gt,

        # Timestamps.
        "timestamp_a":
            np.asarray(
                pose_a[
                    "rgb_timestamp"
                ],
                dtype=np.float64,
            ),

        "timestamp_b":
            np.asarray(
                pose_b[
                    "rgb_timestamp"
                ],
                dtype=np.float64,
            ),

        "depth_timestamp_a":
            np.asarray(
                pose_a[
                    "depth_timestamp"
                ],
                dtype=np.float64,
            ),

        "depth_timestamp_b":
            np.asarray(
                pose_b[
                    "depth_timestamp"
                ],
                dtype=np.float64,
            ),

        "groundtruth_timestamp_a":
            np.asarray(
                pose_a[
                    "groundtruth_timestamp"
                ],
                dtype=np.float64,
            ),

        "groundtruth_timestamp_b":
            np.asarray(
                pose_b[
                    "groundtruth_timestamp"
                ],
                dtype=np.float64,
            ),

        "assignment_method":
            np.asarray(
                assignment_method
            ),
    }

    manifest_row = {
        "case_id":
            case.case_id,

        "frame_a":
            frame_a,

        "frame_b":
            frame_b,

        "n_planes_a":
            len(planes_a),

        "n_planes_b":
            len(planes_b),

        "n_assignment_candidates":
            int(
                np.sum(
                    case.candidate_mask
                )
            ),

        "n_assigned":
            len(assigned_pairs),

        "known_good":
            known_good,

        "known_bad":
            known_bad,

        "ambiguous":
            ambiguous,

        "evaluable_assigned":
            int(
                np.sum(
                    known_mask
                )
            ),

        "bad_ratio_known":
            bad_ratio_known,

        "translation_gt_m":
            float(
                np.linalg.norm(t_gt)
            ),

        "rotation_gt_deg":
            rotation_angle_deg(
                R_gt
            ),

        "assignment_method":
            assignment_method,
    }

    return (
        output,
        manifest_row,
    )


# ============================================================
# Main preparation
# ============================================================

def prepare(
    assignment_data_dir: Path,
    assignment_reference_dir: Path,
    assignment_parameter_file: Path,
    feature_dir: Path,
    frame_dir: Path,
    output_dir: Path,
    assignment_method: str,
    overwrite: bool,
) -> None:

    if (
        assignment_method
        not in VALID_ASSIGNMENT_METHODS
    ):
        raise ValueError(
            f"Unknown Assignment method: "
            f"{assignment_method}"
        )

    if not assignment_data_dir.exists():
        raise FileNotFoundError(
            f"Assignment data directory not found:\n"
            f"{assignment_data_dir}"
        )

    case_paths = sorted(
        assignment_data_dir.glob(
            "tum_*.npz"
        )
    )

    if not case_paths:
        raise RuntimeError(
            f"No TUM Assignment cases found in:\n"
            f"{assignment_data_dir}"
        )

    params = load_frozen_parameters(
        assignment_parameter_file
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = []

    print()
    print("=" * 82)
    print("PREPARE TUM BAD-MATCH REJECTION DATA")
    print("=" * 82)

    print(
        f"Assignment method : "
        f"{assignment_method}"
    )

    print(
        f"Assignment cases  : "
        f"{len(case_paths)}"
    )

    print(
        f"Plane features    : "
        f"{feature_dir}"
    )

    print(
        f"TUM frames        : "
        f"{frame_dir}"
    )

    print()

    for index, case_path in enumerate(
        case_paths,
        start=1,
    ):

        output, row = prepare_case(
            case_path=
                case_path,

            assignment_reference_dir=
                assignment_reference_dir,

            feature_dir=
                feature_dir,

            frame_dir=
                frame_dir,

            params=
                params,

            assignment_method=
                assignment_method,
        )

        output_path = (
            output_dir
            / f"{row['case_id']}.npz"
        )

        if (
            output_path.exists()
            and not overwrite
        ):
            raise FileExistsError(
                f"Output already exists:\n"
                f"{output_path}\n\n"
                "Use --overwrite only if you intentionally "
                "want to replace prepared real-data cases."
            )

        np.savez_compressed(
            output_path,
            **output,
        )

        manifest.append(
            row
        )

        print(
            f"[{index:02d}/"
            f"{len(case_paths):02d}] "
            f"{row['case_id']} | "
            f"planes="
            f"{row['n_planes_a']}x"
            f"{row['n_planes_b']} | "
            f"assigned="
            f"{row['n_assigned']:2d} | "
            f"good="
            f"{row['known_good']:2d} | "
            f"bad="
            f"{row['known_bad']:2d} | "
            f"amb="
            f"{row['ambiguous']:2d}"
        )

    manifest_path = (
        output_dir
        / "manifest.csv"
    )

    save_csv(
        manifest_path,
        manifest,
    )

    parameters_used = {
        "dataset":
            "TUM fr1_xyz + ETH3D-derived persistent plane identities",

        "assignment_method":
            assignment_method,

        "assignment_parameter_policy":
            "Frozen Assignment parameters",

        "gt_policy":
            {
                "good":
                    "gt_label == 1",

                "bad":
                    "gt_label == 0",

                "ambiguous":
                    "gt_label == -1; excluded from later classification metrics",
            },

        "plane_geometry_policy":
            (
                "Loaded from the same upstream per-frame plane-feature "
                "arrays using AssignmentCase current_ids/reference_ids; "
                "no plane re-detection or re-ordering."
            ),

        "relative_pose_convention":
            (
                "A->B, with x_B = R_gt x_A + t_gt; "
                "R_gt = R_WB^T R_WA, "
                "t_gt = R_WB^T(t_WA - t_WB)"
            ),

        "frozen_assignment_parameters":
            params,
    }

    parameter_path = (
        output_dir
        / "parameters_used.json"
    )

    with parameter_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            parameters_used,
            f,
            indent=4,
        )

    total_assigned = sum(
        row["n_assigned"]
        for row in manifest
    )

    total_good = sum(
        row["known_good"]
        for row in manifest
    )

    total_bad = sum(
        row["known_bad"]
        for row in manifest
    )

    total_ambiguous = sum(
        row["ambiguous"]
        for row in manifest
    )

    print()
    print("=" * 82)
    print("TUM BAD-MATCH PREPARATION COMPLETE")
    print("=" * 82)

    print(
        f"Cases prepared       : "
        f"{len(manifest)}"
    )

    print(
        f"Assigned pairs       : "
        f"{total_assigned}"
    )

    print(
        f"Known good pairs     : "
        f"{total_good}"
    )

    print(
        f"Known bad pairs      : "
        f"{total_bad}"
    )

    print(
        f"Ambiguous pairs      : "
        f"{total_ambiguous}"
    )

    print()
    print(
        f"Output directory:\n"
        f"  {output_dir}"
    )

    print(
        f"Manifest:\n"
        f"  {manifest_path}"
    )

    print(
        f"Parameters:\n"
        f"  {parameter_path}"
    )


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Prepare TUM real-data inputs for "
            "Bad-Match Rejection."
        )
    )

    parser.add_argument(
        "--assignment-data-dir",
        type=Path,
        default=
            DEFAULT_ASSIGNMENT_DATA_DIR,
    )

    parser.add_argument(
        "--assignment-reference-dir",
        type=Path,
        default=
            DEFAULT_ASSIGNMENT_REFERENCE_DIR,
    )

    parser.add_argument(
        "--assignment-parameter-file",
        type=Path,
        default=
            DEFAULT_ASSIGNMENT_PARAMETER_FILE,
    )

    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=
            DEFAULT_FEATURE_DIR,
    )

    parser.add_argument(
        "--frame-dir",
        type=Path,
        default=
            DEFAULT_FRAME_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=
            DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--assignment-method",
        type=str,
        choices=sorted(
            VALID_ASSIGNMENT_METHODS
        ),
        default=
            DEFAULT_ASSIGNMENT_METHOD,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    prepare(
        assignment_data_dir=
            args.assignment_data_dir,

        assignment_reference_dir=
            args.assignment_reference_dir,

        assignment_parameter_file=
            args.assignment_parameter_file,

        feature_dir=
            args.feature_dir,

        frame_dir=
            args.frame_dir,

        output_dir=
            args.output_dir,

        assignment_method=
            args.assignment_method,

        overwrite=
            args.overwrite,
    )


if __name__ == "__main__":
    main()
