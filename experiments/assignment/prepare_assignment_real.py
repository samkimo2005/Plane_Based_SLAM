"""
prepare_assignment_real.py

Prepare real TUM + ETH3D-derived data for the Assignment stage.

Inputs
------
results/candidate_gating/candidate_pairs.csv
results/similarity/real_final.csv

Outputs
-------
data/assignment/tum/
    tum_0003_0004.npz
    tum_0004_0005.npz
    ...
    manifest.csv

data/assignment/tum_reference/
    tum_0003_0004_gt.npz
    tum_0004_0005_gt.npz
    ...

Important
---------
1. The Assignment algorithms receive the REAL candidate graph produced
   by the preceding Candidate Gating + Similarity stages.

2. Raw continuous similarity scores are used. The binary
   predicted_match column is NOT used.

3. Real-data GT is incomplete:
       1  = true correspondence
       0  = known non-correspondence
      -1  = unknown / ignore

4. Unknown pairs remain valid candidate edges but are preserved in a
   sidecar GT file so that later evaluation can ignore them correctly.

5. AssignmentCase.gt_matches contains only GT correspondences that
   actually survive into the Assignment candidate graph. This isolates
   Assignment performance from correspondences already removed by the
   upstream Candidate Gating stage.

   The complete physical GT is retained separately in the sidecar file.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from assignment_common import (
    AssignmentCase,
    save_assignment_case,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PAIR_FILE = (
    PROJECT_ROOT
    / "results"
    / "candidate_gating"
    / "candidate_pairs.csv"
)

SIMILARITY_FILE = (
    PROJECT_ROOT
    / "results"
    / "similarity"
    / "real_final.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "assignment"
    / "tum"
)

REFERENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "assignment"
    / "tum_reference"
)


# Change this if your final Similarity WDM selects another method.
DEFAULT_SIMILARITY_METHOD = "Geometric Weighted"


# ============================================================
# Load complete candidate-pair reference
# ============================================================

def load_candidate_pairs(path: Path):

    if not path.exists():
        raise FileNotFoundError(
            f"Candidate-pair file not found:\n{path}"
        )

    pair_lookup = {}

    transitions = defaultdict(list)

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        required = {
            "pair_id",
            "frame_a",
            "plane_a",
            "frame_b",
            "plane_b",
            "gt_label",
            "gt_plane_id_a",
            "gt_plane_id_b",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                "candidate_pairs.csv missing columns: "
                f"{sorted(missing)}"
            )

        for row in reader:

            item = {
                "pair_id": int(row["pair_id"]),
                "frame_a": int(row["frame_a"]),
                "plane_a": int(row["plane_a"]),
                "frame_b": int(row["frame_b"]),
                "plane_b": int(row["plane_b"]),
                "gt_label": int(row["gt_label"]),
                "gt_plane_id_a":
                    int(row["gt_plane_id_a"]),
                "gt_plane_id_b":
                    int(row["gt_plane_id_b"]),
            }

            if item["gt_label"] not in {-1, 0, 1}:
                raise ValueError(
                    f"Invalid GT label for pair "
                    f"{item['pair_id']}: "
                    f"{item['gt_label']}"
                )

            pair_id = item["pair_id"]

            if pair_id in pair_lookup:
                raise ValueError(
                    f"Duplicate pair_id: {pair_id}"
                )

            pair_lookup[pair_id] = item

            key = (
                item["frame_a"],
                item["frame_b"],
            )

            transitions[key].append(item)

    transitions = dict(
        sorted(transitions.items())
    )

    print(
        f"Complete candidate pairs : "
        f"{len(pair_lookup)}"
    )

    print(
        f"Frame transitions        : "
        f"{len(transitions)}"
    )

    return pair_lookup, transitions


# ============================================================
# Load real Similarity output
# ============================================================

def load_similarity(
    path: Path,
    method: str,
):

    if not path.exists():
        raise FileNotFoundError(
            f"Similarity file not found:\n{path}"
        )

    rows = []

    available_methods = set()

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        required = {
            "pair_id",
            "current_frame_id",
            "reference_frame_id",
            "method",
            "gt_label",
            "score",
            "success",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                "real_final.csv missing columns: "
                f"{sorted(missing)}"
            )

        for row in reader:

            available_methods.add(
                row["method"]
            )

            if row["method"] != method:
                continue

            score = float(row["score"])

            if not np.isfinite(score):
                raise ValueError(
                    f"Non-finite score for "
                    f"pair_id={row['pair_id']}"
                )

            rows.append(
                {
                    "pair_id":
                        int(row["pair_id"]),

                    "current_frame_id":
                        int(row["current_frame_id"]),

                    "reference_frame_id":
                        int(row["reference_frame_id"]),

                    "gt_label":
                        int(row["gt_label"]),

                    "score":
                        score,

                    "success":
                        int(row["success"]),
                }
            )

    if not rows:

        raise ValueError(
            f"No real Similarity rows found for "
            f"method '{method}'.\n"
            f"Available methods: "
            f"{sorted(available_methods)}"
        )

    ids = [
        row["pair_id"]
        for row in rows
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            f"Duplicate pair IDs found for "
            f"Similarity method '{method}'."
        )

    successful = sum(
        row["success"] == 1
        for row in rows
    )

    print(
        f"Similarity method        : {method}"
    )

    print(
        f"Similarity candidate rows: {len(rows)}"
    )

    print(
        f"Successful scores        : {successful}"
    )

    return {
        row["pair_id"]: row
        for row in rows
    }


# ============================================================
# Recover persistent plane GT IDs
# ============================================================

def recover_plane_gt_ids(
    pair_rows,
    current_ids,
    reference_ids,
):

    current_values = {
        plane_id: set()
        for plane_id in current_ids
    }

    reference_values = {
        plane_id: set()
        for plane_id in reference_ids
    }

    for row in pair_rows:

        current_values[
            row["plane_a"]
        ].add(
            row["gt_plane_id_a"]
        )

        reference_values[
            row["plane_b"]
        ].add(
            row["gt_plane_id_b"]
        )

    current_gt_ids = np.full(
        len(current_ids),
        -1,
        dtype=np.int32,
    )

    reference_gt_ids = np.full(
        len(reference_ids),
        -1,
        dtype=np.int32,
    )

    for i, plane_id in enumerate(current_ids):

        values = current_values[plane_id]

        if len(values) != 1:
            raise RuntimeError(
                f"Inconsistent persistent GT ID "
                f"for current plane {plane_id}: "
                f"{values}"
            )

        current_gt_ids[i] = next(
            iter(values)
        )

    for j, plane_id in enumerate(reference_ids):

        values = reference_values[plane_id]

        if len(values) != 1:
            raise RuntimeError(
                f"Inconsistent persistent GT ID "
                f"for reference plane {plane_id}: "
                f"{values}"
            )

        reference_gt_ids[j] = next(
            iter(values)
        )

    return (
        current_gt_ids,
        reference_gt_ids,
    )


# ============================================================
# Build one real Assignment case
# ============================================================

def build_case(
    frame_a: int,
    frame_b: int,
    pair_rows,
    similarity_lookup,
    method: str,
):

    # --------------------------------------------------------
    # Plane sets
    # --------------------------------------------------------

    current_ids = sorted({
        row["plane_a"]
        for row in pair_rows
    })

    reference_ids = sorted({
        row["plane_b"]
        for row in pair_rows
    })

    current_index = {
        plane_id: i
        for i, plane_id
        in enumerate(current_ids)
    }

    reference_index = {
        plane_id: j
        for j, plane_id
        in enumerate(reference_ids)
    }

    n_current = len(current_ids)
    n_reference = len(reference_ids)

    expected_full_pairs = (
        n_current * n_reference
    )

    if len(pair_rows) != expected_full_pairs:

        raise RuntimeError(
            f"Frame {frame_a}->{frame_b}: "
            f"expected complete Cartesian set "
            f"{n_current} x {n_reference} = "
            f"{expected_full_pairs}, "
            f"but found {len(pair_rows)}."
        )

    # --------------------------------------------------------
    # Main Assignment matrices
    # --------------------------------------------------------

    similarity = np.zeros(
        (n_current, n_reference),
        dtype=np.float64,
    )

    candidate_mask = np.zeros(
        (n_current, n_reference),
        dtype=bool,
    )

    # Complete real GT:
    #
    #  1 = true correspondence
    #  0 = known negative
    # -1 = unknown
    #
    gt_label_matrix = np.full(
        (n_current, n_reference),
        -1,
        dtype=np.int8,
    )

    pair_id_matrix = np.full(
        (n_current, n_reference),
        -1,
        dtype=np.int64,
    )

    # Full physical GT, before gating.
    full_gt_matches = set()

    # --------------------------------------------------------
    # Fill complete GT
    # --------------------------------------------------------

    for row in pair_rows:

        i = current_index[
            row["plane_a"]
        ]

        j = reference_index[
            row["plane_b"]
        ]

        gt_label_matrix[i, j] = (
            row["gt_label"]
        )

        pair_id_matrix[i, j] = (
            row["pair_id"]
        )

        if row["gt_label"] == 1:

            full_gt_matches.add(
                (i, j)
            )

    # --------------------------------------------------------
    # Fill gated candidates + raw Similarity scores
    # --------------------------------------------------------

    failed_similarity = 0

    for row in pair_rows:

        pair_id = row["pair_id"]

        if pair_id not in similarity_lookup:
            continue

        sim = similarity_lookup[
            pair_id
        ]

        if (
            sim["current_frame_id"]
            != frame_a
            or
            sim["reference_frame_id"]
            != frame_b
        ):
            raise RuntimeError(
                f"Frame mismatch for pair_id "
                f"{pair_id}."
            )

        if (
            sim["gt_label"]
            != row["gt_label"]
        ):
            raise RuntimeError(
                f"GT mismatch for pair_id "
                f"{pair_id}: "
                f"candidate_pairs="
                f"{row['gt_label']}, "
                f"real_final="
                f"{sim['gt_label']}"
            )

        if sim["success"] != 1:

            failed_similarity += 1

            continue

        score = float(
            sim["score"]
        )

        # Current Assignment implementation expects s_ij in [0, 1].
        if (
            score < -1e-9
            or
            score > 1.0 + 1e-9
        ):
            raise ValueError(
                f"Similarity score outside "
                f"[0,1]: pair_id={pair_id}, "
                f"score={score}"
            )

        score = float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )

        i = current_index[
            row["plane_a"]
        ]

        j = reference_index[
            row["plane_b"]
        ]

        candidate_mask[i, j] = True

        similarity[i, j] = score

    # --------------------------------------------------------
    # Reachable Assignment GT
    # --------------------------------------------------------
    #
    # The solver cannot recover a correspondence that Candidate
    # Gating has already removed. Therefore Assignment-stage GT
    # consists only of true GT edges present in candidate_mask.
    #
    # Full physical GT is retained separately for pipeline analysis.
    # --------------------------------------------------------

    assignment_gt_matches = {
        (i, j)
        for (i, j)
        in full_gt_matches
        if candidate_mask[i, j]
    }

    # --------------------------------------------------------
    # Persistent plane IDs
    # --------------------------------------------------------

    (
        current_gt_ids,
        reference_gt_ids,
    ) = recover_plane_gt_ids(
        pair_rows,
        current_ids,
        reference_ids,
    )

    current_gt_known = (
        current_gt_ids >= 0
    )

    reference_gt_known = (
        reference_gt_ids >= 0
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    n_candidate_pairs = int(
        np.sum(candidate_mask)
    )

    n_positive = int(
        np.sum(
            gt_label_matrix == 1
        )
    )

    n_negative = int(
        np.sum(
            gt_label_matrix == 0
        )
    )

    n_unknown = int(
        np.sum(
            gt_label_matrix == -1
        )
    )

    n_evaluable = (
        n_positive
        + n_negative
    )

    n_reachable_positive = len(
        assignment_gt_matches
    )

    n_missed_upstream = (
        n_positive
        - n_reachable_positive
    )

    candidate_known_positive = int(
        np.sum(
            candidate_mask
            & (gt_label_matrix == 1)
        )
    )

    candidate_known_negative = int(
        np.sum(
            candidate_mask
            & (gt_label_matrix == 0)
        )
    )

    candidate_unknown = int(
        np.sum(
            candidate_mask
            & (gt_label_matrix == -1)
        )
    )

    case_id = (
        f"tum_{frame_a:04d}_"
        f"{frame_b:04d}"
    )

    metadata = {
        "dataset":
            "TUM fr1_xyz + ETH3D-derived plane identities",

        "split":
            "tum_real",

        "scenario":
            "real",

        "frame_current":
            frame_a,

        "frame_reference":
            frame_b,

        "similarity_method":
            method,

        "n_current":
            n_current,

        "n_reference":
            n_reference,

        "n_full_pairs":
            expected_full_pairs,

        "n_candidate_pairs":
            n_candidate_pairs,

        "n_evaluable_full_pairs":
            n_evaluable,

        "n_unknown_full_pairs":
            n_unknown,

        "n_positive_full_gt":
            n_positive,

        "n_negative_full_gt":
            n_negative,

        "n_assignment_gt_matches":
            n_reachable_positive,

        "n_gt_matches_missed_upstream":
            n_missed_upstream,

        "candidate_known_positive":
            candidate_known_positive,

        "candidate_known_negative":
            candidate_known_negative,

        "candidate_unknown":
            candidate_unknown,

        "failed_similarity":
            failed_similarity,

        "gt_mode":
            "reachable_stage_gt_with_full_gt_sidecar",
    }

    # --------------------------------------------------------
    # Existing AssignmentCase format
    # --------------------------------------------------------

    case = AssignmentCase(
        case_id=case_id,
        current_ids=current_ids,
        reference_ids=reference_ids,
        similarity=similarity,
        candidate_mask=candidate_mask,
        gt_matches=assignment_gt_matches,
        metadata=metadata,
    )

    # Additional real-data GT that the original AssignmentCase does
    # not contain.
    reference = {
        "case_id":
            case_id,

        "gt_label_matrix":
            gt_label_matrix,

        "pair_id_matrix":
            pair_id_matrix,

        "current_gt_plane_ids":
            current_gt_ids,

        "reference_gt_plane_ids":
            reference_gt_ids,

        "current_gt_known":
            current_gt_known,

        "reference_gt_known":
            reference_gt_known,

        "full_gt_matches":
            np.asarray(
                sorted(full_gt_matches),
                dtype=np.int32,
            ).reshape(-1, 2),

        "assignment_gt_matches":
            np.asarray(
                sorted(
                    assignment_gt_matches
                ),
                dtype=np.int32,
            ).reshape(-1, 2),
    }

    return case, reference


# ============================================================
# Save GT sidecar
# ============================================================

def save_reference(
    path: Path,
    reference: dict,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,

        case_id=np.asarray(
            reference["case_id"]
        ),

        gt_label_matrix=
            reference[
                "gt_label_matrix"
            ],

        pair_id_matrix=
            reference[
                "pair_id_matrix"
            ],

        current_gt_plane_ids=
            reference[
                "current_gt_plane_ids"
            ],

        reference_gt_plane_ids=
            reference[
                "reference_gt_plane_ids"
            ],

        current_gt_known=
            reference[
                "current_gt_known"
            ],

        reference_gt_known=
            reference[
                "reference_gt_known"
            ],

        full_gt_matches=
            reference[
                "full_gt_matches"
            ],

        assignment_gt_matches=
            reference[
                "assignment_gt_matches"
            ],
    )


# ============================================================
# Manifest
# ============================================================

def save_manifest(
    path: Path,
    rows,
):

    if not rows:
        raise ValueError(
            "No manifest rows."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# Main preparation
# ============================================================

def prepare(
    pair_file: Path,
    similarity_file: Path,
    output_dir: Path,
    reference_dir: Path,
    method: str,
):

    pair_lookup, transitions = (
        load_candidate_pairs(
            pair_file
        )
    )

    similarity_lookup = (
        load_similarity(
            similarity_file,
            method,
        )
    )

    # --------------------------------------------------------
    # Validate similarity pair IDs
    # --------------------------------------------------------

    for pair_id in similarity_lookup:

        if pair_id not in pair_lookup:

            raise KeyError(
                f"pair_id={pair_id} exists in "
                f"real_final.csv but not in "
                f"candidate_pairs.csv."
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = []

    total_candidates = 0

    total_full_gt = 0

    total_reachable_gt = 0

    total_missed_upstream = 0

    print()
    print("=" * 78)
    print("PREPARE REAL ASSIGNMENT DATA")
    print("=" * 78)

    print(
        f"Similarity method : {method}"
    )

    print(
        f"Transitions       : "
        f"{len(transitions)}"
    )

    print()

    # --------------------------------------------------------
    # Build all frame transitions
    # --------------------------------------------------------

    for index, (
        (frame_a, frame_b),
        pair_rows,
    ) in enumerate(
        transitions.items(),
        start=1,
    ):

        case, reference = (
            build_case(
                frame_a=frame_a,
                frame_b=frame_b,
                pair_rows=pair_rows,
                similarity_lookup=
                    similarity_lookup,
                method=method,
            )
        )

        case_path = (
            output_dir
            / f"{case.case_id}.npz"
        )

        gt_path = (
            reference_dir
            / f"{case.case_id}_gt.npz"
        )

        # Existing Assignment save function
        save_assignment_case(
            case,
            case_path,
        )

        save_reference(
            gt_path,
            reference,
        )

        m = case.metadata

        manifest.append(
            {
                "case_id":
                    case.case_id,

                "frame_current":
                    frame_a,

                "frame_reference":
                    frame_b,

                "n_current":
                    m["n_current"],

                "n_reference":
                    m["n_reference"],

                "full_pairs":
                    m["n_full_pairs"],

                "candidate_pairs":
                    m["n_candidate_pairs"],

                "full_gt_matches":
                    m["n_positive_full_gt"],

                "assignment_gt_matches":
                    m[
                        "n_assignment_gt_matches"
                    ],

                "gt_missed_upstream":
                    m[
                        "n_gt_matches_missed_upstream"
                    ],

                "candidate_positive":
                    m[
                        "candidate_known_positive"
                    ],

                "candidate_negative":
                    m[
                        "candidate_known_negative"
                    ],

                "candidate_unknown":
                    m[
                        "candidate_unknown"
                    ],

                "failed_similarity":
                    m[
                        "failed_similarity"
                    ],

                "similarity_method":
                    method,
            }
        )

        total_candidates += (
            m["n_candidate_pairs"]
        )

        total_full_gt += (
            m["n_positive_full_gt"]
        )

        total_reachable_gt += (
            m[
                "n_assignment_gt_matches"
            ]
        )

        total_missed_upstream += (
            m[
                "n_gt_matches_missed_upstream"
            ]
        )

        print(
            f"[{index:02d}/"
            f"{len(transitions):02d}] "
            f"{case.case_id} | "
            f"{m['n_current']} x "
            f"{m['n_reference']} | "
            f"candidate="
            f"{m['n_candidate_pairs']:3d} | "
            f"GT(full)="
            f"{m['n_positive_full_gt']:2d} | "
            f"GT(reachable)="
            f"{m['n_assignment_gt_matches']:2d}"
        )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest_file = (
        output_dir
        / "manifest.csv"
    )

    save_manifest(
        manifest_file,
        manifest,
    )

    # --------------------------------------------------------
    # Check every successful similarity row was consumed
    # --------------------------------------------------------

    successful_similarity_ids = {
        pair_id
        for pair_id, row
        in similarity_lookup.items()
        if row["success"] == 1
    }

    consumed_ids = set()

    for pair_id, row in pair_lookup.items():

        if pair_id in successful_similarity_ids:
            consumed_ids.add(pair_id)

    if (
        successful_similarity_ids
        != consumed_ids
    ):

        missing = (
            successful_similarity_ids
            - consumed_ids
        )

        raise RuntimeError(
            "Some Similarity candidate pairs "
            "were not consumed:\n"
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("REAL ASSIGNMENT PREPARATION COMPLETE")
    print("=" * 78)

    print(
        f"Assignment cases          : "
        f"{len(manifest)}"
    )

    print(
        f"Candidate edges           : "
        f"{total_candidates}"
    )

    print(
        f"Full physical GT matches  : "
        f"{total_full_gt}"
    )

    print(
        f"Reachable Assignment GT   : "
        f"{total_reachable_gt}"
    )

    print(
        f"GT removed upstream       : "
        f"{total_missed_upstream}"
    )

    print()

    print(
        f"Assignment cases:\n"
        f"  {output_dir}"
    )

    print(
        f"GT sidecars:\n"
        f"  {reference_dir}"
    )

    print(
        f"Manifest:\n"
        f"  {manifest_file}"
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Prepare real TUM data for "
            "the Assignment stage."
        )
    )

    parser.add_argument(
        "--pair-file",
        type=Path,
        default=PAIR_FILE,
    )

    parser.add_argument(
        "--similarity-file",
        type=Path,
        default=SIMILARITY_FILE,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )

    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=REFERENCE_DIR,
    )

    parser.add_argument(
        "--method",
        type=str,
        default=DEFAULT_SIMILARITY_METHOD,
    )

    args = parser.parse_args()

    prepare(
        pair_file=args.pair_file,
        similarity_file=args.similarity_file,
        output_dir=args.output_dir,
        reference_dir=args.reference_dir,
        method=args.method,
    )


if __name__ == "__main__":
    main()