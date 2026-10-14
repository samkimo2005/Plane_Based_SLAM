"""
validate_bad_match_rejection_tum.py

Validate the prepared TUM real-data inputs for the Bad-Match Rejection
experiment before running any rejection algorithm.

Checks
------
- required NPZ fields exist;
- plane arrays and centroids have consistent shapes;
- assigned-pair indices are valid and one-to-one;
- GT labels are internally consistent;
- plane normals are unit length;
- R_gt is a valid rotation matrix;
- t_gt and timestamps are finite where required.

Summary
-------
The script reports:
- number of prepared cases;
- distribution of assigned correspondence counts;
- cases with >= 3 assigned correspondences;
- cases with >= 3 evaluable correspondences;
- total good / bad / ambiguous labels;
- number of cases containing known bad matches;
- number of all-ambiguous cases;
- concentration of known bad matches across cases.

No bad-match rejection algorithm is run here.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "data"
    / "bad_match_rejection"
    / "tum"
)

DEFAULT_CASE_STATS_FILE = (
    DEFAULT_DATA_DIR
    / "validation_case_stats.csv"
)


REQUIRED_FIELDS = {
    "case_id",
    "frame_a",
    "frame_b",
    "planes_a",
    "planes_b",
    "centroids_a",
    "centroids_b",
    "plane_ids_a",
    "plane_ids_b",
    "assigned_pairs",
    "assigned_similarity",
    "gt_match_labels",
    "gt_match_ambiguous",
    "gt_match_labels_int8",
    "R_gt",
    "t_gt",
    "timestamp_a",
    "timestamp_b",
    "assignment_method",
}


# ============================================================
# Helpers
# ============================================================

def scalar_text(value) -> str:
    arr = np.asarray(value)

    if arr.ndim == 0:
        return str(arr.item())

    if arr.size == 1:
        return str(arr.reshape(-1)[0])

    return str(value)


def scalar_int(value) -> int:
    return int(
        np.asarray(value).item()
    )


def scalar_float(value) -> float:
    return float(
        np.asarray(value).item()
    )


def rotation_angle_deg(
    R: np.ndarray,
) -> float:

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
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
    )

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
# Validate one case
# ============================================================

def validate_case(
    path: Path,
) -> dict:

    errors: list[str] = []

    with np.load(
        path,
        allow_pickle=True,
    ) as data:

        missing = (
            REQUIRED_FIELDS
            - set(data.files)
        )

        if missing:
            raise KeyError(
                f"{path.name}: missing fields "
                f"{sorted(missing)}"
            )

        case_id = scalar_text(
            data["case_id"]
        )

        frame_a = scalar_int(
            data["frame_a"]
        )

        frame_b = scalar_int(
            data["frame_b"]
        )

        assignment_method = scalar_text(
            data["assignment_method"]
        )

        planes_a = np.asarray(
            data["planes_a"],
            dtype=np.float64,
        )

        planes_b = np.asarray(
            data["planes_b"],
            dtype=np.float64,
        )

        centroids_a = np.asarray(
            data["centroids_a"],
            dtype=np.float64,
        )

        centroids_b = np.asarray(
            data["centroids_b"],
            dtype=np.float64,
        )

        plane_ids_a = np.asarray(
            data["plane_ids_a"],
            dtype=np.int64,
        ).reshape(-1)

        plane_ids_b = np.asarray(
            data["plane_ids_b"],
            dtype=np.int64,
        ).reshape(-1)

        assigned_pairs = np.asarray(
            data["assigned_pairs"],
            dtype=np.int64,
        ).reshape(-1, 2)

        assigned_similarity = np.asarray(
            data["assigned_similarity"],
            dtype=np.float64,
        ).reshape(-1)

        gt_match_labels = np.asarray(
            data["gt_match_labels"],
            dtype=bool,
        ).reshape(-1)

        gt_match_ambiguous = np.asarray(
            data["gt_match_ambiguous"],
            dtype=bool,
        ).reshape(-1)

        gt_labels_int8 = np.asarray(
            data["gt_match_labels_int8"],
            dtype=np.int8,
        ).reshape(-1)

        R_gt = np.asarray(
            data["R_gt"],
            dtype=np.float64,
        )

        t_gt = np.asarray(
            data["t_gt"],
            dtype=np.float64,
        ).reshape(-1)

        timestamp_a = scalar_float(
            data["timestamp_a"]
        )

        timestamp_b = scalar_float(
            data["timestamp_b"]
        )

    # --------------------------------------------------------
    # Plane array shapes
    # --------------------------------------------------------

    if (
        planes_a.ndim != 2
        or planes_a.shape[1] != 4
    ):
        errors.append(
            f"planes_a shape={planes_a.shape}"
        )

    if (
        planes_b.ndim != 2
        or planes_b.shape[1] != 4
    ):
        errors.append(
            f"planes_b shape={planes_b.shape}"
        )

    n_a = (
        planes_a.shape[0]
        if planes_a.ndim == 2
        else -1
    )

    n_b = (
        planes_b.shape[0]
        if planes_b.ndim == 2
        else -1
    )

    if centroids_a.shape != (n_a, 3):
        errors.append(
            f"centroids_a shape={centroids_a.shape}"
        )

    if centroids_b.shape != (n_b, 3):
        errors.append(
            f"centroids_b shape={centroids_b.shape}"
        )

    if len(plane_ids_a) != n_a:
        errors.append(
            "plane_ids_a length mismatch"
        )

    if len(plane_ids_b) != n_b:
        errors.append(
            "plane_ids_b length mismatch"
        )

    # --------------------------------------------------------
    # Finite geometry + unit normals
    # --------------------------------------------------------

    if not np.all(
        np.isfinite(planes_a)
    ):
        errors.append(
            "planes_a contains non-finite values"
        )

    if not np.all(
        np.isfinite(planes_b)
    ):
        errors.append(
            "planes_b contains non-finite values"
        )

    if not np.all(
        np.isfinite(centroids_a)
    ):
        errors.append(
            "centroids_a contains non-finite values"
        )

    if not np.all(
        np.isfinite(centroids_b)
    ):
        errors.append(
            "centroids_b contains non-finite values"
        )

    if n_a > 0:
        norms_a = np.linalg.norm(
            planes_a[:, :3],
            axis=1,
        )

        if not np.allclose(
            norms_a,
            1.0,
            atol=1e-3,
        ):
            errors.append(
                "planes_a normals not unit length"
            )

    if n_b > 0:
        norms_b = np.linalg.norm(
            planes_b[:, :3],
            axis=1,
        )

        if not np.allclose(
            norms_b,
            1.0,
            atol=1e-3,
        ):
            errors.append(
                "planes_b normals not unit length"
            )

    # --------------------------------------------------------
    # Assigned-pair structure
    # --------------------------------------------------------

    k = len(
        assigned_pairs
    )

    if len(
        assigned_similarity
    ) != k:
        errors.append(
            "assigned_similarity length mismatch"
        )

    if len(
        gt_match_labels
    ) != k:
        errors.append(
            "gt_match_labels length mismatch"
        )

    if len(
        gt_match_ambiguous
    ) != k:
        errors.append(
            "gt_match_ambiguous length mismatch"
        )

    if len(
        gt_labels_int8
    ) != k:
        errors.append(
            "gt_match_labels_int8 length mismatch"
        )

    if k > 0:

        if np.any(
            assigned_pairs[:, 0] < 0
        ) or np.any(
            assigned_pairs[:, 0] >= n_a
        ):
            errors.append(
                "assigned current index out of bounds"
            )

        if np.any(
            assigned_pairs[:, 1] < 0
        ) or np.any(
            assigned_pairs[:, 1] >= n_b
        ):
            errors.append(
                "assigned reference index out of bounds"
            )

        if (
            len(
                np.unique(
                    assigned_pairs[:, 0]
                )
            )
            != k
        ):
            errors.append(
                "current-plane assignment is not one-to-one"
            )

        if (
            len(
                np.unique(
                    assigned_pairs[:, 1]
                )
            )
            != k
        ):
            errors.append(
                "reference-plane assignment is not one-to-one"
            )

    if not np.all(
        np.isfinite(
            assigned_similarity
        )
    ):
        errors.append(
            "non-finite assigned similarity"
        )

    if np.any(
        assigned_similarity < -1e-9
    ) or np.any(
        assigned_similarity > 1.0 + 1e-9
    ):
        errors.append(
            "assigned similarity outside [0,1]"
        )

    # --------------------------------------------------------
    # GT consistency
    # --------------------------------------------------------

    valid_gt = np.isin(
        gt_labels_int8,
        [-1, 0, 1],
    )

    if not np.all(valid_gt):
        errors.append(
            "invalid GT labels"
        )

    expected_good = (
        gt_labels_int8 == 1
    )

    expected_ambiguous = (
        gt_labels_int8 == -1
    )

    if not np.array_equal(
        gt_match_labels,
        expected_good,
    ):
        errors.append(
            "gt_match_labels inconsistent with int8 GT"
        )

    if not np.array_equal(
        gt_match_ambiguous,
        expected_ambiguous,
    ):
        errors.append(
            "gt_match_ambiguous inconsistent with int8 GT"
        )

    if np.any(
        gt_match_labels
        & gt_match_ambiguous
    ):
        errors.append(
            "pair marked both good and ambiguous"
        )

    # --------------------------------------------------------
    # Ground-truth pose
    # --------------------------------------------------------

    if R_gt.shape != (3, 3):
        errors.append(
            f"R_gt shape={R_gt.shape}"
        )

    if t_gt.shape != (3,):
        errors.append(
            f"t_gt shape={t_gt.shape}"
        )

    if R_gt.shape == (3, 3):

        if not np.all(
            np.isfinite(R_gt)
        ):
            errors.append(
                "R_gt contains non-finite values"
            )

        orthogonality_error = float(
            np.linalg.norm(
                R_gt.T @ R_gt
                - np.eye(3)
            )
        )

        det_R = float(
            np.linalg.det(R_gt)
        )

        if orthogonality_error > 1e-6:
            errors.append(
                "R_gt is not orthonormal"
            )

        if abs(
            det_R - 1.0
        ) > 1e-6:
            errors.append(
                f"det(R_gt)={det_R:.8f}"
            )

    else:
        orthogonality_error = np.nan
        det_R = np.nan

    if not np.all(
        np.isfinite(t_gt)
    ):
        errors.append(
            "t_gt contains non-finite values"
        )

    if not (
        np.isfinite(timestamp_a)
        and np.isfinite(timestamp_b)
    ):
        errors.append(
            "non-finite RGB timestamp"
        )

    # --------------------------------------------------------
    # Case statistics
    # --------------------------------------------------------

    n_good = int(
        np.sum(
            gt_labels_int8 == 1
        )
    )

    n_bad = int(
        np.sum(
            gt_labels_int8 == 0
        )
    )

    n_ambiguous = int(
        np.sum(
            gt_labels_int8 == -1
        )
    )

    n_evaluable = (
        n_good
        + n_bad
    )

    bad_ratio_evaluable = (
        float(
            n_bad
            / n_evaluable
        )
        if n_evaluable > 0
        else np.nan
    )

    row = {
        "case_id":
            case_id,

        "frame_a":
            frame_a,

        "frame_b":
            frame_b,

        "assignment_method":
            assignment_method,

        "n_planes_a":
            n_a,

        "n_planes_b":
            n_b,

        "n_assigned":
            k,

        "n_evaluable":
            n_evaluable,

        "n_good":
            n_good,

        "n_bad":
            n_bad,

        "n_ambiguous":
            n_ambiguous,

        "bad_ratio_evaluable":
            bad_ratio_evaluable,

        "assigned_ge_3":
            int(k >= 3),

        "evaluable_ge_3":
            int(
                n_evaluable >= 3
            ),

        "contains_bad":
            int(n_bad > 0),

        "all_ambiguous":
            int(
                k > 0
                and n_evaluable == 0
            ),

        "zero_assigned":
            int(k == 0),

        "rotation_gt_deg":
            (
                rotation_angle_deg(
                    R_gt
                )
                if R_gt.shape == (3, 3)
                else np.nan
            ),

        "translation_gt_m":
            (
                float(
                    np.linalg.norm(t_gt)
                )
                if t_gt.shape == (3,)
                else np.nan
            ),

        "R_orthogonality_error":
            orthogonality_error,

        "det_R":
            det_R,

        "valid":
            int(
                len(errors) == 0
            ),

        "errors":
            " | ".join(errors),
    }

    return row


# ============================================================
# Aggregate report
# ============================================================

def print_distribution(
    counter: Counter,
) -> None:

    for n_assigned in sorted(
        counter
    ):

        print(
            f"  {n_assigned:>2} assigned : "
            f"{counter[n_assigned]:>3} cases"
        )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Validate prepared TUM "
            "Bad-Match Rejection data."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )

    parser.add_argument(
        "--case-stats-file",
        type=Path,
        default=
            DEFAULT_CASE_STATS_FILE,
    )

    args = parser.parse_args()

    files = sorted(
        args.data_dir.glob(
            "tum_*.npz"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No prepared TUM bad-match cases found in:\n"
            f"{args.data_dir}"
        )

    rows = []

    for path in files:

        row = validate_case(
            path
        )

        rows.append(
            row
        )

    save_csv(
        args.case_stats_file,
        rows,
    )

    invalid_rows = [
        row
        for row in rows
        if row["valid"] == 0
    ]

    assigned_distribution = Counter(
        row["n_assigned"]
        for row in rows
    )

    total_assigned = sum(
        row["n_assigned"]
        for row in rows
    )

    total_evaluable = sum(
        row["n_evaluable"]
        for row in rows
    )

    total_good = sum(
        row["n_good"]
        for row in rows
    )

    total_bad = sum(
        row["n_bad"]
        for row in rows
    )

    total_ambiguous = sum(
        row["n_ambiguous"]
        for row in rows
    )

    cases_assigned_ge_3 = sum(
        row["assigned_ge_3"]
        for row in rows
    )

    cases_evaluable_ge_3 = sum(
        row["evaluable_ge_3"]
        for row in rows
    )

    cases_with_bad = sum(
        row["contains_bad"]
        for row in rows
    )

    all_ambiguous_cases = sum(
        row["all_ambiguous"]
        for row in rows
    )

    zero_assigned_cases = sum(
        row["zero_assigned"]
        for row in rows
    )

    bad_counts = sorted(
        (
            int(row["n_bad"])
            for row in rows
            if row["n_bad"] > 0
        ),
        reverse=True,
    )

    max_bad_one_case = (
        bad_counts[0]
        if bad_counts
        else 0
    )

    bad_ratio_global = (
        float(
            total_bad
            / total_evaluable
        )
        if total_evaluable > 0
        else np.nan
    )

    print()
    print("=" * 78)
    print("TUM BAD-MATCH PREPARED-DATA VALIDATION")
    print("=" * 78)

    print(
        f"Cases checked                : "
        f"{len(rows)}"
    )

    print(
        f"Valid cases                  : "
        f"{len(rows) - len(invalid_rows)}"
    )

    print(
        f"Invalid cases                : "
        f"{len(invalid_rows)}"
    )

    print()

    print(
        f"Assigned pairs               : "
        f"{total_assigned}"
    )

    print(
        f"Evaluable pairs              : "
        f"{total_evaluable}"
    )

    print(
        f"Known good pairs             : "
        f"{total_good}"
    )

    print(
        f"Known bad pairs              : "
        f"{total_bad}"
    )

    print(
        f"Ambiguous pairs              : "
        f"{total_ambiguous}"
    )

    print(
        f"Known-bad ratio              : "
        f"{bad_ratio_global:.4f}"
    )

    print()

    print(
        f"Cases with >=3 assigned      : "
        f"{cases_assigned_ge_3}"
        f"/{len(rows)}"
    )

    print(
        f"Cases with >=3 evaluable     : "
        f"{cases_evaluable_ge_3}"
        f"/{len(rows)}"
    )

    print(
        f"Cases with known bad match   : "
        f"{cases_with_bad}"
        f"/{len(rows)}"
    )

    print(
        f"All-ambiguous cases          : "
        f"{all_ambiguous_cases}"
    )

    print(
        f"Zero-assignment cases        : "
        f"{zero_assigned_cases}"
    )

    print(
        f"Max known bad in one case    : "
        f"{max_bad_one_case}"
    )

    print()
    print(
        "Assigned-pair distribution:"
    )

    print_distribution(
        assigned_distribution
    )

    print()

    if invalid_rows:

        print(
            "INVALID CASES:"
        )

        for row in invalid_rows:

            print(
                f"  {row['case_id']}: "
                f"{row['errors']}"
            )

        raise RuntimeError(
            f"{len(invalid_rows)} prepared cases "
            f"failed validation."
        )

    print(
        "All prepared cases passed structural validation."
    )

    print(
        f"Per-case statistics saved to:\n"
        f"  {args.case_stats_file}"
    )


if __name__ == "__main__":
    main()
