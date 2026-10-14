"""
retune_distance_offset_only.py

Canonical re-tuning step after restoring the original Distance method.

What this does
--------------
- Uses ONLY Synthetic DEV data.
- Re-tunes ONLY the restored offset-only Distance method.
- Keeps the already-corrected SelMaP-inspired parameters from tuning_v2.
- Keeps Normal, RANSAC and Trilateration-inspired unchanged.
- Removes the temporary spatial_threshold_m field from Distance.
- Updates:
    experiments/results/bad_match_rejection/tuning_v2/best_parameters.json

This avoids creating another competing tuning folder.

Selection policy
----------------
1. mean per-case Macro-F1 on non-clean DEV cases;
2. pooled Macro-F1 on non-clean DEV cases;
3. clean-case correct-match retention;
4. completion rate;
5. lower median runtime.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import shutil
import time
from pathlib import Path

import numpy as np

import distance_spatial_rejection as distance


# ============================================================
# Paths
# ============================================================

HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent

DEFAULT_DEV_DIR = (
    EXPERIMENTS_DIR
    / "data"
    / "bad_match_rejection"
    / "dev"
)

DEFAULT_TUNING_DIR = (
    EXPERIMENTS_DIR
    / "results"
    / "bad_match_rejection"
    / "tuning_v2"
)

DEFAULT_PARAMETER_FILE = (
    DEFAULT_TUNING_DIR
    / "best_parameters.json"
)

DEFAULT_GRID_FILE = (
    DEFAULT_TUNING_DIR
    / "distance_offset_grid_results.csv"
)

DEFAULT_SUMMARY_FILE = (
    DEFAULT_TUNING_DIR
    / "distance_offset_tuning_summary.json"
)


# ============================================================
# Offset-only grid
# ============================================================

OFFSET_GRID_M = [
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.15,
    0.20,
]


# ============================================================
# Helpers
# ============================================================

def safe_divide(
    numerator: float,
    denominator: float,
) -> float:

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


def f1_from_counts(
    tp: int,
    fp: int,
    fn: int,
) -> float:

    return safe_divide(
        2.0 * tp,
        2.0 * tp + fp + fn,
    )


def macro_f1_from_counts(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> float:

    bad_f1 = f1_from_counts(
        tp,
        fp,
        fn,
    )

    correct_f1 = f1_from_counts(
        tn,
        fn,
        fp,
    )

    return float(
        (bad_f1 + correct_f1) / 2.0
    )


def classification_counts(
    keep_mask: np.ndarray,
    gt_good: np.ndarray,
) -> tuple[int, int, int, int]:

    keep_mask = np.asarray(
        keep_mask,
        dtype=bool,
    ).reshape(-1)

    gt_good = np.asarray(
        gt_good,
        dtype=bool,
    ).reshape(-1)

    if len(keep_mask) != len(gt_good):
        raise ValueError(
            "keep_mask / gt_match_labels length mismatch."
        )

    pred_bad = ~keep_mask
    gt_bad = ~gt_good

    tp = int(
        np.sum(
            pred_bad & gt_bad
        )
    )

    fp = int(
        np.sum(
            pred_bad & gt_good
        )
    )

    fn = int(
        np.sum(
            keep_mask & gt_bad
        )
    )

    tn = int(
        np.sum(
            keep_mask & gt_good
        )
    )

    return tp, fp, fn, tn


def finite_mean(values) -> float:

    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if len(array) == 0:
        return np.nan

    return float(
        np.mean(array)
    )


def finite_median(values) -> float:

    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if len(array) == 0:
        return np.nan

    return float(
        np.median(array)
    )


def save_csv(
    path: Path,
    rows: list[dict],
) -> None:

    if not rows:
        raise ValueError(
            "No tuning rows to save."
        )

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
# Audit restored Distance implementation
# ============================================================

def audit_distance_interface() -> None:

    signature = inspect.signature(
        distance.reject_bad_matches
    )

    parameters = set(
        signature.parameters
    )

    required = {
        "planes_a",
        "planes_b",
        "assigned_pairs",
        "offset_threshold",
    }

    missing = (
        required
        - parameters
    )

    if missing:
        raise RuntimeError(
            "Distance implementation is not the restored "
            "offset-only version. Missing parameters: "
            f"{sorted(missing)}"
        )

    forbidden = {
        "centroids_a",
        "centroids_b",
        "spatial_threshold",
    }

    present_forbidden = (
        forbidden
        & parameters
    )

    if present_forbidden:
        raise RuntimeError(
            "The imported Distance implementation still contains "
            "the temporary spatial/centroid version. Found: "
            f"{sorted(present_forbidden)}"
        )


# ============================================================
# DEV loading
# ============================================================

def find_dev_files(
    dev_dir: Path,
) -> list[Path]:

    for pattern in (
        "dev_*.npz",
        "sample_*.npz",
        "*.npz",
    ):

        files = sorted(
            dev_dir.glob(pattern)
        )

        if files:
            return files

    return []


def load_case(
    path: Path,
) -> dict:

    with np.load(
        path,
        allow_pickle=True,
    ) as data:

        required = {
            "planes_a",
            "planes_b",
            "assigned_pairs",
            "gt_match_labels",
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

        planes_a = np.asarray(
            data["planes_a"],
            dtype=float,
        )

        planes_b = np.asarray(
            data["planes_b"],
            dtype=float,
        )

        assigned_pairs = np.asarray(
            data["assigned_pairs"],
            dtype=int,
        ).reshape(-1, 2)

        gt_good = np.asarray(
            data["gt_match_labels"],
            dtype=bool,
        ).reshape(-1)

    if len(
        assigned_pairs
    ) != len(
        gt_good
    ):
        raise ValueError(
            f"{path.name}: assigned_pairs / "
            "gt_match_labels length mismatch."
        )

    return {
        "name":
            path.name,

        "planes_a":
            planes_a,

        "planes_b":
            planes_b,

        "assigned_pairs":
            assigned_pairs,

        "gt_good":
            gt_good,

        "is_clean":
            bool(
                np.all(
                    gt_good
                )
            ),
    }


def load_dev_cases(
    dev_dir: Path,
) -> list[dict]:

    files = find_dev_files(
        dev_dir
    )

    if not files:
        raise FileNotFoundError(
            "No Synthetic DEV .npz files found in:\n"
            f"{dev_dir}"
        )

    cases = [
        load_case(path)
        for path in files
    ]

    n_clean = sum(
        case["is_clean"]
        for case in cases
    )

    n_nonclean = (
        len(cases)
        - n_clean
    )

    if n_nonclean == 0:
        raise RuntimeError(
            "DEV set contains no bad-match cases."
        )

    print(
        f"DEV cases       : {len(cases)}"
    )

    print(
        f"Clean cases     : {n_clean}"
    )

    print(
        f"Non-clean cases : {n_nonclean}"
    )

    return cases


# ============================================================
# Evaluate one threshold
# ============================================================

def evaluate_threshold(
    cases: list[dict],
    offset_threshold: float,
) -> dict:

    case_macro_nonclean = []
    clean_retention = []
    runtimes = []

    pooled_tp = 0
    pooled_fp = 0
    pooled_fn = 0
    pooled_tn = 0

    completed = 0
    failure_count = 0

    for case in cases:

        start = (
            time.perf_counter()
        )

        try:

            result = (
                distance.reject_bad_matches(
                    case["planes_a"],
                    case["planes_b"],
                    case["assigned_pairs"],
                    offset_threshold=
                        offset_threshold,
                )
            )

            keep_mask = np.asarray(
                result["keep_mask"],
                dtype=bool,
            ).reshape(-1)

            if len(
                keep_mask
            ) != len(
                case["assigned_pairs"]
            ):
                raise RuntimeError(
                    "keep_mask length mismatch."
                )

            completed += 1

        except Exception:

            # DEV tuning penalises actual execution failure.
            keep_mask = np.zeros(
                len(
                    case["assigned_pairs"]
                ),
                dtype=bool,
            )

            failure_count += 1

        runtime_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        runtimes.append(
            runtime_ms
        )

        tp, fp, fn, tn = (
            classification_counts(
                keep_mask,
                case["gt_good"],
            )
        )

        if case["is_clean"]:

            clean_retention.append(
                safe_divide(
                    tn,
                    tn + fp,
                )
            )

        else:

            pooled_tp += tp
            pooled_fp += fp
            pooled_fn += fn
            pooled_tn += tn

            case_macro_nonclean.append(
                macro_f1_from_counts(
                    tp,
                    fp,
                    fn,
                    tn,
                )
            )

    return {
        "offset_threshold_m":
            float(
                offset_threshold
            ),

        "mean_case_macro_f1_nonclean":
            finite_mean(
                case_macro_nonclean
            ),

        "pooled_macro_f1_nonclean":
            macro_f1_from_counts(
                pooled_tp,
                pooled_fp,
                pooled_fn,
                pooled_tn,
            ),

        "bad_precision_nonclean":
            safe_divide(
                pooled_tp,
                pooled_tp + pooled_fp,
            ),

        "bad_recall_nonclean":
            safe_divide(
                pooled_tp,
                pooled_tp + pooled_fn,
            ),

        "correct_retention_nonclean":
            safe_divide(
                pooled_tn,
                pooled_tn + pooled_fp,
            ),

        "clean_retention":
            finite_mean(
                clean_retention
            ),

        "completion_rate":
            safe_divide(
                completed,
                len(cases),
            ),

        "failure_count":
            failure_count,

        "runtime_median_ms":
            finite_median(
                runtimes
            ),
    }


def selection_key(
    row: dict,
):

    return (
        float(
            row[
                "mean_case_macro_f1_nonclean"
            ]
        ),

        float(
            row[
                "pooled_macro_f1_nonclean"
            ]
        ),

        float(
            row[
                "clean_retention"
            ]
        ),

        float(
            row[
                "completion_rate"
            ]
        ),

        -float(
            row[
                "runtime_median_ms"
            ]
        ),
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Re-tune the restored offset-only Distance "
            "bad-match rejection method."
        )
    )

    parser.add_argument(
        "--dev-dir",
        type=Path,
        default=DEFAULT_DEV_DIR,
    )

    parser.add_argument(
        "--parameter-file",
        type=Path,
        default=DEFAULT_PARAMETER_FILE,
    )

    args = parser.parse_args()

    print()
    print("=" * 78)
    print("DISTANCE OFFSET-ONLY DEV TUNING")
    print("=" * 78)

    audit_distance_interface()

    cases = load_dev_cases(
        args.dev_dir
    )

    if not (
        args.parameter_file.exists()
    ):
        raise FileNotFoundError(
            "Cannot find tuning_v2 best_parameters.json:\n"
            f"{args.parameter_file}"
        )

    with args.parameter_file.open(
        "r",
        encoding="utf-8",
    ) as f:

        best_parameters = json.load(f)

    rows = []

    print()

    for index, threshold in enumerate(
        OFFSET_GRID_M,
        start=1,
    ):

        row = evaluate_threshold(
            cases,
            threshold,
        )

        rows.append(
            row
        )

        print(
            f"[{index:02d}/{len(OFFSET_GRID_M):02d}] "
            f"offset={threshold:.3f} m | "
            f"caseMacro="
            f"{row['mean_case_macro_f1_nonclean']:.4f} | "
            f"pooledMacro="
            f"{row['pooled_macro_f1_nonclean']:.4f} | "
            f"clean="
            f"{row['clean_retention']:.4f}"
        )

    best_row = max(
        rows,
        key=selection_key,
    )

    # Backup the current v2 parameter file once before replacing
    # the temporary spatial Distance entry.
    backup_file = (
        args.parameter_file.parent
        / "best_parameters_before_distance_restore.json"
    )

    if not backup_file.exists():
        shutil.copy2(
            args.parameter_file,
            backup_file,
        )

    # Replace ONLY Distance.
    best_parameters[
        "Distance"
    ] = {
        "offset_threshold_m":
            float(
                best_row[
                    "offset_threshold_m"
                ]
            ),
    }

    metadata = best_parameters.get(
        "_metadata",
        {},
    )

    metadata[
        "distance_method"
    ] = (
        "restored_offset_only"
    )

    metadata[
        "distance_retuned_on"
    ] = (
        "Synthetic DEV only"
    )

    metadata[
        "distance_offset_grid_m"
    ] = (
        OFFSET_GRID_M
    )

    # Remove the obsolete statement that Distance/spatial was
    # one of the corrected changed methods, while preserving
    # the SelMaP correction metadata.
    metadata[
        "version"
    ] = (
        "final_restored_distance_corrected_selmap"
    )

    best_parameters[
        "_metadata"
    ] = metadata

    save_csv(
        DEFAULT_GRID_FILE,
        rows,
    )

    with args.parameter_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            best_parameters,
            f,
            indent=4,
        )

    boundary_hit = (
        best_row[
            "offset_threshold_m"
        ]
        in (
            min(
                OFFSET_GRID_M
            ),
            max(
                OFFSET_GRID_M
            ),
        )
    )

    summary = {
        "best_distance":
            best_row,

        "boundary_hit":
            boundary_hit,

        "parameter_file_updated":
            str(
                args.parameter_file
            ),

        "backup_file":
            str(
                backup_file
            ),
    }

    with DEFAULT_SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    print()
    print("=" * 78)
    print("DISTANCE OFFSET-ONLY TUNING COMPLETE")
    print("=" * 78)

    print(
        f"Best offset threshold : "
        f"{best_row['offset_threshold_m']:.3f} m"
    )

    print(
        f"DEV case Macro-F1     : "
        f"{best_row['mean_case_macro_f1_nonclean']:.4f}"
    )

    print(
        f"DEV pooled Macro-F1   : "
        f"{best_row['pooled_macro_f1_nonclean']:.4f}"
    )

    print(
        f"Clean retention       : "
        f"{best_row['clean_retention']:.4f}"
    )

    print(
        f"Boundary hit          : "
        f"{boundary_hit}"
    )

    print()
    print(
        "Updated canonical parameter file:"
    )

    print(
        f"  {args.parameter_file}"
    )


if __name__ == "__main__":
    main()
