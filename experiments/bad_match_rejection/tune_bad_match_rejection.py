"""
tune_bad_match_rejection_v2.py

Revised Synthetic DEV tuning for Bad-Match Rejection after correcting:

1. SelMaP-inspired:
   angle consistency + parallel-plane separation consistency.

2. Distance/spatial:
   plane-offset residual + transformed-centroid spatial residual.

Only the TWO changed methods are re-tuned. The unchanged methods
(Normal, RANSAC, Trilateration-inspired) retain their previous frozen
parameters when an old best_parameters.json is available; otherwise
their current module defaults are copied explicitly.

The script performs a full schema/interface audit BEFORE any grid search.
If the Synthetic DEV samples do not contain centroids_a / centroids_b,
the script stops immediately rather than silently running an incomplete
Distance/spatial method.

Selection policy
----------------
Primary:
    mean per-case Macro-F1 over DEV cases containing bad matches.

Tie-breakers:
    1. pooled Macro-F1 on non-clean cases;
    2. correct-match retention on clean (0% bad) cases;
    3. algorithm completion rate;
    4. lower median runtime.

No FINAL or TUM data are used for tuning.

Outputs
-------
experiments/results/bad_match_rejection/tuning_v2/
    distance_grid_results.csv
    selmap_grid_results.csv
    best_parameters.json
    tuning_summary.json
"""

from __future__ import annotations

import argparse
import csv
import inspect
import itertools
import json
import time
from pathlib import Path

import numpy as np

import normal_compatibility as normal
import distance_spatial_rejection as distance
import ransac_geometric_verification as ransac
import selmap_compatibility as selmap
import trilateration_rejection as trilateration


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

DEFAULT_OLD_PARAMETER_FILE = (
    EXPERIMENTS_DIR
    / "results"
    / "bad_match_rejection"
    / "tuning"
    / "best_parameters.json"
)

DEFAULT_OUTPUT_DIR = (
    EXPERIMENTS_DIR
    / "results"
    / "bad_match_rejection"
    / "tuning_v2"
)


# ============================================================
# Tuning grids
# ============================================================

DISTANCE_OFFSET_GRID_M = [
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
]

DISTANCE_SPATIAL_GRID_M = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
]

SELMAP_ANGLE_GRID_DEG = [
    3.0,
    6.0,
    9.0,
    12.0,
]

SELMAP_SEPARATION_GRID_M = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
]

SELMAP_COMPATIBLE_FRACTION_GRID = [
    0.50,
    0.70,
    0.90,
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
        2.0 * tp
        + fp
        + fn,
    )


def macro_f1_from_counts(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> float:
    """
    Positive class = bad / rejected.

    Bad-class:
        TP = bad rejected
        FP = good rejected
        FN = bad retained

    Correct-class:
        TP_correct = TN
        FP_correct = FN
        FN_correct = FP
    """

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
        (
            bad_f1
            + correct_f1
        )
        / 2.0
    )


def classification_counts(
    keep_mask: np.ndarray,
    gt_good: np.ndarray,
) -> tuple[int, int, int, int]:
    """
    gt_good:
        True  = correct correspondence
        False = bad correspondence

    Positive class for rejection evaluation:
        bad correspondence.
    """

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

    pred_bad = (
        ~keep_mask
    )

    gt_bad = (
        ~gt_good
    )

    tp = int(
        np.sum(
            pred_bad
            & gt_bad
        )
    )

    fp = int(
        np.sum(
            pred_bad
            & gt_good
        )
    )

    fn = int(
        np.sum(
            keep_mask
            & gt_bad
        )
    )

    tn = int(
        np.sum(
            keep_mask
            & gt_good
        )
    )

    return (
        tp,
        fp,
        fn,
        tn,
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


def finite_mean(
    values,
) -> float:

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


def finite_median(
    values,
) -> float:

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


# ============================================================
# Interface audit
# ============================================================

def audit_algorithm_interfaces() -> None:
    """
    Refuse to run if the old incomplete implementations are still
    imported accidentally.
    """

    distance_signature = (
        inspect.signature(
            distance.reject_bad_matches
        )
    )

    distance_parameters = set(
        distance_signature.parameters
    )

    required_distance = {
        "centroids_a",
        "centroids_b",
        "spatial_threshold",
    }

    missing_distance = (
        required_distance
        - distance_parameters
    )

    if missing_distance:

        raise RuntimeError(
            "The imported distance_spatial_rejection.py is still "
            "the OLD implementation.\n"
            "Missing function parameters: "
            f"{sorted(missing_distance)}\n\n"
            "Replace it with the updated Distance/spatial file "
            "before tuning."
        )

    selmap_signature = (
        inspect.signature(
            selmap.reject_bad_matches
        )
    )

    selmap_parameters = set(
        selmap_signature.parameters
    )

    required_selmap = {
        "angle_threshold_deg",
        "separation_threshold",
        "min_compatible_fraction",
    }

    missing_selmap = (
        required_selmap
        - selmap_parameters
    )

    if missing_selmap:

        raise RuntimeError(
            "The imported selmap_compatibility.py is still "
            "the OLD implementation.\n"
            "Missing function parameters: "
            f"{sorted(missing_selmap)}\n\n"
            "Replace it with the updated SelMaP-inspired file "
            "before tuning."
        )


# ============================================================
# DEV data loading + audit
# ============================================================

def find_dev_files(
    dev_dir: Path,
) -> list[Path]:

    patterns = [
        "dev_*.npz",
        "sample_*.npz",
    ]

    for pattern in patterns:

        files = sorted(
            dev_dir.glob(
                pattern
            )
        )

        if files:
            return files

    files = sorted(
        path
        for path
        in dev_dir.glob("*.npz")
        if not path.name.startswith(".")
    )

    return files


def load_dev_case(
    path: Path,
) -> dict:

    with np.load(
        path,
        allow_pickle=True,
    ) as data:

        required = {
            "planes_a",
            "planes_b",
            "centroids_a",
            "centroids_b",
            "assigned_pairs",
            "gt_match_labels",
        }

        missing = (
            required
            - set(data.files)
        )

        if missing:

            raise KeyError(
                f"{path.name} is missing fields:\n"
                f"  {sorted(missing)}\n\n"
                "The revised Distance/spatial method requires "
                "centroids_a and centroids_b. If these are absent, "
                "the Synthetic DEV generator must be updated before "
                "tuning."
            )

        planes_a = np.asarray(
            data["planes_a"],
            dtype=float,
        )

        planes_b = np.asarray(
            data["planes_b"],
            dtype=float,
        )

        centroids_a = np.asarray(
            data["centroids_a"],
            dtype=float,
        )

        centroids_b = np.asarray(
            data["centroids_b"],
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

    if (
        planes_a.ndim != 2
        or planes_a.shape[1] != 4
    ):

        raise ValueError(
            f"{path.name}: planes_a shape "
            f"{planes_a.shape}, expected (N,4)."
        )

    if (
        planes_b.ndim != 2
        or planes_b.shape[1] != 4
    ):

        raise ValueError(
            f"{path.name}: planes_b shape "
            f"{planes_b.shape}, expected (N,4)."
        )

    if (
        centroids_a.shape
        != (len(planes_a), 3)
    ):

        raise ValueError(
            f"{path.name}: centroids_a shape "
            f"{centroids_a.shape}, expected "
            f"({len(planes_a)},3)."
        )

    if (
        centroids_b.shape
        != (len(planes_b), 3)
    ):

        raise ValueError(
            f"{path.name}: centroids_b shape "
            f"{centroids_b.shape}, expected "
            f"({len(planes_b)},3)."
        )

    if (
        len(assigned_pairs)
        != len(gt_good)
    ):

        raise ValueError(
            f"{path.name}: assigned_pairs / "
            "gt_match_labels length mismatch."
        )

    if len(assigned_pairs) > 0:

        if (
            np.min(
                assigned_pairs[:, 0]
            )
            < 0
            or
            np.max(
                assigned_pairs[:, 0]
            )
            >= len(planes_a)
        ):

            raise IndexError(
                f"{path.name}: plane-A index out of range."
            )

        if (
            np.min(
                assigned_pairs[:, 1]
            )
            < 0
            or
            np.max(
                assigned_pairs[:, 1]
            )
            >= len(planes_b)
        ):

            raise IndexError(
                f"{path.name}: plane-B index out of range."
            )

    if not (
        np.all(
            np.isfinite(
                planes_a
            )
        )
        and
        np.all(
            np.isfinite(
                planes_b
            )
        )
        and
        np.all(
            np.isfinite(
                centroids_a
            )
        )
        and
        np.all(
            np.isfinite(
                centroids_b
            )
        )
    ):

        raise ValueError(
            f"{path.name}: non-finite geometry."
        )

    return {
        "name":
            path.name,

        "planes_a":
            planes_a,

        "planes_b":
            planes_b,

        "centroids_a":
            centroids_a,

        "centroids_b":
            centroids_b,

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


def load_and_audit_dev_cases(
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

    cases = []

    for index, path in enumerate(
        files,
        start=1,
    ):

        cases.append(
            load_dev_case(
                path
            )
        )

        if (
            index % 100 == 0
            or index == len(files)
        ):

            print(
                f"  audited "
                f"{index}/{len(files)}"
            )

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
            "Synthetic DEV contains no cases with bad matches."
        )

    print()
    print(
        "DEV DATA AUDIT PASSED"
    )

    print(
        f"  cases          : "
        f"{len(cases)}"
    )

    print(
        f"  clean cases    : "
        f"{n_clean}"
    )

    print(
        f"  non-clean cases: "
        f"{n_nonclean}"
    )

    return cases


# ============================================================
# Evaluate one parameter configuration
# ============================================================

def evaluate_configuration(
    cases: list[dict],
    runner,
) -> dict:

    nonclean_case_macro_f1 = []
    clean_case_retention = []

    pooled_tp = 0
    pooled_fp = 0
    pooled_fn = 0
    pooled_tn = 0

    completion_count = 0
    runtime_ms = []

    failure_count = 0

    for case in cases:

        start = (
            time.perf_counter()
        )

        try:

            result = runner(
                case
            )

            elapsed = (
                time.perf_counter()
                - start
            ) * 1000.0

            keep_mask = np.asarray(
                result[
                    "keep_mask"
                ],
                dtype=bool,
            ).reshape(-1)

            if (
                len(keep_mask)
                != len(
                    case[
                        "assigned_pairs"
                    ]
                )
            ):

                raise RuntimeError(
                    "keep_mask length mismatch"
                )

            completion_count += 1

        except Exception:

            elapsed = (
                time.perf_counter()
                - start
            ) * 1000.0

            failure_count += 1

            # A failed algorithm is penalised rather than silently
            # converted to an all-keep result during DEV tuning.
            keep_mask = np.zeros(
                len(
                    case[
                        "assigned_pairs"
                    ]
                ),
                dtype=bool,
            )

        runtime_ms.append(
            elapsed
        )

        (
            tp,
            fp,
            fn,
            tn,
        ) = classification_counts(
            keep_mask,
            case["gt_good"],
        )

        if case["is_clean"]:

            clean_case_retention.append(
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

            nonclean_case_macro_f1.append(
                macro_f1_from_counts(
                    tp,
                    fp,
                    fn,
                    tn,
                )
            )

    pooled_macro_f1 = (
        macro_f1_from_counts(
            pooled_tp,
            pooled_fp,
            pooled_fn,
            pooled_tn,
        )
    )

    pooled_bad_precision = (
        safe_divide(
            pooled_tp,
            pooled_tp
            + pooled_fp,
        )
    )

    pooled_bad_recall = (
        safe_divide(
            pooled_tp,
            pooled_tp
            + pooled_fn,
        )
    )

    pooled_correct_retention = (
        safe_divide(
            pooled_tn,
            pooled_tn
            + pooled_fp,
        )
    )

    return {
        "mean_case_macro_f1_nonclean":
            finite_mean(
                nonclean_case_macro_f1
            ),

        "pooled_macro_f1_nonclean":
            pooled_macro_f1,

        "bad_precision_nonclean":
            pooled_bad_precision,

        "bad_recall_nonclean":
            pooled_bad_recall,

        "correct_retention_nonclean":
            pooled_correct_retention,

        "clean_retention":
            (
                finite_mean(
                    clean_case_retention
                )
                if clean_case_retention
                else np.nan
            ),

        "completion_rate":
            safe_divide(
                completion_count,
                len(cases),
            ),

        "failure_count":
            failure_count,

        "runtime_mean_ms":
            finite_mean(
                runtime_ms
            ),

        "runtime_median_ms":
            finite_median(
                runtime_ms
            ),
    }


def selection_key(
    row: dict,
):
    """
    Larger tuple is better.

    Runtime is negated because lower is preferred.
    """

    clean_retention = row[
        "clean_retention"
    ]

    if not np.isfinite(
        clean_retention
    ):

        clean_retention = 1.0

    runtime = row[
        "runtime_median_ms"
    ]

    if not np.isfinite(
        runtime
    ):

        runtime = np.inf

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
            clean_retention
        ),
        float(
            row[
                "completion_rate"
            ]
        ),
        -float(runtime),
    )


# ============================================================
# Distance/spatial tuning
# ============================================================

def tune_distance(
    cases: list[dict],
) -> tuple[dict, list[dict]]:

    grid = list(
        itertools.product(
            DISTANCE_OFFSET_GRID_M,
            DISTANCE_SPATIAL_GRID_M,
        )
    )

    rows = []

    print()
    print("=" * 78)
    print("TUNING DISTANCE / SPATIAL")
    print("=" * 78)

    for index, (
        offset_threshold,
        spatial_threshold,
    ) in enumerate(
        grid,
        start=1,
    ):

        metrics = (
            evaluate_configuration(
                cases,
                runner=lambda case: (
                    distance.reject_bad_matches(
                        case["planes_a"],
                        case["planes_b"],
                        case[
                            "assigned_pairs"
                        ],
                        case[
                            "centroids_a"
                        ],
                        case[
                            "centroids_b"
                        ],
                        offset_threshold=
                            offset_threshold,
                        spatial_threshold=
                            spatial_threshold,
                    )
                ),
            )
        )

        row = {
            "method":
                "Distance",

            "offset_threshold_m":
                offset_threshold,

            "spatial_threshold_m":
                spatial_threshold,

            **metrics,
        }

        rows.append(
            row
        )

        print(
            f"[{index:02d}/{len(grid):02d}] "
            f"d={offset_threshold:.3f} "
            f"c={spatial_threshold:.3f} | "
            f"caseMacro="
            f"{metrics['mean_case_macro_f1_nonclean']:.4f} | "
            f"clean="
            f"{metrics['clean_retention']:.4f}"
        )

    best_row = max(
        rows,
        key=selection_key,
    )

    best_parameters = {
        "offset_threshold_m":
            float(
                best_row[
                    "offset_threshold_m"
                ]
            ),

        "spatial_threshold_m":
            float(
                best_row[
                    "spatial_threshold_m"
                ]
            ),
    }

    return (
        best_parameters,
        rows,
    )


# ============================================================
# SelMaP-inspired tuning
# ============================================================

def tune_selmap(
    cases: list[dict],
) -> tuple[dict, list[dict]]:

    grid = list(
        itertools.product(
            SELMAP_ANGLE_GRID_DEG,
            SELMAP_SEPARATION_GRID_M,
            SELMAP_COMPATIBLE_FRACTION_GRID,
        )
    )

    rows = []

    print()
    print("=" * 78)
    print("TUNING SELMAP-INSPIRED")
    print("=" * 78)

    for index, (
        angle_threshold,
        separation_threshold,
        min_fraction,
    ) in enumerate(
        grid,
        start=1,
    ):

        metrics = (
            evaluate_configuration(
                cases,
                runner=lambda case: (
                    selmap.reject_bad_matches(
                        case["planes_a"],
                        case["planes_b"],
                        case[
                            "assigned_pairs"
                        ],
                        angle_threshold_deg=
                            angle_threshold,
                        separation_threshold=
                            separation_threshold,
                        min_compatible_fraction=
                            min_fraction,
                    )
                ),
            )
        )

        row = {
            "method":
                "SelMaP-inspired",

            "angle_threshold_deg":
                angle_threshold,

            "separation_threshold_m":
                separation_threshold,

            "min_compatible_fraction":
                min_fraction,

            **metrics,
        }

        rows.append(
            row
        )

        print(
            f"[{index:02d}/{len(grid):02d}] "
            f"a={angle_threshold:.1f} "
            f"s={separation_threshold:.3f} "
            f"f={min_fraction:.2f} | "
            f"caseMacro="
            f"{metrics['mean_case_macro_f1_nonclean']:.4f} | "
            f"clean="
            f"{metrics['clean_retention']:.4f}"
        )

    best_row = max(
        rows,
        key=selection_key,
    )

    best_parameters = {
        "angle_threshold_deg":
            float(
                best_row[
                    "angle_threshold_deg"
                ]
            ),

        "separation_threshold_m":
            float(
                best_row[
                    "separation_threshold_m"
                ]
            ),

        "min_compatible_fraction":
            float(
                best_row[
                    "min_compatible_fraction"
                ]
            ),

        "parallel_threshold_deg":
            float(
                getattr(
                    selmap,
                    "PARALLEL_THRESHOLD_DEG",
                    10.0,
                )
            ),
    }

    return (
        best_parameters,
        rows,
    )


# ============================================================
# Preserve unchanged method parameters
# ============================================================

def current_default_parameters() -> dict:

    return {
        "Normal":
            {
                "normal_threshold_deg":
                    float(
                        normal.NORMAL_THRESHOLD_DEG
                    ),
            },

        "Distance":
            {},

        "RANSAC":
            {
                "normal_threshold_deg":
                    float(
                        ransac.NORMAL_THRESHOLD_DEG
                    ),

                "offset_threshold_m":
                    float(
                        ransac.OFFSET_THRESHOLD
                    ),

                "iterations":
                    int(
                        ransac.RANSAC_ITERATIONS
                    ),
            },

        "SelMaP-inspired":
            {},

        "Trilateration-inspired":
            {
                "cluster_radius_m":
                    float(
                        trilateration.CLUSTER_RADIUS
                    ),

                "min_support":
                    float(
                        trilateration.MIN_SUPPORT
                    ),

                "max_subsets":
                    int(
                        trilateration.MAX_SUBSETS
                    ),
            },
    }


def load_unchanged_parameters(
    old_parameter_file: Path,
) -> tuple[dict, str]:

    defaults = (
        current_default_parameters()
    )

    if not old_parameter_file.exists():

        return (
            defaults,
            "current_module_defaults",
        )

    try:

        with old_parameter_file.open(
            "r",
            encoding="utf-8",
        ) as f:

            old = json.load(f)

    except Exception:

        return (
            defaults,
            "current_module_defaults",
        )

    for method in (
        "Normal",
        "RANSAC",
        "Trilateration-inspired",
    ):

        if (
            method in old
            and isinstance(
                old[method],
                dict,
            )
        ):

            defaults[
                method
            ] = old[
                method
            ]

    return (
        defaults,
        str(
            old_parameter_file
        ),
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Revised Synthetic DEV tuning for "
            "Bad-Match Rejection."
        )
    )

    parser.add_argument(
        "--dev-dir",
        type=Path,
        default=DEFAULT_DEV_DIR,
    )

    parser.add_argument(
        "--old-parameter-file",
        type=Path,
        default=
            DEFAULT_OLD_PARAMETER_FILE,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=
            DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    print()
    print("=" * 78)
    print("BAD-MATCH REJECTION TUNING V2")
    print("=" * 78)

    print(
        "Changed methods only: "
        "Distance/spatial and SelMaP-inspired"
    )

    print(
        "FINAL/TUM data are not used."
    )

    print()

    # --------------------------------------------------------
    # Mandatory pre-run audit.
    # --------------------------------------------------------

    audit_algorithm_interfaces()

    cases = (
        load_and_audit_dev_cases(
            args.dev_dir
        )
    )

    # --------------------------------------------------------
    # Tune changed methods.
    # --------------------------------------------------------

    (
        distance_best,
        distance_rows,
    ) = tune_distance(
        cases
    )

    (
        selmap_best,
        selmap_rows,
    ) = tune_selmap(
        cases
    )

    # --------------------------------------------------------
    # Preserve unchanged frozen parameters.
    # --------------------------------------------------------

    (
        best_parameters,
        unchanged_source,
    ) = load_unchanged_parameters(
        args.old_parameter_file
    )

    best_parameters[
        "Distance"
    ] = distance_best

    best_parameters[
        "SelMaP-inspired"
    ] = selmap_best

    best_parameters[
        "_metadata"
    ] = {
        "version":
            "v2_corrected_distance_spatial_and_selmap",

        "tuned_on":
            "Synthetic DEV only",

        "changed_methods_retuned":
            [
                "Distance",
                "SelMaP-inspired",
            ],

        "unchanged_parameter_source":
            unchanged_source,

        "selection_policy":
            (
                "Lexicographic: mean per-case Macro-F1 on "
                "non-clean DEV, pooled Macro-F1, clean retention, "
                "completion rate, then lower median runtime."
            ),

        "distance_grid":
            {
                "offset_threshold_m":
                    DISTANCE_OFFSET_GRID_M,

                "spatial_threshold_m":
                    DISTANCE_SPATIAL_GRID_M,
            },

        "selmap_grid":
            {
                "angle_threshold_deg":
                    SELMAP_ANGLE_GRID_DEG,

                "separation_threshold_m":
                    SELMAP_SEPARATION_GRID_M,

                "min_compatible_fraction":
                    SELMAP_COMPATIBLE_FRACTION_GRID,
            },
    }

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    distance_file = (
        args.output_dir
        / "distance_grid_results.csv"
    )

    selmap_file = (
        args.output_dir
        / "selmap_grid_results.csv"
    )

    best_file = (
        args.output_dir
        / "best_parameters.json"
    )

    summary_file = (
        args.output_dir
        / "tuning_summary.json"
    )

    save_csv(
        distance_file,
        distance_rows,
    )

    save_csv(
        selmap_file,
        selmap_rows,
    )

    with best_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            best_parameters,
            f,
            indent=4,
        )

    best_distance_row = max(
        distance_rows,
        key=selection_key,
    )

    best_selmap_row = max(
        selmap_rows,
        key=selection_key,
    )

    tuning_summary = {
        "distance_best":
            best_distance_row,

        "selmap_best":
            best_selmap_row,
    }

    with summary_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            tuning_summary,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Terminal summary.
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("TUNING V2 COMPLETE")
    print("=" * 78)

    print()
    print("Distance/spatial:")

    print(
        f"  offset threshold  : "
        f"{distance_best['offset_threshold_m']:.3f} m"
    )

    print(
        f"  spatial threshold : "
        f"{distance_best['spatial_threshold_m']:.3f} m"
    )

    print(
        f"  DEV case Macro-F1 : "
        f"{best_distance_row['mean_case_macro_f1_nonclean']:.4f}"
    )

    print(
        f"  clean retention   : "
        f"{best_distance_row['clean_retention']:.4f}"
    )

    print()
    print("SelMaP-inspired:")

    print(
        f"  angle threshold   : "
        f"{selmap_best['angle_threshold_deg']:.1f} deg"
    )

    print(
        f"  separation thr.   : "
        f"{selmap_best['separation_threshold_m']:.3f} m"
    )

    print(
        f"  compatible frac.  : "
        f"{selmap_best['min_compatible_fraction']:.2f}"
    )

    print(
        f"  DEV case Macro-F1 : "
        f"{best_selmap_row['mean_case_macro_f1_nonclean']:.4f}"
    )

    print(
        f"  clean retention   : "
        f"{best_selmap_row['clean_retention']:.4f}"
    )

    print()
    print(
        f"Best parameters:\n"
        f"  {best_file}"
    )

    print(
        f"Distance grid:\n"
        f"  {distance_file}"
    )

    print(
        f"SelMaP grid:\n"
        f"  {selmap_file}"
    )


if __name__ == "__main__":
    main()
