from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

import normal_compatibility as normal
import distance_spatial_rejection as distance
import ransac_geometric_verification as ransac
import selmap_compatibility as selmap
import trilateration_rejection as trilateration

from bad_match_rejection_evaluation import (
    evaluate_result,
)


# ============================================================
# Paths
# ============================================================

HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent

DATA_DIR = (
    EXPERIMENTS_DIR
    / "data"
    / "bad_match_rejection"
    / "final"
)

MANIFEST_FILE = (
    DATA_DIR
    / "manifest.csv"
)

# Canonical corrected tuning file:
# - Distance restored to offset-only and re-tuned
# - SelMaP-inspired corrected and re-tuned
# - other methods unchanged
PARAMETER_FILE = (
    EXPERIMENTS_DIR
    / "results"
    / "bad_match_rejection"
    / "tuning_v2"
    / "best_parameters.json"
)

RESULT_DIR = (
    EXPERIMENTS_DIR
    / "results"
    / "bad_match_rejection"
    / "final_corrected"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    RESULT_DIR
    / "final_results.csv"
)

SUMMARY_CSV = (
    RESULT_DIR
    / "final_summary.csv"
)

PARAMETERS_USED_FILE = (
    RESULT_DIR
    / "parameters_used.json"
)


FINAL_RANSAC_SEED = 20261002


# ============================================================
# Load frozen parameters
# ============================================================

def load_frozen_parameters():

    if not PARAMETER_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{PARAMETER_FILE}"
        )

    with PARAMETER_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        best = json.load(f)

    # Normal
    normal.NORMAL_THRESHOLD_DEG = float(
        best["Normal"][
            "normal_threshold_deg"
        ]
    )

    # Distance: offset-only
    distance.OFFSET_THRESHOLD = float(
        best["Distance"][
            "offset_threshold_m"
        ]
    )

    # RANSAC
    ransac.NORMAL_THRESHOLD_DEG = float(
        best["RANSAC"][
            "normal_threshold_deg"
        ]
    )

    ransac.OFFSET_THRESHOLD = float(
        best["RANSAC"][
            "offset_threshold_m"
        ]
    )

    if "iterations" in best["RANSAC"]:
        ransac.RANSAC_ITERATIONS = int(
            best["RANSAC"][
                "iterations"
            ]
        )

    # SelMaP-inspired
    selmap.ANGLE_THRESHOLD_DEG = float(
        best["SelMaP-inspired"][
            "angle_threshold_deg"
        ]
    )

    selmap.SEPARATION_THRESHOLD = float(
        best["SelMaP-inspired"][
            "separation_threshold_m"
        ]
    )

    selmap.MIN_COMPATIBLE_FRACTION = float(
        best["SelMaP-inspired"][
            "min_compatible_fraction"
        ]
    )

    if (
        "parallel_threshold_deg"
        in best["SelMaP-inspired"]
    ):
        selmap.PARALLEL_THRESHOLD_DEG = float(
            best["SelMaP-inspired"][
                "parallel_threshold_deg"
            ]
        )

    # Trilateration-inspired
    trilateration.CLUSTER_RADIUS = float(
        best["Trilateration-inspired"][
            "cluster_radius_m"
        ]
    )

    trilateration.MIN_SUPPORT = float(
        best["Trilateration-inspired"][
            "min_support"
        ]
    )

    if (
        "max_subsets"
        in best["Trilateration-inspired"]
    ):
        trilateration.MAX_SUBSETS = int(
            best["Trilateration-inspired"][
                "max_subsets"
            ]
        )

    return best


# ============================================================
# Main
# ============================================================

def main():

    best = load_frozen_parameters()

    files = sorted(
        DATA_DIR.glob("sample_*.npz")
    )

    if not files:
        raise FileNotFoundError(
            f"No FINAL samples found in:\n{DATA_DIR}"
        )

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_FILE}"
        )

    manifest = pd.read_csv(
        MANIFEST_FILE
    )

    manifest_lookup = (
        manifest
        .set_index("sample")
        .to_dict("index")
    )

    print("=" * 76)
    print("BAD-MATCH REJECTION CORRECTED FINAL EVALUATION")
    print("=" * 76)

    print(
        f"FINAL samples: {len(files)}"
    )

    print()

    print("Frozen parameters:")

    print(
        "  Normal:",
        best["Normal"][
            "normal_threshold_deg"
        ],
        "deg",
    )

    print(
        "  Distance:",
        best["Distance"][
            "offset_threshold_m"
        ],
        "m offset",
    )

    print(
        "  RANSAC:",
        best["RANSAC"][
            "normal_threshold_deg"
        ],
        "deg /",
        best["RANSAC"][
            "offset_threshold_m"
        ],
        "m",
    )

    print(
        "  SelMaP-inspired:",
        best["SelMaP-inspired"][
            "angle_threshold_deg"
        ],
        "deg /",
        best["SelMaP-inspired"][
            "separation_threshold_m"
        ],
        "m /",
        best["SelMaP-inspired"][
            "min_compatible_fraction"
        ],
    )

    print(
        "  Trilateration-inspired:",
        best["Trilateration-inspired"][
            "cluster_radius_m"
        ],
        "m /",
        best["Trilateration-inspired"][
            "min_support"
        ],
    )

    print()
    print(
        "Parameters are frozen from Synthetic DEV. "
        "No FINAL tuning is performed."
    )
    print()

    rows = []

    for sample_index, file in enumerate(
        files
    ):

        data = np.load(file)

        planes_a = data[
            "planes_a"
        ]

        planes_b = data[
            "planes_b"
        ]

        assigned_pairs = data[
            "assigned_pairs"
        ]

        gt_match_labels = data[
            "gt_match_labels"
        ]

        R_gt = data[
            "R_gt"
        ]

        t_gt = data[
            "t_gt"
        ]

        metadata = (
            manifest_lookup.get(
                file.name,
                {},
            )
        )

        methods = {
            "Normal":
                lambda:
                    normal.reject_bad_matches(
                        planes_a,
                        planes_b,
                        assigned_pairs,
                        normal_threshold_deg=float(
                            best["Normal"][
                                "normal_threshold_deg"
                            ]
                        ),
                    ),

            "Distance":
                lambda:
                    distance.reject_bad_matches(
                        planes_a,
                        planes_b,
                        assigned_pairs,
                        offset_threshold=float(
                            best["Distance"][
                                "offset_threshold_m"
                            ]
                        ),
                    ),

            "RANSAC":
                lambda:
                    ransac.reject_bad_matches(
                        planes_a,
                        planes_b,
                        assigned_pairs,
                        rng=np.random.default_rng(
                            FINAL_RANSAC_SEED
                            + sample_index
                        ),
                    ),

            "SelMaP-inspired":
                lambda:
                    selmap.reject_bad_matches(
                        planes_a,
                        planes_b,
                        assigned_pairs,
                        angle_threshold_deg=float(
                            best["SelMaP-inspired"][
                                "angle_threshold_deg"
                            ]
                        ),
                        separation_threshold=float(
                            best["SelMaP-inspired"][
                                "separation_threshold_m"
                            ]
                        ),
                        min_compatible_fraction=float(
                            best["SelMaP-inspired"][
                                "min_compatible_fraction"
                            ]
                        ),
                    ),

            "Trilateration-inspired":
                lambda:
                    trilateration.reject_bad_matches(
                        planes_a,
                        planes_b,
                        assigned_pairs,
                    ),
        }

        for method_name, runner in methods.items():

            start = time.perf_counter()

            try:
                result = runner()

                runtime_ms = (
                    time.perf_counter()
                    - start
                ) * 1000.0

                keep_mask = np.asarray(
                    result["keep_mask"],
                    dtype=bool,
                )

                metrics = evaluate_result(
                    planes_a=planes_a,
                    planes_b=planes_b,
                    assigned_pairs=
                        assigned_pairs,
                    keep_mask=
                        keep_mask,
                    gt_match_labels=
                        gt_match_labels,
                    R_gt=
                        R_gt,
                    t_gt=
                        t_gt,
                )

                rows.append({
                    "sample":
                        file.name,

                    "method":
                        method_name,

                    **metadata,

                    "rejection_completed":
                        True,

                    "pose_usable":
                        bool(
                            result["success"]
                        ),

                    "runtime_ms":
                        runtime_ms,

                    "kept":
                        int(
                            keep_mask.sum()
                        ),

                    "rejected":
                        int(
                            (~keep_mask).sum()
                        ),

                    **metrics,

                    "error":
                        "",
                })

            except Exception as error:

                runtime_ms = (
                    time.perf_counter()
                    - start
                ) * 1000.0

                rows.append({
                    "sample":
                        file.name,

                    "method":
                        method_name,

                    **metadata,

                    "rejection_completed":
                        False,

                    "pose_usable":
                        False,

                    "runtime_ms":
                        runtime_ms,

                    "kept":
                        np.nan,

                    "rejected":
                        np.nan,

                    "error":
                        str(error),
                })

        if (
            (sample_index + 1) % 25 == 0
            or
            sample_index + 1
            == len(files)
        ):
            print(
                f"Processed "
                f"{sample_index + 1}"
                f"/{len(files)}"
            )

    # ========================================================
    # Save detailed results
    # ========================================================

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ========================================================
    # Summary
    # ========================================================

    summary_columns = [
        "precision",
        "recall",
        "f1",
        "correct_retention",
        "rotation_error_deg",
        "translation_error_m",
        "runtime_ms",
    ]

    summary = (
        df
        .groupby("method")[
            summary_columns
        ]
        .mean(
            numeric_only=True
        )
    )

    if "pose_success" in df.columns:

        pose_success = (
            df
            .groupby("method")[
                "pose_success"
            ]
            .mean()
            .rename(
                "pose_success_rate"
            )
        )

        summary = summary.join(
            pose_success
        )

    completion = (
        df
        .groupby("method")[
            "rejection_completed"
        ]
        .mean()
        .rename(
            "rejection_completion_rate"
        )
    )

    summary = summary.join(
        completion
    )

    summary.to_csv(
        SUMMARY_CSV
    )

    with PARAMETERS_USED_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "parameter_source":
                    str(
                        PARAMETER_FILE
                    ),

                "dataset":
                    str(
                        DATA_DIR
                    ),

                "final_ransac_seed":
                    FINAL_RANSAC_SEED,

                "frozen_parameters":
                    best,
            },
            f,
            indent=4,
        )

    print()
    print("=" * 76)
    print("CORRECTED FINAL EVALUATION COMPLETE")
    print("=" * 76)

    print(
        f"Rows generated: "
        f"{len(df)}"
    )

    print(
        f"Expected rows: "
        f"{len(files) * 5}"
    )

    print()
    print(summary)

    print()
    print(
        f"Detailed results:\n"
        f"{OUTPUT_CSV}"
    )

    print()
    print(
        f"Summary:\n"
        f"{SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()
