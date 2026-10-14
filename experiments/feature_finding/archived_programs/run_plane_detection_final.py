from __future__ import annotations

import csv
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

import run_multiplane_plane_detection_benchmark as mp


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic_multiplane_benchmark_final"
)

TUNING_FILE = (
    PROJECT_ROOT
    / "results"
    / "plane_detection"
    / "tuning"
    / "best_detection_parameters.json"
)

FINAL_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "plane_detection"
    / "final"
)

FINAL_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    FINAL_RESULTS_DIR
    / "multiplane_detection_final_results.csv"
)

PARAMETER_SNAPSHOT_FILE = (
    FINAL_RESULTS_DIR
    / "frozen_detection_parameters_used.json"
)

EXPECTED_FINAL_CASES = 60

DO_WARMUP = True


# ============================================================
# 2. Load the frozen DEV-selected configuration
# ============================================================

def load_frozen_configuration():
    if not TUNING_FILE.exists():
        raise FileNotFoundError(
            "Frozen parameter file not found:\n"
            f"{TUNING_FILE}\n\n"
            "Run tune_plane_detection_parameters.py first."
        )

    with open(
        TUNING_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        tuning_data = json.load(
            f
        )

    configuration = tuning_data[
        "full_frozen_multiplane_configuration"
    ]

    return (
        configuration,
        tuning_data,
    )


def apply_frozen_configuration(
    configuration,
):
    """
    Apply values only in this Python process.

    The original source file is not rewritten.
    """

    for name, value in configuration.items():
        if not hasattr(
            mp,
            name,
        ):
            raise AttributeError(
                "Frozen configuration contains parameter "
                f"'{name}', but the imported benchmark module "
                "does not contain that variable."
            )

        setattr(
            mp,
            name,
            deepcopy(
                value
            ),
        )


# ============================================================
# 3. Load FINAL samples
# ============================================================

def load_final_sample_files():
    sample_files = sorted(
        FINAL_DATA_DIR.glob(
            "P*_N*_O*_R*.npz"
        )
    )

    if len(
        sample_files
    ) != EXPECTED_FINAL_CASES:
        raise RuntimeError(
            f"Expected {EXPECTED_FINAL_CASES} FINAL samples, "
            f"found {len(sample_files)} in:\n"
            f"{FINAL_DATA_DIR}\n\n"
            "Run generate_plane_detection_final.py first."
        )

    return sample_files


# ============================================================
# 4. Save an exact parameter snapshot
# ============================================================

def save_parameter_snapshot(
    configuration,
    original_tuning_data,
):
    snapshot = {
        "dataset_split":
            "FINAL",

        "source_parameter_file":
            str(
                TUNING_FILE
            ),

        "tuning_dataset":
            original_tuning_data.get(
                "tuning_dataset"
            ),

        "dev_repeat_ids":
            original_tuning_data.get(
                "dev_repeat_ids"
            ),

        "frozen_parameters_used":
            configuration,
    }

    with open(
        PARAMETER_SNAPSHOT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            snapshot,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# 5. Main FINAL benchmark
# ============================================================

def main():
    print(
        "=" * 80
    )

    print(
        "Plane Detection FINAL Multi-Plane Benchmark"
    )

    print(
        "=" * 80
    )

    print()
    print(
        "This program uses ONLY:"
    )

    print(
        "  1. the frozen DEV-selected parameter JSON"
    )

    print(
        "  2. the unseen FINAL dataset"
    )

    print()
    print(
        "FINAL results must not be used for retuning."
    )

    (
        frozen_configuration,
        original_tuning_data,
    ) = load_frozen_configuration()

    apply_frozen_configuration(
        frozen_configuration
    )

    save_parameter_snapshot(
        frozen_configuration,
        original_tuning_data,
    )

    sample_files = (
        load_final_sample_files()
    )

    print()
    print(
        "Frozen parameter file:"
    )

    print(
        TUNING_FILE
    )

    print()
    print(
        "FINAL samples:",
        len(
            sample_files
        )
    )

    # --------------------------------------------------------
    # Untimed warm-up
    # --------------------------------------------------------

    if DO_WARMUP:
        print()
        print(
            "Running one un-timed warm-up per method..."
        )

        (
            warm_points,
            _,
            _,
            _,
        ) = mp.load_sample(
            sample_files[
                0
            ]
        )

        for method_index, (
            method_name,
            detector,
        ) in enumerate(
            mp.METHODS.items()
        ):
            rng = np.random.default_rng(
                9_900_000
                + method_index
            )

            detector(
                warm_points,
                rng,
            )

            print(
                "  warm-up:",
                method_name
            )

    # --------------------------------------------------------
    # FINAL runs
    # --------------------------------------------------------

    rows = []

    total_runs = (
        len(
            sample_files
        )
        * len(
            mp.METHODS
        )
    )

    run_number = 0

    for npz_file in sample_files:
        (
            points,
            labels,
            metadata,
            gt_planes,
        ) = mp.load_sample(
            npz_file
        )

        if metadata.get(
            "dataset_split"
        ) != "FINAL":
            raise RuntimeError(
                f"{npz_file.name} is not marked as FINAL."
            )

        for method_index, (
            method_name,
            detector,
        ) in enumerate(
            mp.METHODS.items()
        ):
            run_number += 1

            # Keep the same algorithm-RNG convention as the
            # original benchmark, but use each FINAL sample's
            # new dataset seed.
            algorithm_seed = (
                int(
                    metadata[
                        "seed"
                    ]
                )
                + method_index
                * 100000
            )

            rng = np.random.default_rng(
                algorithm_seed
            )

            start = time.perf_counter()

            models = detector(
                points,
                rng,
            )

            end = time.perf_counter()

            runtime_ms = (
                end
                - start
            ) * 1000.0

            evaluation = mp.match_planes(
                models,
                gt_planes,
                labels,
            )

            gt_count = len(
                gt_planes
            )

            detected_count = len(
                models
            )

            matched_count = evaluation[
                "matched_count"
            ]

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
                "dataset_split":
                    "FINAL",

                "method":
                    method_name,

                "sample_id":
                    metadata[
                        "sample_id"
                    ],

                "plane_count_gt":
                    gt_count,

                "noise_sigma_m":
                    metadata[
                        "noise_sigma_m"
                    ],

                "outlier_ratio":
                    metadata[
                        "outlier_ratio"
                    ],

                "repeat_id":
                    metadata[
                        "repeat_id"
                    ],

                "dataset_seed":
                    metadata[
                        "seed"
                    ],

                "algorithm_seed":
                    int(
                        algorithm_seed
                    ),

                "num_points":
                    len(
                        points
                    ),

                "detected_plane_count":
                    detected_count,

                "matched_plane_count":
                    matched_count,

                "false_plane_count":
                    false_plane_count,

                "missed_plane_count":
                    missed_plane_count,

                "correct_plane_count":
                    correct_plane_count,

                "plane_precision":
                    evaluation[
                        "plane_precision"
                    ],

                "plane_recall":
                    evaluation[
                        "plane_recall"
                    ],

                "plane_f1":
                    evaluation[
                        "plane_f1"
                    ],

                "matched_mean_iou":
                    evaluation[
                        "mean_iou"
                    ],

                "mean_normal_error_deg":
                    (
                        ""
                        if np.isnan(
                            evaluation[
                                "mean_normal_error_deg"
                            ]
                        )
                        else evaluation[
                            "mean_normal_error_deg"
                        ]
                    ),

                "mean_offset_error_m":
                    (
                        ""
                        if np.isnan(
                            evaluation[
                                "mean_offset_error_m"
                            ]
                        )
                        else evaluation[
                            "mean_offset_error_m"
                        ]
                    ),

                "runtime_ms":
                    runtime_ms,

                "success":
                    int(
                        matched_count
                        > 0
                    ),
            }

            rows.append(
                row
            )

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

    expected_rows = (
        EXPECTED_FINAL_CASES
        * len(
            mp.METHODS
        )
    )

    if len(
        rows
    ) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} result rows, "
            f"got {len(rows)}."
        )

    fieldnames = list(
        rows[
            0
        ].keys()
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print()
    print(
        "=" * 80
    )

    print(
        "FINAL BENCHMARK COMPLETE"
    )

    print(
        "=" * 80
    )

    print()
    print(
        "Result rows:",
        len(
            rows
        )
    )

    print()
    print(
        "Results:"
    )

    print(
        OUTPUT_CSV
    )

    print()
    print(
        "Frozen parameter snapshot:"
    )

    print(
        PARAMETER_SNAPSHOT_FILE
    )

    print()
    print(
        "Do not use these FINAL results to change "
        "the detector parameters."
    )


if __name__ == "__main__":
    main()
