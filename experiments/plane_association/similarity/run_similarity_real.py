"""
run_similarity_real.py

Run the five Pairwise Plane Similarity methods on the
real TUM/ETH3D-derived candidate pairs.

Important:
- Uses the same frozen parameters and thresholds selected on Synthetic DEV.
- No threshold tuning is performed on real data.
- gt_label = -1 samples are still processed, but must later be
  excluded from Precision / Recall / F1 calculation.
"""

import csv
import json
import time
from pathlib import Path

import numpy as np

from similarity_common import load_samples_jsonl

import descriptor_6d_similarity
import hia_tcd_similarity
import planematch_similarity
import popup_weighted_similarity
import geometric_weighted_similarity


# ============================================================
# Paths
# ============================================================

DATA_FILE = Path(
    "data/similarity/real_samples.jsonl"
)

PARAM_FILE = Path(
    "results/similarity/frozen_params.json"
)

OUTPUT_FILE = Path(
    "results/similarity/real_final.csv"
)


# ============================================================
# Methods
# ============================================================

METHODS = {
    "6D Descriptor":
        descriptor_6d_similarity.compute_score,

    "HIA-TCD":
        hia_tcd_similarity.compute_score,

    "PlaneMatch":
        planematch_similarity.compute_score,

    "Pop-up Weighted":
        popup_weighted_similarity.compute_score,

    "Geometric Weighted":
        geometric_weighted_similarity.compute_score,
}


# ============================================================
# Load frozen configuration
# ============================================================

def load_frozen_parameters():

    if not PARAM_FILE.exists():
        raise FileNotFoundError(
            f"Frozen parameter file not found:\n"
            f"{PARAM_FILE}"
        )

    with PARAM_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load real samples
    # --------------------------------------------------------

    samples = load_samples_jsonl(
        DATA_FILE
    )

    samples = [
        sample
        for sample in samples
        if sample.data_source == "real"
    ]

    if not samples:
        raise RuntimeError(
            "No real similarity samples found."
        )

    # --------------------------------------------------------
    # Frozen Synthetic-DEV parameters
    # --------------------------------------------------------

    frozen = load_frozen_parameters()

    output_rows = []

    print()
    print(
        "=== Real Pairwise Similarity Test ==="
    )

    print(
        f"Input pairs : {len(samples)}"
    )

    print()

    # --------------------------------------------------------
    # Run all five methods on exactly the same samples
    # --------------------------------------------------------

    for method_name, function in METHODS.items():

        if method_name not in frozen:
            raise KeyError(
                f"No frozen parameters found for "
                f"{method_name}"
            )

        config = frozen[
            method_name
        ]

        threshold = float(
            config["threshold"]
        )

        params = config.get(
            "params",
            {}
        )

        successful = 0
        failed = 0

        runtimes = []

        for sample in samples:

            gt_label = int(
                sample.metadata.get(
                    "gt_label",
                    -1,
                )
            )

            try:

                start = (
                    time.perf_counter()
                )

                score = function(
                    sample,
                    params=params,
                )

                runtime_ms = (
                    time.perf_counter()
                    - start
                ) * 1000.0

                score = float(score)

                if not np.isfinite(score):
                    raise ValueError(
                        "Non-finite similarity score."
                    )

                predicted_match = int(
                    score >= threshold
                )

                success = 1
                successful += 1

                runtimes.append(
                    runtime_ms
                )

            except Exception as exc:

                score = np.nan
                runtime_ms = np.nan
                predicted_match = 0
                success = 0

                failed += 1

                print(
                    f"[FAIL] "
                    f"{method_name} | "
                    f"{sample.sample_id} | "
                    f"{exc}"
                )

            output_rows.append(
                {
                    "sample_id":
                        sample.sample_id,

                    "pair_id":
                        sample.metadata.get(
                            "pair_id",
                            -1,
                        ),

                    "current_frame_id":
                        sample.current_frame_id,

                    "reference_frame_id":
                        sample.reference_frame_id,

                    "method":
                        method_name,

                    "gt_label":
                        gt_label,

                    "score":
                        score,

                    "threshold":
                        threshold,

                    "predicted_match":
                        predicted_match,

                    "runtime_ms":
                        runtime_ms,

                    "success":
                        success,
                }
            )

        mean_runtime = (
            np.mean(runtimes)
            if runtimes
            else np.nan
        )

        print(
            f"{method_name:<20} "
            f"success={successful:3d}/{len(samples):3d}  "
            f"failed={failed:3d}  "
            f"runtime={mean_runtime:.3f} ms"
        )

    # ========================================================
    # Save
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "pair_id",
                "current_frame_id",
                "reference_frame_id",
                "method",
                "gt_label",
                "score",
                "threshold",
                "predicted_match",
                "runtime_ms",
                "success",
            ],
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    print()
    print(
        f"Results saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()