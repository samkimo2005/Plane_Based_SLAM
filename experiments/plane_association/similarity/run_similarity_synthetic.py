"""
run_similarity_synthetic.py

Controlled FINAL experiment for Pairwise Plane Similarity
using the synthetic dataset.
"""

import csv
import json
import time
from pathlib import Path

from similarity_common import load_samples_jsonl

import descriptor_6d_similarity as descriptor_6d
import hia_tcd_similarity as hia_tcd
import planematch_similarity as planematch
import popup_weighted_similarity as popup_weighted
import geometric_weighted_similarity as geometric_weighted


# ============================================================
# Paths
# ============================================================

DATA_FILE = Path(
    "data/similarity/synthetic_samples.jsonl"
)

PARAM_FILE = Path(
    "results/similarity/frozen_params.json"
)

OUTPUT_FILE = Path(
    "results/similarity/synthetic_final.csv"
)


# ============================================================
# Methods
# ============================================================

METHODS = {
    "6D Descriptor": descriptor_6d.compute_score,
    "HIA-TCD": hia_tcd.compute_score,
    "PlaneMatch": planematch.compute_score,
    "Pop-up Weighted": popup_weighted.compute_score,
    "Geometric Weighted": geometric_weighted.compute_score,
}


# ============================================================
# Main
# ============================================================

def main():

    samples = load_samples_jsonl(DATA_FILE)

    # FINAL synthetic samples only
    samples = [
        s for s in samples
        if s.data_source == "synthetic"
        and s.split == "final"
    ]

    if not samples:
        raise RuntimeError(
            "No FINAL synthetic samples found."
        )

    with PARAM_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = json.load(f)

    rows = []

    print(
        f"Running Synthetic Similarity Benchmark "
        f"({len(samples)} samples)"
    )

    for method_name, method in METHODS.items():

        cfg = config[method_name]

        params = cfg.get("params", {})
        threshold = float(cfg["threshold"])

        print(f"  {method_name}...")

        for sample in samples:

            start = time.perf_counter()

            try:
                score = float(
                    method(
                        sample,
                        params=params,
                    )
                )

                runtime_ms = (
                    time.perf_counter() - start
                ) * 1000.0

                predicted = score >= threshold

                success = 1

            except Exception as exc:

                runtime_ms = (
                    time.perf_counter() - start
                ) * 1000.0

                score = float("nan")
                predicted = False
                success = 0

                print(
                    f"    Failed: "
                    f"{sample.sample_id}: {exc}"
                )

            rows.append(
                {
                    "sample_id":
                        sample.sample_id,

                    "method":
                        method_name,

                    "condition":
                        sample.condition,

                    "gt_match":
                        int(sample.gt_match),

                    "score":
                        score,

                    "threshold":
                        threshold,

                    "predicted_match":
                        int(predicted),

                    "runtime_ms":
                        runtime_ms,

                    "success":
                        success,
                }
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nFinished."
        f"\nResults: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()