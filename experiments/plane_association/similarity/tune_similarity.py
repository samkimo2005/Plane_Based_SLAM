"""
tune_similarity.py

Tune similarity thresholds on DEV synthetic samples and save
the frozen parameters for FINAL evaluation.
"""

import json
from pathlib import Path

import numpy as np

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

OUTPUT_FILE = Path(
    "results/similarity/frozen_params.json"
)


# ============================================================
# Methods and initial parameters
# ============================================================

METHODS = {
    "6D Descriptor": {
        "function": descriptor_6d.compute_score,
        "params": {
            "pos_scale": 0.5,
        },
    },

    "HIA-TCD": {
        "function": hia_tcd.compute_score,
        "params": {
            "bins": 12,
        },
    },

    "PlaneMatch": {
        "function": planematch.compute_score,
        "params": {},
    },

    "Pop-up Weighted": {
        "function": popup_weighted.compute_score,
        "params": {
            "w_normal": 0.50,
            "w_centroid": 0.30,
            "w_area": 0.20,
        },
    },

    "Geometric Weighted": {
        "function": geometric_weighted.compute_score,
        "params": {
            "w_normal": 0.40,
            "w_offset": 0.40,
            "w_centroid": 0.20,
        },
    },
}


# ============================================================
# Metrics
# ============================================================

def f1_score(gt, pred):
    gt = np.asarray(gt, dtype=bool)
    pred = np.asarray(pred, dtype=bool)

    tp = np.sum(gt & pred)
    fp = np.sum(~gt & pred)
    fn = np.sum(gt & ~pred)

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    if precision + recall == 0:
        return 0.0

    return (
        2.0
        * precision
        * recall
        / (precision + recall)
    )


# ============================================================
# Threshold search
# ============================================================

def find_best_threshold(scores, gt):
    scores = np.asarray(scores, dtype=float)
    gt = np.asarray(gt, dtype=bool)

    valid = np.isfinite(scores)

    scores = scores[valid]
    gt = gt[valid]

    if len(scores) == 0:
        raise RuntimeError(
            "No valid similarity scores available."
        )

    thresholds = np.linspace(
        scores.min(),
        scores.max(),
        200,
    )

    best_threshold = thresholds[0]
    best_f1 = -1.0

    for threshold in thresholds:
        pred = scores >= threshold

        score = f1_score(
            gt,
            pred,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return float(best_threshold), float(best_f1)


# ============================================================
# Main
# ============================================================

def main():

    samples = load_samples_jsonl(
        DATA_FILE
    )

    dev_samples = [
        s for s in samples
        if s.data_source == "synthetic"
        and s.split == "dev"
    ]

    if not dev_samples:
        raise RuntimeError(
            "No synthetic DEV samples found."
        )

    print(
        f"DEV samples: {len(dev_samples)}"
    )

    frozen = {}

    for method_name, method_data in METHODS.items():

        print(
            f"\nTuning {method_name}..."
        )

        function = method_data["function"]
        params = method_data["params"]

        scores = []
        gt = []

        for sample in dev_samples:

            try:
                score = function(
                    sample,
                    params=params,
                )

                scores.append(
                    float(score)
                )

                gt.append(
                    bool(sample.gt_match)
                )

            except Exception as exc:

                print(
                    f"  Failed "
                    f"{sample.sample_id}: {exc}"
                )

        threshold, best_f1 = (
            find_best_threshold(
                scores,
                gt,
            )
        )

        frozen[method_name] = {
            "threshold": threshold,
            "higher_is_better": True,
            "params": params,
            "dev_f1": best_f1,
        }

        print(
            f"  threshold = "
            f"{threshold:.4f}"
        )

        print(
            f"  DEV F1 = "
            f"{best_f1:.4f}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            frozen,
            f,
            indent=4,
        )

    print(
        f"\nFrozen parameters saved to:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()