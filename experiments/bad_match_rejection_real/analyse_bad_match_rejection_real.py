"""Evaluate RealSense Bad-Match Rejection outputs against held-out Surface-ID labels.

This is the ONLY one of the three scripts that reads GT. The positive class
for Precision/Recall/F1 is BAD correspondence (rejection desired):
TP = bad rejected; FP = good rejected; FN = bad retained; TN = good retained.

Two scoring modes are emitted:
- operational_with_fallback: structural failure means keep-all, as deployed;
- valid_decisions_only: scores only cases where an algorithm made a decision.

A clean (0-bad) baseline has UNDEFINED bad-rejection recall/F1, not 100%.
Do not pool baseline, controlled corruption, and exploratory real outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def project_root() -> Path:
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for parent in (start, *start.parents):
            if parent.name.lower() == "experiments":
                return parent
            if (parent / "experiments").is_dir():
                return parent / "experiments"
    return Path(__file__).resolve().parent


def safe_div(numer: int, denom: int) -> float:
    return float(numer / denom) if denom else float("nan")


def score_correspondences(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {name: np.nan for name in (
            "tp", "fp", "fn", "tn", "gt_bad", "gt_good", "predicted_rejected",
            "bad_precision", "bad_recall", "bad_f1", "correct_precision",
            "correct_recall", "correct_f1", "correct_retention",
            "false_rejection_rate", "macro_f1", "overall_accuracy")}
    gt_good = df["gt_match_label"].astype(int).to_numpy() == 1
    keep = df["effective_keep"].astype(bool).to_numpy()
    bad, rejected = ~gt_good, ~keep
    tp = int(np.sum(bad & rejected))
    fp = int(np.sum(gt_good & rejected))
    fn = int(np.sum(bad & keep))
    tn = int(np.sum(gt_good & keep))
    n_bad, n_good = int(np.sum(bad)), int(np.sum(gt_good))
    bad_f1 = safe_div(2 * tp, 2 * tp + fp + fn) if n_bad else np.nan
    correct_f1 = safe_div(2 * tn, 2 * tn + fn + fp) if n_good else np.nan
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "gt_bad": n_bad, "gt_good": n_good, "predicted_rejected": int(np.sum(rejected)),
        "bad_precision": safe_div(tp, tp + fp),
        "bad_recall": safe_div(tp, tp + fn),
        "bad_f1": bad_f1,
        "correct_precision": safe_div(tn, tn + fn),
        "correct_recall": safe_div(tn, tn + fp),
        "correct_f1": correct_f1,
        "correct_retention": safe_div(tn, tn + fp),
        "false_rejection_rate": safe_div(fp, tn + fp),
        "macro_f1": (bad_f1 + correct_f1) / 2 if n_bad and n_good else np.nan,
        "overall_accuracy": safe_div(tp + tn, tp + fp + fn + tn),
    }


def audit_and_join(decisions: pd.DataFrame, labels: pd.DataFrame,
                   cases: pd.DataFrame) -> pd.DataFrame:
    d_required = {"prepared_case_id", "pair_index", "scenario", "method",
                  "source_plane_id", "target_plane_id", "effective_keep",
                  "decision_available", "fallback_used"}
    l_required = {"prepared_case_id", "pair_index", "gt_match_label",
                  "source_plane_id", "target_plane_id", "source_surface_id", "target_surface_id"}
    c_required = {"prepared_case_id", "scenario", "method", "decision_available",
                  "fallback_used", "runtime_median_ms", "pose_usable"}
    for title, df, required in (("decisions", decisions, d_required),
                                ("GT labels", labels, l_required),
                                ("case results", cases, c_required)):
        if missing := required - set(df.columns):
            raise ValueError(f"{title} missing {sorted(missing)}")
    keys = ["prepared_case_id", "pair_index"]
    if labels.duplicated(keys).any():
        raise ValueError("Duplicate GT label keys")
    if decisions.duplicated(keys + ["method"]).any():
        raise ValueError("Duplicate algorithm decision keys")
    if cases.duplicated(["prepared_case_id", "method"]).any():
        raise ValueError("Duplicate method-case keys")
    merged = decisions.merge(
        labels, on=keys, how="left", suffixes=("", "_gt"),
        validate="many_to_one", indicator=True
    )
    if (merged._merge != "both").any():
        bad = merged[merged._merge != "both"]
        raise ValueError(f"Missing GT for {len(bad)} method-correspondence decisions")
    for col in ("source_plane_id", "target_plane_id"):
        if not (merged[col].astype(int) == merged[f"{col}_gt"].astype(int)).all():
            raise ValueError(f"GT key mismatch on {col}; check input/version mixing")
    if not merged.gt_match_label.astype(int).isin([0, 1]).all():
        raise ValueError("GT label must be 0 (bad) or 1 (correct)")
    surface_good = (merged.source_surface_id == merged.target_surface_id).astype(int)
    if not (merged.gt_match_label.astype(int) == surface_good).all():
        raise ValueError("GT labels disagree with stored Surface ID pairs")
    # Cast explicitly: pandas might otherwise interpret text 'False' as truthy.
    for col in ("effective_keep", "decision_available", "fallback_used"):
        if merged[col].dtype != bool:
            normalized = merged[col].astype(str).str.strip().str.lower()
            if not normalized.isin(["true", "false", "1", "0"]).all():
                raise ValueError(f"Unexpected boolean values in {col}")
            merged[col] = normalized.isin(["true", "1"])
    if ((~merged.decision_available) & (~merged.effective_keep)).any():
        raise ValueError("A structurally failed case rejected correspondences despite keep-all fallback")
    for col in ("decision_available", "fallback_used", "pose_usable"):
        if cases[col].dtype != bool:
            values = cases[col].astype(str).str.strip().str.lower()
            if not values.isin(["true", "false", "1", "0"]).all():
                raise ValueError(f"Invalid case-level boolean {col}")
            cases[col] = values.isin(["true", "1"])
    return merged.drop(columns="_merge")


def make_case_metrics(joined: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in cases.itertuples(index=False):
        group = joined[(joined.prepared_case_id == rec.prepared_case_id) &
                       (joined.method == rec.method)]
        available = bool(rec.decision_available)
        for scoring_mode in ("operational_with_fallback", "valid_decisions_only"):
            sample = group if scoring_mode == "operational_with_fallback" or available else group.iloc[:0]
            rows.append({
                "prepared_case_id": rec.prepared_case_id,
                "original_case_id": rec.original_case_id,
                "scenario": rec.scenario, "source_frame": int(rec.source_frame),
                "target_frame": int(rec.target_frame), "method": rec.method,
                "scoring_mode": scoring_mode,
                "n_scored_matches": len(sample),
                "decision_available": available,
                "fallback_used": bool(rec.fallback_used),
                "pose_usable": bool(rec.pose_usable),
                "runtime_median_ms": float(pd.to_numeric(rec.runtime_median_ms, errors="coerce")),
                "failure_reason": getattr(rec, "failure_reason", ""),
                **score_correspondences(sample),
            })
    return pd.DataFrame(rows)


def summarise(joined: pd.DataFrame, cases: pd.DataFrame, extra_group: list[str] | None = None):
    dimensions = ["scenario", "method"] + (extra_group or [])
    rows = []
    for group_key, subcases in cases.groupby(dimensions, dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        selection = np.ones(len(joined), dtype=bool)
        for column, value in zip(dimensions, group_key):
            selection &= joined[column].to_numpy() == value
        matching = joined.loc[selection]
        for mode in ("operational_with_fallback", "valid_decisions_only"):
            data = matching if mode == "operational_with_fallback" else matching[matching.decision_available]
            latency = pd.to_numeric(subcases.runtime_median_ms, errors="coerce").to_numpy(float)
            latency = latency[np.isfinite(latency)]
            row = {**dict(zip(dimensions, group_key)), "scoring_mode": mode,
                   "cases_total": len(subcases),
                   "cases_with_decision": int(subcases.decision_available.astype(bool).sum()),
                   "fallback_cases": int(subcases.fallback_used.astype(bool).sum()),
                   "pose_usable_cases": int(subcases.pose_usable.astype(bool).sum()),
                   "cases_scored": int(data.prepared_case_id.nunique()),
                   "matches_scored": len(data),
                   "runtime_mean_ms": float(np.mean(latency)) if len(latency) else np.nan,
                   "runtime_median_ms": float(np.median(latency)) if len(latency) else np.nan,
                   "runtime_p95_ms": float(np.percentile(latency, 95)) if len(latency) else np.nan,
                   **score_correspondences(data)}
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=root / "results" / "bad_match_rejection" / "real")
    parser.add_argument("--labels", type=Path,
                        default=root / "data" / "bad_match_rejection" / "real"
                                / "labels_EVALUATION_ONLY.csv")
    args = parser.parse_args()
    results = args.results_dir.expanduser().resolve()
    labels_file = args.labels.expanduser().resolve()
    case_file = results / "case_results.csv"
    decision_file = results / "match_decisions.csv"
    for path in (labels_file, case_file, decision_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    cases = pd.read_csv(case_file, keep_default_na=False)
    decisions = pd.read_csv(decision_file)
    labels = pd.read_csv(labels_file)
    joined = audit_and_join(decisions, labels, cases)
    if len(joined) == 0:
        raise ValueError("There are no decisions to score")
    case_metrics = make_case_metrics(joined, cases)
    summary = summarise(joined, cases)
    by_frame_pair = summarise(joined, cases, ["source_frame", "target_frame"])
    # Detail audit only: each GT-negative record and the chosen decision.
    bad_details = joined.loc[
        joined.gt_match_label.astype(int).eq(0),
        ["prepared_case_id", "original_case_id", "scenario", "method",
         "pair_index", "source_frame", "target_frame", "source_plane_id",
         "target_plane_id", "source_surface_id", "target_surface_id",
         "decision_available", "fallback_used", "effective_keep"]
    ].copy()
    bad_details["bad_rejected"] = ~bad_details.effective_keep.astype(bool)
    outputs = {
        "real_case_metrics.csv": case_metrics,
        "real_summary.csv": summary,
        "real_by_frame_pair.csv": by_frame_pair,
        "real_known_bad_decisions.csv": bad_details,
    }
    for name, frame in outputs.items():
        frame.to_csv(results / name, index=False)
    notes = (
        "SCORING CONVENTION\n"
        "GT bad (different manual physical Surface IDs) is the positive class.\n"
        "TP=bad rejected; FP=good rejected; FN=bad retained; TN=good retained.\n"
        "Undefined Precision/Recall/F1 metrics are NaN (e.g. no GT negatives).\n"
        "operational_with_fallback: keep upstream matches on structural failure.\n"
        "valid_decisions_only: exclude cases with no usable rejection decision.\n"
        "Do not compare methods solely on valid_decisions_only without checking fallback counts.\n"
        "Baseline, controlled_corruption and exploratory are SEPARATE scenarios.\n"
        "Controlled corruption uses real measured planes but artificially permuted targets.\n"
        "Exploratory configuration cases reuse the SAME four underlying frame pairs;\n"
        "the 282 earlier associations are NOT 282 independent physical observations.\n"
        "Plane patch over-segmentation means Surface IDs do not establish patch identity.\n"
        "No ground-truth camera poses are available in these prepared RealSense records;\n"
        "estimated motion diagnostics in case_results.csv are NOT pose errors.\n"
    )
    (results / "evaluation_notes.txt").write_text(notes, encoding="utf-8")
    print("Evaluation complete. No algorithm was re-run.")
    print(f"Files saved under {results}:")
    for filename in outputs:
        print("  ", filename)
    print("\nOperational scoring by scenario:")
    print(summary[summary.scoring_mode == "operational_with_fallback"][[
        "scenario", "method", "cases_total", "fallback_cases",
        "matches_scored", "gt_bad", "gt_good", "bad_recall",
        "correct_retention", "runtime_median_ms",
    ]].to_string(index=False))
    print("\nIMPORTANT: Baseline bad-match recall is undefined with zero GT negatives.")


if __name__ == "__main__":
    main()
