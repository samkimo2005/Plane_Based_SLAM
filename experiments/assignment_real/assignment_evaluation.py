"""
assignment_evaluation.py

Common evaluation functions for the one-to-one plane correspondence
Assignment experiments.

All Assignment algorithms are evaluated using exactly the same metrics.

Primary metrics:
    - correspondence precision
    - correspondence recall
    - correspondence F1
    - unmatched-plane F1
    - runtime

Supporting metrics:
    - exact assignment
    - unmatched precision / recall
    - false forced-match rate (FFMR)
    - mean selected-match similarity
"""

from __future__ import annotations

from typing import Any

import numpy as np

from assignment_common import (
    AssignmentCase,
    AssignmentResult,
    get_gt_unmatched_current,
    get_gt_unmatched_reference,
    validate_assignment_case,
)


# ============================================================
# Basic metric helpers
# ============================================================

def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """
    Safe division returning 0 when the denominator is zero.
    """

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def precision_recall_f1(
    tp: int,
    fp: int,
    fn: int,
) -> tuple[float, float, float]:
    """
    Compute precision, recall and F1.
    """

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2.0
            * precision
            * recall
            / (precision + recall)
        )

    return (
        precision,
        recall,
        f1,
    )


# ============================================================
# Result validation
# ============================================================

def validate_assignment_result(
    case: AssignmentCase,
    result: AssignmentResult,
) -> None:
    """
    Validate an AssignmentResult before evaluation.

    Checks:
        - correct case ID
        - one-to-one correspondence constraint
        - valid plane indices
        - matches belong to the candidate graph
        - unmatched lists are consistent with selected matches
    """

    validate_assignment_case(case)

    if result.case_id != case.case_id:
        raise ValueError(
            f"Result case_id '{result.case_id}' "
            f"does not match case '{case.case_id}'"
        )

    matched_current: list[int] = []
    matched_reference: list[int] = []

    for i, j in result.matches:

        if not (0 <= i < case.n_current):
            raise ValueError(
                f"{case.case_id}: invalid current index {i}"
            )

        if not (0 <= j < case.n_reference):
            raise ValueError(
                f"{case.case_id}: invalid reference index {j}"
            )

        if not case.candidate_mask[i, j]:
            raise ValueError(
                f"{case.case_id}: result contains "
                f"non-candidate pair ({i}, {j})"
            )

        matched_current.append(i)
        matched_reference.append(j)

    # One-to-one requirement
    if len(matched_current) != len(set(matched_current)):
        raise ValueError(
            f"{case.case_id}: a current plane is assigned "
            f"more than once"
        )

    if len(matched_reference) != len(set(matched_reference)):
        raise ValueError(
            f"{case.case_id}: a reference plane is assigned "
            f"more than once"
        )

    expected_unmatched_current = (
        set(range(case.n_current))
        - set(matched_current)
    )

    expected_unmatched_reference = (
        set(range(case.n_reference))
        - set(matched_reference)
    )

    if set(result.unmatched_current) != expected_unmatched_current:
        raise ValueError(
            f"{case.case_id}: unmatched_current is inconsistent "
            f"with returned matches"
        )

    if set(result.unmatched_reference) != expected_unmatched_reference:
        raise ValueError(
            f"{case.case_id}: unmatched_reference is inconsistent "
            f"with returned matches"
        )

    if result.runtime_ms < 0:
        raise ValueError(
            f"{case.case_id}: runtime_ms cannot be negative"
        )


# ============================================================
# Correspondence evaluation
# ============================================================

def evaluate_correspondences(
    case: AssignmentCase,
    result: AssignmentResult,
) -> dict[str, Any]:
    """
    Evaluate selected real plane correspondences against GT.
    """

    predicted = set(result.matches)
    gt = set(case.gt_matches)

    tp_set = predicted & gt
    fp_set = predicted - gt
    fn_set = gt - predicted

    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)

    precision, recall, f1 = precision_recall_f1(
        tp=tp,
        fp=fp,
        fn=fn,
    )

    exact_assignment = int(
        predicted == gt
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_assignment": exact_assignment,
    }


# ============================================================
# Unmatched-plane evaluation
# ============================================================

def _combined_unmatched_labels(
    current_indices: set[int],
    reference_indices: set[int],
) -> set[tuple[str, int]]:
    """
    Represent unmatched planes from both frames in one label set.

    Example:
        ('current', 2)
        ('reference', 5)
    """

    labels: set[tuple[str, int]] = set()

    for i in current_indices:
        labels.add(
            ("current", int(i))
        )

    for j in reference_indices:
        labels.add(
            ("reference", int(j))
        )

    return labels


def evaluate_unmatched_planes(
    case: AssignmentCase,
    result: AssignmentResult,
) -> dict[str, Any]:
    """
    Evaluate the ability to leave planes unmatched.

    Current and reference planes are treated as separate labelled
    entities and pooled into one unmatched classification problem.
    """

    gt_current = get_gt_unmatched_current(case)
    gt_reference = get_gt_unmatched_reference(case)

    pred_current = set(
        result.unmatched_current
    )

    pred_reference = set(
        result.unmatched_reference
    )

    gt_labels = _combined_unmatched_labels(
        gt_current,
        gt_reference,
    )

    pred_labels = _combined_unmatched_labels(
        pred_current,
        pred_reference,
    )

    tp = len(
        gt_labels & pred_labels
    )

    fp = len(
        pred_labels - gt_labels
    )

    fn = len(
        gt_labels - pred_labels
    )

    # Special case:
    # if neither GT nor prediction contains unmatched planes,
    # the unmatched decision is perfectly correct.
    if len(gt_labels) == 0 and len(pred_labels) == 0:
        precision = 1.0
        recall = 1.0
        f1 = 1.0

    else:
        precision, recall, f1 = precision_recall_f1(
            tp=tp,
            fp=fp,
            fn=fn,
        )

    return {
        "unmatched_tp": tp,
        "unmatched_fp": fp,
        "unmatched_fn": fn,
        "unmatched_precision": precision,
        "unmatched_recall": recall,
        "unmatched_f1": f1,
        "n_gt_unmatched_current": len(
            gt_current
        ),
        "n_gt_unmatched_reference": len(
            gt_reference
        ),
        "n_pred_unmatched_current": len(
            pred_current
        ),
        "n_pred_unmatched_reference": len(
            pred_reference
        ),
    }


# ============================================================
# False forced-match rate
# ============================================================

def calculate_ffmr(
    case: AssignmentCase,
    result: AssignmentResult,
) -> float:
    """
    Calculate False Forced-Match Rate (FFMR).

    FFMR measures the proportion of GT-unmatched plane endpoints that
    were incorrectly forced into a real correspondence.

        FFMR =
            incorrectly matched GT-unmatched planes
            ---------------------------------------
            total GT-unmatched planes

    Lower is better.

    Returns 0 if the case contains no GT-unmatched planes.
    """

    gt_unmatched_current = get_gt_unmatched_current(
        case
    )

    gt_unmatched_reference = get_gt_unmatched_reference(
        case
    )

    total_gt_unmatched = (
        len(gt_unmatched_current)
        + len(gt_unmatched_reference)
    )

    if total_gt_unmatched == 0:
        return 0.0

    matched_current = {
        i
        for i, _ in result.matches
    }

    matched_reference = {
        j
        for _, j in result.matches
    }

    forced_current = (
        gt_unmatched_current
        & matched_current
    )

    forced_reference = (
        gt_unmatched_reference
        & matched_reference
    )

    n_false_forced = (
        len(forced_current)
        + len(forced_reference)
    )

    return safe_divide(
        n_false_forced,
        total_gt_unmatched,
    )


# ============================================================
# Similarity statistics
# ============================================================

def calculate_selected_similarity(
    case: AssignmentCase,
    result: AssignmentResult,
) -> dict[str, float]:
    """
    Calculate similarity statistics for selected real matches.
    """

    if len(result.matches) == 0:
        return {
            "mean_selected_similarity": 0.0,
            "min_selected_similarity": 0.0,
            "max_selected_similarity": 0.0,
        }

    scores = np.asarray(
        [
            case.similarity[i, j]
            for i, j in result.matches
        ],
        dtype=np.float64,
    )

    return {
        "mean_selected_similarity":
            float(np.mean(scores)),

        "min_selected_similarity":
            float(np.min(scores)),

        "max_selected_similarity":
            float(np.max(scores)),
    }


# ============================================================
# Complete evaluation
# ============================================================

def evaluate_assignment_result(
    case: AssignmentCase,
    result: AssignmentResult,
) -> dict[str, Any]:
    """
    Evaluate one algorithm result on one AssignmentCase.

    Returns one flat dictionary suitable for saving directly to CSV.
    """

    validate_assignment_result(
        case,
        result,
    )

    correspondence_metrics = (
        evaluate_correspondences(
            case,
            result,
        )
    )

    unmatched_metrics = (
        evaluate_unmatched_planes(
            case,
            result,
        )
    )

    similarity_metrics = (
        calculate_selected_similarity(
            case,
            result,
        )
    )

    ffmr = calculate_ffmr(
        case,
        result,
    )

    return {
        # ----------------------------------------------------
        # Identification
        # ----------------------------------------------------
        "case_id":
            case.case_id,

        "algorithm":
            result.algorithm,

        "scenario":
            case.metadata.get(
                "scenario",
                "unknown",
            ),

        "split":
            case.metadata.get(
                "split",
                "unknown",
            ),

        # ----------------------------------------------------
        # Problem size
        # ----------------------------------------------------
        "n_current":
            case.n_current,

        "n_reference":
            case.n_reference,

        "n_candidates":
            int(
                np.count_nonzero(
                    case.candidate_mask
                )
            ),

        "n_gt_matches":
            len(case.gt_matches),

        "n_pred_matches":
            len(result.matches),

        # ----------------------------------------------------
        # Correspondence accuracy
        # ----------------------------------------------------
        **correspondence_metrics,

        # ----------------------------------------------------
        # Unmatched handling
        # ----------------------------------------------------
        **unmatched_metrics,

        "ffmr":
            ffmr,

        # ----------------------------------------------------
        # Runtime
        # ----------------------------------------------------
        "runtime_ms":
            float(result.runtime_ms),

        # ----------------------------------------------------
        # Supporting similarity statistics
        # ----------------------------------------------------
        **similarity_metrics,
    }


# ============================================================
# Aggregate result summaries
# ============================================================

def aggregate_evaluation_rows(
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Aggregate a collection of per-case evaluation rows.

    Intended for:
        - one algorithm
        - one scenario or full dataset split

    Primary values are macro-averaged across cases.
    """

    if not rows:
        raise ValueError(
            "Cannot aggregate an empty result list"
        )

    def values(key: str) -> np.ndarray:
        return np.asarray(
            [
                float(row[key])
                for row in rows
            ],
            dtype=np.float64,
        )

    f1_values = values("f1")
    precision_values = values("precision")
    recall_values = values("recall")

    unmatched_f1_values = values(
        "unmatched_f1"
    )

    runtime_values = values(
        "runtime_ms"
    )

    exact_values = values(
        "exact_assignment"
    )

    ffmr_values = values(
        "ffmr"
    )

    return {
        "n_cases":
            len(rows),

        "precision_mean":
            float(
                np.mean(precision_values)
            ),

        "recall_mean":
            float(
                np.mean(recall_values)
            ),

        "f1_mean":
            float(
                np.mean(f1_values)
            ),

        "f1_std":
            float(
                np.std(f1_values)
            ),

        "exact_assignment_rate":
            float(
                np.mean(exact_values)
            ),

        "unmatched_f1_mean":
            float(
                np.mean(unmatched_f1_values)
            ),

        "ffmr_mean":
            float(
                np.mean(ffmr_values)
            ),

        "runtime_mean_ms":
            float(
                np.mean(runtime_values)
            ),

        "runtime_median_ms":
            float(
                np.median(runtime_values)
            ),

        "runtime_p95_ms":
            float(
                np.percentile(
                    runtime_values,
                    95,
                )
            ),
    }