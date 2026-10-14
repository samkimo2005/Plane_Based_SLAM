from __future__ import annotations

from collections import defaultdict


# ============================================================
# 1. Basic helper
# ============================================================

def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """
    Divide safely.

    Returns 0 if the denominator is zero.
    """

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


# ============================================================
# 2. Build ground-truth correspondence set
# ============================================================

def build_gt_pair_set(
    case_data: dict,
) -> set:
    """
    Convert the benchmark GT correspondence list into:

        {
            ("A0", "B1"),
            ("A1", "B4"),
            ...
        }

    IMPORTANT:
        GT must only be used for evaluation.
        Similarity algorithms must never use it.
    """

    return {
        (
            match["a_obs_id"],
            match["b_obs_id"],
        )
        for match
        in case_data["gt_matches"]
    }


# ============================================================
# 3. Convert predicted pairs into a set
# ============================================================

def build_predicted_pair_set(
    predicted_pairs: list,
) -> set:
    """
    Expected prediction format:

        [
            {
                "a_obs_id": "A0",
                "b_obs_id": "B1",
                ...
            },
            ...
        ]

    Extra fields such as cost or geometry are ignored.
    """

    predicted_set = set()

    for pair in predicted_pairs:

        predicted_set.add(
            (
                pair["a_obs_id"],
                pair["b_obs_id"],
            )
        )

    return predicted_set


# ============================================================
# 4. Main pair-classification metrics
# ============================================================

def calculate_pair_metrics(
    gt_pairs: set,
    predicted_pairs: set,
) -> dict:
    """
    Definitions:

        TP:
            predicted candidate is a true physical
            correspondence

        FP:
            predicted candidate is not a true
            correspondence

        FN:
            a true correspondence was not retained
            as a candidate
    """

    true_positive_pairs = (
        gt_pairs
        & predicted_pairs
    )

    false_positive_pairs = (
        predicted_pairs
        - gt_pairs
    )

    false_negative_pairs = (
        gt_pairs
        - predicted_pairs
    )

    tp = len(
        true_positive_pairs
    )

    fp = len(
        false_positive_pairs
    )

    fn = len(
        false_negative_pairs
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    f1 = safe_divide(
        2.0 * precision * recall,
        precision + recall,
    )

    return {
        "tp":
            tp,

        "fp":
            fp,

        "fn":
            fn,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "true_positive_pairs":
            sorted(
                true_positive_pairs
            ),

        "false_positive_pairs":
            sorted(
                false_positive_pairs
            ),

        "false_negative_pairs":
            sorted(
                false_negative_pairs
            ),
    }


# ============================================================
# 5. Candidate-count structure
# ============================================================

def calculate_candidate_structure(
    case_data: dict,
    predicted_pairs: set,
    gt_pairs: set,
) -> dict:
    """
    Measure how many candidate B planes are retained
    for each Frame-A plane.

    This is important because Similarity is tested BEFORE
    one-to-one assignment.

    Therefore one A plane may legitimately have:

        A0 -> B1
        A0 -> B2
        A0 -> B3

    as three candidate pairs.
    """

    frame_a_planes = (
        case_data[
            "frame_a"
        ][
            "planes"
        ]
    )

    frame_b_planes = (
        case_data[
            "frame_b"
        ][
            "planes"
        ]
    )

    a_ids = [
        plane["obs_id"]
        for plane in frame_a_planes
    ]

    b_ids = [
        plane["obs_id"]
        for plane in frame_b_planes
    ]

    # --------------------------------------------------------
    # Count predicted candidates for every A plane
    # --------------------------------------------------------

    candidates_by_a = defaultdict(
        list
    )

    for a_id, b_id in predicted_pairs:

        candidates_by_a[
            a_id
        ].append(
            b_id
        )

    candidate_counts_by_a = {
        a_id: len(
            candidates_by_a[
                a_id
            ]
        )
        for a_id in a_ids
    }

    # --------------------------------------------------------
    # Determine which A planes really have a GT match
    # --------------------------------------------------------

    gt_matched_a_ids = {
        a_id
        for a_id, _
        in gt_pairs
    }

    all_a_ids = set(
        a_ids
    )

    gt_unmatched_a_ids = (
        all_a_ids
        - gt_matched_a_ids
    )

    # ========================================================
    # Overall candidate statistics
    # ========================================================

    total_candidates = len(
        predicted_pairs
    )

    total_possible_pairs = (
        len(a_ids)
        * len(b_ids)
    )

    mean_candidates_per_a = (
        safe_divide(
            total_candidates,
            len(a_ids),
        )
    )

    candidate_density = (
        safe_divide(
            total_candidates,
            total_possible_pairs,
        )
    )

    # ========================================================
    # Ambiguity
    # ========================================================

    ambiguous_a_ids = {
        a_id
        for a_id in a_ids
        if candidate_counts_by_a[
            a_id
        ] > 1
    }

    ambiguous_a_rate = (
        safe_divide(
            len(
                ambiguous_a_ids
            ),
            len(
                a_ids
            ),
        )
    )

    # More useful:
    # ambiguity specifically among planes that really
    # should have a correspondence.

    ambiguous_matched_a_ids = (
        ambiguous_a_ids
        & gt_matched_a_ids
    )

    ambiguous_matched_a_rate = (
        safe_divide(
            len(
                ambiguous_matched_a_ids
            ),
            len(
                gt_matched_a_ids
            ),
        )
    )

    # ========================================================
    # Matched-plane coverage
    # ========================================================

    # These planes SHOULD have at least one candidate
    # because a true correspondence exists.

    missed_matched_a_ids = {
        a_id
        for a_id in gt_matched_a_ids
        if candidate_counts_by_a[
            a_id
        ] == 0
    }

    missed_matched_a_rate = (
        safe_divide(
            len(
                missed_matched_a_ids
            ),
            len(
                gt_matched_a_ids
            ),
        )
    )

    # Average candidate count only for planes
    # that genuinely have correspondence.

    matched_candidate_total = sum(
        candidate_counts_by_a[
            a_id
        ]
        for a_id
        in gt_matched_a_ids
    )

    mean_candidates_per_matched_a = (
        safe_divide(
            matched_candidate_total,
            len(
                gt_matched_a_ids
            ),
        )
    )

    # ========================================================
    # Unmatched-plane rejection
    # ========================================================

    # These A planes have NO physical correspondence in B.
    #
    # Correct behaviour:
    #
    #     zero candidates

    correctly_rejected_unmatched_a_ids = {
        a_id
        for a_id in gt_unmatched_a_ids
        if candidate_counts_by_a[
            a_id
        ] == 0
    }

    false_candidate_unmatched_a_ids = {
        a_id
        for a_id in gt_unmatched_a_ids
        if candidate_counts_by_a[
            a_id
        ] > 0
    }

    unmatched_rejection_rate = (
        safe_divide(
            len(
                correctly_rejected_unmatched_a_ids
            ),
            len(
                gt_unmatched_a_ids
            ),
        )
    )

    unmatched_false_candidate_rate = (
        safe_divide(
            len(
                false_candidate_unmatched_a_ids
            ),
            len(
                gt_unmatched_a_ids
            ),
        )
    )

    return {
        # Dataset structure
        "num_a_planes":
            len(
                a_ids
            ),

        "num_b_planes":
            len(
                b_ids
            ),

        "num_possible_pairs":
            total_possible_pairs,

        "num_gt_matched_a":
            len(
                gt_matched_a_ids
            ),

        "num_gt_unmatched_a":
            len(
                gt_unmatched_a_ids
            ),

        # Candidate quantity
        "num_candidates":
            total_candidates,

        "mean_candidates_per_a":
            mean_candidates_per_a,

        "mean_candidates_per_matched_a":
            mean_candidates_per_matched_a,

        "candidate_density":
            candidate_density,

        # Ambiguity
        "ambiguous_a_count":
            len(
                ambiguous_a_ids
            ),

        "ambiguous_a_rate":
            ambiguous_a_rate,

        "ambiguous_matched_a_count":
            len(
                ambiguous_matched_a_ids
            ),

        "ambiguous_matched_a_rate":
            ambiguous_matched_a_rate,

        # Missed matched planes
        "missed_matched_a_count":
            len(
                missed_matched_a_ids
            ),

        "missed_matched_a_rate":
            missed_matched_a_rate,

        # Unmatched-plane rejection
        "correctly_rejected_unmatched_a_count":
            len(
                correctly_rejected_unmatched_a_ids
            ),

        "false_candidate_unmatched_a_count":
            len(
                false_candidate_unmatched_a_ids
            ),

        "unmatched_rejection_rate":
            unmatched_rejection_rate,

        "unmatched_false_candidate_rate":
            unmatched_false_candidate_rate,

        # Diagnostic lists
        "candidate_counts_by_a":
            candidate_counts_by_a,

        "ambiguous_a_ids":
            sorted(
                ambiguous_a_ids
            ),

        "ambiguous_matched_a_ids":
            sorted(
                ambiguous_matched_a_ids
            ),

        "missed_matched_a_ids":
            sorted(
                missed_matched_a_ids
            ),

        "correctly_rejected_unmatched_a_ids":
            sorted(
                correctly_rejected_unmatched_a_ids
            ),

        "false_candidate_unmatched_a_ids":
            sorted(
                false_candidate_unmatched_a_ids
            ),
    }


# ============================================================
# 6. Complete candidate-similarity evaluation
# ============================================================

def evaluate_similarity_candidates(
    case_data: dict,
    candidate_pairs: list,
) -> dict:
    """
    Common evaluator for pairwise similarity methods.

    Intended for:

        1. Weighted Geometric Cost
        2. Hard Geometric Thresholds
        3. Descriptor-Based Similarity

    It evaluates the candidate set BEFORE assignment.
    """

    gt_pairs = build_gt_pair_set(
        case_data
    )

    predicted_pairs = (
        build_predicted_pair_set(
            candidate_pairs
        )
    )

    classification = (
        calculate_pair_metrics(
            gt_pairs,
            predicted_pairs,
        )
    )

    structure = (
        calculate_candidate_structure(
            case_data,
            predicted_pairs,
            gt_pairs,
        )
    )

    exact_candidate_set = (
        predicted_pairs
        == gt_pairs
    )

    return {
        "sample_id":
            case_data[
                "sample_id"
            ],

        "scene_type":
            case_data[
                "scene_type"
            ],

        "prior_level":
            case_data[
                "prior_level"
            ],

        "num_gt_matches":
            len(
                gt_pairs
            ),

        "exact_candidate_set":
            bool(
                exact_candidate_set
            ),

        **classification,

        **structure,
    }


# ============================================================
# 7. Generic correspondence evaluator
# ============================================================

def evaluate_predicted_pairs(
    case_data: dict,
    predicted_pairs: list,
) -> dict:
    """
    Generic pair evaluator.

    This is useful later for the fourth method:

        Joint Association / Pose Optimisation

    because the joint method may return FINAL correspondences
    rather than an intermediate candidate set.

    For that reason, this function reports only TP/FP/FN,
    Precision, Recall and F1, without candidate ambiguity
    statistics.
    """

    gt_pairs = build_gt_pair_set(
        case_data
    )

    predicted_set = (
        build_predicted_pair_set(
            predicted_pairs
        )
    )

    metrics = (
        calculate_pair_metrics(
            gt_pairs,
            predicted_set,
        )
    )

    return {
        "sample_id":
            case_data[
                "sample_id"
            ],

        "scene_type":
            case_data[
                "scene_type"
            ],

        "prior_level":
            case_data[
                "prior_level"
            ],

        "num_gt_matches":
            len(
                gt_pairs
            ),

        "num_predicted_pairs":
            len(
                predicted_set
            ),

        "exact_correspondence_set":
            bool(
                predicted_set
                == gt_pairs
            ),

        **metrics,
    }


# ============================================================
# 8. Optional helper for parameter tuning
# ============================================================

def annotate_pair_results_with_gt(
    case_data: dict,
    pair_results: list,
) -> list:
    """
    Add:

        is_gt_match

    to every pair result.

    Useful later in:
        tune_similarity_parameters.py

    This function must NOT be used by the similarity algorithm
    itself. It is strictly for offline DEV-set tuning/evaluation.
    """

    gt_pairs = build_gt_pair_set(
        case_data
    )

    annotated = []

    for row in pair_results:

        new_row = dict(
            row
        )

        pair = (
            row["a_obs_id"],
            row["b_obs_id"],
        )

        new_row[
            "is_gt_match"
        ] = (
            pair
            in gt_pairs
        )

        annotated.append(
            new_row
        )

    return annotated